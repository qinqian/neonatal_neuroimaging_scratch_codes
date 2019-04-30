library(future)
require(ANTsR)
require(ANTsRNet)
require(tensorflow)
require(keras)
require(jsonlite)
library(reticulate)
library(magick)
library(abind)
library(rhdf5)
#require(fslr)
#library(extrantsr)
library(ggplot2)
library(doMC)
library(foreach)
args <- commandArgs(T)
model = args[1]
#model = 'dense'
#model = 'vgg'

seed = args[2]

##seed = 999
set.seed(seed)
mask = antsImageRead('/data1/qinqian/MRI//data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalAtlas2/brainmask.nii.gz')
mask = splitNDImageToList(mask)[[17]]

all = read.table('all_pvl_images.txt', stringsAsFactors = F)

trainProportion = 0.7*0.7

subindex = sample(1:nrow(all), nrow(all)*trainProportion, replace = F)

validProportion = 0.21
valindex <- sample((1:nrow(all))[-subindex], nrow(all)*validProportion, replace=F)

testindex <- (1:nrow(all))[-c(subindex, valindex)]
labelso <- all[,ncol(all)]

print(table(labelso[subindex]))
print(table(labelso[valindex]))
print(table(labelso[testindex]))

labels <- to_categorical(labelso)
#quit()

registerDoMC(4)
training_generator <- function(sample_index, batch_size, aug=F) {
    imgs.cur <- all[sample_index, ]
    label.cur <- labels[sample_index, ]
    imgs.next <- all[sample_index, ]
    label.next <- labels[sample_index,]
    function() {
        ## start new epoch
        if (nrow(imgs.cur) < batch_size) {
            imgs.cur <<- imgs.next
            label.cur <<- label.next
        }
        batch_ind <- sample(1:nrow(imgs.cur), batch_size, replace=F) # shuffle current batch
        batch_images_list <- imgs.cur[batch_ind, ]
        imgs.cur <<- imgs.cur[-batch_ind, ]
        batch_label_list <- label.cur[batch_ind,]
        label.cur <<- label.cur[-batch_ind,]
        ## data augmentation by random inverse transformation to each of the inviduals
        x_y.batch <- foreach(i=1:batch_size) %do% {
            load_one_modal <- function(j) {
                if (aug) {
                    x = imageFileNames2ImageList(batch_images_list[j,1:4])
                    for (z in seq_along(x)) {
                    if (runif(1)<=0.3) {
                        if (runif(1)<=0.5) {
                            rand = randomImageTransformAugmentation( x[[z]], list(list(x[[z]])), list(x[[z]]), n=1, typeOfTransform='AffineAndDeformation')
                            x[[z]] = rand$outputPredictorList[[1]][[1]]
                        } else {
                            if (runif(1) <= 0.5) {
                            x[[z]] = x[[z]] + array(rnorm(prod(dim(x[[z]]))), dim=dim(x[[z]]))
                            }
                        }
                    }}
                } else {
                    x = imageFileNames2ImageList(batch_images_list[j,1:4])
                }
                y <- lapply(x, function(y0) {
                    # 107x144x111
                     z = iMath(cropImage(y0, mask), 'Normalize')
                    ## z = as.array(cropIndices(z, c(20, 20, 20), c(99, 99, 99)))
                    #z = as.array(resampleImage(y0, c(100, 100, 100), useVoxels=1))
                    #z = as.array(y0)[10:99, 10:99, 10:99]
                    -(z-mean(z))/sd(z)
                })
                y <- abind(y, along=4) # 1 * x * y * z * channels
                y
            }
            #x0 = future({ load_one_modal(i) }) %plan% multiprocess
            x0 = load_one_modal(i)
            #list(x0=value(x0), y=batch_label_list[i, ])
            list(x0=x0, y=batch_label_list[i, ])
        }
        x_y.batch <- purrr::transpose(x_y.batch)
        x.batch <- do.call(abind, c(x_y.batch$x0, list(along=0))) # batch * x * y * z * channels
        y.batch <-  do.call(rbind, x_y.batch$y)
        result <- list(keras_array(x.batch), keras_array(y.batch))
    }
}

source('models.R')
backend()$clear_session()

if (model == 'vgg') {
vggModel <- createVggModel3D(#c(117, 159, 126, 2), ## image size x modality # 107x144x111
                             #c(100, 100, 100, 2),
                             c(80, 80, 80, 1),
                             #c(107, 144, 111, 5),
                            numberOfClassificationLabels = 2,
                            layers = c(1),
                            lowestResolution = 8,
                            convolutionKernelSize = c(3, 3, 3),
                            poolSize = c( 2, 2, 2 ),
                            strides = c( 2, 2, 2 ),
                            denseUnits = 10,   #  10
                            dropoutRate = 0.2,   # 0.2
                            weightDecay = 0.001,
                            style = 19)
} 
if (model == 'dense') {
vggModel <- createDenseNetModel3D(#c(117, 159, 126, 1),
                                 c(80, 80, 80, 1),
                                 numberOfClassificationLabels = 2,
                                 numberOfFilters = 8,
                                 depth = 7,
                                 numberOfDenseBlocks = 2,
                                 growthRate = 12,
                                 dropoutRate = 0.2,
                                 weightDecay = 0.001
                                 )
}

if (model == 'res') {
vggModel = createResNetModel3D(c(107, 144, 111,4), numberOfClassificationLabels=2, layers=1:2, residualBlockSchedule=c(3,4), lowestResolution = 64, cardinality=16)
#vggModel = createResNetModel3D(c(90, 90, 90, 2), numberOfClassificationLabels=2, layers=1:2, residualBlockSchedule=c(3,4), lowestResolution = 64, cardinality=16)
}

print(vggModel)
##tensorboard("logs/run_a")
## lr = args[1]

lr = 0.0001

print(lr)
vggModel %>% compile(
  loss = 'binary_crossentropy',
  #optimizer = optimizer_rmsprop(lr=0.0001),
  optimizer = optimizer_adam(lr = as.double(lr)),
  metrics=c('binary_crossentropy', 'accuracy'))

## batch = args[5]
batch.size = 4

train.iterator <- py_iterator(training_generator(subindex, batch_size=batch.size, aug=F)) ## augmented dataset
valid.iterator <- py_iterator(training_generator(valindex, batch_size=batch.size)) # no augmentation
test.iterator <- py_iterator(training_generator(testindex, batch_size=batch.size))

## schedule = args[4]
schedule = 's'
lr_schedule <- function(epoch, lr) {
   if (epoch %% 3 == 0) {
       lr = lr / 3
       if (lr < 1e-7) {
          lr = 1e-7
       }
   }
   lr
}

if (schedule == 's') {
   lr_callback=callback_learning_rate_scheduler(lr_schedule)
} else {
   lr_callback=callback_reduce_lr_on_plateau(monitor = "val_loss",
                                             factor = 0.1,
                                             patience = 10,
                                             min_delta = 1e-3,
                                             verbose = 1)
}

## prefix = paste0(args[3], args[2], args[1], args[4], args[5])
prefix = paste0(model, '_', schedule, '_', seed, '_', lr, '_', batch.size)
write.table(all[subindex,], sep='\t', file=paste0(prefix, '_split_traindata.txt'), quote=F)
write.table(all[testindex,], sep='\t', file=paste0(prefix, '_split_testdata.txt'), quote=F)
write.table(all[valindex,], sep='\t', file=paste0(prefix, '_split_valdata.txt'), quote=F)

callbacks_list <- list(
        callback_tensorboard(paste0(prefix, "_logs_r")),
        #callback_early_stopping(monitor = "val_loss",
        #                        min_delta = 1e-4,
        #                        patience = 3,
        #                        verbose = 1),
        lr_callback,
        callback_model_checkpoint(filepath = paste0("weights_r_", prefix, "/", prefix, "_{epoch:0004d}.h5"),
                                  monitor = "val_loss",
                                  save_best_only = TRUE,
                                  save_weights_only = TRUE))


system('mkdir -p weights_r_aug')
track = vggModel %>% fit_generator(
                         train.iterator,
                         #steps_per_epoch=500, # may be too long
                         steps_per_epoch=10,
                         epochs=50,
                         callbacks=callbacks_list,
                         validation_data=valid.iterator,
                         validation_steps=length(valindex)/batch.size,
                         verbose=1)

print(vggModel %>% evaluate_generator(test.iterator, steps=length(testindex)/batch.size))
pdf(paste0(prefix, 'aug.pdf'))
g=plot(track)
print(g)
dev.off()

df = as.data.frame(track)
write.table(df, file=paste0(prefix, '_performance.txt'), quote=F, sep='\t')

## Save the model
save_model_weights_hdf5(
  vggModel, filepath = paste0( './', prefix, '_Weights.h5' ) )
#save_model_hdf5(
#  vggModel, filepath = paste0( './', prefix, '_Model.h5' ), overwrite = TRUE )

