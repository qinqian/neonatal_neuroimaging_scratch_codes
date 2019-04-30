library(future)
imgs.warp3.f <- future({ read.table("augmented_list.txt", stringsAsFactors=F) }) %plan% multiprocess

library(tfestimators)
require(tensorflow)
require(keras)
require(ANTsR)
require(jsonlite)
library(reticulate)
library(magick)
library(abind)
library(rhdf5)
require(fslr)
library(extrantsr)
library(ggplot2)
library(doMC)
library(foreach)

args <- commandArgs(T)

workspace <- "/home/qinqian/data1/MRI/01_PVL_AI"
downsample <- c(64, 64, 19)
#downsample <- c(32, 32, 19)
setwd(workspace)

loadMeta <- function() {
    pvl <- read.table("output_pvl/participants.tsv", header=T, stringsAsFactors = F)
    pvl[,4] <- 'pvl'
    pvl <- pvl[file.exists(paste0('output_pvl/', pvl[,1], '/anat/', pvl[,1], "_run-2_T2w.nii.gz")),]
    pvl[,1] <- paste0('output_pvl/', pvl[,1])
    cont <- read.table("output_control/participants.tsv", header=T, sep='\t', stringsAsFactors = F)
    cont <- cont[!(cont[,1]%in%basename(pvl[,1])), ]
    cont <- cont[file.exists(paste0('output_control/', cont[,1], '/anat/', cont[,1], "_run-2_T2w.nii.gz")), ]
    cont[,1] <- paste0('output_control/', cont[,1])
    meta <- rbind(pvl, cont)
    group = meta[,4]
    labels = rep(0, length(group))
    labels[group=='pvl'] = 1
    meta[,4] = labels
    meta
}

imagesRead <- function(subject.dir) {
    load.img <- function(x, modal='t2') {
        if (modal=='t2') {
            protocal1 <- read_json(paste0(x, ".json"))$ProtocolName
            modal.1 <- antsImageRead(paste0(x, ".nii.gz"))
            modal.1s <- resampleImage(modal.1, interpType=4,
                                     useVoxels=T, resampleParams=downsample)
        } else {
            protocal1 <- read_json(paste0(x, ".json"))$ProtocolName
            if (length(grep("sag", protocal1, ignore.case=T))>=1) {
                protocal1 <- read_json(paste0(gsub('run-1', 'run-2', x), ".json"))$ProtocolName
                modal.1 <- antsImageRead(paste0(gsub('run-1', 'run-2', x), ".nii.gz"))
            } else {
                modal.1 <- antsImageRead(paste0(x, ".nii.gz"))
            }
            modal.1s <- resampleImage(modal.1, interpType=4,
                                      useVoxels=T, resampleParams=downsample)
        }
    }
    load.subject <- function(x) {
        n <- basename(x[1])
        t21 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T2w"))
        t22 <- load.img(paste0(x[1], "/", "anat/", n, "_run-2_T2w"))
        t1 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T1w"), 't1')

        x1 <- paste0("training_data/", n, "_t21.nii.gz")
        x2 <- paste0("training_data/", n, "_t22.nii.gz")
        x3 <- paste0("training_data/", n, "_t1.nii.gz")

        antsImageWrite(t21, x1)
        antsImageWrite(t22, x2)
        antsImageWrite(t1, x3)
        #createAntsrTransform(type='AffineTransform', precision="float", dimension=3)
        ##return(abind(a, along=3)) ## height x width x (z * modality)
        return(c(x1, x2, x3))
    }
    load.subject(subject.dir)
}

meta <- loadMeta()

system("mkdir -p training_data")

trainProportion = 0.6
set.seed(1234)
#set.seed( 9999 )
#set.seed( 1234 )
#set.seed( 6666 )
#set.seed( 1111 )
subindex = sample(1:nrow(meta), nrow(meta)*trainProportion, replace = F)

validProportion = 0.2
valindex <- sample((1:nrow(meta))[-subindex], nrow(meta)*validProportion, replace=F)

testindex <- (1:nrow(meta))[-c(subindex, valindex)]

#registerDoMC(8)
#imgs.list <- foreach(i=subindex, .combine=rbind) %dopar% {
#    print(i)
#    imagesRead(as.character(meta[i,]))
#}

#imgs.list[,1] <- basename(imgs.list[,1])
#imgs.list[,2] <- basename(imgs.list[,2])
#imgs.list[,3] <- basename(imgs.list[,3])

## write.table(imgs.list, file="template.list", quote=F, sep=",", row.names=F, col.names=F)
## system("bash test_template.sh")

## oa=createAntsrTransform(matrix=matrix(rnorm(9), nrow=3), dimension=3,translation=c(3,1,2))
## img2 = applyAntsrTransformToImage(a, imgs.arr[[1]], imgs.arr[[2]])
## matrix(getAntsrTransformParameters(a), nrow=3)


labels <- h5read("augmented_images.h5", 'pvl/labels')

c(m, s) %<-% read.table("mean_sd.txt")

imgs.warp3 = value(imgs.warp3.f)

registerDoMC(8)
training_generator <- function(sample_index, batch_size) {
    imgs.cur <- imgs.warp3[sample_index, ]
    label.cur <- labels[sample_index,]
    imgs.next <- imgs.warp3[sample_index, ]
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
        x_y.batch <- foreach(i=1:batch_size) %dopar% {
            load_one_modal <- function(j) {
                x = antsImageRead(batch_images_list[i, j+3])
                x.t = antsImageRead(batch_images_list[i, j])
                x.inv <- antsApplyTransforms(fixed=x, moving=x.t,
                                              transformlist = c(batch_images_list[i, 9], batch_images_list[i, 10]),
                                              whichtoinvert=c(T, F))
                x.inv <- as.array(iMath(x.inv, "Normalize"))
                x.inv <- (x.inv-m)/s
            }
            
            x0.inv = future({ load_one_modal(3) }) %plan% multiprocess
            x1.inv = future({ load_one_modal(4) }) %plan% multiprocess
            x2.inv = future({ load_one_modal(5) }) %plan% multiprocess
            list(x0=value(x0.inv), x1=value(x1.inv), x2=value(x2.inv), y=batch_label_list[i, ])
        }

        x_y.batch <- purrr::transpose(x_y.batch)
        x0 <- do.call(abind, c(x_y.batch$x0, list(along=0)))
        x1 <- do.call(abind, c(x_y.batch$x1, list(along=0)))
        x2 <- do.call(abind, c(x_y.batch$x2, list(along=0)))
        
        x.batch <- abind(list(x0, x1, x2), along=5) # batch * x * y * z * channels
        y.batch <-  do.call(rbind, x_y.batch$y)
        result <- list(keras_array(x.batch), keras_array(y.batch))
    }
}

valid_generator <- function(sample_index, batch_size) {
    imgs.cur <- imgs.warp3[sample_index, ]
    label.cur <- labels[sample_index,]
    imgs.next <- imgs.warp3[sample_index, ]
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

        x_y.batch <- foreach(i=1:batch_size) %dopar% {
            ## no data augmentation
            load_one_modal <- function(j) {
                x = antsImageRead(batch_images_list[i, j+3])
                x <- as.array(iMath(x, "Normalize"))
                x <- (x-m)/s
            }
            
            x0 = future({ load_one_modal(3) }) %plan% multiprocess
            x1 = future({ load_one_modal(4) }) %plan% multiprocess
            x2 = future({ load_one_modal(5) }) %plan% multiprocess
            list(x0=value(x0), x1=value(x1), x2=value(x2), y=batch_label_list[i, ])
        }

        x_y.batch <- purrr::transpose(x_y.batch)
        x0 <- do.call(abind, c(x_y.batch$x0, list(along=0)))
        x1 <- do.call(abind, c(x_y.batch$x1, list(along=0)))
        x2 <- do.call(abind, c(x_y.batch$x2, list(along=0)))
        x.batch <- abind(list(x0, x1, x2), along=5) # batch * x * y * z * channels
        y.batch <- do.call(rbind, x_y.batch$y)
        result <- list(keras_array(x.batch), keras_array(y.batch))
    }
}

source('models.R')

backend()$clear_session()

model = args[3]
if (model == 'vgg') {
vggModel <- createVggModel3D(c(64, 64, 19, 3), ## image size x modality
                            numberOfClassificationLabels = 2,
                            layers = c(1, 2),
                            lowestResolution = 8,
                            convolutionKernelSize = c(3, 3, 3),
                            poolSize = c( 2, 2, 2 ),
                            strides = c( 2, 2, 2 ),
                            denseUnits = 100,   #  10 
                            dropoutRate = 0.6,   # 0.2
                            weightDecay = as.double(args[2]),
                            style = 19)
} else {
vggModel <- createDenseNetModel3D(c(64, 64, 19, 3),
                                 numberOfClassificationLabels = 2,
                                 numberOfFilters = 16, 
                                 depth = 7,
                                 numberOfDenseBlocks = 1,
                                 growthRate = 12,
                                 dropoutRate = 0.6,
                                 weightDecay = as.double(args[2])
                                 )
}

##tensorboard("logs/run_a")
lr = args[1]

print(lr)
vggModel %>% compile(
  loss = 'binary_crossentropy',
  #optimizer = optimizer_rmsprop(lr=0.0001),
  optimizer = optimizer_adam(lr = as.double(lr)),
  metrics=c('binary_crossentropy', 'accuracy'))

batch = args[5]
batch.size = as.double(batch)
train.iterator <- py_iterator(training_generator(which(imgs.warp3[,1] %in% subindex), batch_size=batch.size)) ## augmented dataset
valid.iterator <- py_iterator(valid_generator(which((imgs.warp3[,1] %in% valindex)&imgs.warp3[,2]), batch_size=batch.size)) # no augmentation
test.iterator <- py_iterator(valid_generator(which((imgs.warp3[,1] %in% testindex)&imgs.warp3[,2]), batch_size=batch.size))

schedule = args[4]

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
   lr_callback=callback_reduce_lr_on_plateau(monitor = "loss",
                                      factor = 0.1,
                                      patience = 10,
                                      min_delta = 1e-4,
                                      verbose = 1)
}

prefix = paste0(args[3], args[2], args[1], args[4], args[5])
callbacks_list <- list(
	callback_tensorboard(paste0(prefix, "_logs_r")),
	#callback_early_stopping(monitor = "val_loss",
        #                        min_delta = 1e-4,
        #                        patience = 3,
        #                        verbose = 1),
        lr_callback,
	callback_model_checkpoint(filepath = paste0("weights_r_aug/", prefix, "_{epoch:0004d}.h5"),
                                  monitor = "val_loss",
                                  save_best_only = TRUE,
                                  save_weights_only = TRUE))

system('mkdir -p weights_r_aug')
track = vggModel %>% fit_generator(
                         train.iterator,
                         ##steps_per_epoch=as.integer(nrow(imgs.warp3)/batch.size),
                         #steps_per_epoch=500, # may be too long
                         steps_per_epoch=20,
                         epochs=500,
                         callbacks=callbacks_list,
                         validation_data=valid.iterator,
                         validation_steps=length(valindex)/batch.size,
                         verbose=1)


pdf(paste0(prefix, 'aug.pdf'))
g=plot(track)
print(g)
dev.off()

df = as.data.frame(track)
write.table(df, file=paste0(prefix, 'aug.txt'), quote=F, sep='\t')

## Save the model
save_model_weights_hdf5(
  vggModel, filepath = paste0( './', prefix, '_Weights.h5' ) )
save_model_hdf5(
  vggModel, filepath = paste0( './', prefix, '_Model.h5' ), overwrite = TRUE )

print(vggModel %>% evaluate_generator(test.iterator, steps=length(testindex)/batch.size))

