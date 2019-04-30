#' 3-D implementation of the Vgg deep learning architecture.
#' Creates a keras model of the Vgg deep learning architecture for image
#' recognition based on the paper
#'
#' K. Simonyan and A. Zisserman, Very Deep Convolutional Networks for
#'   Large-Scale Image Recognition
#'
#' available here:
#'
#'         https://arxiv.org/abs/1409.1556
#'
#' learn from ANTsRNet
#' 
#' 
library(tfestimators)
require(tensorflow)
require(keras)
require(ANTsR)
require(jsonlite)
library(reticulate)
library(abind)
library(rhdf5)
require(fslr)
library(foreach)
library(doMC)
library(extrantsr)
library(ggplot2)

keras = import("keras")

fn <- keras$wrappers$scikit_learn$KerasClassifier
metrics <- import('sklearn.metrics')

"
metrics$roc_auc_score
metrics$roc_curve
metrics$recall_score
metrics$precision_score
metrics$f1_score
metrics$average_precision_score
metrics$precision_recall_curve
"

model = commandArgs(T)
model = model[1] #
print(model)
system("mkdir -p weights_noaug")

res = 64
load.img <- function(x, modal='t2') {
    if (modal=='t2') {
        protocal1 <- read_json(paste0(x, ".json"))$ProtocolName
        modal.1 <- antsImageRead(paste0(x, ".nii.gz"))
        modal.1s <- resampleImage(modal.1, interpType=4, useVoxels=T, resampleParams=c(res, res, 19))
    } else {
        protocal1 <- read_json(paste0(x, ".json"))$ProtocolName
        if (length(grep("sag", protocal1, ignore.case=T))>=1) {
            protocal1 <- read_json(paste0(gsub('run-1', 'run-2', x), ".json"))$ProtocolName
            modal.1 <- antsImageRead(paste0(gsub('run-1', 'run-2', x), ".nii.gz"))
        } else {
            modal.1 <- antsImageRead(paste0(x, ".nii.gz"))
        }
        modal.1s <- resampleImage(modal.1, interpType=4, useVoxels=T, resampleParams=c(res, res, 19))
    }
}
## t21 <- load.img("output_pvl/sub-10441512_20141111/anat/sub-10441512_20141111_run-1_T2w")
## t22 <- load.img("output_pvl/sub-10441512_20141111/anat/sub-10441512_20141111_run-2_T2w")
## t1 <- load.img("output_pvl/sub-10441512_20141111/anat/sub-10441512_20141111_run-1_T1w", 't1')

load.batch <- function(x) {
    n <- basename(x[1])
    t21 <- as.array(iMath(load.img(paste0(x[1], "/", "anat/", n, "_run-1_T2w")), "Normalize"))
    t22 <- as.array(iMath(load.img(paste0(x[1], "/", "anat/", n, "_run-2_T2w")), "Normalize"))
    t1 <- as.array(iMath(load.img(paste0(x[1], "/", "anat/", n, "_run-1_T1w"), 't1'), "Normalize"))
    result <- list(t1, t21, t22)
    result <- abind(result, along=4)
    return(result)
}

setwd("/home/qinqian/data1/MRI/01_PVL_AI")
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

source("models.R")

if (file.exists('all_images.h5')) {
    h=H5Fopen('all_images.h5')
    allArray = h$pvl$images
    labels = h$pvl$labels
    H5Fclose(h)
} else {
    registerDoMC(16)
    imgs.list <- foreach(i=1:nrow(meta)) %dopar% {
        load.batch(as.character(meta[i,]))
    }
    allArray <- abind(imgs.list, along=5)
    allArray <- aperm(allArray, c(5,1,2,3,4)) ## sample x imgx x imgy x imgz x channel(modality)
    labels <- to_categorical(meta[,4])
    h5createFile('all_images.h5')
    h5createGroup('all_images.h5', 'pvl')
    print(h5ls('all_images.h5'))
    h5write(allArray, file='all_images.h5', name='pvl/images')
    h5write(labels, file='all_images.h5', name='pvl/labels')
}

print(dim(allArray))
allArray <- (allArray - mean(allArray)) / sd(allArray)

seeds <- c(100, 111, 1234, 5000, 9999, 1111, 6666, 8000, 9000, 10000, 2000, 1989)
results <- rep(0, length(seeds))

for (i in seq_along(seeds)) {
    set.seed(seeds[i])
    trainProportion = 0.8
    subindex = sample(1:nrow(meta), nrow(meta)*trainProportion, replace = F)
    backend()$clear_session()
    #model = 'dense'
    #model = 'vgg'
    if (model == 'vgg') {
    vggModel <- createVggModel3D(dim(allArray)[2:5],
                                numberOfClassificationLabels = 2,
                                layers = c(1, 2),
                                lowestResolution = 8,
                                convolutionKernelSize = c(3, 3, 3),
                                poolSize = c( 2, 2, 2 ),
                                strides = c( 2, 2, 2 ),
                                #denseUnits = 64,  ## 100, 0.6 very well, 
                                denseUnits = 100, 
                                dropoutRate = 0.6,  ## try 0.1
                                weightDecay = 0.01,
                                style=19)
    print(vggModel)
    vggModel %>% compile(
                     loss = 'binary_crossentropy',
                     #optimizer = optimizer_rmsprop(lr=0.0001),
                     optimizer = optimizer_adam(lr = 0.001),
                     #optimizer =  optimizer_sgd(lr = 0.001, momentum = 0.9, nesterov = TRUE),
                     #optimizer = optimizer_adadelta(lr = 0.001, decay=1e-4),
                                        #optimizer = optimizer_sgd(lr = 0.01),
                     metrics=c('accuracy'))
    callbacks_list <- list(
	callback_tensorboard("logs_r"),
	callback_early_stopping(monitor = "val_loss",
                                min_delta = 1e-4,
                                patience = 5,
                                verbose = 1),
        #callback_learning_rate_scheduler(lr_schedule),
	callback_reduce_lr_on_plateau(monitor = "val_loss",
                                      factor = 0.3333,
                                      patience = 3,
                                      verbose = 1),
        ## callback_model_checkpoint(paste0( "./", "vggtWeights.h5"),
        ##                           monitor = 'val_loss', save_best_only=T),
	callback_model_checkpoint(filepath = paste0("weights_noaug/", model, "_lr001_{epoch:02d}.h5"),
                                  monitor = "val_loss",
                                  save_best_only = TRUE,
                                  save_weights_only = TRUE))
        batch_size=16
        epochs=100
    } else {
    vggModel <- createDenseNetModel3D(dim(allArray)[2:5],
                                      numberOfClassificationLabels = 2,
                                      numberOfFilters = 16, 
                                      depth = 7,
                                      numberOfDenseBlocks = 1,
                                      growthRate = 8,
                                      dropoutRate = 0.2,
                                      weightDecay = 1e-8
                                     )
    vggModel %>% compile(
                     loss = 'binary_crossentropy',
                     #optimizer = optimizer_rmsprop(lr=0.01),
                     optimizer = optimizer_adam(lr = 1),
                     #optimizer =  optimizer_sgd(lr = 0.01, momentum = 0.9, nesterov = TRUE),
                     #optimizer = optimizer_adadelta(lr = 0.001, decay=1e-4),
                                        #optimizer = optimizer_sgd(lr = 0.01),
                     metrics=c('accuracy'))
    lr_schedule <- function(epoch, lr) {
       if (epoch %% 5 == 0) {
           lr = lr / 3
            
           if (lr < 0.00001) {
              lr = 0.00001
           }
       }
       lr
    }

    callbacks_list <- list(
	callback_tensorboard("logs_r"),
	callback_early_stopping(monitor = "val_loss",
                                min_delta = 1e-4,
                                patience = 8,
                                verbose = 1),
        callback_learning_rate_scheduler(lr_schedule),
	## callback_reduce_lr_on_plateau(monitor = "val_loss",
        ##                               factor = 0.33,
        ##                               patience = 2,
        ##                               verbose = 1),
        ## callback_model_checkpoint(paste0( "./", "vggtWeights.h5"),
        ##                           monitor = 'val_loss', save_best_only=T),
	callback_model_checkpoint(filepath = paste0("weights_noaug/", model, "_lr001_{epoch:02d}.h5"),
                                  monitor = "val_loss",
                                  save_best_only = TRUE,
                                  save_weights_only = TRUE))
        batch_size=16
        epochs=100
    }
    #tensorboard("logs/run_a")

    track <- vggModel %>% fit(allArray[subindex,,,,], labels[subindex,],
                             epochs = epochs, batch_size = batch_size, verbose = 1, shuffle = TRUE,
                             callbacks = callbacks_list,
                             validation_split = 0.2)

    pdf(paste0(i, "_lr001", model, ".pdf"))
    g <- plot(track)
    print(g)
    dev.off()

    df <- as.data.frame(track)
    write.table(df, file=paste0(i, "__", model, "_accuracy.txt"), quote=F, sep="\t")
  
    result <- vggModel %>% evaluate(allArray[-subindex,,,,], labels[-subindex,])
    print(result)
    results[i] <- result$acc
    print(results)

    ## prediction = vggModel %>% predict_classes(allArray[-subindex,,,,])
    ## vggModel %>% evaluate(allArray.s[subindex,,,,], labels[subindex,])
    ## vggModel <- createVggModel3D(dim(allArray)[2:5],
    ##                             numberOfClassificationLabels = 2,
    ##                             layers = c(1, 2),
    ##                             lowestResolution = 8,
    ##                             convolutionKernelSize = c(3, 3, 3),
    ##                             poolSize = c( 2, 2, 2 ),
    ##                             strides = c( 2, 2, 2 ),
    ##                             denseUnits = 100,   #  10 
    ##                             dropoutRate = 0.5,   # 0.2
    ##                             style = 19)
    ## vggModel %>% compile(
    ##   loss = 'binary_crossentropy',
    ##   #optimizer = optimizer_rmsprop(lr=0.0001),
    ##   optimizer = optimizer_adam(lr = 0.001),
    ##   #optimizer = optimizer_adadelta(lr = 0.001, decay=1e-4),
    ##   #optimizer = optimizer_sgd(lr = 0.01),
    ##   metrics=c('binary_crossentropy', 'accuracy'))
    ## callback <- list(
    ##     callback_tensorboard(log_dir="logs/run_a", write_graph=T)
    ## )
    ## track <- vggModel %>% fit(allArray.s[subindex,,,,], labels[subindex,],
    ##                          epochs = 50, batch_size = 64, verbose = 1, shuffle = TRUE,
    ##                          callbacks = callback)
    ## vggModel %>% evaluate(allArray.s[-subindex,,,,], labels[-subindex,])
    ## ## Save the model
    save_model_weights_hdf5(
      vggModel, filepath = paste0( './', i, '__', model, 'Weights_all.h5' ) )
    save_model_hdf5(
      vggModel, filepath = paste0( './', i, '__', model, 'Model_all.h5' ), overwrite = TRUE )
}

#write.table(results, file="vgg_accuracy.txt", quote=F, sep="\t")
print(summary(results))
write.table(results, file=paste0(model, "_accuracy.txt"), quote=F, sep="\t")

H5close()
