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
model = commandArgs(T)
model = model[1] # 
print(model)
# weights_noaug_crossval
system("mkdir -p weights_noaug_crossval")

createDenseNetModel3D <- function( inputImageSize, 
                                   numberOfClassificationLabels = 1000,
                                   numberOfFilters = 16, 
                                   depth = 7,
                                   numberOfDenseBlocks = 1,
                                   growthRate = 12,
                                   dropoutRate = 0.2,
                                   weightDecay = 1e-4
                                 )
{

  convolutionFactory3D <- function( model, numberOfFilters, kernelSize = c( 3, 3, 3 ), 
                                    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    model <- model %>% layer_batch_normalization( axis = -1,
      gamma_regularizer = regularizer_l2( weightDecay ), 
      beta_regularizer = regularizer_l2( weightDecay ) )  

    #model <- model %>% layer_activation( activation = 'relu' )
    model <- model %>% layer_activation_elu( alpha=0.3 )

    model <- model %>% layer_conv_3d( filters = numberOfFilters, 
      kernel_size = kernelSize, kernel_initializer = 'he_uniform', padding = 'same', 
      use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) ) 

    if( dropoutRate > 0.0 )  
      {
      model <- model %>% layer_dropout( rate = dropoutRate )  
      }
    return( model )
    }

  transition3D <- function( model, numberOfFilters, dropoutRate = 0.0, 
                            weightDecay = 1e-4 )  
    {
    model <- convolutionFactory3D( model, numberOfFilters, kernelSize = c( 1, 1, 1 ),
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    model <- model %>% layer_average_pooling_3d( pool_size = c( 2, 2, 2 ), 
      strides = c( 2, 2, 2 ) )
    return( model )
    }

  createDenseBlocks3D <- function( model, numberOfFilters, depth, growthRate, 
    dropoutRate = 0.0, weightDecay = 1e-4 )
    {
    #  Theano:  [batchSize, channelSize, widthSize, heightSize]
    #  tensorflow:  [batchSize, widthSize, heightSize, channelSize]

    K <- keras::backend()  

    concatenationAxis <- 1  
    if( K$image_data_format() == 'channels_last' )
      {
      concatenationAxis <- -1 
      }

    denseBlockLayers <- list( model )
    for( i in 1:depth )  
      {
      model <- convolutionFactory3D( model, numberOfFilters = growthRate, 
        kernelSize = c( 3, 3, 3 ), dropoutRate = dropoutRate, weightDecay = weightDecay )  
      denseBlockLayers[[i+1]] <- model
      model <- layer_concatenate( denseBlockLayers, axis = concatenationAxis )
      numberOfFilters <- numberOfFilters + growthRate
      }

    return( list( model = model, numberOfFilters = numberOfFilters ) )  
    }

  if( ( depth - 4 ) %% 3 != 0 )
    {
    stop( "Depth must be equal to 3*N+4 where N is an integer." )  
    }
  numberOfLayers = as.integer( ( depth - 4 ) / 3 )

  inputs <- layer_input( shape = inputImageSize )

  outputs <- inputs %>% layer_conv_3d( filters = numberOfFilters, 
    kernel_size = c( 3, 3, 3 ), kernel_initializer = 'he_uniform', padding = 'same',
    use_bias = FALSE, kernel_regularizer = regularizer_l2( weightDecay ) )
  
  # Add dense blocks

  nFilters <- numberOfFilters

  for( i in 1:( numberOfDenseBlocks - 1 ) )
    {
    denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters, 
      depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
      weightDecay = weightDecay )
    outputs <- denseBlockLayer$model
    nFilters <- denseBlockLayer$numberOfFilters

    outputs <- transition3D( outputs, numberOfFilters = nFilters, 
      dropoutRate = dropoutRate, weightDecay = weightDecay )
    }

  denseBlockLayer <- createDenseBlocks3D( outputs, numberOfFilters = nFilters, 
    depth = numberOfLayers, growthRate = growthRate, dropoutRate = dropoutRate,
    weightDecay = weightDecay )
  outputs <- denseBlockLayer$model
  nFilters <- denseBlockLayer$numberOfFilters
  
  outputs <- outputs %>% layer_batch_normalization( axis = -1, 
    gamma_regularizer = regularizer_l2( weightDecay ), 
    beta_regularizer = regularizer_l2( weightDecay ) )  
  
  outputs <- outputs %>% layer_activation_elu(alpha=0.3) # layer_activation( activation = 'relu' )
  #outputs <- outputs %>% layer_activation( activation = 'relu' )
  outputs <- outputs %>% layer_global_average_pooling_3d()
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, 
    activation = 'sigmoid', kernel_regularizer = regularizer_l2( weightDecay ), 
    bias_regularizer = regularizer_l2( weightDecay ) )

  denseNetModel <- keras_model( inputs = inputs, outputs = outputs )
  return( denseNetModel )
}

createVggModel3D <- function(inputImageSize,
                            numberOfClassificationLabels = 1000,
                            layers = c( 1, 2, 3, 4, 4 ),
                            lowestResolution = 64,
                            convolutionKernelSize = c( 3, 3, 3 ),
                            poolSize = c( 2, 2, 2 ),
                            strides = c( 2, 2, 2 ),
                            denseUnits = 4096,
                            dropoutRate = 0.0,
                            weightDecay = 1e-4,
                            style = 19)
{
  if( !usePkg( "keras" ) )
    {
    stop( "Please install the keras package." )
    }

  if( style != 19 && style != 16 )
    {
    stop( "Incorrect style.  Must be either '16' or '19'." )
    }

  vggModel <- keras_model_sequential()

  for( i in 1:length( layers ) )
    {
        numberOfFilters <- lowestResolution * 2 ^ ( layers[i] - 1 )

        if( i == 1 )
        {
            vggModel %>%
                layer_conv_3d(input_shape = inputImageSize, filters = numberOfFilters,
                              kernel_size = convolutionKernelSize,
                              padding = 'same' ) %>%
                #layer_batch_normalization() %>%
                layer_batch_normalization( axis = -1,
                     gamma_regularizer = regularizer_l2( weightDecay ),
                     beta_regularizer = regularizer_l2( weightDecay ) ) %>%
                layer_activation_elu(alpha=0.3) %>%
                layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                              padding = 'same' ) %>%
                #layer_batch_normalization() %>%
                layer_batch_normalization( axis = -1,
                     gamma_regularizer = regularizer_l2( weightDecay ),
                     beta_regularizer = regularizer_l2( weightDecay ) ) %>%
                layer_activation_elu(alpha=0.3)
        } else if( i == 2 ) {
            vggModel %>%
                layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                               padding = 'same' ) %>%
                #layer_batch_normalization() %>%
                layer_batch_normalization( axis = -1,
                     gamma_regularizer = regularizer_l2( weightDecay ),
                     beta_regularizer = regularizer_l2( weightDecay ) ) %>%
                layer_activation_elu(alpha=0.3) %>%
                layer_conv_3d(filters = numberOfFilters, kernel_size = convolutionKernelSize,
                              padding = 'same' ) %>%
                #layer_batch_normalization() %>%
                layer_batch_normalization( axis = -1,
                     gamma_regularizer = regularizer_l2( weightDecay ),
                     beta_regularizer = regularizer_l2( weightDecay ) ) %>%
                layer_activation_elu(alpha=0.3)
        }  else {
            if( style == 16 )
            {
                vggModel %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_batch_normalization() %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_batch_normalization() %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_batch_normalization() %>%
                    layer_activation_elu(alpha=0.3)
            } else {  # style == 19
                vggModel %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_batch_normalization() %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                   padding = 'same' ) %>%
                    layer_batch_normalization() %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                   padding = 'same' ) %>%
                    layer_batch_normalization() %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                   padding = 'same' ) %>%
                    layer_batch_normalization() %>%
                    layer_activation_elu(alpha=0.3) 
            }
        }
        vggModel %>% layer_max_pooling_3d( pool_size = poolSize, strides = strides )
    }

  vggModel %>% layer_flatten()
  vggModel %>% layer_dense( units = denseUnits ) %>% 
        #layer_batch_normalization() %>%
        layer_batch_normalization( axis = -1,
             gamma_regularizer = regularizer_l2( weightDecay ),
             beta_regularizer = regularizer_l2( weightDecay ) ) %>%
        layer_activation_elu(alpha=0.3)

  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }

  vggModel %>% layer_dense( units = denseUnits)  %>% 
          #layer_batch_normalization() %>% 
          layer_batch_normalization( axis = -1,
               gamma_regularizer = regularizer_l2( weightDecay ),
               beta_regularizer = regularizer_l2( weightDecay ) ) %>%
          layer_activation_elu(alpha=0.3) 
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }

  vggModel %>% layer_dense(units = numberOfClassificationLabels,
                           kernel_regularizer = regularizer_l2(weightDecay),
                           activation = 'sigmoid' )
  return( vggModel )
}

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

lr_schedule <- function(epoch, lr) {
    if (epoch %% 5 == 0) {
        lr = lr / 3
        if (lr < 0.00001) {
            lr = 0.00001
        }
    }
    lr
}

set.seed(999)
npp = 6
params <- data.frame(
    dropout=runif(npp, 0.01, 0.99),
    batch=sample(c(8, 16, 24, 32, 64, 128))
)
params.index <- expand.grid(sample(1:npp),sample(1:npp))
params <- cbind(params[,1][params.index[,1]],
                params[,2][params.index[,2]])
epochs=200
val.metrics.all <- matrix(0, nrow=nrow(params), ncol=2)
for (m in seq_along(seeds)) {
    set.seed(seeds[m])
    trainProportion = 0.7
    subindex = sample(1:nrow(meta), nrow(meta)*trainProportion, replace = F)
    backend()$clear_session()
    k <- 3
    indices <- sample(1:length(subindex))
    folds <- cut(indices, breaks=k, labels=F)
    for (i in 1:nrow(params)) {
        cat("param #", i, "\n")
        print(params[i, ])
        val.metrics <- matrix(0, nrow=k, ncol=2)
        for (j in 1:k) {
            cat("processing fold #", j, "\n")
            val.index <- which(folds==j, arr.ind=T)
            val.data <- allArray[subindex[indices][val.index],,,,]
            val.targets <- labels[subindex[indices][val.index],]
            partial.train <- allArray[subindex[indices][-val.index],,,,]
            partial.targets <- labels[subindex[indices][-val.index],]

            if (model == 'vgg') {
                vggModel <- createVggModel3D(dim(allArray)[2:5],
                                            numberOfClassificationLabels = 2,
                                            layers = c(1, 2),
                                            lowestResolution = 8,
                                            convolutionKernelSize = c(3, 3, 3),
                                            poolSize = c( 2, 2, 2 ),
                                            strides = c( 2, 2, 2 ),
                                            denseUnits = 100,
                                            dropoutRate = params[i, 1],  ## try 0.1
                                            weightDecay = 0,
                                            style=19)
            } else {
                vggModel <- createDenseNetModel3D(dim(allArray)[2:5],
                                                 numberOfClassificationLabels = 2,
                                                 numberOfFilters = 16,
                                                 depth = 7,
                                                 numberOfDenseBlocks = 1,
                                                 growthRate = 12,
                                                 dropoutRate = params[i, 1],
                                                 weightDecay = 0,
                                                 )
            }
            vggModel %>% compile(
                             loss = 'binary_crossentropy',
                                        #optimizer = optimizer_rmsprop(lr=0.001),
                             optimizer = optimizer_adam(lr = 0.001),
                                        #optimizer =  optimizer_sgd(lr = 0.001, momentum = 0.9, nesterov = TRUE),
                                        #optimizer = optimizer_adadelta(lr = 0.001, decay=1e-4)
                             metrics=c('accuracy'))
            callbacks_list <- list(
                callback_tensorboard("logs_cross"),
                callback_early_stopping(monitor = "val_loss",
                                        min_delta = 1e-4,
                                        patience = 8,
                                        verbose = 1),
                ##callback_learning_rate_scheduler(lr_schedule),
                callback_reduce_lr_on_plateau(monitor = "val_loss",
                                              factor = 0.1, 
                                              patience = 2,
                                              verbose = 1),
                callback_model_checkpoint(filepath = paste0("weights_noaug_crossval/", i,"_", j,"_",m,"_",model, "_lr001_{epoch:02d}.h5"),
                                          monitor = "val_loss",
                                          save_best_only = TRUE,
                                          save_weights_only = TRUE))
            ## tensorboard("logs/run_a")
            ## training
            track <- vggModel %>% fit(partial.train, partial.targets,
                                     epochs = epochs, batch_size = params[i, 2], verbose = 1, shuffle = TRUE,
                                     callbacks = callbacks_list,
                                     validation_data=list(val.data, val.targets))
            df <- as.data.frame(track)
            result <- vggModel %>% evaluate(val.data, val.targets)
            val.metrics[j, 1] <- result$loss
            val.metrics[j, 2] <- result$acc
            print(result)
            print(val.metrics)
        }
        val.metrics <- colMeans(val.metrics) # loss, acc
        val.metrics.all[i, ] <- val.metrics
        print(val.metrics.all)
    }
    write.table(cbind(params, val.metrics.all), file=paste0(m, model, "_crossval.txt"), sep='\t', quote=F)

    params <- params[which.min(val.metrics.all[, 1]), ]
    if (model == 'vgg') {
        vggModel <- createVggModel3D(dim(allArray)[2:5],
                                    numberOfClassificationLabels = 2,
                                    layers = c(1, 2),
                                    lowestResolution = 8,
                                    convolutionKernelSize = c(3, 3, 3),
                                    poolSize = c( 2, 2, 2 ),
                                    strides = c( 2, 2, 2 ),
                                    denseUnits = 100,
                                    dropoutRate = params[1],  ## try 0.1
                                    weightDecay = 0,
                                    style=19)
    } else {
        vggModel <- createDenseNetModel3D(dim(allArray)[2:5],
                                         numberOfClassificationLabels = 2,
                                         numberOfFilters = 16,
                                         depth = 7,
                                         numberOfDenseBlocks = 1,
                                         growthRate = 12,
                                         dropoutRate = params[1],
                                         weightDecay = 1e-8
                                         )
    }
    vggModel %>% compile(
                     loss = 'binary_crossentropy',
                                        #optimizer = optimizer_rmsprop(lr=0.001),
                     optimizer = optimizer_adam(lr = 0.001),
                                        #optimizer =  optimizer_sgd(lr = 0.001, momentum = 0.9, nesterov = TRUE),
                                        #optimizer = optimizer_adadelta(lr = 0.001, decay=1e-4)
                     metrics=c('accuracy'))
    callbacks_list <- list(
        callback_tensorboard("logs_cross"),
        callback_early_stopping(monitor = "val_loss",
                                min_delta = 1e-4,
                                patience = 8,
                                verbose = 1),
        ##callback_learning_rate_scheduler(lr_schedule),
        callback_reduce_lr_on_plateau(monitor = "val_loss",
                                      factor = 0.1, 
                                      patience = 2,
                                      verbose = 1),
        callback_model_checkpoint(filepath = paste0("weights_noaug_crossval/", m, model, "_lr001_{epoch:02d}.h5"),
                                  monitor = "val_loss",
                                  save_best_only = TRUE,
                                  save_weights_only = TRUE))
    ## tensorboard("logs/run_a")
    ## training
    track <- vggModel %>% fit(allArray[subindex,,,,], labels[subindex,],
                             epochs = epochs, batch_size = params[2], verbose = 1, shuffle = TRUE,
                             callbacks = callbacks_list)

    result <- vggModel %>% evaluate(allArray[-subindex,,,,], labels[-subindex,])
    print(result)
    results[m] <- result$acc
    print(results)

    pdf(paste0(m, "_lr001", model, "_cross.pdf"))
    g <- plot(track)
    print(g)
    dev.off()
    write.table(df, file=paste0(m, "__", model, "_cross_accuracy.txt"), quote=F, sep="\t")
##     test <- do.call(cbind, test)
##     test.loss <- subset(test, metric=='loss')
##     test.aveloss <- rowMeans(test.loss[ ,seq(2, ncol(test), 4)])
##     test.acc <- subset(test, metric=='')
##     test.aveloss <- rowMeans(test.loss[ ,seq(2, ncol(test), 4)])

    save_model_weights_hdf5(
        vggModel, filepath = paste0( './', m, '__', model, '_cross_Weights_all.h5' ) )
    save_model_hdf5(
        vggModel, filepath = paste0( './', m, '__', model, '_cross_Model_all.h5' ), overwrite = TRUE )
}

print(summary(results))
write.table(results, file=paste0(model, "_cross_accuracy.txt"), quote=F, sep="\t")
H5close()

