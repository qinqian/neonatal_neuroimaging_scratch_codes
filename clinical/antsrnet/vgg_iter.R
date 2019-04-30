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
library(extrantsr)
library(ggplot2)
library(foreach)
library(doMC)
registerDoMC(8)

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
labels <- to_categorical(labels)

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

system('mkdir -p weights_noaug')

vol_generator <- function(sample_index, batch_size) {
    res <- 64
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
        list(protocal1, modal.1s)
    }
    load.batch <- function(x) {
        n <- basename(x[1])
        t21 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T2w"))
        t22 <- load.img(paste0(x[1], "/", "anat/", n, "_run-2_T2w"))
        t1 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T1w"), 't1')
        return(list(t1[[2]], t21[[2]], t22[[2]]))
    }
    imgs.cur <- meta[sample_index, ]
    label.cur <- labels[sample_index, ]
    imgs.next <- meta[sample_index, ]
    label.next <- labels[sample_index, ]
    function() {
        ## start new epoch
        if (nrow(imgs.cur) < batch_size) {
            imgs.cur <<- imgs.next
            label.cur <<- label.next
        }
        batch_ind <- sample(1:nrow(imgs.cur), batch_size) # shuffle current batch
        batch_images_list <- imgs.cur[batch_ind, ]
        imgs.cur <<- imgs.cur[-batch_ind, ]
        batch_label_list <- label.cur[batch_ind,]
        label.cur <<- label.cur[-batch_ind,]
        ## data augmentation by random inverse transformation to each of the inviduals
        x_y.batch <- foreach(i=1:batch_size) %dopar% {
            x = load.batch(as.character(batch_images_list[i, ]))
            x2.inv <- as.array(iMath(x[[3]], "Normalize"))
            ##x2.inv <- (x2.inv-mean(x2.inv))/sd(x2.inv)
            x1.inv <- as.array(iMath(x[[2]], "Normalize"))
            ##x1.inv <- (x1.inv-mean(x1.inv))/sd(x1.inv)
            list(x1=x1.inv, x2=x2.inv, y=batch_label_list[i, ])
        }
        x_y.batch <- purrr::transpose(x_y.batch)
        x1 <- do.call(abind, c(x_y.batch$x1, list(along=0)))
        x2 <- do.call(abind, c(x_y.batch$x2, list(along=0)))

        x.batch <- abind(list(x1, x2), along=5) # batch * x * y * z * channels
        y.batch <-  do.call(rbind, x_y.batch$y)
        result <- list(keras_array(x.batch), keras_array(y.batch))
    }
}

batch.size = 64
train.iterator <- py_iterator(vol_generator(subindex, batch_size=batch.size))
valid.iterator <- py_iterator(vol_generator(valindex, batch_size=batch.size))
test.iterator <- py_iterator(vol_generator(testindex, batch_size=batch.size))

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
    model <- model %>% layer_batch_normalization( axis = 1, 
      gamma_regularizer = regularizer_l2( weightDecay ), 
      beta_regularizer = regularizer_l2( weightDecay ) )  
    model <- model %>% layer_activation( activation = 'relu' )
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
  
  outputs <- outputs %>% layer_batch_normalization( axis = 1, 
    gamma_regularizer = regularizer_l2( weightDecay ), 
    beta_regularizer = regularizer_l2( weightDecay ) )  
  
  outputs <- outputs %>% layer_activation( activation = 'relu' )
  outputs <- outputs %>% layer_global_average_pooling_3d()
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, 
    activation = 'softmax', kernel_regularizer = regularizer_l2( weightDecay ), 
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
                              dilation_rate = c(2, 2, 2),
                              padding = 'same' ) %>%
                layer_activation_elu(alpha=0.3) %>%
                layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                              padding = 'same' ) %>%
                layer_activation_elu(alpha=0.3)
        } else if( i == 2 ) {
            vggModel %>%
                layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                               padding = 'same' ) %>%
                layer_activation_elu(alpha=0.3) %>%
                layer_conv_3d(filters = numberOfFilters, kernel_size = convolutionKernelSize,
                              padding = 'same' ) %>%
                layer_activation_elu(alpha=0.3)
        }  else {
            if( style == 16 )
            {
                vggModel %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_activation_elu(alpha=0.3)
            } else {  # style == 19
                vggModel %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                  padding = 'same' ) %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                   padding = 'same' ) %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                   padding = 'same' ) %>%
                    layer_activation_elu(alpha=0.3) %>%
                    layer_conv_3d( filters = numberOfFilters, kernel_size = convolutionKernelSize,
                                   padding = 'same' ) %>%
                    layer_activation_elu(alpha=0.3) 
            }
        }

        vggModel %>% layer_max_pooling_3d( pool_size = poolSize, strides = strides )
    }

  vggModel %>% layer_flatten()

  vggModel %>% layer_batch_normalization()

  vggModel %>% layer_dense( units = denseUnits ) %>% layer_activation_elu(alpha=0.3) 
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }
  vggModel %>% layer_dense( units = denseUnits)  %>% layer_activation_elu(alpha=0.3) 
  if( dropoutRate > 0.0 )
    {
    vggModel %>% layer_dropout( rate = dropoutRate )
    }

  vggModel %>% layer_dense(units = numberOfClassificationLabels,
#                           kernel_regularizer = regularizer_l1_l2(l1=1e-3, l2=1e-2),
                           activation = 'sigmoid' )

  return( vggModel )
}

seeds <- c(1234, 5000, 9999, 1111, 6666, 8000, 9000, 10000, 2000, 1989)
results <- rep(0, length(seeds))
for (i in seq_along(seeds)) {
    set.seed(seeds[i])
    backend()$clear_session()
    gc()
    vggModel <- createVggModel3D(c(64, 64, 19, 2),
                                numberOfClassificationLabels = 2,
                                layers = c(1, 2),
                                lowestResolution = 8,
                                convolutionKernelSize = c(3, 3, 3),
                                poolSize = c( 2, 2, 2 ),
                                strides = c( 2, 2, 2 ),
                                denseUnits = 50,   #  10 
                                dropoutRate = 0.5,   # 0.2
                                style = 19)
    ## vggModel <- createDenseNetModel3D(dim(allArray)[2:5],
    ##                                  numberOfClassificationLabels = 2,
    ##                                  numberOfFilters = 8, 
    ##                                  depth = 10,
    ##                                  numberOfDenseBlocks = 5,
    ##                                  growthRate = 12,
    ##                                  dropoutRate = 0.2,
    ##                                  weightDecay = 1e-4
    ##                                  )
    ##tensorboard("logs/run_a")
    vggModel %>% compile(
                     loss = 'binary_crossentropy',
                     ##optimizer = optimizer_rmsprop(lr=0.0001),
                     optimizer = optimizer_adam(lr = 0.001),
                     ##optimizer = optimizer_adadelta(lr = 0.001, decay=1e-4),
                     ##optimizer = optimizer_sgd(lr = 0.01),
                     metrics=c('accuracy'))

    lr_schedule <- function(epoch) {
        if(epoch <= 20) {
            0.001
        } else if(epoch > 20 && epoch <= 40) {
            0.0005
        } else {
            0.0001
        }
    }

    callbacks_list <- list(
	callback_tensorboard("logs_r"),
	callback_early_stopping(monitor = "loss",
                                min_delta = 1e-4,
                                patience = 8,
                                verbose = 1),
	callback_reduce_lr_on_plateau(monitor = "val_loss",
                                      factor = 0.1,
                                      patience = 2,
                                      verbose = 1),
        ## callback_model_checkpoint(paste0( "./", "vggtWeights.h5"),
        ##                           monitor = 'val_loss', save_best_only=T),
	callback_model_checkpoint(filepath = "weights_noaug/vgg_{epoch:02d}.h5",
                                  monitor = "val_loss",
                                  save_best_only = TRUE,
                                  save_weights_only = TRUE))

    track <- vggModel %>% fit_generator(train.iterator,
                                        steps_per_epoch=as.integer(length(subindex)/batch.size),
                                        validation_steps=length(valindex)/batch.size,
                                        epochs = 50, verbose = 1,
                                        callbacks = callbacks_list,
                                        validation_data = valid.iterator)

    pdf(paste0(i, "_vgg.pdf"))
    g <- plot(track)
    print(g)
    dev.off()

    result <- vggModel %>% evaluate_generator(test.iterator, steps=length(testindex)/batch.size)
    results[i] <- result$acc

    prediction = vggModel %>% predict_generator(test.iterator, steps=length(testindex)/batch.size)
    print(dim(prediction))
    print(head(prediction))
    ## ## Save the model
    save_model_weights_hdf5(
      vggModel, filepath = paste0( './', i, '__vggWeights_all.h5' ) )
    save_model_hdf5(
      vggModel, filepath = paste0( './', i, '__vggModel_all.h5' ), overwrite = TRUE )
}

write.table(results, file="vgg_accuracy.txt", quote=F, sep="\t")
