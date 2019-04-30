library(tensorflow)
library(keras)

createResNetModel3D <- function( inputImageSize, 
                                 numberOfClassificationLabels = 1000,
                                 layers = 1:4, 
                                 residualBlockSchedule = c( 3, 4, 6, 3 ),
                                 lowestResolution = 64,
                                 cardinality = 1
                               )
{

  addCommonLayers <- function( model )
    {
    model <- model %>% layer_batch_normalization()
    #model <- model %>% layer_activation_leaky_relu()
    model <- model %>% layer_activation_elu( alpha=0.3 )

    return( model )
    }

  groupedConvolutionLayer3D <- function( model, numberOfFilters, strides )
    {
    K <- keras::backend()
    if( cardinality == 1 )
      {
      groupedModel <- model %>% layer_conv_3d( filters = numberOfFilters, 
        kernel_size = c( 3, 3, 3 ), strides = strides, padding = 'same' ) %>% layer_spatial_dropout_3d(0.5)
      return( groupedModel )
      }

    if( numberOfFilters %% cardinality != 0 )  
      {
      stop( "numberOfFilters %% cardinality != 0" )  
      }

    numberOfGroupFilters <- as.integer( numberOfFilters / cardinality )

    convolutionLayers <- list()
    for( j in 1:cardinality )
      {
      convolutionLayers[[j]] <- model %>% layer_lambda( function( z ) 
        { 
        K$set_image_data_format( 'channels_last' )
        z[,,,, ( ( j - 1 ) * numberOfGroupFilters + 1 ):( j * numberOfGroupFilters )]
        } )
      convolutionLayers[[j]] <- convolutionLayers[[j]] %>% 
        layer_conv_3d( filters = numberOfGroupFilters, 
          kernel_size = c( 3, 3, 3 ), strides = strides, padding = 'same' )
      }

    groupedModel <- layer_concatenate( convolutionLayers )
    return( groupedModel )
    }

  residualBlock3d <- function( model, numberOfFiltersIn, numberOfFiltersOut, 
    strides = c( 1, 1, 1 ), projectShortcut = FALSE )
    {
    shortcut <- model

    model <- model %>% layer_conv_3d( filters = numberOfFiltersIn, 
      kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
    model <- addCommonLayers( model )

    # ResNeXt (identical to ResNet when `cardinality` == 1)
    model <- groupedConvolutionLayer3D( model, numberOfFilters = numberOfFiltersIn, 
      strides = strides )
    model <- addCommonLayers( model ) 

    model <- model %>% layer_conv_3d( filters = numberOfFiltersOut, 
      kernel_size = c( 1, 1, 1 ), strides = c( 1, 1, 1 ), padding = 'same' )
    model <- model %>% layer_batch_normalization() 

    if( projectShortcut == TRUE || prod( strides == c( 1, 1, 1 ) ) == 0 )
      {
      shortcut <- shortcut %>% layer_conv_3d( filters = numberOfFiltersOut, 
        kernel_size = c( 1, 1, 1 ), strides = strides, padding = 'same' )
      shortcut <- shortcut %>% layer_batch_normalization()  
      }

    model <- layer_add( list( shortcut, model ) )

    #model <- model %>% layer_activation_leaky_relu()
    model <- model %>% layer_activation_elu( alpha=0.3 )

    return( model )
    }

  inputs <- layer_input( shape = inputImageSize )

  nFilters <- lowestResolution

  outputs <- inputs %>% layer_conv_3d( filters = nFilters, 
    kernel_size = c( 7, 7, 7 ), strides = c( 2, 2, 2 ), padding = 'same' )
  outputs <- addCommonLayers( outputs )  
  outputs <- outputs %>% layer_max_pooling_3d( pool_size = c( 3, 3, 3 ), 
    strides = c( 2, 2, 2 ), padding = 'same' )

  for( i in 1:length( layers ) )
    {
    nFiltersIn <- lowestResolution * 2 ^ ( layers[i] )
    nFiltersOut <- 2 * nFiltersIn
    for( j in 1:residualBlockSchedule[i] )  
      {
      projectShortcut <- FALSE
      if( i == 1 && j == 1 )  
        {
        projectShortcut <- TRUE  
        }
      if( i > 1 && j == 1 )
        {
        strides <- c( 2, 2, 2 )
        } else {
        strides <- c( 1, 1, 1 )  
        }
      outputs <- residualBlock3d( outputs, numberOfFiltersIn = nFiltersIn, 
        numberOfFiltersOut = nFiltersOut, strides = strides, 
        projectShortcut = projectShortcut )  
      }
    }

  outputs <- outputs %>% layer_global_average_pooling_3d()
  outputs <- outputs %>% layer_dense( units = numberOfClassificationLabels, 
                                      activation = 'softmax' )

  resNetModel <- keras_model( inputs = inputs, outputs = outputs )

  return( resNetModel )
}



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
