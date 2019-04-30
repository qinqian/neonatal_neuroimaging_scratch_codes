"""modify from https://github.com/selimsef/dsb2018_topcoders/"""

import tensorflow as tf
#from tensorflow.python.keras._impl.keras.applications.inception_resnet_v2 import conv2d_bn
#from tensorflow.python.keras._impl.keras.applications.densenet import dense_block
layer = tf.keras.layers
#from keras_deeplabv3plus import BilinearUpsampling

def conv_bn_relu(input, num_channel, kernel_size, stride, name, padding='same', bn_axis=-1, bn_momentum=0.99,
                 bn_scale=True, use_bias=True):
    x = layer.Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = layer.BatchNormalization(name=name + '_bn', scale=bn_scale, axis=bn_axis, momentum=bn_momentum, epsilon=1.001e-5)(x)
    x = layer.Activation('relu', name=name + '_relu')(x)
    return x

#def dense_block(x, blocks, name):
#    """A dense block.
#
#    Arguments:
#        x: input tensor.
#        blocks: integer, the number of building blocks.
#        name: string, block label.
#
#    Returns:
#        output tensor for the block.
#    """
#    for i in range(blocks):
#        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
#    return x
#
#
#def transition_block(x, reduction, name):
#    """A transition block.
#
#    Arguments:
#        x: input tensor.
#        reduction: float, compression rate at transition layers.
#        name: string, block label.
#
#    Returns:
#        output tensor for the block.
#    """
#    K = tf.keras.backend
#    layer = tf.keras.layers
#    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
#
#    x = layer.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + '_bn')(x)
#    x = layer.Activation('relu', name=name + '_relu')(x)
#    x = layer.Conv2D(
#        int(K.int_shape(x)[bn_axis] * reduction),
#        1,
#        use_bias=False,
#        name=name + '_conv')(
#        x)
#    x = layer.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
#    return x
#
#
#def conv_block(x, growth_rate, name):
#    """A building block for a dense block.
#
#    Arguments:
#        x: input tensor.
#        growth_rate: float, growth rate at dense layers.
#        name: string, block label.
#
#    Returns:
#        output tensor for the block.
#    """
#    K = tf.keras.backend
#    layer = tf.keras.layers
#    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
#
##    x1 = layer.BatchNormalization(
#        axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(
#        x)
#    x1 = layer.Activation('relu', name=name + '_0_relu')(x1)
#    x1 = layer.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + '_1_conv')(x1)
#    x1 = layer.BatchNormalization(
#        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x1)
#    x1 = layer.Activation('relu', name=name + '_1_relu')(x1)
#    x1 = layer.Conv2D(
#        growth_rate, 3, padding='same', use_bias=False, name=name + '_2_conv')(
#        x1)
#    x = layer.Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
#    return x

def conv_relu(input, num_channel, kernel_size, stride, name, padding='same', use_bias=True, activation='relu'):
    x = layer.Conv2D(filters=num_channel, kernel_size=(kernel_size, kernel_size),
               strides=stride, padding=padding,
               kernel_initializer="he_normal",
               use_bias=use_bias,
               name=name + "_conv")(input)
    x = layer.Activation(activation, name=name + '_relu')(x)
    return x


def create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    P5 = layer.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5', kernel_initializer="he_normal")(C5)
    P5_upsampled = layer.UpSampling2D(name='P5_upsampled')(P5)

    P4 = layer.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced',
                      kernel_initializer="he_normal")(C4)
    P4 = layer.Add(name='P4_merged')([P5_upsampled, P4])

    P4 = layer.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4', kernel_initializer="he_normal")(P4)
    P4_upsampled = layer.UpSampling2D(name='P4_upsampled')(P4)

    P3 = layer.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced',
                      kernel_initializer="he_normal")(C3)
    P3 = layer.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = layer.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3', kernel_initializer="he_normal")(P3)
    P3_upsampled = layer.UpSampling2D(name='P3_upsampled')(P3)

    P2 = layer.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced',
                kernel_initializer="he_normal")(C2)
    P2 = layer.Add(name='P2_merged')([P3_upsampled, P2])
    P2 = layer.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P2', kernel_initializer="he_normal")(P2)
    P2_upsampled = layer.UpSampling2D(size=(2, 2), name='P2_upsampled')(P2)

    P1 = layer.Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced',
                kernel_initializer="he_normal")(C1)
    P1 = layer.Add(name='P1_merged')([P2_upsampled, P1])
    P1 = layer.Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P1', kernel_initializer="he_normal")(P1)
    return P1, P2, P3, P4, P5

def inception_resnet_v2_unet(input_shape):
    inception_resv2 = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=(256,256,3), weights='imagenet', include_top=False)
    return


def prediction_fpn_block(x, name, upsample=None):
    x = conv_relu(x, 128, 3, stride=1, name="predcition_" + name + "_1")
    x = conv_relu(x, 128, 3, stride=1, name="prediction_" + name + "_2")

    if upsample:
        x = layer.UpSampling2D(upsample)(x)
    return x

def decoder_block_no_bn(input, filters, skip, block_name, activation='relu'):
    x = layer.UpSampling2D()(input)
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv1', activation=activation)
    x = layer.concatenate([x, skip], axis=-1, name=block_name + '_concat')
    x = conv_relu(x, filters, 3, stride=1, padding='same', name=block_name + '_conv2', activation=activation)
    return x

def densenet_unet(channels, input_shape=(256, 256, 3)):
    densenet =  tf.keras.applications.DenseNet169(input_shape=input_shape, weights='imagenet', include_top=False)
    layer = tf.keras.layers
    conv1 = densenet.get_layer('conv1/relu').output
    conv2 = densenet.get_layer('pool2_relu').output
    conv3 = densenet.get_layer('pool3_relu').output
    conv4 = densenet.get_layer('pool4_relu').output
    conv5 = densenet.get_layer('bn').output
    conv5 = layer.Activation('relu', name='conv5_relu')(conv5)

    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)

    x = layer.concatenate(
        [
            prediction_fpn_block(P5, 'P5', (8, 8)),
            prediction_fpn_block(P4, 'P4', (4, 4)),
            prediction_fpn_block(P3, 'P3', (2, 2)),
            prediction_fpn_block(P2, 'P2')
        ]
    )

    x = conv_bn_relu(x, 256, 3, (1, 1), name='aggregation')

    x = decoder_block_no_bn(x, 128, conv1, 'up4')

    x = layer.UpSampling2D(name='up_after_4')(x)

    x = conv_relu(x, 64, 3, (1, 1), name='up5_conv1')
    x = conv_relu(x, 64, 3, (1, 1), name='up5_conv2')

    x = layer.Conv2D(channels, (1,1), activation='sigmoid', name='mask')(x)
    model = tf.keras.Model(densenet.input, x)
    return model

def edd_net(channels, input_shape, n=2, activation='relu'):
    """ Fully automatic acute ischemic lesion segmentation in DWI using convolutional neural networks
    n: how many deconvolution network to stack
    """
    input = tf.keras.Input(shape=input_shape)
    x = input
    deconvs = []
    for i in range(n):
        x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='conv1_%s' % i, activation=tf.nn.elu)(input)
        x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='conv1_1_%s' % i, activation=tf.nn.elu)(x)
        x = layer.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_%s' % i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_1_%s' % i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_2_%s' % i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_3_%s' % i, activation=tf.nn.elu)(x)
        x = layer.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_1_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_2_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_3_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_4_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_5_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_1_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_2_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_3_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_4_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_5_%s' %i, activation=tf.nn.elu)(x)
        x = layer.UpSampling2D()(x) # 64
        x = layer.Conv2DTranspose(64, (3,3), strides=1, padding='SAME', name='deconv2_1_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(64, (3,3), strides=1, padding='SAME', name='deconv2_2_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(64, (3,3), strides=1, padding='SAME', name='deconv2_3_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(32, (3,3), strides=1, padding='SAME', name='deconv2_4_%s' %i, activation=tf.nn.elu)(x)
        x = layer.UpSampling2D()(x)
        x = layer.Conv2DTranspose(32, (3,3), strides=1, padding='SAME', name='deconv4_4_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(2, (3,3), strides=1, padding='SAME', name='deconv3_4_%s' %i, activation=tf.nn.elu)(x)
        deconvs.append(x)
    
    ds = layer.Add(name='add')(deconvs)
    x = layer.concatenate(
        [ds, input]
    )
    x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='convlast_1_%s'% i, activation=tf.nn.elu)(x)
    x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='convlast_2_%s'% i, activation=tf.nn.elu)(x)
    x = layer.Conv2D(2, (3,3), strides=1, padding='SAME', name='convlast_3_%s'% i, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    return model

def resunet(channels, input_shape, n=2):
    """ White matter hyperintensity and stroke lesion segmentation and differentiation using convolutional neural networks
    """
    input = tf.keras.Input(shape=input_shape)
    x = input
    deconvs = []
    for i in range(n):
        x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='conv1_%s' % i, activation=tf.nn.elu)(input)
        x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='conv1_1_%s' % i, activation=tf.nn.elu)(x)
        x = layer.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_%s' % i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_1_%s' % i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_2_%s' % i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(64, (3,3), strides=1, padding='SAME', name='conv2_3_%s' % i, activation=tf.nn.elu)(x)
        x = layer.MaxPooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_1_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_2_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_3_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_4_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2D(128, (3,3), strides=1, padding='SAME', name='conv3_5_%s'% i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_1_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_2_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_3_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_4_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(128, (3,3), strides=1, padding='SAME', name='deconv1_5_%s' %i, activation=tf.nn.elu)(x)
        x = layer.UpSampling2D()(x) # 64
        x = layer.Conv2DTranspose(64, (3,3), strides=1, padding='SAME', name='deconv2_1_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(64, (3,3), strides=1, padding='SAME', name='deconv2_2_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(64, (3,3), strides=1, padding='SAME', name='deconv2_3_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(32, (3,3), strides=1, padding='SAME', name='deconv2_4_%s' %i, activation=tf.nn.elu)(x)
        x = layer.UpSampling2D()(x)
        x = layer.Conv2DTranspose(32, (3,3), strides=1, padding='SAME', name='deconv4_4_%s' %i, activation=tf.nn.elu)(x)
        x = layer.Conv2DTranspose(2, (3,3), strides=1, padding='SAME', name='deconv3_4_%s' %i, activation=tf.nn.elu)(x)
        deconvs.append(x)
    
    ds = layer.Add(name='add')(deconvs)
    x = layer.concatenate(
        [ds, input]
    )
    x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='convlast_1_%s'% i, activation=tf.nn.elu)(x)
    x = layer.Conv2D(32, (3,3), strides=1, padding='SAME', name='convlast_2_%s'% i, activation=tf.nn.elu)(x)
    x = layer.Conv2D(2, (3,3), strides=1, padding='SAME', name='convlast_3_%s'% i, activation=tf.nn.sigmoid)(x)

    model = tf.keras.Model(inputs=input, outputs=x)
    return model

if __name__ == '__main__':
    #inception_resnet_v2_unet()
    #densenet =  tf.keras.applications.DenseNet121(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
    #densenet =  tf.keras.applications.DenseNet169(input_shape=(256, 256, 3), weights='imagenet', include_top=False)
    #print(densenet.summary())
    #model = densenet_unet(2)
    model = edd_net(2, (64,64,1))
    print(model.summary())
    #print(inception_resv2.summary())
