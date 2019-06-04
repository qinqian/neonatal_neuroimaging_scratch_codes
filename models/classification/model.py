import tensorflow as tf

layers = tf.keras.layers

def vgg(shape, output_shape):
    strides = (2,2,2)
    inputvol = tf.keras.Input(shape=shape)
    #x = layers.Conv3D(filters=8, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(inputvol)
    #x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=8, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(inputvol)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), strides=strides)(x)

    x = layers.Conv3D(filters=16, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=16, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), strides=strides)(x)

    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=32, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), strides=strides)(x)

    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=64, kernel_size=(3,3,3), padding='same', activation=tf.nn.elu)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2,2,2), strides=strides)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation=tf.nn.elu)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation=tf.nn.elu)(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(output_shape, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputvol, outputs=x)
    return model

def inception(x, c1, c2, c3, c4):
    p1_1 = layers.Conv3D(c1, kernel_size=1, activation=tf.nn.elu)
    p2_1 = layers.Conv3D(c2[0], kernel_size=1, activation=tf.nn.elu)
    p2_2 = layers.Conv3D(c2[1], kernel_size=3, padding='SAME', activation=tf.nn.elu)

    p3_1 = layers.Conv3D(c3[0], kernel_size=1, activation=tf.nn.elu)
    p3_2 = layers.Conv3D(c3[1], kernel_size=5, padding='SAME', activation=tf.nn.elu)

    p4_1 = layers.MaxPooling3D(pool_size=(3,3,3), strides=1, padding='SAME')
    p4_2 = layers.Conv3D(c4, kernel_size=1, activation=tf.nn.elu)

    p1 = p1_1(x)
    p2 = p2_2(p2_1(x))
    p3 = p3_2(p3_1(x))
    p4 = p4_2(p4_1(x))
    return layers.concatenate([p1, p2, p3, p4], axis=-1)

def inception_v3(shape, output_shape):
    inputvol = tf.keras.Input(shape=shape)
    strides = (2,2,2)
    #x = layers.Conv3D(filters=64, kernel_size=(7,7,7), strides=strides, padding='valid', activation=tf.nn.elu)(inputvol)
    #x = layers.MaxPooling3D(pool_size=(3,3,3), strides=strides, padding='same')(x)

    #x = layers.Conv3D(filters=64, kernel_size=(1,1,1), padding='same', activation=tf.nn.relu)(inputvol)

    x = layers.Conv3D(filters=192, kernel_size=(3,3,3), strides=strides, padding='same', activation=tf.nn.elu)(inputvol)
    x = layers.MaxPooling3D(pool_size=(3,3,3), strides=strides)(x)

    x = inception(x, 64,  (96,  128), (16, 32),  32)
    x = inception(x, 128, (128, 192), (32, 96),  64)
    x = layers.MaxPooling3D(pool_size=(3,3,3), strides=strides)(x)

    x = inception(x, 192, (96, 208), (16, 48),   64)
    x = inception(x, 160, (112, 224), (24, 64),  64)
    x = inception(x, 128, (128, 256), (24, 64),  64)
    x = inception(x, 112, (144, 288), (32, 64),  64)
    x = inception(x, 256, (160, 320), (32, 128), 128)
    x = layers.MaxPooling3D(pool_size=(3,3,3), strides=strides)(x)

    x = inception(x, 256, (160, 320), (32, 128), 128)
    #x = inception(x, 384, (192, 384), (48, 128), 128)
    x = layers.GlobalAveragePooling3D()(x)

    x = layers.Dense(output_shape, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputvol, outputs=x)
    return model

def resnext(shape):
    inputvol = tf.keras.Input(shape=shape)
    strides = (2,2,2)
    inputvol = tf.keras.Input(shape=shape)
    x = layers.Conv3D(filters=64, kernel_size=(7,7,7), strides=strides, padding='valid', activation=tf.nn.elu)(inputvol)
    model = tf.keras.Model(inputs=inputvol, outputs=x)
    return model

if __name__ == '__main__':
    print(inception_v3((87, 87, 112 ,5), 2).summary())
