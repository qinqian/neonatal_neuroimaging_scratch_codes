import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-5):
    K = tf.keras.backend
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - (dice_coef(y_true, y_pred))

def sigmoid_dice_loss(y_true, y_pred):
    K = tf.keras.backend
    return K.mean(K.binary_crossentropy(y_true, y_pred)) * 0.5 + dice_coef_loss(y_true, y_pred) * 0.5

def binary_crossentropy(y_true, y_pred):
    K = tf.keras.backend
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def double_head_loss(y_true, y_pred):
    mask = sigmoid_dice_loss(y_true[..., 0], y_pred[..., 0])
    contour = sigmoid_dice_loss(y_true[..., 1], y_pred[..., 1])
    return mask+contour

def softmax_dice_loss(y_true, y_pred):
    K = tf.keras.backend
    #    K.categorical_crossentropy
    return K.categorical_crossentropy(y_true, y_pred) * 0.5 + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.3 + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2

def hard_dice_coef(y_true, y_pred, smooth=1e-5):
    K = tf.keras.backend
    y_true_f = K.flatten(K.round(y_true[..., 0]))
    y_pred_f = K.flatten(K.round(y_pred[..., 0]))
    intersection = K.sum(y_true_f * y_pred_f)
    #return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    dice = 2 * (intersection) / (K.sum(y_true_f)+K.sum(y_pred_f))
    if smooth>0:
        #return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return tf.clip_by_value(dice, 0, 1 - smooth)
    else:
        return dice

def hard_dice_coef_ch1(y_true, y_pred, smooth=1e-5):
    K = tf.keras.backend
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)

    dice = 2 * (intersection) / (K.sum(y_true_f)+K.sum(y_pred_f))
    if smooth>0:
        #return 100. * (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return tf.clip_by_value(dice, 0, 1 - smooth)
    else:
        return dice

def focal_loss(gamma=2., alpha=.25):
    K = tf.keras.backend
    def focal_loss_fixed(y_true, y_pred):
        pt = tf.where(y_true > 0, y_pred, 1 - y_pred)
        t = tf.ones_like(y_true)
        alpha_t = 0.3 * t
        alpha = tf.where(y_true > 0, alpha_t, t - alpha_t)
        loss = -alpha * (tf.pow(1-pt, gamma)) * tf.log(tf.minimum(pt + 1e-12, 1))
        return tf.reduce_sum(loss)
    focal_loss_fixed.__name__ = 'focal_loss'
    return focal_loss_fixed

def focal_loss_2(gamma=2., alpha=.5):
    """ weird with negative values """
    def focal_loss_fixed(y_true, y_pred):
        K = tf.keras.backend
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0)))
    return focal_loss_fixed

def pixwise_crossentropy(y_true, y_pred):
    output = tf.clip_by_value(y_pred, 1e-8, 1.-1e-8)
    return -tf.reduce_sum(y_true * tf.log(y_pred))
   
