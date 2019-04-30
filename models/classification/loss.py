import tensorflow as tf
from itertools import product
from functools import partial

K = tf.keras.backend
def w_categorical_crossentropy(y_true, y_pred, weights):
    """https://github.com/keras-team/keras/issues/2115"""
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)

    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

#w_array = np.ones((3,3))
#w_array[2,1] = 1.2
#w_array[1,2] = 1.2
#ncce = partial(w_categorical_crossentropy, weights=w_array)
#ncce.__name__ ='w_categorical_crossentropy'

def focal_loss(gamma=2., alpha=.75):
    """ adapted from https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py"""
    K = tf.keras.backend
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1))-K.mean((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    focal_loss_fixed.__name__ = 'focal_loss'
    return focal_loss_fixed

def class_balance_ce1(y_true, y_pred):
    K = tf.keras.backend
    """https://blog.csdn.net/weixin_35653315/article/details/78327408"""
    ce = K.binary_crossentropy(y_pred, y_true)
    num_samp = tf.cast(tf.reduce_prod(tf.shape(y_true)), tf.float32)

    pos_weight = tf.cast(tf.equal(y_true, 1), tf.float32)
    neg_weight = 1 - pos_weight
    n_pos = tf.reduce_sum(pos_weight)
    n_neg = tf.reduce_sum(neg_weight)

    def has_pos():
        return tf.reduce_sum(ce * pos_weight)/n_pos
    def has_neg():
        return tf.reduce_sum(ce * neg_weight)/n_neg
    def no():
        return tf.constant(0.0)

    pos_loss = tf.cond(n_pos > 0, has_pos, no)
    neg_loss = tf.cond(n_neg > 0, has_neg, no)

    return (pos_loss + neg_loss)/2

def class_balance_ce2(y_true, y_pred):
    """https://arxiv.org/pdf/1504.06375.pdf"""
    K = tf.keras.backend
    y = tf.cast(y_true, tf.float32)

    n_pos = tf.reduce_sum(y)
    n_neg = tf.reduce_sum(1.-y)

    beta = tf.ones_like(y_pred) * n_neg / (n_pos + n_neg)
    weight = tf.where(tf.equal(y, 1.), beta, 1-beta)

    return K.mean(K.binary_crossentropy(y_pred, y_true) * weight)

def combined_loss(y_true, y_pred):
    #return class_balance_ce2(y_true, y_pred) + f2_loss(y_true, y_pred)
    K = tf.keras.backend
    #return K.binary_crossentropy(y_true, y_pred) + f2_loss(y_true, y_pred)
    return K.mean(K.binary_crossentropy(y_true, y_pred)) + f2_loss(y_true, y_pred)

def f1(y_true, y_pred):
    """ https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras """
    K = tf.keras.backend
    beta = 1
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred)[...,1]
    y_true = y_true[...,1]
    tp = K.sum(y_true * y_pred_bin)
    c2 = K.sum(y_pred_bin)
    c3 = K.sum(y_true)

    precision = tp/(c2+K.epsilon())
    recall = tp/(c3+K.epsilon())

def f1_multi(y_true, y_pred, threshold_shift=0):
    """ https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras """
    K = tf.keras.backend
    beta = 1
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

def fbeta(y_true, y_pred, threshold_shift=0):
    """ https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras """
    K = tf.keras.backend
    beta = 2
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)
    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)), axis=1)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

def false_pos(y_true, y_pred):
    K = tf.keras.backend
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred)
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    return fp

def true_pos(y_true, y_pred):
    K = tf.keras.backend
    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred)
    tp = K.sum(K.round(y_true * y_pred_bin), axis=1)
    return tp

def true_neg(y_true, y_pred):
    K = tf.keras.backend
    # just in case of hipster activation at the final layer
    y_pred_neg = K.clip(1-y_pred, 0, 1)
    # shifting the prediction threshold from .5 if needed
    #y_pred_bin = K.round(y_pred_neg)
    #tn = K.sum(K.round((1-y_true) * y_pred_bin), axis=1)
    y_pred_bin = K.round(y_pred_neg)[..., 1]
    tn = K.sum(K.round((1-y_true[..., 1]) * y_pred_bin))
    return tn

def false_neg(y_true, y_pred):
    K = tf.keras.backend
    y_pred = K.clip(y_pred, 0, 1)
    y_pred_bin = K.round(y_pred)[..., 1]
    fn = K.sum(K.round(K.clip(y_true[..., 1] - y_pred_bin, 0, 1)))
    return fn

def f2_loss(y_true, y_pred):
    """https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809#205663"""
    K = tf.keras.backend
    smooth = 0.1
    epsilon = 1e-4
    beta = 2
    beta2 = beta ** 2
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    y_true = K.clip(y_true, epsilon, 1.0 - epsilon)
    true_and_pred = y_true * y_pred
    ttp_sum = K.sum(true_and_pred, axis=1)
    tpred_sum = K.sum(y_pred, axis=1)
    ttrue_sum = K.sum(y_true, axis=1)
    tprecision = ttp_sum / tpred_sum
    trecall = ttp_sum / ttrue_sum
    tf_score = ((1 + beta2) * tprecision * trecall + smooth) / (beta2 * tprecision + trecall + smooth)
    return -K.mean(tf_score)

f2_loss.__name__ = 'f2_loss'

#def fbeta_bce_loss(self, y_true, y_pred, beta = 2):
#    """https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/35899"""
#    K = tf.keras.backend
#    beta_sq = beta ** 2
#    tp_loss = K.sum(y_true * (1 - K.binary_crossentropy(y_pred, y_true)), axis=-1)
#    fp_loss = K.sum((1 - y_true) * K.binary_crossentropy(y_pred, y_true), axis=-1)
#    return - K.mean((1 + beta_sq) * tp_loss / ((beta_sq * K.sum(y_true, axis = -1)) + tp_loss + fp_loss))

