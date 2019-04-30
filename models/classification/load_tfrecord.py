import tensorflow as tf
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()
import glob
import numpy as np

def reduce_var(x, axis=None, keepdims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

def reduce_std(x, axis=None, keepdims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))


def test_load(ds_train):
    iterator = ds_train.make_initializable_iterator()
    filenames = tf.placeholder(dtype=tf.string, shape=[None])
    next_elem = iterator.get_next()
    with tf.Session() as sess:
        #files = glob.glob('nii/*tfrecord')
        files = glob.glob('tfrecords/*tfrecord')
        sess.run(iterator.initializer, feed_dict={filenames: files})
        for i in files:
            image, label = sess.run(next_elem)
            print(image.shape, label.shape)
            #print(image.sum(axis=-1))

def _parse_record(example, shape, select=[], multilabel=True, aug=False):
    featdef = {
        'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'label': tf.FixedLenFeature(shape=[], dtype=tf.string)
    }
    example = tf.parse_single_example(example, featdef)
    im = tf.decode_raw(example['image'], tf.int16)
    im = tf.cast(tf.reshape(im, shape), tf.float32)

    if len(select) > 0:
        im = tf.split(im, shape[-1], axis=-1)
        ims = []
        for i in select:
            ims.append(im[i])
        im = tf.concat(ims, axis=3)

    sd = reduce_std(im)

    #mean = tf.reduce_mean(im)
    im = (im - tf.reduce_mean(im))/sd
    #im = im - tf.reduce_mean(im)
    label = tf.decode_raw(example['label'], tf.int16)
    if multilabel > 10:
        return im, label

    if isinstance(multilabel, int):
        label = tf.cast(label[multilabel], tf.int32)
        label = tf.one_hot(label, depth=2)

    return im, label

if __name__ == '__main__':
    sess = tf.Session()
    files = glob.glob('nii/*lower*tfrecord')
    shape=(80,80,60,5)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    ds_train = tf.data.TFRecordDataset(files, num_parallel_reads=28).map(lambda x: _parse_record(x, shape, [1,3]))
    ds_train = ds_train.shuffle(20).batch(5).repeat().prefetch(5)
    test_load(ds_train)

