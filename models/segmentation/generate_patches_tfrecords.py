import tensorflow as tf
import nibabel as nib
import os
import numpy as np
import pandas as pd
from losses import double_head_loss, sigmoid_dice_loss, hard_dice_coef, hard_dice_coef_ch1
from prediction_valid import predict_whole_brain, dice_ratio

def split_patches(image, patch_size, overlap_size):
    """
    :param image:  4D image, last dimension is channel, now support nii
    :param patch_size: sliding window size
    :param overlap_size: overlap between windows
    :return:
    """
    D, H, W, C = image.shape
    overlap_size = int(overlap_size)
    drange = list(range(0, D-patch_size+1, overlap_size))
    hrange = list(range(0, H-patch_size+1, overlap_size))
    if (D-patch_size) % overlap_size != 0:
        drange.append(D-patch_size)
    if (H-patch_size) % overlap_size != 0:
        hrange.append(H-patch_size)
    patches = []
    for d in drange:
        for h in hrange:
            patches.append((d,h))
    return patches

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def get_tfrecord(niis, crop_size, label, output_file):
    """
    :param model:
    :return:
    """
    images = []
    for i in niis:
        images.append(nib.load(i).get_data())
    #images = np.array(images)
    images = np.stack(images, axis=0)
    images = np.transpose(images, (1,2,3,0)) # channels last

    patches = split_patches(images, crop_size, crop_size//2)
    #patches = view_as_windows(images, crop_size, crop_size//2)
    #output = np.zeros((images.shape[0], images.shape[1], images.shape[2], 2), dtype=np.float32)
    writer = tf.python_io.TFRecordWriter(output_file)
    slices_image = []
    for s in range(images.shape[2]):
        for d, h in patches:
            image = images[:, :, s, :]
            image = image/np.max(image)*255
            image = image[d:(d+crop_size), h:(h+crop_size), :]
            slices_image.append(image)

    slices_image = np.stack(slices_image, axis=0)
    slices_image = slices_image.astype(np.uint8)
    label = label.astype(np.uint8)
    patches = np.array(patches)
    example = tf.train.Example(features=tf.train.Features(
              feature={'image': _bytes_feature([slices_image.tostring()]),
                       'original_shape': _int64_feature(images.shape),
                       'patchid': _bytes_feature([patches.tostring()]),
	               'label': _bytes_feature([label.tostring()])}
              ))
    writer.write(example.SerializeToString())
    writer.close()

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def generate_tfrecords(df):
    df = pd.read_csv(df)
    os.system('mkdir -p patch64')
    for i in range(df.shape[0]):
        niis = [df.loc[:, 'flair'][i], df.loc[:, 'dwi'][i], df.loc[:, 't1'][i]]
        label = [df.loc[:, 'label'][i], df.loc[:, 'label2'][i]]
        label = nib.load(label[0]).get_data() | nib.load(label[1]).get_data().astype(np.int32)
        get_tfrecord(niis, 64, label, "patch64/%s.tfrecord"%i)
    return

def _parse_record(example):
    featdef = {
        'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'label': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'patchid': tf.FixedLenFeature(shape=[], dtype=tf.string),
        'original_shape': tf.FixedLenFeature(4, tf.int64)
    }
    example = tf.parse_single_example(example, featdef)
    im = tf.decode_raw(example['image'], tf.uint8)
    label = tf.decode_raw(example['label'], tf.uint8)
    patch = tf.decode_raw(example['patchid'], tf.int64)
    size = example['original_shape']
    return im, label, patch, size

def input_fn(files, batch, patch):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=10).map(lambda x: _parse_record(x), num_parallel_calls=28)
    #if train:
    #    ds = ds.shuffle(32)
    #ds = ds.repeat().batch(batch)
    iterator = ds.make_initializable_iterator()
    next_elem = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    a, y, p, s = sess.run(next_elem)
    p = np.reshape(p, (p.shape[0]//2, 2))
    a = np.reshape(a, (p.shape[0]*s[2], patch, patch,3))
    y = np.reshape(y, (s[:-1]))
    return a, y, p, s

def test_load(batch, crop_size, model):
    import glob
    dices_all = []
    for i in range(102):
        x, y, patch, size = input_fn(glob.glob('patch64/%s.tfrecord' % i), 16, 64)
        output = np.zeros(tuple(list(size[:-1])+[2]), dtype=np.float32)
        overlap_count = np.zeros(tuple(list(size[:-1])+[2]), dtype=np.float32)
        x = x/127.5 - 1
        prediction = model.predict(x, batch_size=batch)
        for s in range(size[2]):
            for p, k in zip(range(patch.shape[0]), range(s*patch.shape[0], patch.shape[0]*(s+1))): # # NOTE: k*(s+1) should be there to match the results
                d, h = patch[p, :]
                output[d:(d+crop_size), h:(h+crop_size), s, :] += prediction[k, :, :, :] 
                overlap_count[d:(d+crop_size), h:(h+crop_size), s, :] += 1
        output /= overlap_count
        output = output.argmax(axis=-1)
        dices_all.append(dice_ratio(output, y))
        #break
    dices_all = np.array(dices_all)
    print(dices_all)
    print(np.mean(dices_all))
    print((dices_all>0).mean())

if __name__ == '__main__':
    #generate_tfrecords('data_3channel_with_mask.xls')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from unet import edd_net, densenet_unet
    model = edd_net(2, (64,64,3), n=2)
    print(len(model.layers))
    model = densenet_unet(2, (64,64,3))
    print(len(model.layers))
    #model.load_weights('models_store.bak/edd_seed_double_00_64_batch16/best_edd_293_0.03125_0.96261.h5')
    #model.load_weights('models_store.bak/edd_seed_bce_00_64_batch16/best_edd_170_0.00045_0.99814.h5')
    #model.load_weights('models_store.bak2/edd_seed_bce_00_64_batch16/best_edd_279_0.00020_0.99925.h5')
    #adam = tf.keras.optimizers.Adam(lr=2e-4, amsgrad=True)
    #model.compile(loss = double_head_loss, optimizer=adam,
    #              metrics=[sigmoid_dice_loss, hard_dice_coef, hard_dice_coef_ch1, tf.keras.metrics.binary_crossentropy])
    #test_load(16, 64, model)
