"""
cam and grad-cam
cam: http://cnnlocalization.csail.mit.edu/ wi*Last_conv
grad-cam: keras book ave_channel_last_conv_grad * Last_conv
"""
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from model import vgg, inception_v3
from load_tfrecord import _parse_record
from sklearn.model_selection import train_test_split
import numpy as np
import glob
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import ants

K=tf.keras.backend
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
shape = [80,80,60,5]
select = [1,2]
files = np.array(glob.glob('nii/*lower*tfrecord'))
train, test = train_test_split(np.arange(len(files)), test_size=0.3, random_state=99)

def input_fn(files, batch, shape, select=[], multilabel=True, train=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=batch).map(lambda x: _parse_record(x, shape, select, multilabel))
    if train:
        ds = ds.shuffle(96)
    ds = ds.repeat().batch(batch)
    return ds

def load_batch():
    ds_test = input_fn(files[test], 2, shape, select, 1, False)
    test_steps = int(len(test) / 2)
    iterator = ds_test.make_initializable_iterator()
    next_elem = iterator.get_next()
    s = tf.Session()
    s.run(iterator.initializer)
    ys = []
    for i in range(test_steps):
        _, y = s.run(next_elem)
        ys.append(y)
    y = np.concatenate(ys, axis=0)
    return ds_test, test_steps, y

def load_model():
    shape[-1] = len(select)
    m = vgg(shape, 2)
    #m = inception_v3(shape, 2)
    print(m.summary())
    with open('checkpoints-2-1/checkpoint') as f:
    #with open('checkpoints-1-1/checkpoint') as f:
        z = f.readline().split()[-1].replace('"', '')
        
    from loss import f2_loss, focal_loss, class_balance_ce1, class_balance_ce2, fbeta, combined_loss
    #m.load_weights(os.path.join('checkpoints-1-1', z))
    m.load_weights(os.path.join('checkpoints-2-1', z))
    m.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=[class_balance_ce2, f2_loss, focal_loss(2, 0.5), fbeta, class_balance_ce1, 'binary_accuracy', 'binary_crossentropy'])
    return m

def load_one(i):
    shape = [80,80,60,5]
    ds_test = input_fn([i], 2, shape, select, multilabel=1, train=False)
    iterator = ds_test.make_initializable_iterator()
    next_elem = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)
    X, label = sess.run(next_elem)
    return X, label

d, s, l = load_batch()
m = load_model()
eval = m.evaluate(d, steps=s)
print(eval)
preds = m.predict(d, steps=s)
print(preds[:5])
print(l[:5])
pred = preds.argmax(axis=1)
print(pred)
print(l[:,1])
print(np.mean(pred == l[:, 1]))
confusion = confusion_matrix(l[:, 1], pred, labels=[0, 1])

pred = (preds[:, 1] > 0.2).astype('int')
print(np.mean(pred == l[:, 1]))
confusion = confusion_matrix(l[:, 1], pred, labels=[0, 1])

print(confusion)
pred = (preds[:, 1] > 0.5).astype('int')
print(np.mean(pred == l[:, 1]))
confusion = confusion_matrix(l[:, 1], pred, labels=[0, 1])
print(confusion)

X, l = load_one('nii/1546__lower_1mm_merge_chan.tfrecord')
print(m.predict(X))
print(l)

grads = K.gradients(m.output[:, 1], m.get_layer('conv3d_11').output)[0]
iterate = K.function([m.input], [grads, m.get_layer('conv3d_11').output])
#grads = K.gradients(m.output[:, 1], m.get_layer('conv3d_56').output)[0]
#iterate = K.function([m.input], [grads, m.get_layer('conv3d_56').output])
g, c = iterate([X])

for i in range(64):
#for i in range(128):
    c[:, :, :, :, i] *= g[:, :, :, :, i]

heatmap = np.mean(c, axis=-1)
print(heatmap.shape)

print(files[test][3])
print(files[test][4])
print(files[test][8])

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
heatmap[heatmap<=0.5] = 0
print(heatmap)

#:ants
#:plt.matshow(heatmap)
#:plt.savefig('test.pdf')

import ants
def resample_plot(i):
    z=ants.image_read('nii/%s__iso_1mm_merge_chan.nii.gz' % i)
    z=z.split_channels()
    ok=ants.resample_image(z[2], [10,10,7], True) # vgg
    #ok=ants.resample_image(z[2], [4,4,2], True)  # inception
    ok[:, :, :] = heatmap[0, :, :, :]
    final = ants.resample_image(ok, z[2].shape,True, interp_type=4)
    print(final.shape)
    from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
    print(z[2].shape)
    ants.plot(z[2], nslices=60, ncol=6, filename='%s_raw.pdf' % i, dpi=750, axis=2)
    ants.plot(z[2], final, nslices=60, ncol=6, filename='%s.pdf' % i, dpi=750, axis=2, cbar=True, overlay_alpha=0.6, overlay_cmap='jet') # nilearn_cmaps['blue_transparent_full_alpha_range'])

resample_plot('1546')
