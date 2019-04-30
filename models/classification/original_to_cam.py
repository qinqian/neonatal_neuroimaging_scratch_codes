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
#select = [0,1,2,3,4]
select = [0,1,2,3]
files = np.array(glob.glob('nii/*lower*tfrecord'))
seed = 99
train, test = train_test_split(np.arange(len(files)), test_size=0.3, random_state=seed)
train, valid = train_test_split(np.arange(len(files)), test_size=0.3, random_state=seed)

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
    print(len(m.layers))
    m.load_weights('model_store/vgg_gpu2_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99/weights.006-0.455-0.826-0.583.hdf5')
    #m.load_weights('model_store.bak/vgg_gpu2_batch32_label1_modal0_1_2_3_4_lr0.0001_lossbce_seed99/weights.011-0.584-0.853-0.852.hdf5')
    from loss import f2_loss, focal_loss, class_balance_ce1, class_balance_ce2, fbeta, combined_loss
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

def get_grad_cam(i):
    X, l = load_one(i)
    print(X.shape)
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
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #heatmap[heatmap<=0.1] = 0
    heatmap[heatmap<=0.2] = 0
    #print(heatmap)
    return heatmap

def resample_plot(i):
    import ants
    z=ants.image_read('nii/%s__iso_1mm_merge_chan.nii.gz' % i)
    z=z.split_channels()
    ok=ants.resample_image(z[2], [10,10,7], True) # vgg
    #ok=ants.resample_image(z[2], [4,4,2], True)  # inception
    ok[:, :, :] = heatmap[0, :, :, :]
    final = ants.resample_image(ok, z[2].shape,True, interp_type=4)
    print(final.shape)
    from nilearn.plotting.cm import _cmap_d as nilearn_cmaps
    print(z[2].shape)
    ##ants.plot(z[2], nslices=60, ncol=6, filename='%s_raw.pdf' % i, dpi=750, axis=2)
    ants.plot(z[2], final, nslices=60, ncol=6, filename='%s.pdf' % i, dpi=750, axis=2, cbar=True, overlay_alpha=0.6, overlay_cmap='jet') # nilearn_cmaps['blue_transparent_full_alpha_range'])

d, s, l = load_batch()
m = load_model()
#eval = m.evaluate(d, steps=s)
#print(eval)
preds = m.predict(d, steps=s)
##index, = np.where(preds[:, 1]>0.6)
index, = np.where(preds[:, 1]<0.5) ## false neg
for i in index:
    print(preds[:,1][i])
    print(l[i])
    #if l[i][1] == 0:# False pos
    if l[i][1] == 1: # False neg
        print(l[i])
        candidate = files[test][i]
        #pred = preds.argmax(axis=1)
        #print(np.mean(pred == l[:, 1]))
        #confusion = confusion_matrix(l[:, 1], pred, labels=[0, 1])
        #print(confusion)
        heatmap = get_grad_cam(candidate)
        print(candidate)
        resample_plot(os.path.basename(candidate).split('_')[0])
