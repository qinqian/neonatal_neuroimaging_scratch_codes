import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
from model import vgg, inception_v3
import os
from load_tfrecord import _parse_record
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from model import vgg
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from loss import f2_loss, focal_loss, class_balance_ce1, class_balance_ce2, fbeta, combined_loss, f1, true_neg, false_neg, true_pos, false_pos
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
models = [
  #'model_store_bak2_preserve_best_only/vgg_gpu2_batch32_label1_modal0_1_2_3_4_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/vgg_gpu2_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/vgg_gpu2_batch32_label1_modal0_1_2_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/vgg_gpu2_batch32_label1_modal0_1_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/vgg_gpu2_batch32_label1_modal0_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/inception_gpu0_batch32_label1_modal0_1_2_3_4_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/inception_gpu0_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/inception_gpu0_batch32_label1_modal0_1_2_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/inception_gpu0_batch32_label1_modal0_1_lr0.0001_lossbce_seed99',
  #'model_store_bak2_preserve_best_only/inception_gpu0_batch32_label1_modal0_lr0.0001_lossbce_seed99'
  'model_store/vgg_gpu2_batch32_label1_modal0_1_2_3_4_lr0.0001_lossbce_seed99',
  'model_store/vgg_gpu2_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99',
  #'model_store/vgg_gpu2_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99',
  'model_store/vgg_gpu2_batch32_label1_modal0_1_2_lr0.0001_lossbce_seed99',
  'model_store/vgg_gpu2_batch32_label1_modal0_1_lr0.0001_lossbce_seed99',
  'model_store/vgg_gpu2_batch32_label1_modal0_lr0.0001_lossbce_seed99',
  'model_store/inception_gpu0_batch32_label1_modal0_1_2_3_4_lr0.0001_lossbce_seed99',
  #'model_store/inception_gpu0_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99/weights.005-0.372-0.838-0.599.hdf5',
  'model_store/inception_gpu0_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99',
  #'model_store/inception_gpu0_batch32_label1_modal0_1_2_3_lr0.0001_lossbce_seed99',
  'model_store/inception_gpu0_batch32_label1_modal0_1_2_lr0.0001_lossbce_seed99',
  'model_store/inception_gpu0_batch32_label1_modal0_1_lr0.0001_lossbce_seed99',
  'model_store/inception_gpu0_batch32_label1_modal0_lr0.0001_lossbce_seed99'
]

def load_model(i,last=True):
    modals = list(map(int, models[i].replace('_lr0.0001_lossbce_seed99', '').split('modal')[1].strip().split('_')))
    #modals = list(map(int, os.path.basename(os.path.dirname(models[i])).replace('_lr0.0001_lossbce_seed99', '').split('modal')[1].strip().split('_')))
    if 'vgg' in models[i]:
        v = vgg((80,80,60,len(modals)), 2)
    else:
        v = inception_v3((80,80,60,len(modals)), 2)
    
    if last:
        with open(os.path.join(models[i], 'checkpoint')) as inf:## 
            inf.readline()
            test = inf.readline().split(':')[1].strip().replace('"','')
            v.load_weights(os.path.join(models[i], test))
    else: # load by the best val_f1
        allmodels = glob.glob(os.path.join(models[i], 'weights.*hdf5.index'))
        allvalf1 = np.array(list(map(lambda x: float(x.split('-')[-1].replace('.hdf5.index','')), allmodels)))
        index = np.argmax(allvalf1)
        v.load_weights(allmodels[index].replace('.index', ''))
    return v, modals

def input_fn(files, batch, shape, select, multilabel=True, train=True, return_iter=False):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=batch).map(lambda x: _parse_record(x, shape, select, multilabel))
    if train:
        ds = ds.shuffle(96)
    ds = ds.repeat(1).batch(batch)
    if return_iter:
        sess = tf.Session()
        iterator = ds.make_initializable_iterator()
        next_elem = iterator.get_next()
        sess.run(iterator.initializer)
        return sess, next_elem
    return ds

def evaluate():
    #multilabel = 10
    multilabel = 1
    files = np.array(glob.glob('nii/*lower*tfrecord'))
    train, test = train_test_split(np.arange(len(files)), test_size=0.3, random_state=99)
    train, valid = train_test_split(np.arange(len(train)), test_size=0.3, random_state=99)
    #print(len(train))
    #print(len(valid))
    #print(len(test))
    #ds_test = input_fn(files[test], 20, (80,80,60,5), True, False)
    plt.figure()
    alltestaucs = []
    #for i in range(10):
    for i in [1, 7]:
        model, select = load_model(i, False)
        print(select)
        ds_test = input_fn(files[test], 20, (80,80,60,5), select, 1, False)
        ds_train = input_fn(files[train], 20, (80,80,60,5), select, 1, False)
        ds_valid = input_fn(files[valid], 20, (80,80,60,5), select, 1, False)

        optimizer = tf.keras.optimizers.Adam(lr=0.001)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=[f1, true_neg, false_neg, true_pos, false_pos, class_balance_ce2, f2_loss, focal_loss(2, 0.5), fbeta, class_balance_ce1, 'binary_accuracy', 'binary_crossentropy'])
        #test = model.evaluate(ds_test, steps=len(test)//16)
        #print(['binary', 'f1', 'f1', 'true_neg', 'false_neg', 'true_pos', 'false_pos', 'class_balance_ce2', 'f2_loss', 'focal_loss', 'fbeta', 'class_balance_ce1', 'binary_accuracy', 'binary_crossentropy'])
        #print(test)

        #prediction = model.predict(ds_train, steps=len(train)//20)[:, 1]
        #sess, ds = input_fn(files[train], 20, (80,80,60,5), select, multilabel, False, True)
        #ys = []
        #for j in range(len(train)//20):
        #    x, y = sess.run(ds)
        #    ys.append(y)
        #ys = np.concatenate(ys, axis=0)
        ##print(prediction.shape)
        ##print(ys.shape)
        #print(f1_score(ys[:, 1], prediction>0.5)) 
        #c = confusion_matrix(ys[:, 1], prediction>0.5, labels=[0,1])
        #np.save("%s_train.npy" % models[i], c)

        #prediction = model.predict(ds_valid, steps=len(valid)//20)[:, 1]
        #sess, ds = input_fn(files[valid], 20, (80,80,60,5), select, multilabel, False, True)
        #ys = []
        #for j in range(len(valid)//20):
        #    x, y = sess.run(ds)
        #    ys.append(y)
        #ys = np.concatenate(ys, axis=0)
        #print(prediction.shape)
        #print(ys.shape)
        #print(f1_score(ys[:, 1], prediction>0.5))
        #c = confusion_matrix(ys[:, 1], prediction>0.5, labels=[0,1])
        #np.save("%s_valid.npy" % models[i], c)

        prediction = model.predict(ds_test, steps=len(test)//20)[:, 1]
        sess, ds = input_fn(files[test], 20, (80,80,60,5), select, multilabel, False, True)
        ys = []
        for j in range(len(test)//20):
            x, y = sess.run(ds)
            ys.append(y)
        ys = np.concatenate(ys, axis=0)
        print(prediction.shape)
        print(ys.shape)
        fff = f1_score(ys[:, 1], prediction>0.5)

        fpr, tpr, thresholds = roc_curve(ys[:, 1], prediction, pos_label=1)
        rocauc = auc(fpr, tpr)
        print(rocauc)
        alltestaucs.append(rocauc)
        c = confusion_matrix(ys[:, 1], prediction>0.5, labels=[0,1])
        np.save("%s_test.npy" % models[i], c)

        if i==1 or i == 7:
            if i == 1:
                lw = 2
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve (area = %0.3f), f1 (%0.3f)' % (rocauc, fff))
            else:
                lw = 2.5
                plt.plot(fpr, tpr, color='red',
                         lw=lw, label='ROC curve (area = %0.3f), f1 (%0.3f)' % (rocauc, fff))

    np.save('%s_testauc.npy', np.array(alltestaucs))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('PVL Classification Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig("%s_merge_test_auc.pdf" % models[i])
    plt.savefig("Figure2_merge_test_auc.pdf")

    #file[test][(aa[:, 1] == 0) & (y[:, 1] == 1)]
    #files[test][(aa[:, 1] == 0) & (y[:, 1] == 1)]
    #files[test][(aa[:, 1] == 0) & (y[:, 1] == 1)]
    #files[test][(aa[:, 1] == 0) & (y[:, 1] == 1)]
    #a1=files[test][(aa[:, 1] == 0) & (y[:, 1] == 1)]
    #a0=files[test][(aa[:, 0] == 0) & (y[:, 0] == 1)]
    #a2=files[test][(aa[:, 2] == 0) & (y[:, 2] == 1)]
    #a3=files[test][(aa[:, 3] == 0) & (y[:, 3] == 1)]
    #a4=files[test][(aa[:, 4] == 0) & (y[:, 4] == 1)]
    #a5=files[test][(aa[:, 5] == 0) & (y[:, 5] == 1)]
    #a6=files[test][(aa[:, 6] == 0) & (y[:, 6] == 1)]
    #np.hstack([a0, a1, a2, a3, a4, a5, a6])
    #np.hstack([a0, a1, a2, a3, a4, a5, a6]).shape
    #np.unique(np.hstack([a0, a1, a2, a3, a4, a5, a6]), return_counts=True)

if __name__ == '__main__':
    evaluate()
