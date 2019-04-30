"""
multi-label classification
github.com/keras-team/keras/issues/2166
binary_crossentropy: 1-or-more 0/1
categorical_crossentropy: 1-of-N (one-hot)

https://stats.stackexchange.com/questions/207794/what-loss-function-for-multi-class-multi-label-classification-tasks-in-neural-n
https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/

F2 loss: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809#205663
a good tutorial: http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/
                 https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/36809#205663
"""

import tensorflow as tf
import numpy as np
import glob
import argparse

import random
from keras.backend import binary_crossentropy
from loss import f2_loss, focal_loss, class_balance_ce1, class_balance_ce2, fbeta, combined_loss, f1, true_neg, false_neg, true_pos, false_pos
from keras.metrics import categorical_accuracy, binary_accuracy
from sklearn.metrics import confusion_matrix

from load_tfrecord import _parse_record
from model import vgg, inception_v3
from sklearn.model_selection import train_test_split
import os

def convert_to_estimator(model, train_fn, valid_fn, steps):
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
    config = tf.estimator.RunConfig(train_distribute=strategy)
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model, config=config, model_dir='model_dir')

    early_stopping = tf.contrib.estimator.stop_if_no_decrease_hook(
        estimator,
        metric_name='loss',
        max_steps_without_decrease=1000,
        min_steps=50)
    val_summary_hook = tf.contrib.training.SummaryAtEndHook('model_dir_new/eval')
    step_cnt_hook = tf.train.StepCounterHook(every_n_steps=50, output_dir='model_dir_new')
    for _ in range(10):
        estimator.train(input_fn=train_fn, steps=50, hooks=[early_stopping, step_cnt_hook])
        val = estimator.evaluate(input_fn=valid_fn, steps=steps, hooks=[val_summary_hook])  # steps=int(0.3 * len(files) / batch)
        print(val)
    #tf.estimator.train_and_evaluate(estimator,
    #                                train_spec=tf.estimator.TrainSpec(lambda: ds_train, hooks=[early_stopping]),
    #                                eval_spec=tf.estimator.EvalSpec(lambda: ds_test))


def input_fn(files, batch, shape, select=[], multilabel=True, is_train=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=28).map(lambda x: _parse_record(x, shape, select, multilabel), num_parallel_calls=28)
    if is_train:
        ds = ds.shuffle(32)
    ds = ds.repeat().batch(batch)
    return ds

def split_data(df):
    """ stratified by platform and label proportion
    :return:
    """
    from sklearn.model_selection import StratifiedKFold
    skfold = StratifiedKFold(5, shuffle=True, random_state=999)

def train(gpus, batch, model, multilabel, modal, lr, model_dir, loss, seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    if multilabel > 10:
        output_shape = 7
    else:
        output_shape = 2

    ## fixed random seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    initial_learning_rate = lr
    select = list(map(int, modal.split('_')))
    #shape = [181, 217, 181, 6]
    #files = np.array(glob.glob('../norm/*tfrecord'))

    #shape = (128, 128, 80, 5)
    #files = np.array(glob.glob('nii/*_136_*tfrecord'))

    shape = [80, 80, 60, 5]
    files = np.array(glob.glob('nii/*lower*tfrecord'))

    #files = np.array(glob.glob('tfrecords/*tfrecord'))
    #shape = (192, 192, 128, 5)

    os.system('mkdir -p model_store_new')
    os.system('mkdir -p tb_store_new')
    best_model_path = 'model_store_new/%s/weights.{epoch:003d}-{val_loss:.3f}-{val_fbeta:.3f}-{val_f1:.3f}.hdf5' % model_dir
    os.system('mkdir -p %s' % best_model_path)

    test_metrics = 'model_store_new/%s.npy' % model_dir
    tb_path = 'tb_store_new/%s/' % model_dir
    os.system('mkdir -p %s' % tb_path)

    train, test = train_test_split(np.arange(len(files)), test_size=0.3, random_state=seed)
    train, valid = train_test_split(np.arange(len(train)), test_size=0.3, random_state=seed)
    test_steps = int(len(test) / 2)
    ds_train = input_fn(files[train], batch, shape, select, multilabel)
    ds_valid = input_fn(files[valid], batch, shape, select, multilabel, False)
    ds_test = input_fn(files[test], 2, shape, select, multilabel, False)

    if len(select) > 0:
        shape[-1] = len(select) # selected modality

    if model == 'vgg':
        model = vgg(shape, output_shape)
    else:
        model = inception_v3(shape, output_shape)

    print(model.summary())
    print(len(model.layers))

    if loss=='bce':
        loss='binary_crossentropy'
    elif loss=='focal':
        loss=focal_loss(2, 0.75)
    elif loss=='weight':
        loss=class_balance_ce1
    else:
        loss=combined_loss

    #optimizer = tf.train.AdamOptimizer(learning_rate = initial_learning_rate)
    optimizer = tf.keras.optimizers.Adam(lr=initial_learning_rate)
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[f1, true_neg, false_neg, true_pos, false_pos, class_balance_ce2, f2_loss, focal_loss(2, 0.5), fbeta, class_balance_ce1, 'binary_accuracy', 'binary_crossentropy'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir=tb_path),
        tf.keras.callbacks.ModelCheckpoint(monitor='val_f1', save_best_only=False, save_weights_only=True, mode='max',
        #tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min',
                                           filepath=best_model_path),
                                           #filepath='checkpoints-2-%s/weights.{epoch:02d}-{val_loss:.2f}.hdf5' % str(multilabel)),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]

    val_steps = int(len(valid) / batch)
    history = model.fit(ds_train, steps_per_epoch=50, epochs=60, callbacks=callbacks,
                        validation_data=ds_valid, validation_steps=val_steps)
    history = history.history
    print(history)

    print(model.evaluate(ds_test, steps=test_steps))
    y_pred = model.predict(ds_test, steps=test_steps)
    print(y_pred)

    if not isinstance(multilabel, bool):
        y_pred = y_pred.argmax(axis=1)
    else:
        #y_pred[y_pred>0.3] = 1
        #y_pred[y_pred<=0.3] = 0
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0

    print(y_pred.shape)
    print(y_pred.sum())

    iterator = ds_test.make_initializable_iterator()
    next_elem = iterator.get_next()
    sess = tf.Session()
    sess.run(iterator.initializer)

    ys = []
    for i in range(test_steps):
        _, y = sess.run(next_elem)
        ys.append(y)

    yval = np.concatenate(ys, axis=0)
    if not isinstance(multilabel, bool):
        yval = yval[:, 1]

    print((y_pred==yval).mean())
    print(len(files[test]))
    print(len(y_pred))

    if not isinstance(multilabel, bool):
        testfile = files[test][:len(y_pred)]
        confusion = confusion_matrix(yval, y_pred, labels=[0, 1])
        np.save(test_metrics, confusion)
        print(testfile[(y_pred == 0) & (yval == 1)])
        print(testfile[(y_pred == 1) & (yval == 0)])
    else:
        testfile = files[test][:y_pred.shape[0]]
        for i in range(7):
            confusion = confusion_matrix(yval[:, i], y_pred[:, i], labels=[0, 1])
            np.save(test_metrics, confusion)
            print(i)
            print(confusion)
            print(testfile[(y_pred[:, i] == 0) & (yval[:, i] == 1)])
            print(testfile[(y_pred[:, i] == 1) & (yval[:, i] == 0)])
    sess.close()


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('-gpu', type=str)
    parse.add_argument('-batch', type=int, default=32)
    parse.add_argument('-multilabel', default=1, type=int)
    parse.add_argument('-model', default='vgg', type=str)
    parse.add_argument('-modal', default='0_1_2_3_4', type=str)
    parse.add_argument('--model_dir', type=str)
    parse.add_argument('-lr', default=0.00002, type=float)
    parse.add_argument('-loss', default='bce')
    parse.add_argument('-seed', default=999, type=int)
    parse.add_argument('-resume', default=False, type=bool, help='resume training from the best checkpoint') ## important TODO list:::
    args = parse.parse_args()
    train(args.gpu, args.batch, args.model, args.multilabel, args.modal, args.lr, args.model_dir, args.loss, args.seed)


if __name__ == '__main__':
    main()
