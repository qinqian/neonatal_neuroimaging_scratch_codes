#!/usr/bin/env python
import os
import sys
import ants
import nibabel as nib
import tensorflow as tf
import numpy as np
import argparse

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=value))

def write_tfrecord(nii):
    import pandas as pd
    df = pd.read_table('results/data_for_model_manual_fixed_resample_failure.xls')
    #meta = df.loc[df.iloc[:, 0] == int(os.path.basename(nii).replace('__not_iso_merge_chan.nii.gz', '')), :]  # 1559__iso_1mm_merge_chan.nii.gz
    meta = df.loc[df.iloc[:, 0] == int(os.path.basename(nii).replace('__lower_1mm_merge_chan.nii.gz', '')), :]  # 1559__iso_1mm_merge_chan.nii.gz
    #meta = df.loc[df.iloc[:, 0] == int(os.path.basename(nii).replace('___136__1mm_merge_chan.nii.gz', '')), :]  # 1559__iso_1mm_merge_chan.nii.gz
    labels = meta.iloc[0, 7:].values.astype(np.int16)
    print(labels)

    z = ants.image_read(nii)
    z = z.numpy()
    z = z.astype(np.int16)
    #z = nib.load('/home/qinqian/MRI_project_cleaned_data/100__merge_chan.nii.gz')
    #z = z.get_data()
    writer = tf.python_io.TFRecordWriter('%s.tfrecord' % nii.replace('.nii.gz', ''))
    exam = tf.train.Example(features=tf.train.Features(
           feature={'image': _bytes_feature([z.tostring()]),
                    'label': _bytes_feature([labels.tostring()])}
                    #'label': _int_feature([labels.tostring()])}
    ))
    writer.write(exam.SerializeToString())
    writer.close()
    

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--infile', help='input nii.gz')
    args = p.parse_args()
    if not args.infile:
        p.print_help()
        sys.exit(1)
    write_tfrecord(args.infile)

if __name__ == '__main__':
    main()
