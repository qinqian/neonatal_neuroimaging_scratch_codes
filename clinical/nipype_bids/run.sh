#!/bin/bash -ex

mkdir -p /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/tmp
export TMPDIR=/data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/tmp
echo $TMPDIR

mkdir -p cleaned_data/pvl
mkdir -p cleaned_data/control

main() {
#  neo_lowres run --bids /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/ --clinical NA --candidate age_less_than_1month_positive.txt --output /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/pvl/ --backend=None
  neo_lowres run --bids /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/ --clinical NA --candidate age_less_than_1month_positive.txt --output /data1/qinqian/MRI/01_PVL_AI/cleaned_data/pvl/ --backend=p

#  python group_analysis.py --list0 /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/pvl/align/*/*1/*nii.gz --list /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/pvl/transform/*/*Warp.nii.gz --list2 /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/pvl/transform/*/*mat --output /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/pvl/

#  neo_lowres run --bids /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/ --clinical NA --candidate age_less_than_1month.txt --output /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/pvl/ --backend=sge

#  neo_lowres run --bids /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/ --clinical NA --candidate control_evidence_with_no_significant_anamoly.txt --output /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/control --backend=None
}

main

#CUDA_VISIBLE_DEVICES="0" Rscript keras_with_generator.R res
#CUDA_VISIBLE_DEVICES="0" Rscript keras_with_generator.R vgg
#CUDA_VISIBLE_DEVICES="2" Rscript keras_with_generator.R res 1000

#CUDA_VISIBLE_DEVICES="0" Rscript keras_with_generator.R vgg
#CUDA_VISIBLE_DEVICES="1" Rscript keras_with_generator.R dense

#CUDA_VISIBLE_DEVICES="2" Rscript keras_with_generator_bak.R res
#CUDA_VISIBLE_DEVICES="0" Rscript keras_with_generator_bak.R res
#python mri_3d_classification.py
#python mri_2d_classification.py

## python mri_3d_classification.py

