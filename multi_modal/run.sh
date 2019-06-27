#!/bin/bash -ex

docker run --rm -t \
   -u $(id -u):$(id -g) \
   -v $PWD/output/sub-102/T1/subject_idsub-102/:/data \
   biomedia/dhcp-structural-pipeline:latest subject1 session1 44 \
           -T1 sub-102_T1w_noise_corrected_reoriented_enh.nii.gz -t 24
