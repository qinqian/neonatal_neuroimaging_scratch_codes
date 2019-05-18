#!/bin/bash -ex

mkdir -p high_template
#
#IFS=","
#while read line; do
#   field=($line)
#   cp training_data_high/${field[0]} high_template/
#   cp training_data_high/${field[1]} high_template/
#   cp training_data_high/${field[2]} high_template/
#   cp training_data_high/${field[3]} high_template/
#done < template_high.csv

cd high_template && cp ../template_high.csv .
antsMultivariateTemplateConstruction2.sh -n 0 -g 0.2 -a 1 -d 3 -i 3 -k 4 -z sub-10002254_20110717_t21.nii.gz -z sub-10002254_20110717_t22.nii.gz -z sub-10002254_20110717_t1.nii.gz -z sub-10002254_20110717_dwi.nii.gz -f 4x2x1 -s 2x1x0vox -q 30x20x4 -t SyN -m CC -c 1 -o FUDAN_PVL_240 template_high.csv
