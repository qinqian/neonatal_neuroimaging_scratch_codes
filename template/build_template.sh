#!/bin/bash

#rm -f template.list
#while read line; do
#  grep ${line}$'\t' ../UGT1A1/phaseA_preproc/fmri_images.list | grep 'DTI' >> template.list
#done < ids

mkdir -p data
cd data
n=0
while read line; do
  fields=($line)
  mkdir -p subj${n}/dti
  dcm2niix -i y -o subj${n}/dti -f ${n}_${fields[0]}_${fields[7]}_"%s" $(echo ${fields[11]} | cut -f 1 -d,)
  let n++
done < ../template.list

#for i in data/subj*/*bold/*nii.gz; do
#    echo $i
#    #slicetimer -r 2 --odd -i $i -o ${i/.nii.gz/}_tshift.nii.gz &
#    #mcflirt -in ${i/.nii.gz/}_tshift.nii.gz -out ${i/.nii.gz/}_tshift_vol.nii.gz &
#done

#for i in data/subj*/*bold/*_tshift_vol.nii.gz; do
#    echo $i
#    #fslmaths ${i} -Tmean ${i/.nii.gz/}_average_time.nii.gz &
#done

#for i in data/subj*/*bold/*_average_time.nii.gz; do
#echo $i
#ln -s $i .
#done

#buildtemplateparallel.sh -j 2 -d 3 -m 80x80x30 -t GR -s CC -c 1 -o fudan_children_hospital_bold_normal_ *nii.gz

