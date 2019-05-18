#!/bin/bash

#rm -f template.list
#while read line; do
#  grep ${line}$'\t' ../UGT1A1/phaseA_preproc/fmri_images.list | grep 'BOLD' >> template.list
#done < ids

#mkdir -p data
#cd data
#n=0
#while read line; do
#  fields=($line)
#  mkdir -p subj${n}/bold
#  #dcm2niix -i y -o subj${n}/bold -f ${n}_${fields[0]}_${fields[6]}_"%s" $(echo ${fields[10]} | cut -f 1 -d,)
#  let n++
#done < ../template.list

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

# manually filter template list
buildtemplateparallel.sh -i 4 -z /data1/qinqian/MRI/data/Atlas/UNC-CEDARS-INFANT-20170420/resolution-adaptive/infant-neo-intensity-withSkull-2_928mm.nii -n 1 -r 0 -j 8 -d 3 -m 50x50x30 -t GR -s CC -c 1 -o fudan_children_hospital_bold_normal_ *nii.gz

