#!/bin/bash -ex

clean_dicoms() {
## a complete example seen in https://gist.github.com/satra/8701942
## use the output_meta to construct the converted patient ids
## output_meta/.heudiconv/10775081/info/dicominfo.tsv
#  heudiconv -d ../mydicom/{subject}/SDY*/*/*DCM -s 10742200 10771864_2016-10-13 10771864_2016-10-26 10775081 10781657 10811927_2017-2-4 10811927_2017-9-7 10863037 -f heuristic.py -c dcm2niix -b -o output --minmeta -a test --anon-cmd "./convert_id.py"

## to debug, run one by one
#for i in `seq 0 232`; do
##for i in 23 47 229 230 231 232; do
##for i in 111; do
#    echo $i
#    echo "----------------"
#    #heudiconv -d data/raw/pvl/PA{subject}/*/*/* -s $i -f heuristic.py -c none -b -o output_meta --minmeta
#    heudiconv -d data/raw/pvl/PA{subject}/*/*/* -s $i -ss $i -f heuristic.py -c dcm2niix -b -o output_pvl_tmp --minmeta -a output_pvl --anon-cmd "./convert_id.py" --overwrite
#done

#for i in data/raw/control/*; do
###for i in 23 47 229 230 231 232; do
###for i in 111; do
#    echo $i
#    echo "----------------"
#    p=$(basename $i)
##    heudiconv -d data/raw/control/{subject}/*/*/* -s $p -f heuristic.py -c none -b -o output_meta2 --minmeta
#    heudiconv -d data/raw/control/{subject}/*/*/* -s $p -f heuristic.py -c dcm2niix -b -o output_control_tmp --minmeta -a output_control --anon-cmd "./convert_id2.py"
#done

#for i in `seq 233 413` 241_1; do
#    echo $i
#    echo "----------------"
#    ## heudiconv -d data/raw/pvl/PA{subject}/*/*/* -s $i -f heuristic.py -c none -b -o output_meta3 --minmeta
#    heudiconv -d data/raw/pvl/PA{subject}/*/*/* -s $i -ss $i -f heuristic.py -c dcm2niix -b -o output_pvl_tmp --minmeta -a output_pvl --anon-cmd "./convert_id3.py" --overwrite
#done
#

echo 1
# get patient id
#rm pvl_patient_id.txt
#for i in data/Shanghai/pvl/dicom/*; do
#    d=$(dirname $i)
#    p=$(basename $i)
#    echo $(readlink -f $i) $(python fetch_patient_id.py ${d}/$p/*/*/*) >> pvl_patient_id.txt
#done
#rm control_patient_id.txt
#for i in data/Shanghai/control/dicom/*; do
#    d=$(dirname $i)
#    p=$(basename $i)
#    echo $p $(python fetch_patient_id.py ${d}/$p/*/*/*) >> control_patient_id.txt
#done

# generate final heudiconv annotation together
#rm -f pvl_shanghai_heudiconv.commands
#for i in data/Shanghai/pvl/dicom/* ; do
#    d=$(dirname $i)
#    p=$(basename $i)
#    echo $d, $p
#    echo -e "heudiconv -d ${d}/{subject}/*/*/* -s $p -f convertall -c none -b -o pvl_shanghai_heudiconv01 --minmeta" >> pvl_shanghai_heudiconv.commands
#done
#parallel -j 32 < pvl_shanghai_heudiconv.commands

#rm -f cont_shanghai_heudiconv.commands
#for i in data/Shanghai/control/dicom/* ; do
#    d=$(dirname $i)
#    p=$(basename $i)
#    echo $d, $p
#    echo -e "heudiconv -d ${d}/{subject}/*/*/* -s $p -f convertall -c none -b -o cont_shanghai_heudiconv01 --minmeta" >> cont_shanghai_heudiconv.commands
#done
#parallel -j 32 < cont_shanghai_heudiconv.commands

# final version for PVL project in Shanghai
## run again with bidskit
# first pass
# docker run -it -v /data1/qinqian/MRI/01_PVL_AI/data/raw/pvl/:/mnt rnair07/bidskit --indir=/mnt --outdir=/mnt/BIDS --no-sessions 
# manually edit the protocol translation json, then second pass, issues: brackets exists, fixed!!
# python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/pvl/dicom/ -o /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/pvl/source/ --no-sessions

## the same for control samples
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/cont/dicom/ -o /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/cont/source/ --no-sessions
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/cont/dicom/ -o /data1/qinqian/MRI/01_PVL_AI/data/Shanghai/cont/source/ --no-sessions

# genetics data from Shanghai
#mkdir -p ../genetics/mydicom
#n=0
#for i in ../../Genetics/CHD7/phaseC_bids/mydicom/* ../../Genetics/data2_yanglin_positive/* ../../Genetics/PWS/data/* ../../Genetics/data/*; do
#   let n++
#   ln -s `readlink -f $i` ../genetics/mydicom/${n}
#done
#unlink /data1/qinqian/MRI/01_PVL_AI/genetics/mydicom/142 # this folder contains previous nifty files without dicom

#rm -f ../genetics/genetics_patient_id.txt
#for i in ../genetics/mydicom/*/*/*00/IMG00000.DCM; do
#    d=$(dirname $(dirname $(dirname $i)))
#    p=$(basename $d)
#    echo -e "../genetics/source/sub-${p}\t$(readlink -f $i)\t$(python fetch_patient_id.py $i)" >> ../genetics/genetics_patient_id.txt
#done
#vim :%s/unknown encoding: ISO 2022 IR \d*\n//g

# genetics nifty files
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/01_PVL_AI/genetics/mydicom/ -o /data1/qinqian/MRI/01_PVL_AI/genetics/source --no-sessions
#python initial_full_bidskit.py Shanghai_protocal/1.json Shanghai_protocal/2.json Shanghai_protocal/3.json ../genetics/derivatives/conversion/Protocol_Translator.json
#cp ../genetics/derivatives/conversion/Protocol_Translator.json ../genetics/derivatives/conversion/Protocol_Translator.json.bak
#cp test.json ../genetics/derivatives/conversion/Protocol_Translator.json
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/01_PVL_AI/genetics/mydicom/ -o /data1/qinqian/MRI/01_PVL_AI/genetics/source --no-sessions

rm ../data/shanghai_patient_id.txt
for i in ../data/Shanghai/pvl/dicom/*/ST0/SE0/IM0 ../data/Shanghai/pvl/dicom/*/st0/se0/im0 ../data/Shanghai/cont/dicom/*/SDY*/SRS*00/*00.DCM; do
    d=$(dirname $(dirname $(dirname $i)))
    p=$(basename $d)
    g=$(basename $(dirname $(dirname $d)))
    echo -e "../data/Shanghai/${g}/source/sub-${p}\t$(readlink -f $i)\t$(python fetch_patient_id.py $i)" >> ../data/shanghai_patient_id.txt
done

#for i in ../data/Shanghai/pvl/dicom/*/*/T2*/; 
#do 
#    echo $(readlink -f $(ls ${i}/*IMA|head -1)) $(python fetch_patient_id.py $(ls ${i}/*IMA|head -1)) >> ../data/shanghai_patient_id.txt
#done

# get low resolution files part from the new sequence 
# mkdir -p ../new_sequence
# rm -f ../new_sequence/new_sequence_patient_id.txt
# for i in /data1/qinqian/MRI/data/MRI/fMRI/*/DICOM/PA0/ST0/SE0/IM0 /data1/qinqian/MRI/data/MRI/fMRI/*/dicom/pa0/st0/se0/im0; do
#     d=$(dirname $(dirname $(dirname $(dirname $(dirname $i)))))
#     p=$(basename $d)
#     echo -e "../new_sequence/source/sub-${p}\t$(readlink -f $i)\t$(python fetch_patient_id.py $i)" >> ../new_sequence/new_sequence_patient_id.txt
# done
# 
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/data/MRI/fMRI/ -o /data1/qinqian/MRI/01_PVL_AI/new_sequence/source --no-sessions
#python initial_full_bidskit.py Shanghai_protocal/1.json Shanghai_protocal/2.json Shanghai_protocal/3.json Shanghai_protocal/4.json ../new_sequence/derivatives/conversion/Protocol_Translator.json
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/data/MRI/fMRI/ -o /data1/qinqian/MRI/01_PVL_AI/new_sequence/source --no-sessions
}

### heudiconv not work for siemmens because of directory layout with only one layer
clean_xiamen() {
    ## to debug, run one by one
#    for i in `ls -d data/Xiamen/xiamen2/*`; do
#        pre=$(basename $i)
#        # echo $pre
#        # heudiconv -d data/Xiamen/xiamen2/{subject}/* -s $pre -f heuristic.py -c none -b -o output_xiamen --minmeta
#    done
#
#    for i in `ls -d data/Xiamen/xiamen2/*`; do
#        pre=$(basename $i)
#        echo $pre
#        heudiconv -d data/Xiamen/xiamen2/{subject}/* -s $pre -f heuristic_xiamen_high_resolution.py -c dcm2niix -b -o output_xiamen_high --minmeta
#    done
#for i in `seq 233 413` 241_1; do
#    echo $i
#    echo "----------------"
#    ## heudiconv -d data/raw/pvl/PA{subject}/*/*/* -s $i -f heuristic.py -c none -b -o output_meta3 --minmeta
#    heudiconv -d data/raw/pvl/PA{subject}/*/*/* -s $i -ss $i -f heuristic.py -c dcm2niix -b -o output_pvl_tmp --minmeta -a output_pvl --anon-cmd "./convert_id3.py" --overwrite
#done

#mkdir -p /data1/qinqian/MRI/01_PVL_AI/data/Xiamen/xiamen2018/source/
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/01_PVL_AI/data/Xiamen/xiamen2018/dicom/ -o /data1/qinqian/MRI/01_PVL_AI/data/Xiamen/xiamen2018/source/ --no-sessions
#python /data1/qinqian/MRI/software/bidskit/dcm2bids.py -i /data1/qinqian/MRI/01_PVL_AI/data/Xiamen/xiamen2018/dicom/ -o /data1/qinqian/MRI/01_PVL_AI/data/Xiamen/xiamen2018/source/ --no-sessions

rm -f ../data/xiamen_patient_id.txt
for i in /data1/qinqian/MRI/01_PVL_AI/data/Xiamen/xiamen2018/dicom/*; do
     ii=${i}/$(ls $i|head -1)
     p=$(basename $(dirname $ii))
     echo -e "../data/Xiamen/xiamen2018/source/sub-${p}\t$(readlink -f $ii)\t$(python fetch_patient_id.py $ii)" >> ../data/xiamen_patient_id.txt
done

## left high-res SWI, Pha, mIP images etc...
#for i in data/Xiamen/xiamen2/work/conversion/*/*nii*; do d=$(fslval $i dim3); if [[ $d -gt 50 ]]; then echo $i $d >> all_xiamen_high_dimensional.txt; fi; done
echo 1
}


main() {
#clean_dicoms
clean_xiamen

#wc -l ../genetics/genetics_patient_id.txt ../new_sequence/new_sequence_patient_id.txt ../data/shanghai_patient_id.txt  ../data/xiamen_patient_id.txt
##all flair 2889
#ls ../genetics/source/sub-*/anat/*run-01_T2w.nii.gz | wc -l
#ls ../new_sequence/source/sub-*/anat/*run-01_T2w.nii.gz | wc -l
#ls ../data/Shanghai/pvl/source/sub-*/anat/*run-01_T2w.nii.gz | wc -l
#ls ../data/Shanghai/cont/source/sub-*/anat/*run-01_T2w.nii.gz | wc -l
#ls ../data/Xiamen/xiamen2018/source/sub-*/anat/*run-01_T2w.nii.gz | wc -l
#ls ../genetics/source/sub-*/anat/*run-01_T2w.nii.gz ../new_sequence/source/sub-*/anat/*run-01_T2w.nii.gz ../data/Shanghai/pvl/source/sub-*/anat/*run-01_T2w.nii.gz ../data/Shanghai/cont/source/sub-*/anat/*run-01_T2w.nii.gz ../data/Xiamen/xiamen2018/source/sub-*/anat/*run-01_T2w.nii.gz | wc -l

#run model
#CUDA_VISIBLE_DEVICES="0" Rscript vgg_model.R vgg

#CUDA_VISIBLE_DEVICES="0" Rscript vgg_all.R dense
#CUDA_VISIBLE_DEVICES="1" Rscript test_model_dense.R

#CUDA_VISIBLE_DEVICES="2" Rscript vgg_all.R vgg
#CUDA_VISIBLE_DEVICES="2" Rscript test_model.R

#CUDA_VISIBLE_DEVICES="1" Rscript vgg_all_darkfluid.R vgg
#CUDA_VISIBLE_DEVICES="1" Rscript vgg_all2.R vgg
#CUDA_VISIBLE_DEVICES="1" Rscript test_model2.R  ## don't use the early stop point, use the highest accuracy of validation 

#CUDA_VISIBLE_DEVICES="1" Rscript vgg_all.R dense
#CUDA_VISIBLE_DEVICES="2" Rscript vgg_all.R dense

#CUDA_VISIBLE_DEVICES="0" Rscript vgg_aug.R 0.001 0 vgg nots 64
#CUDA_VISIBLE_DEVICES="0" Rscript vgg_aug.R 0.01 1e-4 vgg nots 128

#CUDA_VISIBLE_DEVICES="0" Rscript vgg_aug.R 0.001 1e-6 vgg nots 16
#CUDA_VISIBLE_DEVICES="0" Rscript vgg_aug.R 0.001 0 vgg s 8

#CUDA_VISIBLE_DEVICES="2" Rscript vgg_aug.R 0.001 1e-8 dense nots 32
#CUDA_VISIBLE_DEVICES="2" Rscript vgg_aug.R 1 1e-6 dense s 16

#CUDA_VISIBLE_DEVICES="2" Rscript vgg_all2_crossvalidation.R vgg
#CUDA_VISIBLE_DEVICES="1" Rscript vgg_all2_crossvalidation.R dense
#CUDA_VISIBLE_DEVICES="0" Rscript vgg_cross_validation.R vgg

#CUDA_VISIBLE_DEVICES="1" Rscript vgg_cross_validation_small.R vgg
#CUDA_VISIBLE_DEVICES="2" Rscript vgg_cross_validation_decay.R vgg
#echo 1
}

main

