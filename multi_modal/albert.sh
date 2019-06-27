#!/bin/bash -ex
## rm -f dhcp.commands
## mkdir -p /data1/qinqian/MRI/software/dhcp-structural-pipeline/tmp
## while read line; do
## fields=($line)
## i=${fields[0]}
## ls ALBERTs/T1/${i}.nii.gz ALBERTs/T2/${i}.nii.gz
## echo "export TMPDIR=/data1/qinqian/MRI/software/dhcp-structural-pipeline/tmp; ./dhcp-pipeline.sh $i session1 ${fields[1]} -T1 ALBERTs/T1/${i}.nii.gz -T2 ALBERTs/T2/${i}.nii.gz -t 2 -no-cleanup 1 2>&1>${i}.logs" >>dhcp.commands
## 
## #./dhcp-pipeline.sh subject3 session1 33 -T1 test2-T1w.nii.gz -T2 test2-T2w.nii.gz -t 28  -no-cleanup 
## 
## #./dhcp-pipeline.sh subjectX session1 33 -T2 test-low-T2w.nii.gz -t 28 -no-cleanup 
## 
## #./dhcp-pipeline.sh subjectS session1 33 -T2 swi-T2w.nii.gz -t 14 -no-cleanup 
## done <ALBERTs/ages.csv

#bash dhcp-pipeline_try_t1.sh subjectFudan session1 35 -T1 ../../hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/sub-99_T1w_noise_corrected_reoriented_enh_reg.nii.gz -T2 ../../hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/sub-99_T1w_noise_corrected_reoriented_enh_b0.nii.gz

#bash dhcp-pipeline_try_t1.sh subjectFudan session1 35 -T1 ../../hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/sub-99_T1w_noise_corrected_reoriented_enh_reg.nii.gz -T2 ../../hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/sub-99_T1w_noise_corrected_reoriented_enh_b0.nii.gz

#rm -f dhcp_fudan_b0.commands
#for i in ../../hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/*T1w_noise_corrected_reoriented_enh_b0.nii.gz; do
#    label=$(basename $i)
#    label=${label/_T1w_noise_corrected_reoriented_enh_b0.nii.gz/}
#    echo $label
#    cga=$(grep ${label}$'\t' ../../hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/kai_2017_dti_t1_bold_to_be_processed.xls  | awk '{IFS="\t"; if ($39=="NA"){$39=38}; print $3+$39}')
#    echo $cga
#    echo ./dhcp-pipeline.sh ${label/_T1w_noise_corrected_reoriented_enh_b0.nii.gz/dhcp_b0} session1 $cga -T2 $i -t 4 -no-cleanup  >> dhcp_fudan_b0.commands
#done
#head -80 dhcp_fudan_b0.commands > dhcp_fudan_b0.commands.gpu
#tail -102 dhcp_fudan_b0.commands > dhcp_fudan_b0.commands.node13


for i in workdir/sub-*/surfaces/*/workbench/*spec*; do sub=$(echo $i | cut -f 2 -d / | sed 's/-session1//'); echo $sub, session1, $(grep $sub dhcp_fudan_b0.commands | cut -f 4 -d" "); done

