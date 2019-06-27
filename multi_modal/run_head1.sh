#!/bin/bash -ex
#mkdir -p /cluster/home/qinqian/mri/hyperbilirubinemia/output
#python process_T1_DTI_head1_three_modal.py -dti
#python process_T1_DTI_head1_three_modal.py -t1

#bash clean_up_dti.sh ## for the N4 failed ones
#rename 's/_simple_eddy//g' *eddy*

#rm -f dti_t1_bold.txt
#for i in ../phaseA_clean_dicom/output/sub-*; do
#    # output/sub-132/sub-132_FA.nii.gz
#    # output/sub-108/bet_t1/subject_idsub-108/sub-108_T1w_brain_noise_corrected_enh.nii.gz
#    cga=$(grep $(basename $i)$'\t' kai_2017_dti_t1_bold_to_be_processed.xls  | awk '{IFS="\t"; if ($39=="NA"){$39=38}; print $3+$39}')
#
#    if [ -s ${i}/$(basename $i)_FA.nii.gz ]; then
#       #echo $(basename $i) $cga ${i}/$(basename $i)_FA.nii.gz ${i}/$(basename $i)_MD.nii.gz ${i}/$(basename $i)_simple_eddy_b0.nii.gz ${i}/bet_t1/*/*.nii.gz $(basename $i)/func/*bold*nii.gz >> dti_t1_bold.txt
#       echo $(basename $i) $cga ${i}/$(basename $i)_FA.nii.gz ${i}/$(basename $i)_MD.nii.gz ${i}/$(basename $i)_simple_eddy_b0.nii.gz ${i}/T1/*/*.nii.gz $(basename $i)/func/*bold*nii.gz >> dti_t1_bold.txt
#    else
#       #fslroi ${i}/out_file/*/*nii.gz ${i}/$(basename $i)_b0.nii.gz 0 1 
#       #echo $(basename $i) $cga ${i}/out_FA/*/*.nii.gz ${i}/out_MD/*/*nii.gz ${i}/$(basename $i)_b0.nii.gz ${i}/bet_t1/*/*.nii.gz $(basename $i)/func/*bold*nii.gz >> dti_t1_bold.txt
#       echo $(basename $i) $cga ${i}/out_FA/*/*.nii.gz ${i}/out_MD/*/*nii.gz ${i}/$(basename $i)_b0.nii.gz ${i}/T1/*/*.nii.gz $(basename $i)/func/*bold*nii.gz >> dti_t1_bold.txt
#    fi
#done

#python process_T1_DTI_head1_three_modal.py -bold dti_t1_bold.txt -input /data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/ -output results/nipype/

#mkdir -p results/processed
#rm -f bet.commands
#for i in results/processed/*b0*; do
#    echo bet $i ${i/.nii.gz/}_bet -m -f 0.2 >> bet.commands
#    echo $i
#done

#segment wm, csf, performance is not good...
#rm -f t1.commands
#while read line; do
#  fields=($line)
#  t1label=results/processed/$(basename ${fields[5]})
#  echo $t1label
#  if [ ! -s ${t1label/.nii.gz/}_wm_nohwm.nii.gz ]; then
#     echo python register_t1.py  ${fields[4]/.nii.gz/_bet.nii.gz} ${fields[5]} ${fields[1]} >> t1.commands
#  fi
#done <dti_t1_bold.txt

#python process_T1_DTI_head1_three_modal.py -bold dti_t1_bold.txt
#performance is bad, use ANTsPy instead
rm -f rsbold.commands
while read line; do
  fields=($line)
  echo Rscript preprocess_rsbold.R -s /data1/qinqian/MRI/software/dhcp-structural-pipeline/workdir/${fields[0]}-session1/segmentations/${fields[0]}-session1_tissue_labels.nii.gz -b results/processed/${fields[0]}_T1w_noise_corrected_reoriented_enh_b0.nii.gz -a results/cleaned_bold/average_bold/_subject_id_${fields[0]}/${fields[0]}_run-1_bold_roi_dtype_mcf_mask_smooth_mask_gms_mean.nii.gz -i results/smooth_cleaned_bold/_subject_id_${fields[0]}/_maskfunc30/${fields[0]}_run-1_bold_roi_dtype_mcf_mask_smooth_mask.nii.gz -m results/cleaned_bold/bold_mask/_subject_id_${fields[0]}/_dilatemask0/${fields[0]}_run-1_bold_roi_dtype_mcf_bet_thresh_dil.nii.gz -p results/cleaned_bold/motion_par/_subject_id_${fields[0]}/_realign0/${fields[0]}_run-1_bold_roi_dtype_mcf.nii.gz.par -o /data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/${fields[0]}_process_bold >>rsbold.commands
done <dti_t1_bold.txt


