#!/bin/bash -ex

#flirt -in results/cleaned_bold/average_bold/_subject_id_sub-102/sub-102_run-1_bold_roi_dtype_mcf_mask_smooth_mask_gms_mean.nii.gz -ref results/processed/sub-102_b0_bet.nii.gz -out 102_bold2b0 -omat 102_bold2b0.mat
# bold=results/smooth_cleaned_bold/_subject_id_sub-102/_maskfunc30/sub-102_run-1_bold_roi_dtype_mcf_mask_smooth_mask.nii.gz
# prefix=/data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis
# python2.7 ICA-AROMA/ICA_AROMA.py -in ${prefix}/$bold -out ${prefix}/102_ICA_AROMA -affmat ${prefix}/102_bold2b0.mat -mc ${prefix}/results/cleaned_bold/motion_par/_subject_id_sub-102/_realign0/sub-102_run-1_bold_roi_dtype_mcf.nii.gz.par

Rscript preprocess_rsbold.R -c results/processed/sub-99_T1w_noise_corrected_reoriented_enh_csf.nii.gz -w results/processed/sub-99_T1w_noise_corrected_reoriented_enh_wm.nii.gz -b results/processed/sub-99_T1w_noise_corrected_reoriented_enh_b0.nii.gz -a results/cleaned_bold/average_bold/_subject_id_sub-99/sub-99_run-1_bold_roi_dtype_mcf_mask_smooth_mask_gms_mean.nii.gz -i results/smooth_cleaned_bold/_subject_id_sub-99/_maskfunc30/sub-99_run-1_bold_roi_dtype_mcf_mask_smooth_mask.nii.gz -m results/cleaned_bold/bold_mask/_subject_id_sub-99/_dilatemask0/sub-99_run-1_bold_roi_dtype_mcf_bet_thresh_dil.nii.gz -p results/cleaned_bold/motion_par/_subject_id_sub-99/_realign0/sub-99_run-1_bold_roi_dtype_mcf.nii.gz.par -o /data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/sub-99_process_bold

