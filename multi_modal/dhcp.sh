#!/bin/bash -ex

#N4BiasFieldCorrection -d 3 --input-image /cluster/home/qinqian/mri/hyperbilirubinemia/output/preprocessDTI/dwi_workflow/eddy_correct/_subject_id_156/b0_avg/sub-156_run-1_dti_denoised_avg_b0.nii.gz --mask-image /cluster/home/qinqian/mri/hyperbilirubinemia/output/preprocessDTI/dwi_workflow/_subject_id_156/bet_dwi_pre/sub-156_run-1_dti_denoised_baseline_brain_mask.nii.gz --output sub-156_run-1_dti_denoised_avg_b0_corrected.nii.gz

#N4BiasFieldCorrection -d 3 --input-image /cluster/home/qinqian/mri/hyperbilirubinemia/output/preprocessDTI/dwi_workflow/eddy_correct/_subject_id_156/b0_avg/sub-156_run-1_dti_denoised_avg_b0.nii.gz --output sub-156_run-1_dti_denoised_avg_b0_corrected.nii.gz
rm -f dhcp.commands
for i in results/processed/*T1w_noise_corrected_reoriented_enh_b0.nii.gz; do
    label=$(basename $i)
    label=${label/_T1w_noise_corrected_reoriented_enh_b0.nii.gz/}
    echo $label
    cga=$(grep ${label}$'\t' kai_2017_dti_t1_bold_to_be_processed.xls  | awk '{IFS="\t"; if ($39=="NA"){$39=38}; print $3+$39}')
    echo $cga
    echo docker run --rm -t -v /cluster/home/qinqian/mri/hyperbilirubinemia/results/processed/:/data biomedia/dhcp-structural-pipeline:latest  ${label/_T1w_noise_corrected_reoriented_enh_b0.nii.gz/dhcp_b0} session1 $cga -T2 /data/$(basename $i) -t 4 -no-cleanup >> dhcp.commands
done

tail -n 102 dhcp.commands > dhcp.command.102

