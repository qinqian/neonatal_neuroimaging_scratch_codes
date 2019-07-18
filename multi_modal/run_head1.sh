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
#rm -f rsbold.commands
#while read line; do
#  fields=($line)
##  if [ ! -s /data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/${fields[0]}_process_bold_alffinb0.nii.gz ]; then
#  echo Rscript preprocess_rsbold.R -s /data1/qinqian/MRI/software/dhcp-structural-pipeline/workdir/${fields[0]}-session1/segmentations/${fields[0]}-session1_tissue_labels.nii.gz -b results/processed/${fields[0]}_T1w_noise_corrected_reoriented_enh_b0.nii.gz -a results/cleaned_bold/average_bold/_subject_id_${fields[0]}/${fields[0]}_run-1_bold_roi_dtype_mcf_mask_smooth_mask_gms_mean.nii.gz -i results/smooth_cleaned_bold/_subject_id_${fields[0]}/_maskfunc30/${fields[0]}_run-1_bold_roi_dtype_mcf_mask_smooth_mask.nii.gz -m results/cleaned_bold/bold_mask/_subject_id_${fields[0]}/_dilatemask0/${fields[0]}_run-1_bold_roi_dtype_mcf_bet_thresh_dil.nii.gz -p results/cleaned_bold/motion_par/_subject_id_${fields[0]}/_realign0/${fields[0]}_run-1_bold_roi_dtype_mcf.nii.gz.par -o /data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis/results/processed/${fields[0]}_process_bold >>rsbold.commands
##  fi
#done <dti_t1_bold.txt
#
#
rm -f all_files_in_native_space.txt all_files_build_template.com all_files_in_template_space.txt
while read line;do 
  echo $line
  b0=results/processed/${line}_T1w_noise_corrected_reoriented_enh_b0.nii.gz
  t1=results/processed/${line}_T1w_noise_corrected_reoriented_enh_reg.nii.gz
  alffb0=results/processed/${line}_process_bold_alffinb0.nii.gz
  meanbold=results/processed/${line}_process_bold_boldaveinb0.nii.gz
  falffb0=results/processed/${line}_process_bold_falffinb0.nii.gz
  rehob0=results/processed/${line}_process_bold_rehoinb0.nii.gz
  fa=results/processed/${line}_FA.nii.gz
  md=results/processed/${line}_MD.nii.gz
  od=results/processed/${line}/${line}/AMICO/NODDI_${line}/FIT_OD.nii.gz
  icvf=results/processed/${line}/${line}/AMICO/NODDI_${line}/FIT_ICVF.nii.gz
  isovf=results/processed/${line}/${line}/AMICO/NODDI_${line}/FIT_ISOVF.nii.gz
  if [ -s $rehob0 ]; then
     echo $line /data1/qinqian/MRI/software/dhcp-structural-pipeline/workdir/${line}-session1/segmentations/${line}-session1_tissue_labels.nii.gz $b0 $t1 $alffb0 $falffb0 $rehob0 $fa $md $od $icvf $isovf >> all_files_in_native_space.txt
     echo python build_template.py -template results/processed/sub-143_T1w_noise_corrected_reoriented_enh_b0.nii.gz -label $line /data1/qinqian/MRI/software/dhcp-structural-pipeline/workdir/${line}-session1/segmentations/${line}-session1_tissue_labels.nii.gz $b0 $t1 $alffb0 $falffb0 $rehob0 $meanbold $fa $md $od $icvf $isovf >> all_files_build_template.com
     echo $line results/template/${line}_${line}-session1_tissue_labels.nii.gz results/template/${line}_b0.nii.gz results/template/${line}_smooth_fwmh2_b0.nii.gz results/template/${line}_deform_jac_geom.nii.gz results/template/${line}_deform_smoothjac_geom.nii.gz results/template/${line}_$(basename $t1) results/template/${line}_$(basename $alffb0) results/template/${line}_smooth_fwmh2_$(basename $alffb0) results/template/${line}_$(basename $falffb0) results/template/${line}_smooth_fwmh2_$(basename $falffb0) results/template/${line}_$(basename $rehob0) results/template/${line}_smooth_fwmh2_$(basename $rehob0) results/template/${line}_$(basename $fa) results/template/${line}_smooth_fwmh2_$(basename $fa) results/template/${line}_$(basename $md) results/template/${line}_smooth_fwmh2_$(basename $md) results/template/${line}_$(basename $od) results/template/${line}_smooth_fwmh2_$(basename $od) results/template/${line}_$(basename $icvf)  results/template/${line}_smooth_fwmh2_$(basename $icvf) results/template/${line}_$(basename $isovf) results/template/${line}_smooth_fwmh2_$(basename $isovf) >> all_files_in_template_space.txt 
   fi
done < <(cut -f 2 kai_2017_dti_t1_bold_to_be_processed.xls | sed 1d | uniq)


### warp feng shi'rois to native mean bold space
#rm -f fengshi_warpatlas.commands
#while read line; do
#  fields=($line)
#  echo python build_template.py -bold -template results/cleaned_bold/average_bold/_subject_id_${fields[0]}/${fields[0]}_run-1_bold_roi_dtype_mcf_mask_smooth_mask_gms_mean.nii.gz -label ${fields[0]}_fengshi_bold ../../data/Atlas/UNC-CEDARS-INFANT-20170420/resolution-3mm/infant-neo-aal-3mm.nii ../../data/Atlas/UNC-CEDARS-INFANT-20170420/resolution-3mm/infant-neo-intensity-withSkull-3mm.nii ../../data/Atlas/UNC-CEDARS-INFANT-20170420/resolution-3mm/infant-neo-funPar-3mm.nii  >> fengshi_warpatlas.commands
#done <dti_t1_bold.txt

rm -f fengshi_norm.commands
while read line; do
  fields=($line)
  echo python build_template4d.py -bold -template ../../data/Atlas/UNC-CEDARS-INFANT-20170420/resolution-3mm/infant-neo-intensity-3mm.nii  -label ${fields[0]}_fengshi_normbold results/processed/${fields[0]}_process_bold_clean4d.nii.gz results/cleaned_bold/average_bold/_subject_id_${fields[0]}/${fields[0]}_run-1_bold_roi_dtype_mcf_mask_smooth_mask_gms_mean.nii.gz >> fengshi_norm.commands
done <dti_t1_bold.txt

#modal=(label label tissue_label b0 smoothb0 jac smoothjac t1 alff smoothalff falff smoothfalff reho smoothreho fa smoothfa md smoothmd od smoothod icvf smoothicvf isovf smoothisovf)
#for i in `seq 2 23`; do
#    #echo fslmerge -t ${modal[$i]}_local_template.nii.gz $(cut -f $i -d" " all_files_in_template_space.txt | tr '\n' ' ')
#    a=($(cut -f $i -d" " all_files_in_template_space.txt | tr '\n' ' '))
#    echo ${#a[@]}
##    fslmerge -t ${modal[$i]}_local_template.nii.gz $(cut -f $i -d" " all_files_in_template_space.txt | tr '\n' ' ')
##    fslmaths ${modal[$i]}_local_template.nii.gz -Tmean ${modal[$i]}_mean_template.nii.gz
#done

#python build_template.py -template results/processed/sub-143_T1w_noise_corrected_reoriented_enh_b0.nii.gz -label JHU_Atlas /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_b0ss.nii.gz 

#python build_template.py -template results/processed/sub-143_T1w_noise_corrected_reoriented_enh_b0.nii.gz -label UNC_Atlas /data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-1yr-aal.nii /data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-1yr-withCerebellum.nii

#python build_template.py -template ../../software/dhcp-structural-pipeline/workdir/ALBERT_01-session1/T1/ALBERT_01-session1.nii.gz -label ALBERT_01_warp_to_JHU /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_T1ss.nii.gz
#python build_template.py -template ../../software/dhcp-structural-pipeline/workdir/ALBERT_05-session1/T1/ALBERT_05-session1.nii.gz -label ALBERT_05_warp_to_JHU /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_T1ss.nii.gz
#python build_template.py -template ../../software/dhcp-structural-pipeline/workdir/ALBERT_08-session1/T1/ALBERT_08-session1.nii.gz -label ALBERT_08_warp_to_JHU /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_T1ss.nii.gz

#python build_template.py -template ../../software/dhcp-structural-pipeline/workdir/ALBERT_06-session1/T1/ALBERT_06-session1.nii.gz -label ALBERT_06_warp_to_JHU /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_T1ss.nii.gz

