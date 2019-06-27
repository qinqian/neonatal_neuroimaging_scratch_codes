#!/bin/bash -ex
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4388938/
# Alignment of functional data into a common space involved two steps: (1) within-subject rigid alignment [FSL FLIRT (for FMRIB Linear Image Restoration Tool)] between functional and T1-weighted images; and (2) nonlinear [FSL FNIRT (for FMRIB Nonlinear Image Registration Tool)] registration of the T1-weighted images to a T1-weighted template image acquired from an independent subject scanned at 2 weeks of age.

#rm -f dti.left.list
#for i in output/crash/*; do nipypecli crash $i | grep "N4" | uniq >> dti.left.list;done
for sub in $(grep avg_b0_corrected dti.left.list | cut -f 9 -d" "  | cut -f 1 -d_); do
    ls ${sub}/dwi/*dti*.nii.gz ${sub}/dwi/*dti*.bvec ${sub}/dwi/*dti*.bval
    #eddy_correct ${sub}/dwi/*dti*.nii.gz output/${sub}_simple_eddy.nii.gz 0 &
    #mkdir -p output/${sub}
    #mv output/${sub}_simple_eddy.nii.gz output/${sub}
    #fslroi output/${sub}/${sub}_simple_eddy.nii.gz output/${sub}/${sub}_simple_eddy_b0.nii.gz 0 1
    #bet output/${sub}/${sub}_simple_eddy_b0.nii.gz output/${sub}/${sub}_simple_eddy_b0_bet -m -f 0.2 -R
    #dtifit -k output/${sub}/${sub}_simple_eddy.nii.gz -o output/${sub}/${sub} -m output/${sub}/${sub}_simple_eddy_b0_bet_mask.nii.gz -r ${sub}/dwi/*dti*.bvec -b ${sub}/dwi/*dti*.bval
done
