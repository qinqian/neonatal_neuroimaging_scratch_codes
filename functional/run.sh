#!/bin/bash

get_header() {
    find ../../data/MRI/fMRI -iname "dicomdir" >> fmri_dicomdir.list
    while read line;do
       python get_meta.py -d $line >> fmri_images.list
    done < fmri_dicomdir.list
}

convert_xls_to_csv() {
    python convert.py
    }

clean_dicom() {
    awk -F"\t" '{OFS="\t";print $1,$6,$2,$3}' fmri_images.list | sort -k1,1 -k2,2 | uniq > fmri_images.uniq_meta
}

convertnii() {
    modal=$1
    subj=$2
    dicom=$3
    mkdir -p ../data/fmri_nii/subj${subj}/${modal}
    dcm2niix -i y -o ../data/fmri_nii/subj${subj}/${modal} -f "%s" $dicom
}


preproc_one_modal() {
    one_modal=$1
    label=$2
    while read line; do
	fields=($line)
	func=($(grep ${fields[1]}$'\t' fmri_images.list | grep $'\t'${fields[2]}$'\t' | grep -i $one_modal | cut -f 10 |cut -f 1 -d,))
	length=${#func[*]}
	let length--
	for j in `seq 0 $length`;do
	    echo ${func[$j]}
	    if [ -s ${func[$j]} ]; then
		d=$(dirname ${func[$j]})
	    else # to lowercase
		pre=$(echo ${func[$j]} | awk '{print substr($1, 1, 34)}')
		suf=$(echo ${func[$j]} | awk '{print substr($1, 35, 100)}'| tr 'A-Z' 'a-z')
		inew=${pre}${suf}
		d=$(dirname $inew)
	    fi
	    convertnii $label ${fields[0]}_${j} $d
	done
    done < <(sed 1d Table1.csv | awk -F"," '{OFS="\t";print $1,$2,$3}')
}

fsl_bedpost() {
    dti=$1
    bval=$2
    bvec=$3
    cuda=$4
    mask=$5

    export FSLPARALLEL=SGE # use qsub
    export FSLPARALLEL="normal.q" # use qsub
    export SGE_ROOT=/var/lib/gridengine
    folder=$(dirname $dti)/bedpostx

    CUDA_VISIBLE_DEVICES=$cuda
    export CUDA_VISIBLE_DEVICES=$cuda
    if [ ! -d ${folder}.bedpostX ]; then
        mkdir -p $folder
        cp $mask ${folder}/nodif_brain_mask.nii.gz
        cp $bval ${folder}/bvals
        cp $bvec ${folder}/bvecs
        cp $dti ${folder}/data.nii.gz
        bedpostx_gpu $folder -Q normal.q -NJOBS 1
    fi

    # ${folder}.bedpostx/merged_f1samples.nii.gz
    n=0

    for i in $(grep -v "wm" ../../data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv | cut -f 1 -d,); do
        for j in $(grep -v "wm" ../../data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv | cut -f 1 -d,); do
	    rm -f  ${folder}.bedpostX/target.txt
	    echo "../../data/Atlas/jhu_atlas_neonate/nii/${i}.nii.gz" >> ${folder}.bedpostX/target.txt
	    echo "CUDA_VISIBLE_DEVICES=$n probtrackx2_gpu --omatrix3 -x ${dti/_eddy.nii.gz/}_wm_mask.nii.gz -s ${folder}.bedpostX/merged -m ${folder}/nodif_brain_mask.nii.gz --forcedir --dir=${dti/_eddy.nii.gz/}_probtrackx2_gpu_${i}roi_logs --sampvox=0.0 -c 0.25 -P 5000 --fibthresh=0.01 --distthresh=0.0 --os2t --rseed=999 --s2tastext --onewaycondition --steplength=0.5 -V 1 --opd --pd --ompl --targetmasks=${folder}.bedpostX/target.txt --target3=${folder}.bedpostX/target.txt --lrtarget3" >> fsl_tracking.commands
	    if [ $n -ge 2 ];then
	        n=0
	    fi
	    let n++
        done
    done
}

mr_tract() {
   # http://mrtrix.readthedocs.io/en/latest/getting_started/beginner_dwi_tutorial.html
   # http://mrtrix.readthedocs.io/en/latest/quantitative_structural_connectivity/ismrm_hcp_tutorial.html
   i=$1
   echo $i
   #mrconvert $i ${i/_eddy.nii.gz/}.mif -fslgrad ${i/_eddy.nii.gz/}.bvec  ${i/_eddy.nii.gz/}.bval -datatype float32 -stride 0,0,0,1
   #dwiextract ${i/_eddy.nii.gz/}.mif - -bzero | mrmath - mean ${i/_eddy.nii.gz/}.meanb0.mif -axis 3
   #dwi2mask ${i/_eddy.nii.gz/}.mif ${i/_eddy.nii.gz/}.mask.mif 
   # mrconvert ${i/_eddy.nii.gz/}.mask.mif ${i/_eddy.nii.gz/}.mrtrix.mask.nii.gz

   #dwi2response tournier ${i/_eddy.nii.gz/}.mif ${i/_eddy.nii.gz/}.response.txt -voxels ${i/_eddy.nii.gz/}.voxels.mif
   #dwi2fod csd ${i/_eddy.nii.gz/}.mif ${i/_eddy.nii.gz/}.response.txt ${i/_eddy.nii.gz/}.fod.mif -mask ${i/_eddy.nii.gz/}.mask.mif
   #
   #tckgen ${i/_eddy.nii.gz/}.fod.mif ${i/_eddy.nii.gz/}.tck -seed_image ${i/_eddy.nii.gz/}.mask.mif -mask ${i/_eddy.nii.gz/}.mask.mif -select 5M -cutoff 0.1
   #tcksift ${i/_eddy.nii.gz/}.tck ${i/_eddy.nii.gz/}.fod.mif ${i/_eddy.nii.gz/}_1M.tck -term_number 1M
   #mrconvert ${i/_eddy.nii.gz/}_fa_labeled.nii.gz ${i/_eddy.nii.gz/}_fa_labeled.mif -datatype int32

   ## need freesurfer to generate nodes_fixSGM.mif! T1 bravo failed at tk alignment step
   ## try ants aligned atlas convereted mif files instead
   #tck2connectome ${i/_eddy.nii.gz/}_1M.tck ${i/_eddy.nii.gz/}_fa_labeled.mif ${i/_eddy.nii.gz/}_connectome.csv
   ## mrview 12_fa_labeled.mif -connectome.init 12_fa_labeled.mif -connectome.load 12_connectome.csv
}

main() {
    #get_header
    #clean_dicom
    #use map_seq2mri.ipynb to match sequence with dicoms

    #manually clean the dicom annotations, then convert to csv
    #convert_xls_to_csv

    # mkdir -p ../data/fmri_nii
    # preproc_one_modal bold func
    # preproc_one_modal dti dti
    # preproc_one_modal bravo struct
    #preproc_one_modal dki dki # too few for UGT1A1 control
    #preproc_one_modal asl asl # too few for UGT1A1 control

    echo "multi-modal part"
    ### unify multi-modal processing into one script
    # rm -f denoised.commands dwimask.commands
    # for i in ../data/fmri_nii/subj*/dti/??.nii.gz ../data/fmri_nii/subj*/dti/?.nii.gz; do
    #     ## https://neurostars.org/t/determining-bids-phaseencodingdirection-from-dicom/612/8
    #     ## https://github.com/MRtrix3/mrtrix3/issues/747
    #     ## For axial ROW is the L->R dimension and COL is the R->L dimension.
    #     echo $i

    #     #echo "dwidenoise $i ${i/.nii.gz/_denoised.nii.gz} && dwibiascorrect ${i/.nii.gz/_denoised.nii.gz} ${i/.nii.gz/_denoised_bias.nii.gz} -ants -fslgrad ${i/.nii.gz/.bvec} ${i/.nii.gz/.bval} -nthreads 4" >> denoised.commands
    #     echo "mrconvert $i ${i/.nii.gz/}.mif -fslgrad ${i/.nii.gz/}.bvec ${i/.nii.gz/}.bval -datatype float32 -stride 0,0,0,1 && dwi2mask ${i/.nii.gz/.mif} ${i/.nii.gz/}_mask.mif &&  mrconvert ${i/.nii.gz/}_mask.mif ${i/.nii.gz/}_mask.nii.gz " >> dwimask.commands
    # done
    # parallel -j 8 < denoised.commands
    # parallel -j 8 < dwimask.commands # rm *.mif files
    
    # rm -f n4.commands
    # rm -f skull.commands
#    for i in ../data/fmri_nii/subj*/struct/??.nii.gz ../data/fmri_nii/subj*/struct/?.nii.gz ; do
#	echo $i
#    	## echo "N4BiasFieldCorrection -i $i -o ${i/.nii.gz/_bias.nii.gz}" >> n4.commands
#
#       ## @NoisySkullStrip need gui editor, explore later
#    	## echo "3dcalc -a ${i/.nii.gz/_bias.nii.gz} -expr 'a'  -prefix ${i/.nii.gz/_bias} && @NoisySkullStrip -input ${i/.nii.gz/_bias+orig} -3dSkullStrip_opts -use_skull -blur_fwhm 2 -shrink_fac_bot_lim 0.1" >> skull.commands
#    done

    # parallel -j 32 < n4.commands
    # parallel -j 32 < skull.commands

    # rm -f n4.commands
    # for i in  ../data/fmri_nii/subj*/func/??.nii.gz ../data/fmri_nii/subj*/func/?.nii.gz; do
    # 	echo $i
    # 	echo "N4BiasFieldCorrection -d 4 -i $i -o ${i/.nii.gz/_bias.nii.gz}" >> n4.commands
    # done
    # parallel -j 32 < n4.commands

    # Rscript intermodal.R
    echo "---------------------------"
    rm -f  dtifit.commands
    for i in ../data/fmri_nii/subj*/dti/?.nii.gz ../data/fmri_nii/subj*/dti/??.nii.gz; do
        prefix=$(dirname $i)
        #mrconvert ${prefix}/dti.nii.gz ${prefix}/dti.mif -fslgrad ${i/.nii.gz/}.bvec  ${i/.nii.gz/}.bval -datatype float32 -stride 0,0,0,1
        #mrconvert $(dirname $prefix)/brainmask2.nii.gz ${prefix}/brainmask2.mif
        #mrconvert -force $(dirname $prefix)/wmmask.nii.gz ${prefix}/wmmask.mif
        ## estimate response parameters
        # dwi2response tournier ${prefix}/dti.mif ${prefix}/dti.response.txt ## -voxels ${prefix}/dti.voxels.mif
        ## csd model
        # if [ ! -s ${prefix}/structure_connectome.csv ]; then
        # dwi2fod csd ${prefix}/dti.mif ${prefix}/dti.response.txt ${prefix}/dti.fod.mif -mask ${prefix}/brainmask2.mif -nthreads 28
        #    tckgen -force ${prefix}/dti.fod.mif ${prefix}/dti.tck -seed_image ${prefix}/wmmask.mif -mask ${prefix}/brainmask2.mif -select 1M -cutoff 0.1
        #    tcksift ${prefix}/dti.tck ${prefix}/dti.fod.mif ${prefix}/dti.500k.tck -term_number 0.5M
        #    mrconvert $(dirname $prefix)/jhu_native_labels.nii.gz $(dirname $prefix)/jhu_native_labels.mif -datatype int32
        #    tck2connectome ${prefix}/dti.500k.tck $(dirname $prefix)/jhu_native_labels.mif ${prefix}/structure_connectome.csv
        # fi
	## modify the edge part for Seed out of boundary issues
	## /home/qinqian/.local/lib/python2.7/site-packages/dipy/tracking/eudx.py
        ## use probabilistic tracking

        #if [ ! -s ${prefix}/dti_dipy.trk ]; then
        #     echo $prefix
        #    python dipy_roi_connectivity.py -dwi ${prefix}/dti.nii.gz -vec ${i/.nii.gz/}.bvec -val ${i/.nii.gz/}.bval -name ${prefix}/dti_dipy -brainmask $(dirname $prefix)/brainmask2.nii.gz -wmmask $(dirname $prefix)/wmmask.nii.gz -labelatlas $(dirname $prefix)/jhu_native_labels.nii.gz -fa ${prefix}/dti_moco_camino_fa.nii.gz -md ${prefix}/dti_moco_camino_md.nii.gz
        #fi

        # if [ ! -s ${prefix}/dti_dipy.prob.trk ]; then
        #      echo $prefix
        #      python dipy_roi_connectivity.py -dwi ${prefix}/dti.nii.gz -vec ${i/.nii.gz/}.bvec -val ${i/.nii.gz/}.bval -name ${prefix}/dti_dipy -brainmask $(dirname $prefix)/brainmask2.nii.gz -wmmask $(dirname $prefix)/wmmask.nii.gz -labelatlas $(dirname $prefix)/jhu_native_labels.nii.gz -fa ${prefix}/dti_moco_camino_fa.nii.gz -md ${prefix}/dti_moco_camino_md.nii.gz
        # fi
	echo "dtifit -k ${prefix}/dti.nii.gz -o ${prefix}/dti_fsldtifit -m $(dirname $prefix)/brainmask2.nii.gz -r ${i/.nii.gz/}.bvec -b ${i/.nii.gz/}.bval" >> dtifit.commands
        #break
    done

    #########

    ###
    ##############

    # the following process one modal one time ...
    
    ###
    #############

    # first T1 sag bravo failed with freesurfer, try ants instead 
    # surface analysis https://github.com/edickie/ciftify, nilearn and freesurfer

    # it's not possible to avoid freesurfer, let's debug!!!
    # follow http://surfer.nmr.mgh.harvard.edu/fswiki/ReconAllDevTable
    # steps: https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all
    # or recon-all -make all -s subject  skull strip is very bad....
    # gunzip ../data/fmri_nii/*/struct/*gz
    # for i in ../data/fmri_nii/*/struct/*nii; do
    #     echo $i
    #     rm -rf $(dirname $i)/freesurfer
    #     mkdir -p $(dirname $i)/freesurfer
    #     export SUBJECTS_DIR=$(dirname $i)/freesurfer
    #     recon-all -i $i -subject ${i/.nii/} -autorecon1 
    #     #recon-all -make all -s ${i/.nii/}
    #     break
 
    # done

    # rm -f stat
    # for i in $(ls ../data/fmri_nii/*/func/??.nii.gz); do echo $i>>stat;done 
    # for i in $(ls ../data/fmri_nii/*/func/?.nii.gz); do echo $i>>stat;done 

    # for i in `cat stat`; do 
    #     echo $i
        # fs=$(basename $i)
        # cd $(dirname $i)
        #struct=$(ls ../struct/*.nii)
        #structbase=$(basename $struct)
        #3dWarp -prefix rsbold_${fs/.nii.gz/}_deoblique -deoblique $fs 
        #3dWarp -prefix t1_${structbase/.nii/}_deoblique -deoblique $struct
    	# example 10b
        # afni_proc.py -subj_id ${fs/.nii.gz/} \
        #  	     -dsets rsbold_${fs/.nii.gz/}_deoblique+orig \
        #             -copy_anat t1_${structbase/.nii/}_deoblique+orig \
    	#  	     -blocks despike tshift align volreg blur mask regress \
    	#  	     -tcat_remove_first_trs 3             \
        #            -volreg_align_e2a       \
        #            -blur_size 6.0          \
        #            -mask_apply epi         \
        #            -mask_segment_anat yes          \
        #            -mask_segment_erode yes         \
        #            -regress_anaticor               \
     	# 	     -regress_bandpass 0.01 0.1           \
        #            -regress_RSFC                        \
        #            -regress_ROI WMe                        \
    	# 	     -regress_apply_mot_types demean deriv \
        #            -regress_est_blur_errts -regress_run_clustsim no
     	#  	     #-regress_censor_motion 0.2           \
     	#  	     #-regress_censor_outliers 0.1         \
        # output annotation:
        # 1. struct_series+orig: 3dcopy copy T1
        # 2. pb00.9.r01.tcat+orig: 3dTcat copy rsbold, remove first 3 time points
        # 3. pb00.9.r01.tcat+orig: 3dToutcount remove outlier at each time point with 1deval to evaluate outcount
        # 4. pb01.9.r01.despike:   3dDespike
        # 5. pb02.9.r01.tshift :   3dTshift
        # 6. (this step is very poor, replace with ANTs) antsMotionCorr + antsRegistration
	#    vr_base+orig as reference (=>final_epi_vr_base), output final_epi_vr_base+orig anat_final.9+orig aligned to T1 space:
	#                          3dbucket pick the second time point align_epi_anat.py -anat2epi 
        #                          align_epi_anat.py -anat2epi (aligned t1 image: t1_7_deoblique_al_junk+orig.nii + t1_7_deoblique_al_junk_mat.aff12.1D)
        #                          3dcopy & 3dSkullStrip & 3dresample & 3dAllineate &
	#                          3dvolreg (cat_matvec -I (invert transform anat2epi t1_7_deoblique_al_junk_mat.aff12.1D=>mat.r01.warp.aff12.1D)+mask_epi_extents(2nd volreg with NN interpolator)+pb02.9.r01.tshift+orig => pb03.9.r01.volreg)
	# 7. pb03.9.r01.volreg+orig=>pb04.9.r01.blur: 3dmerge -1blur_fwhm 6.0 (input pb03.9.r01.volreg+orig => output pb04.9.r01.blur+orig)
	# 8. 3dmask_tool => 3dresample => 3dABoverlap => get T1 and bold mask
	# 9. (extremely poor segmentation) 3dSeg => 3dresample => 3dmask_tool (T1 anat_final.9+orig => mask_GMe_resam mask_WMe_resam)
	# 10. 3dresample, 1d_tool.py => motion_deriv.1D motion_demean.1D
	# 11. 3dmaskave get average of WM regions
	# 12. 3dDeconvolve => .xmat.1D => (but not run, seem to be taskfMRI) a.3dREMLfit (stats.REML_cmd); b. ExamineXmat.R; c. 3dSynthesize
	# 13. (nuisance regressor) 3dTproject (input: pb04.9.r01.blur+orig.HEAD + X.xmat.1D + full_mask.9+orig) => errts.9.anaticor
	# 14. 3dRSFC (from 3dBandpass) => ALFF, fALFF (input: full_mask.9 errts.9.anaticor+orig(4D) cutoff: 0.01 0.1)
        #     3dRSFC -prefix RSFC/run_01 -nodetrend -mask full_mask.9+orig 0.01 0.1 errts.9.anaticor+orig[0..156]
	# 15. postprocessing: 3dTstat 3dcalc 3dTnorm 3dFWHMx

	# our workflow
	# 1. (AFNI) 3dDespike => 3dAFNItoNIFTI
	# 2. (FSLr) slicetimer
	# 3. (Motion Correction) with median slice (ANTS) => Motion Parameters (Nuisance Variable 1)
        #    (Motion Censoring, i.e. remove outlier)
	# 4. (ANTS) average bold registered to JHU T1 and warped back (label atlas)
	#    get global signal, gray, white signal (Nuisance Variable 2)
	# 5. (R) Nuisance regressor with Nuisance Variable 1+2+linear+constant detrend
	# 6. network analysis filterfMRIforNetworkAnalysis.R makeGraph is useful

	# echo "Rscript /data1/qinqian/MRI/UGT1A1/phaseA_preproc/rsbold.R ${i/.nii.gz/}" >> fmri.commands
	# Rscript /data1/qinqian/MRI/UGT1A1/phaseA_preproc/rsbold.R ${i/.nii.gz/}
	# break
	# (optional): export mocoparams(other nuisance) and corrected images to nilearn.connectome.ConnectivityMeasure: 1. input tshift and motioncorr 4D volume; 2. input nuisance correction and bandpass

	## python nilearn_connectome.py # TODO
	# 7. VBM analysis: (AFNI) 3dRSCF + 3dReHo, if necessary, use 3dcalc to convert nii.gz to +orig data
	# echo $i
	
	# 3dReHo                         \
	#     -mask ${i/nii.gz/}_ants_jhut1_parcel_mask.nii.gz \
	#     -inset ${i/nii.gz/}_ants_scorr_motioncorr_smooth_nuisancereg.nii.gz[0..154] \
	#     -prefix ${i}_REHO          \
	#     -neigh_RAD 2.9             \
	#     -chi_sq
	# break
	# (optional): explore anova p value with clustersim(alphasim)
	# cd -
    # done

    # parallel -j 9 < fmri.commands

    # rm -f eddy.commands
    # for i in ../data/fmri_nii/subj*/dti/*nii.gz; do
    # 	echo "eddy_correct $i ${i/.nii.gz/}_eddy.nii.gz 0" >> eddy.commands
    # done
    # parallel -j 32 < eddy.commands

    # rm -f dipy.commands
    # for i in ../data/fmri_nii/subj*/dti/*_eddy.nii.gz; do
    # 	echo $i
    # 	echo "python dti.py -dwi ${i} -vec ${i/_eddy.nii.gz/}.bvec -val ${i/_eddy.nii.gz/}.bval -name ${i/_eddy.nii.gz/}" >> dipy.commands
    # done
    # parallel -j 32 < dipy.commands

    # fiber tracking
    # n=0
    # rm -f fsl_tracking.commands
    # for i in ../data/fmri_nii/subj*/dti/*_eddy.nii.gz; do
        # dipy version
    	# python fiber.py -dwi ${i} -vec ${i/_eddy.nii.gz/}.bvec -val ${i/_eddy.nii.gz/}.bval -name ${i/_eddy.nii.gz/} -label /data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv -labelatlas ${i/_eddy.nii.gz/_fa_labeled.nii.gz}

        # fsl gpu version, voxel based instead of roi, consume too much memory > 80GB
        # fsl_bedpost $i ${i/_eddy.nii.gz/}.bval ${i/_eddy.nii.gz/}.bvec $n ${i/_eddy.nii.gz/}_mask.nii.gz

        # mrtrix3 version
        # mr_tract $i

    #     break
    #     if [ $n -ge 2 ]; then
    #         n=0
    #     fi
    #     let n++
    # done
}
 
main

