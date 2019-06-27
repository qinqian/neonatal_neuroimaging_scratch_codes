#!/usr/bin/env python

import ants
import amico
amico.core.setup()
import pandas as pd
import numpy as np
import os
import glob
for subject in glob.glob('../output/sub*'):
    subject = os.path.basename(subject)
    if subject == 'sub-122': continue

    ae = amico.Evaluation("results/processed/%s" % subject, subject)
    os.system(f'mkdir -p results/processed/{subject}/{subject}')
    # pointset2scheme -version STEJSKALTANNER -dwparams 0.022 0.04 0.025 0.072 -addzeromeas 6 -inputfile PointSets/Elec060.txt > M6_N60_ST.scheme
    amico.util.fsl2scheme(f"../phaseA_clean_dicom/output/{subject}/dwi/{subject}_run-1_dti.bval",
                          f"../phaseA_clean_dicom/output/{subject}/dwi/{subject}_run-1_dti.bvec")
    
    os.system(f'cp results/processed/{subject}_b0_bet_mask.nii.gz results/processed/{subject}/{subject}')
    os.system(f'cp ../phaseA_clean_dicom/output/{subject}/dwi/{subject}_run-1_dti.scheme results/processed/{subject}/{subject}')
    os.system(f'cp ../output/{subject}/out_file/subject_id{subject}/vol0000_warp_maths_thresh_flirt_merged.nii.gz results/processed/{subject}/{subject}/{subject}_dti.nii.gz')
    
    ae.load_data(dwi_filename=f"{subject}_dti.nii.gz", 
                 scheme_filename=f"{subject}_run-1_dti.scheme",
                 mask_filename=f'{subject}_b0_bet_mask.nii.gz')
    
    ae.set_model("NODDI")
    ##ae.set_model("CylinderZeppelinBall") # http://camino.cs.ucl.ac.uk/index.php?n=Man.Pointset2scheme
    ae.generate_kernels()
    ae.load_kernels()
    ae.fit()
    ae.save_results(subject)
