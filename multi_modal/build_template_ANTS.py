#!/usr/bin/env python
"""
warp three atlas into the native space
output: three warped label image with 
        three corresponding transformation nii.gz
"""
from nipype.pipeline.engine import Node, MapNode, Workflow
from nipype import IdentityInterface, Function, Select
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.ants import N4BiasFieldCorrection, Registration, DenoiseImage, CreateJacobianDeterminantImage, \
    ApplyTransforms
from nipype.workflows.dmri.fsl.utils import b0_average, extract_bval, dwi_flirt, recompose_xfm, recompose_dwi, \
    time_avg, b0_indices
from nipype.interfaces.fsl import Reorient2Std, ExtractROI, DTIFit, ImageMaths, FLIRT, BET

from nipype.workflows.fmri.fsl import create_featreg_preproc

import argparse
import matplotlib
matplotlib.use('Agg')
from skimage import measure

from matplotlib import gridspec
import matplotlib.pyplot as plt
import os
import pandas as pd
import ants
import sys
from skimage.restoration import denoise_wavelet
from skimage import exposure
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-template')
parser.add_argument('input', nargs='+', type=str)
parser.add_argument('-label')
args = parser.parse_args()

def normalize(name='reg', type='SyN'):
    # too slow
    # normalize = Node(Registration(from_file="smri_ants_registration_settings.json"), name='ANTSRegistration')
    # refer to https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
    normalize = Node(Registration(), name=name)
    normalize.inputs.interpolation = 'BSpline'
    #normalize.inputs.interpolation = 'Linear'
    normalize.inputs.use_histogram_matching = True ## within modality 1, across modality 0
    normalize.inputs.initial_moving_transform_com = 1 ## 1-match by center of mass [$t1brain,$template,1]
    normalize.inputs.dimension = 3
    normalize.inputs.winsorize_lower_quantile = 0.025
    normalize.inputs.winsorize_upper_quantile = 0.975
    normalize.inputs.collapse_output_transforms = True
    normalize.inputs.write_composite_transform = False # write one h5 file, hard to see
    normalize.inputs.float = False
    normalize.inputs.transforms = ['Rigid', 'Affine', 'SyN']
    normalize.inputs.transform_parameters = [[0.1], [0.1], [0.1, 3, 0]]
    normalize.inputs.metric = ['MI', 'MI', 'CC']
    normalize.inputs.metric_weight = [1] * 3
    normalize.inputs.sampling_strategy = ['Regular', 'Regular', None]
    normalize.inputs.sampling_percentage = [0.25] * 3
    normalize.inputs.radius_or_number_of_bins = [32, 32, 4]
    normalize.inputs.convergence_window_size = [10] * 3
    normalize.inputs.convergence_threshold = [1e-6] * 3
    normalize.inputs.number_of_iterations = [[100, 70, 50, 30]] * 3
    normalize.inputs.shrink_factors = [[8, 4, 2, 1]] * 3
    normalize.inputs.smoothing_sigmas = [[3, 2, 1, 0]] * 3
    normalize.inputs.sigma_units = ['vox'] * 3
    normalize.inputs.num_threads = 10
    normalize.inputs.use_estimate_learning_rate_once = [True] * 3
    normalize.config = {'execution':
                             {'remove_unnuecessary_outputs' : False}
                       }
    return normalize

align = normalize('build_template', ['Rigid', 'Affine', 'SyN']) 

align.inputs.moving_image = os.path.abspath(args.input[1])
align.inputs.fixed_image = os.path.abspath(args.template)
align.inputs.output_transform_prefix = os.path.abspath(os.path.join('results/template_ANTS/', args.label))
align.run()

