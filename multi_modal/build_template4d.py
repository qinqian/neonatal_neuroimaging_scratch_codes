#!/usr/bin/env python

"""
warp three atlas into the native space
output: three warped label image with 
        three corresponding transformation nii.gz
"""
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
parser.add_argument('-bold', action='store_true')
args = parser.parse_args()

input = ants.image_read(args.input[1])
template = ants.image_read(args.template)

if args.bold:
    reg = ants.registration(template, input, type_of_transform='SyN', reg_iterations=(50, 30, 10), outprefix=f'results/template/{args.label}_native_b0_to_143template')
else:
    reg = ants.registration(template, input, type_of_transform='SyNAggro', reg_iterations=(50, 30, 10), outprefix=f'results/template/{args.label}_native_b0_to_143template')

print(reg)

jac = ants.create_jacobian_determinant_image(template, reg['fwdtransforms'][0], 0, 1)

output = reg['warpedmovout']
ants.image_write(output, f'results/template/{args.label}_b0.nii.gz')

output = ants.smooth_image(output, sigma=(2, 2, 2), FWHM=True, sigma_in_physical_coordinates=True)
ants.image_write(output, f'results/template/{args.label}_smooth_fwmh2_b0.nii.gz')

ants.image_write(jac,    f'results/template/{args.label}_deform_jac_geom.nii.gz')
jac = ants.smooth_image(jac, sigma=(2, 2, 2), FWHM=True, sigma_in_physical_coordinates=True)
ants.image_write(jac,    f'results/template/{args.label}_deform_smoothjac_geom.nii.gz')

other_img = ants.image_read(args.input[0])
output = ants.apply_transforms(template, other_img, transformlist = reg['fwdtransforms'], imagetype=3, interpolator='linear')
other_input_label = os.path.basename(args.input[0])
ants.image_write(output, f'results/template/{args.label}_{other_input_label}')
