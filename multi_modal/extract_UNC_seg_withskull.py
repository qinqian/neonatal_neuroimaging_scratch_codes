#!/usr/bin/env python

import ants

unc = ants.image_read('results/UNC/sub_368_seg-orig.img')
unc_raw = ants.image_read('results/UNC/sub_368_T1w_pad_striped-orig.img')

unc_mask = unc_raw.get_mask()
mask = ants.iMath(unc_mask, 'MD', 5)

raw = ants.image_read('../phaseA_clean_dicom/output/sub-368/anat/sub-368_T1w.nii.gz')
raw = raw.resample_image_to_target(unc_raw)

raw = raw * mask
ants.image_write(raw, 'results/UNC/sub_368_withskull.nii.gz')
