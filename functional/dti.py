import dipy.denoise.noise_estimate as ne
import argparse 
import dipy.reconst.dti as dti
import nibabel as nib
from dipy.io.image import save_nifti
from nibabel.streamlines import save as save_trk
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("-dwi")
p.add_argument("-vec")
p.add_argument("-val")
p.add_argument("-mask")
p.add_argument("-name")
args = p.parse_args()
img = nib.load(args.dwi)
bval, bvec = read_bvals_bvecs(args.val, args.vec)
gtab = gradient_table(bval, bvec)
mask = nib.load(args.mask)
affine = img.get_affine()
data = img.get_data()
sigma = ne.estimate_sigma(data)
#maskdata, mask = median_otsu(data[:, :, :, 0])
dti_restore = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
fit = dti_restore.fit(data, mask)
#fa = np.clip(fit.fa, 0, 1)
fa = fit.fa
md = fit.md
rd = fit.rd
#cfa = dti.color_fa(fa, fit.evecs)
save_nifti("%s_fa.nii.gz" % args.name, fa, affine)
save_nifti("%s_md.nii.gz" % args.name, md, affine)
save_nifti("%s_rd.nii.gz" % args.name, rd, affine)

