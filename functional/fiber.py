import dipy.denoise.noise_estimate as ne
import argparse 
import dipy.reconst.dti as dti
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel, auto_response)
import nibabel as nib
from dipy.io.image import save_nifti
from nibabel.streamlines import save as save_trk
from nibabel.streamlines import Tractogram

from dipy.tracking.streamline import Streamlines
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.tracking.utils import random_seeds_from_mask, seeds_from_mask, target, connectivity_matrix

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.segment.mask import median_otsu
from dipy.data import get_sphere, default_sphere
from dipy.direction import peaks_from_model, ProbabilisticDirectionGetter
from dipy.reconst import peaks, shm
from dipy.viz import window, actor, fvtk
from dipy.viz.colormap import line_colors
from dipy.tracking.eudx import EuDX

import numpy as np
import pandas as pd

def build_label_mask(label, atlas):
    ll = pd.read_csv(label, header=None)
    ll_wm = ll.iloc[:,3].str.contains('wm') # jhu white matter mask
    ll_wm[pd.isnull(ll_wm)] = True
    wm_index = ll.loc[ll_wm, 0]
    labelimg = nib.load(atlas)
    labeldata = labelimg.get_data().astype('int32')
    wm_mask = np.zeros(labeldata.shape)
    for i in wm_index:
        wm_mask[labeldata==i] = 1
    return wm_mask, labeldata

def build_model(data, gtab):
    _, mask = median_otsu(data[:, :, :, 0])
    tensor_model = dti.TensorModel(gtab, fit_method='WLS')
    #sigma = ne.estimate_sigma(data)
    #tensor_model = dti.TensorModel(gtab, fit_method='RESTORE', sigma=sigma)
    tensor_fit = tensor_model.fit(data, mask)
    fa = tensor_fit.fa
    md = tensor_fit.md
    return fa, md, tensor_fit.evecs, mask

def build_track_direction(gtab, data, mask, csd=True):
    if csd:
        response, ratio = auto_response(gtab, data, roi_radius=10, fa_thr=0.5)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
        peak = peaks_from_model(model=csd_model,
                                data=data,
                                sphere=default_sphere,
                                #sphere=get_sphere('symmetric724'),
                                mask=mask,
                                relative_peak_threshold=.5,
                                min_separation_angle=25, 
                                #return_sh=True,
                                parallel=True)
    else:
        csamodel = shm.CsaOdfModel(gtab, 6)
        peak = peaks.peaks_from_model(model=csamodel, data=data, 
                                      sphere=default_sphere,
                                      relative_peak_threshold=.5,
                                      min_separation_angle=25, 
                                      mask=mask)
    return peak

def get_tracking(fa, evecs, dg, affine, seeds, eudx_csa=True, eudx_fa=True):
    if eudx_fa:
        ind = dti.quantize_evecs(evecs, default_sphere.vertices)
        streamline_generator = EuDX(a=fa, ind=ind, seeds=seeds, 
                                    odf_vertices=default_sphere.vertices, 
                                    a_low=.1)

    elif not eudx_csa:
        tissue_classifier = ThresholdTissueClassifier(fa, 0.1)
        # default EuDX
        streamline_generator = LocalTracking(dg, tissue_classifier, seeds,
                                             # affine=np.eye(4),
                                             affine=affine,
                                             step_size=0.5)
        #prob_dg = ProbabilisticDirectionGetter.from_shcoeff(csd_peaks.shm_coeff, max_angle=30, sphere=default_sphere)
        #streamline_generator = LocalTracking(prob_dg, tissue_classifier, seeds, affine=affine, step_size=0.5, max_cross=1)
        #streamlines = Streamlines(streamline_generator)
    else:
        streamline_generator = EuDX(dg.peak_values, dg.peak_indices, 
                                    odf_vertices=peaks.default_sphere.vertices,
                                    affine=affine,
                                    a_low=.05, step_sz=.2, seeds=seeds)

    streamlines = list(streamline_generator)
    #save_trk(Tractogram(streamlines, affine_to_rasmm=affine), "%s.trk" % args.name)
    save_trk(Tractogram(streamlines, affine_to_rasmm=affine), "%s.1000.trk" % args.name)
    return streamline_generator

def plot(streamlines):
    ren = window.Renderer()
    #ren.add(actor.peak_slicer(csd_peaks.peak_dirs,
    #                          csd_peaks.peak_values,
    #                          colors=None))
    #if interactive:
    #    window.show(ren, size=(900, 900))
    #else:
    #    window.record(ren, out_path='csd_direction_field.png', size=(900, 900))
    #ren.clear()
    ren.add(actor.line(streamlines))
    window.record(ren, out_path='det_csd_streamlines.png', size=(900,900))

def get_connectivity(streamlines, atlas, affine):
    print(np.unique(atlas))
    M, groupping = connectivity_matrix(list(streamlines), atlas,
                                       affine=streamlines.affine,
                                       return_mapping=True, mapping_as_streamlines=True)
    M = pd.DataFrame(M)
    M.to_csv('%s.1000.connectivity' % args.name)
    return M


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("-dwi")
    p.add_argument("-vec")
    p.add_argument("-val")
    p.add_argument("-name")
    p.add_argument("-label")
    p.add_argument("-labelatlas")
    p.add_argument("-segs")
    args = p.parse_args()
    
    segs = nib.load(args.labelatlas)

    interactive = False
    #simple = True
    simple = False

    img = nib.load(args.dwi)
    affine = img.affine
    bval, bvec = read_bvals_bvecs(args.val, args.vec)
    gtab = gradient_table(bval, bvec)
    data = img.get_data()

    mask, labeldata = build_label_mask(args.label, args.labelatlas)
    fa, md, evecs, m = build_model(data, gtab)
    mask = np.logical_and(m, np.logical_and(np.logical_or(fa >= 0.3,
                                                          np.logical_and(fa>=0.15, md>=0.0011)), mask)) # combine fa with jhu white matter parts
    # save_nifti("%s_wm_mask.nii.gz" % args.name, mask.astype('int32'), affine)

    dg = None
    if not simple:
       dg = build_track_direction(gtab, data, m, False)

    streamlines = get_tracking(fa, evecs, dg, affine,
                               random_seeds_from_mask(mask, seeds_count=1000, affine=affine), True, False)
                               #random_seeds_from_mask(mask, seeds_count=8, affine=affine), True, False)
                              #seeds_from_mask(mask, density=[2,2,2], affine=affine))
                              # random_seeds_from_mask(mask, seeds_count=8),
    #plot(list(streamlines))
    M = get_connectivity(streamlines, labeldata, affine)
    print(M)

