#!/usr/bin/env python
"""
http://nipy.org/dipy/examples_built/streamline_tools.html#example-streamline-tools
"""

import os, sys
import numpy as np
import pandas as pd
import argparse
from dipy.core.gradients import gradient_table
import pickle

from dipy.tracking.eudx import EuDX
from dipy.tracking import utils
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from nibabel.streamlines import save as save_trk
from nibabel.streamlines import Tractogram
import nibabel as nib

from dipy.viz import window, actor, fvtk
from dipy.viz.colormap import line_colors

from dipy.reconst import peaks, shm
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.direction import peaks_from_model, ProbabilisticDirectionGetter
from dipy.data import default_sphere
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier

def plot(streamlines, name):
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
    #window.record(ren, out_path='%s_eudx_streamlines.png' % name, size=(900,900))
    window.record(ren, out_path='%s_prob_streamlines.png' % name, size=(900,900))

def draw_target_fiber(streamlines, labels, target, affine):
    cc_slice = labels == target
    cc_streamlines = utils.target(streamlines, cc_slice, affine=affine)
    cc_streamlines = list(cc_streamlines)
    other_streamlines = list(utils.target(streamlines, cc_slice, affine=affine,
                                     include=False))
    assert len(other_streamlines) + len(cc_streamlines) == len(streamlines)
    #color = line_colors(cc_streamlines)
    #cc_streamlines_actor = fvtk.line(cc_streamlines, color)
    #
    #cc_ROI_actor = fvtk.contour(cc_slice, levels=[1], colors=[(1., 1., 0.)])
    #
    #vol_actor = fvtk.slicer(t1_data)
    #vol_actor.display(40, None, None)
    #
    #vol_actor2 = vol_actor.copy()
    #vol_actor2.display(None, None, 35)
    #
    #r = fvtk.ren()
    #fvtk.add(r, vol_actor)
    #fvtk.add(r, vol_actor2)
    #fvtk.add(r, cc_ROI_actor)
    #fvtk.add(r, cc_streamlines_actor)
    #fvtk.record(r, n_frames=1, out_path='corpuscallosum_axial.png', size=(800, 800))
    #fvtk.camera(r, [-1, 0, 0], [0,0,0], viewup=[0,0,1])
    #fvtk.record(r, n_frames=1, out_path='corpuscallusum_sagittal.png', size=(400,400))
    return
    
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("-dwi")
    p.add_argument("-vec")
    p.add_argument("-val")
    p.add_argument("-name")
    p.add_argument("-brainmask")
    p.add_argument("-wmmask")
    p.add_argument("-labelatlas")
    p.add_argument("-fa")
    p.add_argument("-md")
    args = p.parse_args()

    gtab = gradient_table(args.val, args.vec)
    rcond=None
    #model = shm.CsaOdfModel(gtab, 6)
    model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=6)

    dwi = nib.load(args.dwi)
    affine = dwi.affine
    labels = nib.load(args.labelatlas)
    labels = labels.get_data().astype('int32')
    print(labels.shape)
    white_matter = nib.load(args.wmmask)
    mask = nib.load(args.brainmask).get_data()
    fa = nib.load(args.fa).get_data()
    md = nib.load(args.md).get_data()

    #white_matter = (white_matter.get_data() * mask)
    #print(np.min(fa), np.max(fa))
    #print(np.min(md), np.max(md))
    #print(np.sum(white_matter.get_data() * mask >0))
    white_matter = np.logical_or(white_matter.get_data() * mask, mask * np.logical_or(fa >= 0.3, np.logical_and(fa>=0.15, md>=0.0011)))

    print(white_matter.shape)
    #print(np.sum(white_matter > 0))

    built_peaks = peaks.peaks_from_model(model=model, data=dwi.get_data(), sphere=peaks.default_sphere, relative_peak_threshold=.8,
                                         min_separation_angle=45,
                                         mask=white_matter)

    #seeds = utils.seeds_from_mask(white_matter, density=3)
    seeds = utils.seeds_from_mask(white_matter, density=2)
    print(seeds[0])
    #seeds = utils.random_seeds_from_mask(white_matter, seeds_count=50)
    x, y, z, g = built_peaks.peak_values.shape
    edge = np.array([x, y, z], dtype=np.float64) - 1.
    new_seeds = []
    print(len(seeds))
    for s in seeds:
        if np.any(s < 0.) or np.any(s > edge):
            continue
        new_seeds.append(s)

    print(len(new_seeds))
    #streamline_generator = EuDX(built_peaks.peak_values, built_peaks.peak_indices,
    #                            odf_vertices=peaks.default_sphere.vertices,
    #                            a_low=0.05, step_sz=0.5, seeds=new_seeds)


    fod_coeff = built_peaks.shm_coeff
    classifier = ThresholdTissueClassifier(fa, .05) # terminate when fa < 0.15
    prob_dg = ProbabilisticDirectionGetter.from_shcoeff(fod_coeff, max_angle=45,
                                                        sphere=default_sphere)
    streamline_generator = LocalTracking(prob_dg, classifier, new_seeds, affine, step_size=.5)

    streamlines = list(streamline_generator)
    print(len(streamlines))
    save_trk(Tractogram(streamlines, affine_to_rasmm=affine), "%s.prob.trk" % args.name)
    #plot(streamlines, args.name)
    
    affine = streamline_generator.affine
    M, grouping = utils.connectivity_matrix(streamlines, labels, affine=affine,
                                            return_mapping=True,
                                            mapping_as_streamlines=True)

    M = pd.DataFrame(M)
    #M.to_csv('%s.eudx.connectivity' % args.name)
    #f = open('%s.eudx.grouping' % args.name, 'w')
    M.to_csv('%s.prob.connectivity' % args.name)
    f = open('%s.prob.grouping' % args.name, 'w')
    pickle.dump(grouping, f)
    f.close()

