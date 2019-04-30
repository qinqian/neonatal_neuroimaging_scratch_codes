import os.path as op
import numpy as np
import mne
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne import setup_volume_source_space, setup_source_space
from mne import make_forward_solution
from mne.io import read_raw_fif
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.connectivity import spectral_connectivity
from mne.viz import circular_layout, plot_connectivity_circle
import pandas as pd
import sys


con = sys.argv[1]
cutoff = float(sys.argv[2])

# First, we reorder the labels based on their location in the left hemi
labels = pd.read_csv('../../data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv', header=None)
labels_names = list(labels.apply(lambda x: "%s-%s" % (x[1], 'lh') if 'left' in x[2] else "%s-%s" % (x[1], 'rh'), axis=1))

coord = pd.read_table('../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_labels_coordinates.xls')
print(coord.head())
print(labels_names)

#conn = pd.read_csv('ugt1a1_dti_mrtrix_connectome_neglog10p_value.csv', index_col=0) # > 2
#conn = pd.read_csv('ugt1a1_dti_mrtrix_connectome_1minusp_value.csv', index_col=0) # 0 ~ 1
#conn = pd.read_csv('dipy_ugt1a1_dti_mrtrix_connectome_1minusp_value.csv', index_col=0) # 0 ~ 1
conn = pd.read_csv(con, index_col=0) # 0 ~ 1
print(conn.shape)
#print(conn.describe())

#lh_labels = [name for name in label_names if name.endswith('lh')]
#
## Get the y-location of the label
#label_ypos = list()
#for name in lh_labels:
#    idx = label_names.index(name)
#    ypos = np.mean(labels[idx].pos[:, 1])
#    label_ypos.append(ypos)
#
## Reorder the labels based on their location
#lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
#
## For the right hemi
#rh_labels = [label[:-2] + 'rh' for label in lh_labels]
#
## Save the plot order and create a circular layout
#node_order = list()
#node_order.extend(lh_labels[::-1])  # reverse the order
#node_order.extend(rh_labels)
#
#node_angles = circular_layout(label_names, node_order, start_pos=90,
#                              group_boundaries=[0, len(label_names) / 2])
#
# Plot the graph using node colors from the FreeSurfer parcellation. We only
fig = plt.figure(num=None, figsize=(18, 18), facecolor='black')
fig, _ = plot_connectivity_circle(conn.values, labels_names, vmin=cutoff, vmax=1, # n_lines=5, #node_angles=node_angles, node_colors=label_colors,
                         #title='All-to-All Structural Connectivity for UGT1A1 ANOVA analysis',
                         fig = fig, interactive=False)
# fig.savefig('Figure2.pdf', facecolor='black')
# fig.savefig('Figure2_dipy.pdf', facecolor='black')
# fig.savefig('Figure3_bold.pdf', facecolor='black')
fig.savefig('%s.pdf' % con, facecolor='black')


