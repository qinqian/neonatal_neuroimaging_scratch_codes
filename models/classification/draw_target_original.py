import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
import matplotlib.pyplot as plt
import ants
z = ants.image_read('nii/520__iso_1mm_merge_chan.nii.gz')
z = z.split_channels()
modal = ['T1', 'T1_Sag', 'T2', 'T2_FLAIR', 'DWI']
for i, c in enumerate(z):
    ants.plot(c, nslices=60, ncol=6, filename='520_%s_raw.pdf' % modal[i], dpi=500, axis=2)
