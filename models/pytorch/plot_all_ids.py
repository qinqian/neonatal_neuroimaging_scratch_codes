import ants
import pandas as pd
import os

os.system('mkdir -p results/check_dups')
df = pd.read_csv('check_images.csv', index_col=0)

from matplotlib.backends.backend_pdf import PdfPages


with PdfPages('check_dup.pdf') as pdf:
    for row in df.index:
        image = ants.image_read('results/processed/%s_t2f_processed.nii.gz' % row)
        image.plot(axis=2, slices=range(image.shape[2]), filename='results/check_dups/%s_%s.pdf' % (row, df.loc[row, 'ID']))
