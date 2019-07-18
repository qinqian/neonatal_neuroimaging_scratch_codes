# Import the Figure module :
from visbrain.gui import Brain
from visbrain.objects import BrainObj
from visbrain.io import download_file

"""Download file if needed
"""
file = download_file('lh.sig.nii.gz', astype='example_data')

b_obj = BrainObj('inflated', translucent=False, sulcus=True)
b_obj.add_activation(file=file, clim=(5., 20.), hide_under=5, cmap='viridis',
                     hemisphere='left')

vb = Brain(brain_obj=b_obj)
vb.rotate('left')
vb.show()
