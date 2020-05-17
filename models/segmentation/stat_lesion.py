from skimage import measure
import numpy as np

import glob

labels = glob.glob("2d_pvls/*label*npy")

all_lesion_slices = []
slice_num = []
for i in labels:
    label_array = np.load(i)
    if (label_array.sum() > 0):
        component = measure.label(label_array, connectivity=1)
        component = measure.regionprops(component)
        a = [p.area for p in component]
        #all_lesion_slices.append(np.mean(a))
        all_lesion_slices.append(np.sum(a))
        slice_num.append(len(a))

print(all_lesion_slices)
print(np.mean(all_lesion_slices))
print(np.std(all_lesion_slices))
print(np.mean(slice_num))
print(np.std(slice_num))

#print(lesions.shape)
#print(np.mean(lesions))
