import argparse
from plot import plot_with_mask

p = argparse.ArgumentParser()
p.add_argument('--mask')
#p.add_argument('--t2')
p.add_argument('--t2flair')
p.add_argument('--name')
args = p.parse_args()
#incomplete plotting with nilearn.plotting weird....
#display = plot_anat(t2flair, display_mode='z', cut_coords=5, draw_cross=False, annotate=False)
#display.add_contours(mask, antialiased=False, linewidths=0.34, levels=[0],
#                     colors=['red'], linestyles='dashed', filled=False, vmin=0, vmax=1.1)
# f, axes = plt.subplots(4, 5, sharey=True)
# print(t2flair.shape)
# n = 0
# mask_data = mask.get_data()
# mask_last = nibabel.Nifti1Image(np.zeros(mask.shape), mask.affine)
# for i in range(axes.shape[0]):
#     for j in range(axes.shape[1]):
#         n += 1
#         if n >= t2flair.shape[2]:
#             display = plot_anat(mask_last, display_mode = 'z', cut_coords=[n], axes=axes[i, j], draw_cross=True)
#             continue
#         print(n)
#         display = plot_anat(t2flair, display_mode = 'z', cut_coords=[n], axes=axes[i, j], draw_cross=True)
#         # judge = np.sum(mask_data[:, :, i]==1)
#         display.add_contours(mask, antialiased=False, linewidths=0.34, levels=[0],
#                              colors=['red'], linestyles='dashed', filled=False, vmin=0, vmax=1.1)
#     if n >= t2flair.shape[2]:
#         display = plot_anat(mask_last, display_mode = 'z', cut_coords=[n], axes=axes[i, j], draw_cross=True)
#        # break

#display.savefig('%s.pdf' % args.name, dpi=800)
#display.close()

plot_with_mask(args.mask, args.t2flair, args.name)
