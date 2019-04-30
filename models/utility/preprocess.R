library(ANTsR)
outputs <- '/data1/qinqian/MRI/01_PVL_AI/phaseB_classification/nii'
niil = commandArgs(T)
i = niil[6]
niil = niil[1:5]
x = imageFileNames2ImageList(niil)
print(x)
x[[5]] <- splitNDImageToList(x[[5]])[[2]]
x <- lapply(x, function(x) {
print(antsGetSpacing(x))
y = iMath(x, 'PadImage', 10)
y = resampleImage(y, c(1,1,1), useVoxels=F)
return(y)
})
for (j in 2:length(x)) {
    warp <- antsRegistration(x[[1]], x[[j]], typeofTransform="Rigid")
    x[[j]] <- warp$warpedmovout
}
mask <- getMask(x[[1]])
mask2 <- getMask(x[[2]])
mask <- thresholdImage(mask+mask2, 1, 3)
x <- lapply(x, cropImage, mask)
x <- mergeChannels(x)
antsImageWrite(x, paste0(outputs, '/', i, '__iso_1mm_merge_chan.nii.gz')) # use index to output
## 1.use resample to keep in the same dimension
x = resampleImage(x, c(192, 192, 128), useVoxels=T)
## or 2. normalize to the neonate template
antsImageWrite(x, paste0(outputs, '/', i, '__not_iso_merge_chan.nii.gz')) # use index to output
