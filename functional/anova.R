library(ANTsR)
library(extrantsr)
library(neurobase)
library(spm12r)

fs <- Sys.glob("data/*/*img")
fsimg <- imageFileNames2ImageList(fs)
fslabel <- factor(basename(dirname(fs)))
## bold.template <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNC-CEDARS-INFANT-20170420/resolution-adaptive/infant-neo-intensity-withSkull-2_928mm.nii")
own.template <- antsImageRead('/data1/qinqian/MRI/build_bold_template/fudan_children_hospital_bold_normal_template.nii.gz')

## own.template.center <- acpc_reorient(infiles=c(bold.template, own.template), modality="T1") # must be the same dimensions

## pdf("template.pdf")
## plot(own.template, nslices=64, ncolumns=6, axis=3)
## plot(bold.template, nslices=48, ncolumns=6, axis=3)
## plot(bold.template, nslices=48, ncolumns=6, axis=3)
## plot(own.template, nslices=64, ncolumns=6, axis=2)
## plot(own.template, nslices=64, ncolumns=6, axis=1)
## ortho2(own.template)
## ortho2(bold.template)
## dev.off()

library(pander)
fsi.ave = fsimg[[1]]
for (i in 2:length(fsimg))
    fsi.ave = fsi.ave + fsimg[[i]]
fsi.ave = fsi.ave/length(fsimg)

reg <- antsRegistration(own.template, fsi.ave, typeofTransform="SyN")

fsimg.reg <- lapply(fsimg, function(x) {
    antsApplyTransforms(own.template, x, transformlist=reg$fwdtransforms)
    })

library(animation)
ani.options(autobrowser=F)
saveGIF({
    for (i in 1:length(fsimg)) {
        double_ortho(as.array(fsimg.reg[[i]]), own.template)
        ## ortho2(fsimg.reg[[i]])
    }
}, movie.name = paste0("check_alff_gt_35weeks_reg.gif"))

mask <- getMask(own.template)
fsi.mat <- imageListToMatrix(fsimg.reg, mask)

means <- rowMeans(fsi.mat)
rfsi.mat.m <- as.matrix(residuals(lm(fsi.mat~means)))

rfsi.mat <- aov(rfsi.mat.m~scale(as.numeric(fslabel)))
## mdl <- bigLMStats(rfsi.mat)

Rfsi.mat.p <- unlist(lapply(summary(rfsi.mat), function(x) {
    return(x[1,5])
    }))

qv <- -log10(p.adjust(rfsi.mat.p, 'BH'))
qvimg = makeImage(mask, qv)

png("TSB_groups.png", width=800, height=600)
plot(own.template, qvimg, color.overlay=c("orange", "red"), window.overlay=c(1.8, 2), alpha=0.9, nslices=36, ncolumns=6, axis=3, colorbar=T)
## plot(own.template, nslices=37, ncolumns=8, axis=3)
## plot(own.template, nslices=37, ncolumns=8, axis=2)
## plot(own.template, nslices=37, ncolumns=8, axis=1)
dev.off()

mypreds = scale(as.numeric(fslabel))
sccan <- sparseDecom2(
    inmatrix = list(rfsi.mat.m, mypreds),
    inmask = c(mask, NA),
    mycoption = 0,
    sparseness = c(0.25, 0.3),
    nvecs=3,
    its=3,
    cthresh=c(20,0),
    smooth=6,
    perms=100, robust=1)

print(sccan$ccasummary)

scimg <- matrixToImages(t(sccan$eig1), mask)
sccansol <- abs(scimg[[1]]) %>% iMath("Normalize")
## sccansol2 <- abs(scimg[[2]]) %>% iMath("Normalize")

pdf("sccan.pdf")
plot(fsi.ave, sccansol, window.overlay=range(c(sccan$eig1)), alpha=0.7, nslices=36, ncolumns=6, axis=3, colorbar=T)
dev.off()
