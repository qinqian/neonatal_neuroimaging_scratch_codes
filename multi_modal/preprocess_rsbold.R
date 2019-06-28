library(optparse)
library(colormap)
library(ANTsR)

parser <- OptionParser()
parser <- add_option(parser, c("-v", "--verbose"),
                    action="store_true",
                    default=T, help='print extra output [default]')
parser <- add_option(parser, c("-i", "--img"), type="character")
parser <- add_option(parser, c("-o", "--output"), type="character")
parser <- add_option(parser, c("-a", "--average"), type="character")
parser <- add_option(parser, c("-p", "--parammotion"), type="character")
parser <- add_option(parser, c("-m", "--mask"), type="character")
parser <- add_option(parser, c("-b", "--b0image"), type="character")
parser <- add_option(parser, c("-s", "--seg"), type="character")

args <- parse_args(parser)
#print(args)

b0 <- antsImageRead(args$b0image)
seg <- antsImageRead(args$seg)
boldmean <- antsImageRead(args$average)

bold <- antsImageRead(args$img)
boldmask <- antsImageRead(args$mask)

moco.raw <- read.table(args$parammotion)
#print(dim(moco.raw))
moco = rbind(rep(0, 6), moco.raw)
moco[, 1:3] = moco[, 1:3] * 50
moco = apply(moco, 2, diff)
moco = abs(moco)
fd = rowSums(moco)
#print(length(fd))
bold.mat <- timeseries2matrix(bold, boldmask)
#print(rowMeans(bold.mat))

cat(fd, sep='\t', file=paste0(args$output, '_all_fds.txt'))
if (sum(fd>=0.2)>0) {
    badtimes = which(fd>=0.2)
    cat(badtimes, '\n', sep='\t', file=paste0(args$output, '_badtimes0.2.txt'))
} else {
    cat(0, '\n', sep='\t', file=paste0(args$output, '_badtimes.txt'))
}

if (sum(fd>=0.5)>0) {
    badtimes = which(fd>=0.5)
    cat(badtimes, '\n', sep='\t', file=paste0(args$output, '_badtimes0.5.txt'))
    goodtimes = which(fd<0.5)
    for (v in badtimes) {
        interp = spline(goodtimes, bold.mat[goodtimes, v], method="natural", xout=badtimes)$y
        bold.mat[badtimes, v] = interp
    }
} else {
    cat(0, '\n', sep='\t', file=paste0(args$output, '_badtimes.txt'))
}

bold.reg <- antsRegistration(b0, boldmean, typeofTransform="SyNBoldAff", outprefix=args$output)
seg.bold <- antsApplyTransforms(boldmean, seg, transformlist=bold.reg[['invtransforms']], interpolator="nearestNeighbor")
csf.bold <- thresholdImage(thresholdImage(seg.bold, 5, 5) + thresholdImage(seg.bold, 1, 1), 1, 2)
#wm.bold  <- thresholdImage(thresholdImage(seg.bold, 3, 3) + thresholdImage(seg.bold, 8, 8) + thresholdImage(seg.bold, 9, 9), 1, 3)
wm.bold  <- thresholdImage(thresholdImage(seg.bold, 3, 3) + thresholdImage(seg.bold, 9, 9), 1, 3)

boldpng <- paste0(args$output, '_boldseg1.png')
plot(boldmean, csf.bold, alpha=0.8, axis=2, nslices=30, color.overlay=c(colormaps$viridis), ncolumns=6, quality=8, outname=boldpng)
boldpng <- paste0(args$output, '_boldseg2.png')
plot(boldmean, wm.bold, alpha=0.8, axis=2, nslices=30, color.overlay=c(colormaps$viridis), ncolumns=6, quality=8, outname=boldpng)

csf.vol <- which(subset(csf.bold, boldmask > 0)>=1)
wm.vol <- which(subset(wm.bold, boldmask > 0)>=1)

print(length(csf.vol))
print(length(wm.vol))

csfnui <- rowMeans(bold.mat[, csf.vol])
wmnui <- rowMeans(bold.mat[, wm.vol])
globalnui <- rowMeans(bold.mat)
tissuenui <- cbind(globalnui, wmnui, csfnui)
tissuederiv <- rbind(rep(0, dim(tissuenui)[2]), diff(tissuenui, 1))
#print(dim(tissuederiv))

dvars <- computeDVARS(bold.mat)
moconui <- cbind(moco.raw, moco.raw*moco.raw)
moconui.diff <- apply(moconui, 2, diff)
#print(head(moconui.diff))
mocoderiv <- rbind(rep(0, dim(moconui)[2]), moconui.diff)
#print(dim(mocoderiv))
compcornui <- compcor(bold.mat, 6)
nuisance.combined <- cbind(1:nrow(compcornui), tissuenui, tissuederiv, moconui, mocoderiv, compcornui, dvars)
write.table(nuisance.combined, file=paste0(args$output, '_all_nuisance.txt'), sep='\t', quote=F)

print(nuisance.combined)
bold.mat <- apply(bold.mat, 2, function(x) (x-mean(x)) / sd(x) ^ as.logical(sd(x)))
nuisance.combined <- apply(nuisance.combined, 2, function(x) (x-mean(x)) / sd(x) ^ as.logical(sd(x)))
print(rowMeans(nuisance.combined))
bold.mat <- residuals(lm(bold.mat ~ nuisance.combined))
bold.mat <- frequencyFilterfMRI(bold.mat, tr = antsGetSpacing(bold)[4],
                               freqLo = 0.009, freqHi = 0.08,
                               opt = "butt")

bold.clean <- matrix2timeseries(bold, boldmask, bold.mat)
clean.bold <- paste0(args$output, '_clean4d.nii.gz')
antsImageWrite(bold.clean, clean.bold)


alff = apply( bold.mat, FUN=function(x) {
    alffmap(x, detrend=F, tr=antsGetSpacing(bold)[4])
}, MARGIN=2 )
alff.clean <- matrixToImages(alff, boldmask)
antsImageWrite(alff.clean[[1]], paste0(args$output, '_alff.nii.gz'))
antsImageWrite(alff.clean[[2]], paste0(args$output, '_falff.nii.gz'))

antsImageWrite(bold.reg[['warpedmovout']], paste0(args$output, '_boldaveinb0.nii.gz'))

system(paste0('3dReHo -mask ', args$mask, ' -inset ', clean.bold, ' -prefix ', paste0(args$output, '_afni'), ' -neigh_RAD 3 '))
system(paste0('cd ', dirname(args$output), ' && 3dAFNItoNIFTI ', paste0(basename(args$output), '_afni+orig.BRIK')))

reho = antsImageRead(paste0(args$output, '_afni.nii'))
alff.b0 = antsApplyTransforms(b0, alff.clean[[1]], transformlist=bold.reg[['fwdtransforms']])
falff.b0 = antsApplyTransforms(b0, alff.clean[[2]], transformlist=bold.reg[['fwdtransforms']])
reho.b0 = antsApplyTransforms(b0, reho, transformlist=bold.reg[['fwdtransforms']])
antsImageWrite(alff.b0,  paste0(args$output, '_alffinb0.nii.gz'))
antsImageWrite(falff.b0, paste0(args$output, '_falffinb0.nii.gz'))
antsImageWrite(reho.b0, paste0(args$output, '_rehoinb0.nii.gz'))

clean.bold.b0 = antsApplyTransforms(b0, bold.clean, transformlist=bold.reg[['fwdtransforms']], imagetype=3)
clean.bold <- paste0(args$output, '_clean4dinb0.nii.gz')
antsImageWrite(clean.bold.b0, clean.bold)
