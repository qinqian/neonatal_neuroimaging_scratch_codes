library(optparse)

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
print(args)

library(ANTsR)

b0 <- antsImageRead(args$b0image)
seg <- antsImageRead(args$seg)
boldmean <- antsImageRead(args$average)
bold.reg <- antsRegistration(b0, boldmean, typeofTransform="SyNBold", outprefix=args$output)
seg.bold <- antsApplyTransforms(boldmean, seg, transformlist=bold.reg[['invtransforms']], interpolator="nearestNeighbor")
csf.bold <- thresholdImage(thresholdImage(seg.bold, 5, 5) + thresholdImage(seg.bold, 1, 1), 1, 2)
wm.bold  <- thresholdImage(seg.bold, 3, 3)

bold <- antsImageRead(args$img)
boldmask <- antsImageRead(args$mask)

moco <- read.table(args$parammotion)
moco = rbind(rep(0, 6), moco)
moco[, 1:3] = moco[, 1:3] * 50
moco = apply(moco, 2, diff)
moco = abs(moco)
fd = rowSums(moco)
bold.mat <- timeseries2matrix(bold, boldmask)
if (sum(fd>=0.5)>0) {
    badtimes = which(fd>=0.5)
    goodtimes = which(fd<0.5)
    for (v in badtimes) {
        interp = spline(goodtimes, bold.mat[goodtimes, v], method="natural", xout=badtimes)$y
        bold.mat[badtimes, v] = interp
    }
}

csf.vol <- which(subset(csf.bold, boldmask > 0)>=0.5)
wm.vol <- which(subset(wm.bold, boldmask > 0)>=0.5)

csfnui <- rowMeans(bold.mat[, csf.vol])
wmnui <- rowMeans(bold.mat[, wm.vol])
globalnui <- rowMeans(bold.mat)

compcornui <- compcor(bold.mat, 6)

nuisance.combined <- scale(cbind(1:nrow(compcornui), csfnui, wmnui, globalnui, moco, compcornui))
bold.mat <- scale(bold.mat)
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

