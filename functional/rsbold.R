library(extrantsr)
library(neurobase)
library(neurohcp)
library(R.utils)
library(matlabr)
print(have_matlab())
library(spm12r)
library(ANTsR)
library(fslr)
library(mFilter)
library(MASS)
library(matrixStats)
library(RColorBrewer)
library(ggplot2)
library(spm12r)

args <- commandArgs(T)
## ## for test
setwd("~/data1/MRI/UGT1A1/data/fmri_nii/subj0_0/func")
args <- c("9")

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F, stringsAsFactors=F)
## volume register average bold to t2 template
label.template <- antsImageRead("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz")

own.template <- antsImageRead("/data1/qinqian/MRI/build_bold_template/fudan_children_hospital_bold_normal_template.nii.gz")

subimg <- paste0(args[1], "_sub.nii.gz")
tshift_inter <- paste0(args[1], "_tshift_interleaved.nii.gz")
mocoimg <- paste0(args[1], "_ants_moco_nuisance_bandpass.nii.gz")
mocoimg.cen <- paste0(args[1], "_ants_moco_nuisance_bandpass_censor.nii.gz")
mocoimg.cen.smooth <- paste0(args[1], "_ants_moco_nuisance_bandpass_censor_smooth.nii.gz")
maskimg <- paste0(args[1], "_ants_mask.nii.gz")

despike <- function(x) {
    fin=paste0(x, ".nii.gz")
    system(paste0("3dDespike -prefix ", x, ".despike -NEW ", fin))
    system(paste0("3dAFNItoNIFTI -prefix ", x, ".despike ", paste0(x, ".despike+orig")))
    return(paste0(x, ".despike.nii"))
}

get_fd <- function(x) {
    mp <- apply(x[,3:5]*50, 2, diff)
    mp = rbind(rep(0, 3), mp)
    mp <- abs(mp)
    return (rowSums(mp))
}

des <- despike(args[1])
b <- readnii(des)
d <- oro2ants(b)
tr <- antsGetSpacing(d)[4]
first.scan = floor(10.0 / tr) + 2  # 10 seconds "stabilization of signal"
subfmri <- subset_4d(b, first.scan:ntim(b))
d <- oro2ants(subfmri)
antsImageWrite(d, subimg)

slicetiming <- function(x, z, tr) {
    ## cmd <- paste0("ImageMath 4 ", y, " SliceTimingCorrection ", x, " 0 sinc")
    ## system(cmd)
    img <- antsImageRead(fsl_slicetimer(file=x, outfile=z, tr=tr, direction="z", acq_order="interleaved", indexing="up"))
    return(img)
    }

## slicing time correction
simg <- slicetiming(subimg, tshift_inter, tr)

## motion correction (realign)
ref.slice <- getAverageOfTimeSeries(simg)

my.preprocessfMRI <- function(boldImage, maskImage = NA, maskingMeanRatioThreshold = 0.75, 
    initialNuisanceVariables = NA, numberOfCompCorComponents = 6, 
    doMotionCorrection = TRUE, useMotionCorrectedImage = FALSE, 
    motionCorrectionAccuracyLevel = 1, meanBoldFixedImageForMotionCorrection = NA, 
    frequencyLowThreshold = NA, frequencyHighThreshold = NA, 
    spatialSmoothingType = "none", spatialSmoothingParameters = 0, 
    residualizeMatrix = TRUE) 
{
    nuisanceVariables <- initialNuisanceVariables
    numberOfTimePoints <- dim(boldImage)[4]
    framewiseDisplacement <- rep(0, numberOfTimePoints)
    if (doMotionCorrection) {
        motionCorrectionResults <- .motion_correction(boldImage, 
            fixed = meanBoldFixedImageForMotionCorrection, moreaccurate = motionCorrectionAccuracyLevel)
        motionCorrectionParameters <- motionCorrectionResults$moco_params
        nuisanceVariables <- as.matrix(motionCorrectionParameters)[, 
            3:ncol(motionCorrectionParameters)]
        for (i in 2:numberOfTimePoints) {
            motionCorrectionParametersAtTime1 <- c(motionCorrectionParameters[i, 
                3:14])
            rotationMatrixAtTime1 <- matrix(as.numeric(motionCorrectionParametersAtTime1[1:9]), 
                ncol = 3, nrow = 3)
            translationAtTime1 <- as.numeric(motionCorrectionParametersAtTime1[10:12])
            motionCorrectionParametersAtTime2 <- c(motionCorrectionParameters[i - 
                1, 3:14])
            rotationMatrixAtTime2 <- matrix(as.numeric(motionCorrectionParametersAtTime2[1:9]), 
                ncol = 3, nrow = 3)
            translationAtTime2 <- as.numeric(motionCorrectionParametersAtTime2[10:12])
            samplePoint <- data.matrix(t(matrix(rep(10, 3), nrow = 1)))
            transformedPointAtTime1 <- data.matrix(rotationMatrixAtTime1) %*% 
                samplePoint + translationAtTime1
            transformedPointAtTime2 <- data.matrix(rotationMatrixAtTime2) %*% 
                samplePoint + translationAtTime2
            framewiseDisplacement[i] <- dist(rbind(transformedPointAtTime2, 
                transformedPointAtTime1))[[1]]
        }
        framewiseDisplacement[1] <- mean(framewiseDisplacement[2:numberOfTimePoints])
        if (useMotionCorrectedImage) {
            boldImage <- motionCorrectionResults$moco_img
        }
    }
    else {
        if (useMotionCorrectedImage) {
            cat("Warning:  if motion correction is not performed then the motion corrected image is unavailable for use.\n")
            useMotionCorrectedImage <- FALSE
        }
    }

    averageImage <- new("antsImage", "float", 3)
    antsMotionCorr(list(d = 3, a = boldImage, o = averageImage))
    if (is.na(maskImage)) {
        maskImage <- getMask(averageImage)
    }

    antsImageWrite(maskImage, maskimg)
    averageImage[maskImage == 0] <- 0
    boldMatrix <- timeseries2matrix(boldImage, maskImage)
    DVARS <- computeDVARS(boldMatrix)
    if (numberOfCompCorComponents > 0) {
        compCorNuisanceVariables <- compcor(boldImage, maskImage, 
            ncompcor = numberOfCompCorComponents, variance_extreme = 0.975)
        if (is.na(nuisanceVariables) || is.null(dim(nuisanceVariables))) {
            nuisanceVariables <- compCorNuisanceVariables
        }
        else {
            nuisanceVariables <- cbind(nuisanceVariables, compCorNuisanceVariables)
        }
    }

    globalSignal = rowMeans(boldMatrix)
    trend <- matrix(1:numberOfTimePoints)
    nuisanceVariables <- cbind(globalSignal, trend, trend**2, trend**3, nuisanceVariables)
    write.csv(nuisanceVariables, file="nuisance_var.csv", quote=F)

    boldMatrix <- scale(boldMatrix, center=T, scale=T)
    if (!is.na(nuisanceVariables[1]) & residualizeMatrix) {
        boldMatrix <- residuals(lm(boldMatrix ~ scale(nuisanceVariables, center=T, scale=T)))
    }

    DVARSpostCleaning <- computeDVARS(boldMatrix)
    cleanBoldImage <- matrix2timeseries(boldImage, maskImage, 
        boldMatrix)
    censored <- aslCensoring(cleanBoldImage, mask=maskImage)
    antsImageWrite(censored$asl.inlier, mocoimg.cen) # ready for ALFF

    boldMatrix <- timeseries2matrix(censored$asl.inlier, maskImage)

    if (!is.na(frequencyHighThreshold) & !is.na(frequencyHighThreshold) & 
        (frequencyLowThreshold != frequencyHighThreshold)) {
        boldMatrix <- frequencyFilterfMRI(boldMatrix, tr = antsGetSpacing(boldImage)[4], 
            freqLo = frequencyLowThreshold, freqHi = frequencyHighThreshold, 
            opt = "trig")
    }

    cleanBoldImage <- matrix2timeseries(censored$asl.inlier, maskImage, boldMatrix)

    smoothCleanBoldImage = cleanBoldImage * 1
    if (spatialSmoothingType == "gaussian") {
        if (length(spatialSmoothingParameters) == 1) {
            sigmaVector <- c(rep(spatialSmoothingParameters[1], 
                3, ), 0)
            smoothCleanBoldImage <- smoothImage(cleanBoldImage, 
                sigmaVector, FWHM=T, sigmaInPhysicalCoordinates=T)
        }
        else {
            stop("Expecting a single scalar parameter.")
        }
    }
    else if (spatialSmoothingType == "perona-malik") {
        if (length(spatialSmoothingParameters) == 2) {
            smoothCleanBoldImage <- iMath(cleanBoldImage, "PeronaMalik", 
                spatialSmoothingParameters[1], spatialSmoothingParameters[2])
        }
        else {
            stop("Expecting a two element vector.")
            return
        }
    }
    else if (spatialSmoothingType != "none") {
        stop("Unrecognized smoothing option.")
    }

    antsImageWrite(smoothCleanBoldImage, mocoimg.cen.smooth) # ready for correlation
    return(list(alff=censored$asl.inlier, cleanBoldImage = smoothCleanBoldImage, maskImage = maskImage, 
        DVARS = DVARS, DVARSpostCleaning = DVARSpostCleaning, 
        FD = framewiseDisplacement, globalSignal = globalSignal, 
        nuisanceVariables = nuisanceVariables))
}

## tweak default preprocessfMRI
clean.fmri <- my.preprocessfMRI(simg, numberOfCompCorComponents=6,
                                meanBoldFixedImageForMotionCorrection = own.template,
                                spatialSmoothingType="gaussian",
                                motionCorrectionAccuracyLevel=1,
                                frequencyLowThreshold=0.009, frequencyHighThreshold=0.08,
                                spatialSmoothingParameters=6, doMotionCorrection = TRUE,
                                useMotionCorrectedImage = T, residualizeMatrix = TRUE)

## pdf(paste0(args[1], "_motion.pdf"))
## plot(cbind(dvars, motionRes$dvars))
## abline(a=0, b=1, col="red")
## dev.off()
## reg.csf.mask <- antsApplyTransforms(motionRes$moco_avg_img, csf.mask, transformlist=c(reg.shi$fwdtransforms))
## rf <- colorRampPalette(rev(brewer.pal(11, "Spectral")))
## r <- rf(32)
## pdf(paste0(args[1], "_scale_tshift_volreg_bold.pdf"))
## mat <- scale(boldMatrix)
## image(x=1:nrow(mat), y=1:ncol(mat), mat, useRaster=T, col=r, xlab="Scan", ylab="Voxel",
##       main=paste0("Dimension", dim(mat)[1], "x", dim(mat)[2]))
## dev.off()
## long <- reshape2::melt(nuisance.comcor)
## pdf(paste0(args[1], "_signal_components.pdf"))
## colnames(long) <- c("scan", "component", "value")
## ggplot(long, aes(x=scan, y=value)) + geom_line() + facet_wrap(~component, ncol=1)
## dev.off()

## ALFF

## check volumes by average and gif 
library(animation)
ani.options(autobrowser=F)
arr = as.array(clean.fmri$cleanBoldImage)
## lab = as.array(label.template)
saveGIF({
    for (i in 1:ntim(clean.fmri$cleanBoldImage)) {
        double_ortho(arr[,,,i], own.template)
        ## ortho2(arr[,,,i])
    }
    }, movie.name = paste0(args[1], "_check.gif"))

ALFF <- function(x, mask) {
    system(paste0("3dcalc -a ", mask, " -expr \'a\' -prefix ", gsub(".nii.gz", "", mask)))
    system(paste0("3dcalc -a ", x, " -expr \'a\' -prefix ", gsub(".nii.gz", "", x)))
    cmd = paste0("3dRSFC -prefix ", gsub(".nii.gz", ".RSFC", x), " -nodetrend -mask ", gsub(".nii.gz", "+orig", mask), " 0.009 0.08 ", gsub(".nii.gz", "", x), "+orig[0..", dim(clean.fmri$alff)[4], "]")
    system(cmd)
    system(paste0("3dAFNItoNIFTI -prefix ", gsub(".nii.gz", ".RSFC_ALFF ", x) , gsub(".nii.gz", ".RSFC_ALFF+orig ", x)))
    system(paste0("3dAFNItoNIFTI -prefix ", gsub(".nii.gz", ".RSFC_fALFF ", x) , gsub(".nii.gz", ".RSFC_fALFF+orig ", x)))
    system(paste0("3dReHo", " -mask ", gsub(".nii.gz", "+orig", mask), " -inset ", gsub(".nii.gz", "+orig", x), " -prefix ", gsub(".nii.gz", ".RSFC_REHO ", x), "-neigh_RAD 2.9 -chi_sq"))
   system(paste0("3dAFNItoNIFTI -prefix ", gsub(".nii.gz", ".RSFC_REHO ", x) , gsub(".nii.gz", ".RSFC_REHO+orig ", x)))
}

ALFF(mocoimg.cen, maskimg)


falff <- antsImageRead(gsub(".nii.gz", ".RSFC_fALFF.nii", mocoimg.cen))
falff.s <- smoothImage(falff, c(3,3,3), FWHM=T, sigmaInPhysicalCoordinates=T)

reho <- antsImageRead(gsub(".nii.gz", ".RSFC_REHO.nii", mocoimg.cen), 3)
reho.s <- smoothImage(reho, c(3,3,3), FWHM=T, sigmaInPhysicalCoordinates=T)

pdf(paste0(args[1], '_falff.pdf'), width=10, height=10)
plot(falff, nslices=60, ncolumns=10)
plot(falff.s, nslices=60, ncolumns=10)
plot(reho, nslices=60, ncolumns=10)
plot(reho.s, nslices=60, ncolumns=10)
dev.off()

## basic network analysis by correlation
## adapted from filterfMRIforNetworkAnalysis (the function seems not finished)
## network <- function(x) {
## ## warpedParcel <- antsApplyTransforms(motionRes$moco_avg_img, label.template, transformlist=breg$fwdtransforms, interpolator='NearestNeighbor')
## ## antsImageWrite(warpedParcel, filename=warpedimg)
## ## antsImageWrite(moco.smooth, filename=mocoimg.smooth)
#### NOTE::::
### =========== remember the brain mask, errors....
##     oulabels <- sort(unique(warpedParcel[warpedParcel>0]))
##     labelvec <- warpedParcel[warpedParcel>0]
##     labmat <- matrix(data = rep(NA, length(oulabels)*nrow(boldMatrix)), nrow=length(oulabels))
##     for (mylab in oulabels) {
##         dd <- labelvec == mylab
##         submat <- boldMatrix[, dd]
##         myavg <- rowMeans(submat) # the first principal component instead of mean values
##         ## myavg <- svd(submat, nu=1, nv=0)$u[,1] # the first principal component instead of mean values
##         labmat[oulabels==mylab, ] <- myavg
##     }
##     ## other advanced methods
##     ## 1. try robust correlation coefficients(MASS:rlm(a~b))[2] later
##     ## 2. nilearn partial correlation or filterfMRIforNetworkAnalysis pcormat (not understand)
##     ## 3. nilearn tangent (the best in terms of prediction not sure the principal)
##     ## 4. sklearn/nilearn glasso and R glasso::glasso
##     conmat <- cor(t(labmat)) ## the most basic exploration
##     write.csv(conmat, file=paste0(args[1], "_fmri_connectivity_basics_correlation.csv"), quote=F)
## }
