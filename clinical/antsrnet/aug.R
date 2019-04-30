## generate augmented volume by building template for 60% of all the data(training)
## then process to get warp/inverse warp and affine transformation files
## finally, generate hdf5 by applying each of the tranformation to each of the data

library(tfestimators)
require(tensorflow)
require(keras)
require(ANTsR)
require(jsonlite)
library(reticulate)
library(magick)
library(abind)
library(rhdf5)
require(fslr)
library(extrantsr)
library(ggplot2)
library(doMC)
library(foreach)

workspace <- "/home/qinqian/data1/MRI/01_PVL_AI"
downsample <- c(64, 64, 19)
#downsample <- c(32, 32, 19)
setwd(workspace)

loadMeta <- function() {
    pvl <- read.table("output_pvl/participants.tsv", header=T, stringsAsFactors = F)
    pvl[,4] <- 'pvl'
    pvl <- pvl[file.exists(paste0('output_pvl/', pvl[,1], '/anat/', pvl[,1], "_run-2_T2w.nii.gz")),]
    pvl[,1] <- paste0('output_pvl/', pvl[,1])
    cont <- read.table("output_control/participants.tsv", header=T, sep='\t', stringsAsFactors = F)
    cont <- cont[!(cont[,1]%in%basename(pvl[,1])), ]
    cont <- cont[file.exists(paste0('output_control/', cont[,1], '/anat/', cont[,1], "_run-2_T2w.nii.gz")), ]
    cont[,1] <- paste0('output_control/', cont[,1])
    meta <- rbind(pvl, cont)
    group = meta[,4]
    labels = rep(0, length(group))
    labels[group=='pvl'] = 1
    meta[,4] = labels
    meta
}

imagesRead <- function(subject.dir, withN4=F) {
    load.img <- function(x, modal='t2') {
        if (modal=='t2') {
            protocal1 <- read_json(paste0(x, ".json"))$ProtocolName
            modal.1 <- antsImageRead(paste0(x, ".nii.gz"))
            if (withN4) {
                modal.1s <- n4BiasFieldCorrection(resampleImage(modal.1, interpType=4,
                                                               useVoxels=T, resampleParams=downsample))
            } else {
                modal.1s <- resampleImage(modal.1, interpType=4,
                                         useVoxels=T, resampleParams=downsample)
            }
        } else {
            protocal1 <- read_json(paste0(x, ".json"))$ProtocolName
            if (length(grep("sag", protocal1, ignore.case=T))>=1) {
                protocal1 <- read_json(paste0(gsub('run-1', 'run-2', x), ".json"))$ProtocolName
                modal.1 <- antsImageRead(paste0(gsub('run-1', 'run-2', x), ".nii.gz"))
            } else {
                modal.1 <- antsImageRead(paste0(x, ".nii.gz"))
            }

            if (withN4) {
                modal.1s <- n4BiasFieldCorrection(resampleImage(modal.1, interpType=4,
                                                               useVoxels=T, resampleParams=downsample))
            }  else {
                modal.1s <- resampleImage(modal.1, interpType=4,
                                         useVoxels=T, resampleParams=downsample)
            }
        }
    }
    load.subject <- function(x) {
        n <- basename(x[1])
        t21 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T2w"))
        t22 <- load.img(paste0(x[1], "/", "anat/", n, "_run-2_T2w"))
        t1 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T1w"), 't1')
        if (withN4) {
            ## due to the suffix .csv/.txt not given
            ## the template.list is given
            ## antsMultivariateTemplateConstruction2.sh
            ## use t1 as template0, t21 as template1, t22 as template2
            x1 <- paste0("training_data/MYtemplate0", n, "_t1Repaired.nii.gz")
            x2 <- paste0("training_data/MYtemplate1", n, "_t21Repaired.nii.gz")
            x3 <- paste0("training_data/MYtemplate2", n, "_t22Repaired.nii.gz")

            antsImageWrite(t1, x1)
            x2.warp <- antsRegistration(t1, t21, typeofTransform = c('QuickRigid'))
            antsImageWrite(x2.warp$warpedmovout, x2)
            x3.warp <- antsRegistration(t1, t22, typeofTransform = c('QuickRigid'))
            antsImageWrite(x3.warp$warpedmovout, x3)
        } else {
            x1 <- paste0("training_data/", n, "_t21.nii.gz")
            x2 <- paste0("training_data/", n, "_t22.nii.gz")
            x3 <- paste0("training_data/", n, "_t1.nii.gz")
            antsImageWrite(t21, x1)
            antsImageWrite(t22, x2)
            antsImageWrite(t1, x3)
        }
        #createAntsrTransform(type='AffineTransform', precision="float", dimension=3)
        ##return(abind(a, along=3)) ## height x width x (z * modality)
        return(c(x1, x2, x3))
    }
    load.subject(subject.dir)
}

meta <- loadMeta()

system("mkdir -p training_data")

trainProportion = 0.6
set.seed(1234)
subindex = sample(1:nrow(meta), nrow(meta)*trainProportion, replace = F)

validProportion = 0.2
valindex <- sample((1:nrow(meta))[-subindex], nrow(meta)*validProportion, replace=F)

testindex <- (1:nrow(meta))[-c(subindex, valindex)]

registerDoMC(4)
imgs.list <- foreach(i=subindex, .combine=rbind) %dopar% {
  print(i)
  imagesRead(as.character(meta[i,]))
}

#imgs.list[,1] <- basename(imgs.list[,1])
#imgs.list[,2] <- basename(imgs.list[,2])
#imgs.list[,3] <- basename(imgs.list[,3])

#imgs.list2 <- foreach(i=c(valindex, testindex), .combine=rbind) %dopar% {
#   print(i)
#   imagesRead(as.character(meta[i,]))
#}
#imgs.list2[,1] <- basename(imgs.list2[,1])
#imgs.list2[,2] <- basename(imgs.list2[,2])
#imgs.list2[,3] <- basename(imgs.list2[,3])

#write.table(rbind(imgs.list, imgs.list2), file="template.csv", quote=F, sep=",", row.names=F, col.names=F)

#system("bash test_template.sh")

## oa=createAntsrTransform(matrix=matrix(rnorm(9), nrow=3), dimension=3,translation=c(3,1,2))
## img2 = applyAntsrTransformToImage(a, imgs.arr[[1]], imgs.arr[[2]])
## matrix(getAntsrTransformParameters(a), nrow=3)

## registerDoMC(12)
## imgs.list <- foreach(i=c(valindex, testindex), .combine=rbind) %dopar% {
##    print(i)
##    imagesRead(as.character(meta[i,]), withN4=T)
## }

template <- imageFileNames2ImageList(Sys.glob("training_data/FUDAN_PVLtemplate?.nii.gz"))

## registerDoMC(12)
## imgs.list <- foreach(i=c(valindex, testindex), .combine=rbind) %dopar% {
##    n <- basename(meta[i, 1])
##    x1 <- antsImageRead(paste0("training_data/MYtemplate0", n, "_t1Repaired.nii.gz"))
##    x1.warp <- antsRegistration(template[[1]], x1, typeofTransform = c('SyN'))
##    x2 <- antsImageRead(paste0("training_data/MYtemplate1", n, "_t21Repaired.nii.gz"))
##    x3 <- antsImageRead(paste0("training_data/MYtemplate2", n, "_t22Repaired.nii.gz"))
##    x2.warp <- antsApplyTransforms(template[[2]], x2, transformlist=c(x1.warp$fwdtransforms))
##    x3.warp <- antsApplyTransforms(template[[3]], x3, transformlist=c(x1.warp$fwdtransforms))
##    system(paste0("mv ", x1.warp$invtransforms[2], " training_data/MY", n, "_t1xxxInverseWarp.nii.gz"))

##    system(paste0("mv ", x1.warp$fwdtransforms[1], " training_data/MY", n, "_t1xxxWarp.nii.gz"))
##    system(paste0("mv ", x1.warp$fwdtransforms[2], " training_data/MY", n, "_t1xxxGenericAffine.mat"))
##    antsImageWrite(x1.warp$warpedmovout, file=paste0("training_data/MYtemplate0", n, "_t1xxxWarpedToTemplate.nii.gz"))
##    antsImageWrite(x2.warp, file=paste0("training_data/MYtemplate1", n, "_t21xxxWarpedToTemplate.nii.gz"))
##    antsImageWrite(x3.warp, file=paste0("training_data/MYtemplate2", n, "_t22xxxWarpedToTemplate.nii.gz"))
## }

## split real data and augmented training data
## registration each of the volume to another invidual and store as hdf5 files
imgs.warp <- cbind(Sys.glob(paste0("training_data/FUDAN_PVL", basename(meta[, 1]), "*Affine.mat")),
                  Sys.glob(paste0("training_data/FUDAN_PVL", basename(meta[, 1]), "*InverseWarp*nii.gz")))

## training data
imgs.warp2 <- cbind(paste0('training_data/FUDAN_PVLtemplate0', basename(meta[, 1]), '_t21Repaired.nii.gz'),
                   Sys.glob(paste0('training_data/FUDAN_PVLtemplate0', basename(meta[ , 1]), '_t21*WarpedTo*nii.gz')),
                   paste0('training_data/FUDAN_PVLtemplate1', basename(meta[ , 1]), '_t22Repaired.nii.gz'),
                   Sys.glob(paste0('training_data/FUDAN_PVLtemplate1', basename(meta[ , 1]), '_t22*WarpedTo*nii.gz')),
                   paste0('training_data/FUDAN_PVLtemplate2', basename(meta[ , 1]), '_t1Repaired.nii.gz'),
                   Sys.glob(paste0('training_data/FUDAN_PVLtemplate2', basename(meta[ , 1]), '_t1*WarpedTo*nii.gz')),
                   imgs.warp)

index <- expand.grid(1:nrow(imgs.warp2), 1:nrow(imgs.warp2))

train.labels <- to_categorical(meta[, 4][index[,1]])

imgs.warp3 <- cbind(
                   imgs.warp2[index[,1], c(2,4,6)],
                   imgs.warp2[index[,2], c(1,3,5)],
                   imgs.warp2[index[,2], c(7,8)])

imgs.warp3 <- as.data.frame(imgs.warp3)
imgs.warp3 <- cbind(data.frame(index=index[,1], real=index[,1] == index[,2]),
                   imgs.warp3)
write.table(imgs.warp3, file="augmented_list.txt", sep="\t", quote=F)

ff = "augmented_images.h5"
h5createFile(ff)
h5createGroup(ff, 'pvl')
h5createDataset(ff, 'pvl/labels', dim(train.labels),
                storage.mode='integer', chunk=c(128, 2), level=5)
h5write(train.labels, file=ff, 'pvl/labels')


h5createDataset(ff, 'pvl/images', c(nrow(imgs.warp3), 64, 64, 19, 3),
                storage.mode='double', chunk=c(128, 64, 64, 19, 3), level=7)

c(m, s) %<-% read.table("mean_sd.txt")

#x_y.batch <- foreach(i=1:nrow(imgs.warp3)) %dopar% {
print(nrow(imgs.warp3))
x_y.batch <- foreach(i=1:nrow(imgs.warp3)) %do% {
#x_y.batch <- foreach(i=1:300) %do% {
    x0 = antsImageRead(as.character(imgs.warp3[i, 6]))
    x0.t = antsImageRead(as.character(imgs.warp3[i, 3]))
    trans = c(as.character(imgs.warp3[i, 9]), as.character(imgs.warp3[i, 10]))
    x0.inv <- antsApplyTransforms(fixed=x0, moving=x0.t,
                                 transformlist = trans,
                                 whichtoinvert=c(T, F))
    x1 = antsImageRead(as.character(imgs.warp3[i, 7]))
    x1.t = antsImageRead(as.character(imgs.warp3[i, 4]))
    x1.inv <- antsApplyTransforms(fixed=x1, moving=x1.t,
                                 transformlist = trans,
                                 whichtoinvert=c(T, F))
    x2 = antsImageRead(as.character(imgs.warp3[i, 8]))
    x2.t = antsImageRead(as.character(imgs.warp3[i, 5]))
    x2.inv <- antsApplyTransforms(fixed=x2, moving=x2.t,
                                 transformlist = trans,
                                 whichtoinvert=c(T, F))
    x0.inv <- as.array(iMath(x0.inv, "Normalize"))
    x1.inv <- as.array(iMath(x1.inv, "Normalize"))
    x2.inv <- as.array(iMath(x2.inv, "Normalize"))
    x0.inv <- (x0.inv-m)/s
    x1.inv <- (x1.inv-m)/s
    x2.inv <- (x2.inv-m)/s
    x.batch <- abind(list(x0.inv, x1.inv, x2.inv), along=4) # batch * x * y * z * channels
    x.batch <- abind(list(x.batch), along=0)
    print(dim(x.batch))
    print(i)
    h5write(x.batch, file=ff, name='pvl/images',
            start=c(i, 1,1,1,1))
}

##h5dump(ff)

H5close()
