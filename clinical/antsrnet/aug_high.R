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
downsample <- c(256, 256, 20)
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
        if (modal=='dwi') {
            ds = Sys.glob(paste0(x, ".json"))
            if (length(ds) == 0) {
                return(F)
            } else {
                for (j in ds) {
                    print(j)
                    protocal1 <- read_json(j)$ProtocolName
                    print(protocal1)
                    if (grepl('DWI', protocal1, ignore.case=T)||grepl('ep2d', protocal1, ignore.case=T)) {
                        print('ok')
                        break
                    }
                }
                modal.1 <- antsImageRead(gsub(".json", ".nii.gz", j), dimension=4)
                if (dim(modal.1)[4]<2) {
                    return(F)
                } else {
                    modal.1 <- splitNDImageToList(modal.1)[[2]]
                }
            }
            if (withN4) {
                modal.1s <- resampleImage(n4BiasFieldCorrection(modal.1), interpType=0,
                                         useVoxels=T, resampleParams=downsample)
            }else{
                modal.1s <- resampleImage(modal.1, interpType=0,
                                         useVoxels=T, resampleParams=downsample)
            }
            return(modal.1s)
        }
        if (modal=='t2') {
            protocal1 <- read_json(paste0(x, ".json"))$ProtocolName
            modal.1 <- antsImageRead(paste0(x, ".nii.gz"))
            if (withN4) {
                modal.1s <- resampleImage(n4BiasFieldCorrection(modal.1), interpType=0,
                                         useVoxels=T, resampleParams=downsample)
            } else {
                modal.1s <- resampleImage(modal.1, interpType=0,
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
                modal.1s <- resampleImage(n4BiasFieldCorrection(modal.1), interpType=0,
                                         useVoxels=T, resampleParams=downsample)
            }  else {
                modal.1s <- resampleImage(modal.1, interpType=0,
                                         useVoxels=T, resampleParams=downsample)
            }
        }
    }
    load.subject <- function(x) {
        n <- basename(x[1])
        x1 <- paste0("training_data_high/", n, "_t21.nii.gz")
        x2 <- paste0("training_data_high/", n, "_t22.nii.gz")
        x3 <- paste0("training_data_high/", n, "_t1.nii.gz")
        x4 <- paste0("training_data_high/", n, "_dwi.nii.gz")
        if (!file.exists(x4)) {
            t21 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T2w"))
            t22 <- load.img(paste0(x[1], "/", "anat/", n, "_run-2_T2w"))
            t1 <- load.img(paste0(x[1], "/", "anat/", n, "_run-1_T1w"), 't1')
            dwi <- load.img(paste0(x[1], "/", "dwi/", n, "_run-*_dwi*"), 'dwi')
            if (class(dwi)!='antsImage') {return(c(NA,NA,NA,NA))}
            if (withN4) {
                antsImageWrite(t1, x1)
                x2.warp <- antsRegistration(t1, t21, typeofTransform = c('Rigid'))
                antsImageWrite(x2.warp$warpedmovout, x2)
                x3.warp <- antsRegistration(t1, t22, typeofTransform = c('Rigid'))
                antsImageWrite(x3.warp$warpedmovout, x3)
                x4.warp <- antsRegistration(t1, dwi, typeofTransform = c('Rigid'))
                antsImageWrite(x4.warp$warpedmovout, x4)
            } else {
                antsImageWrite(t21, x1)
                antsImageWrite(t22, x2)
                antsImageWrite(t1, x3)
                antsImageWrite(dwi, x4)
            }
        }
        return(c(x1, x2, x3, x4))
    }
    load.subject(subject.dir)
}

meta <- loadMeta()

system("mkdir -p training_data_high")

trainProportion = 0.6
set.seed(1234)
subindex = sample(1:nrow(meta), nrow(meta)*trainProportion, replace = F)

validProportion = 0.2
valindex <- sample((1:nrow(meta))[-subindex], nrow(meta)*validProportion, replace=F)

testindex <- (1:nrow(meta))[-c(subindex, valindex)]
imagesRead(as.character(meta[c(subindex, testindex, valindex)[1],]))

registerDoMC(3)
imgs.list <- foreach(i=c(subindex, testindex, valindex), .combine=rbind) %do% {
  print(i)
  imagesRead(as.character(meta[i,]))
}

## remove NA (the ones without DWI data)
## writeout files with labels
## df.out <- cbind(imgs.list, meta[,4])

## df.out <- df.out[complete.cases(df.out), ]
## write.table(df.out, file="high_res_meta.csv", quote=F, sep=",", row.names=F, col.names=F)

## imgs.list.t1 <- df.out[sample(which(df.out[,5] == "1"), 10), 1:4]
## imgs.list.t2 <- df.out[sample(which(df.out[,5] == "0"), 10), 1:4]

## imgs.list.t <- rbind(imgs.list.t1, imgs.list.t2)
## imgs.list.t[,1] <- basename(imgs.list.t[,1])
## imgs.list.t[,2] <- basename(imgs.list.t[,2])
## imgs.list.t[,3] <- basename(imgs.list.t[,3])
## imgs.list.t[,4] <- basename(imgs.list.t[,4])

## write.table(imgs.list.t, file="template_high.csv", quote=F, sep=",", row.names=F, col.names=F)

#system('bash test_template_high.sh')

df.out <- read.csv('high_res_meta.csv',header=F)
