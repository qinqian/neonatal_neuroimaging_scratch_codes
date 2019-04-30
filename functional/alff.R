library(ANTsR)
library(doMC)
library(foreach)
library(reshape2)
library(data.table)
library(foreach)
library(doMC)

rss <- Sys.glob('../data/fmri_nii/subj*/func/rs_clean_without_lowpass.nii.gz')

registerDoMC(3)
foreach(i = rss) %dopar% {
    x = antsImageRead(i)
    m = antsImageRead(paste0(dirname(dirname(i)), '/brainmask2.nii.gz'))
    mat <- timeseries2matrix(x, m)
    fallf = apply( mat, FUN=alffmap, MARGIN=2 )
    alff <- makeImage(m, fallf[1,])
    antsImageWrite(alff,  paste0(dirname(dirname(i)), '/func/alff.nii.gz'))
    falff <- makeImage(m, fallf[2,])
    antsImageWrite(falff, paste0(dirname(dirname(i)), '/func/falff.nii.gz'))
}


## too slow...........give up, try 3dReHo from AFNI instead later
rss <- Sys.glob('../data/fmri_nii/subj*/func/rs_clean_with_lowpass.nii.gz')

registerDoMC(3)
foreach(i = rss) %dopar% {
    x = antsImageRead(i)
    m = antsImageRead(paste0(dirname(dirname(i)), '/brainmask2.nii.gz'))
    rehoi <- reho(x, m, shifter=2)
    antsImageWrite(rehoi, paste0(dirname(dirname(i)), '/func/reho.nii.gz'))
    gc()
}
 
