library(ANTsR)
library(gdata)
library(extrantsr)
library(foreach)
library(doMC)
template <- antsImageRead('../../data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalAtlas2/template_T2.nii.gz')
template <- splitNDImageToList(template)[[17]]

df <- read.xls('../clinical/final_pvl_with_mask_with_control.xlsx')
write.table(df, 'mask_meta.xls', sep='\t', quote=F)
pvl <- with(df, group.1=='pvl'&LABELED=='1-3')#  & cleaned_up_files=='1')

## subset(df, cleaned_file_id=='6' & group.1=='pvl')
## subset(df, cleaned_file_id=='6' & group.1=='control')

ids = as.vector(df$cleaned_file_id)
ids = ids!='#N/A'&(!is.na(ids)) &pvl

df <- df[ids&pvl,]
print(head(df))
quit()

transforms0 <- list()
transforms1 <- list()
cmask.list <- list()
qmask.list <- list()
n <- 0
for (i in df$cleaned_file_id) {
    n <- n+1
    transforms0[[n]] <- Sys.glob(paste0("../cleaned_data/pvl/transform/pvl_", i, "_*/*Generic*"))
    transforms1[[n]] <- Sys.glob(paste0("../cleaned_data/pvl/transform/pvl_", i, "_*/*Warp.nii.gz"))
    if (length(transforms0[[n]]) == 0) {
        transforms0[[n]] <- Sys.glob(paste0("../cleaned_data/pvl/transform/cont_", i, "_*/*Generic*"))
        transforms1[[n]] <- Sys.glob(paste0("../cleaned_data/pvl/transform/cont_", i, "_*/*Warp.nii.gz"))
    }
    cmask.list[[n]] <- Sys.glob(paste0('../phaseA_exploration/remodel_mxnet/mask/all/chen_labels/1-3/', i, '.nii.gz'))
    qmask.list[[n]] <- Sys.glob(paste0('../phaseA_exploration/remodel_mxnet/mask/all/qq_labels/1-3/', i, '.nii.gz'))
}

#registerDoMC(20)
chen <- foreach(i=1:length(cmask.list)) %do% {
    mask <- antsImageRead(cmask.list[[i]])
    print(dim(mask))
    iii <- antsApplyTransforms(template, mask, transformlist = c(transforms1[[i]], transforms0[[i]]), interpolator='GenericLabel')
    return(iii)
}

qin <- foreach(i=1:length(qmask.list)) %do% {
    mask <- antsImageRead(qmask.list[[i]])
    print(dim(mask))
    iii <- antsApplyTransforms(template, mask, transformlist = c(transforms1[[i]], transforms0[[i]]), interpolator='GenericLabel')
    return(iii)
}

qin.lesion = antsImageClone(qin[[1]])
for (i in 1:length(qin)) {
    qin.lesion <- qin.lesion + qin[[i]]
}
qin.lesion <- qin.lesion / length(qin)
antsImageWrite(qin.lesion, 'qin_lesion.nii.gz')

average.lesion = antsImageClone(chen[[1]])
for (i in 1:length(chen)) {
    average.lesion <- average.lesion + chen[[i]]
}
average.lesion <- average.lesion / length(chen)
antsImageWrite(average.lesion, 'chen_lesion.nii.gz')

seg = seq(0, 1, 0.1)
pcols = colorRampPalette(c("blue", "white", "red"))(length(seg)-1)
lesion.slices <- which(apply(average.lesion, 3, sum)>0)
## plot(template, list(average.lesion), nslices=9, axis=3, window.overlay=c(0, 0.7), color.overlay=c('red'), colorbar=T, alpha=0.8, ncolumns=9)
## plot(template, list(qin.lesion), nslices=80, axis=3, window.overlay=c(0, 0.7), color.overlay=c('red'), colorbar=T, alpha=0.8)
## dev.off()
