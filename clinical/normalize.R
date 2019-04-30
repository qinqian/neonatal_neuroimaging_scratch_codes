library(ANTsR)

#'nii/1000__iso_1mm_merge_chan.nii.gz'
ii = commandArgs(T)
template = antsImageRead('../mri/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-withSkull.nii')
a = antsImageRead(ii)
a = splitChannels(a)
trans = antsRegistration(template, a[[1]], typeofTransform = "SyN")
jac=createJacobianDeterminantImage(a[[1]], trans$fwdtransforms[1])

a = lapply(a, function(x) {
    antsApplyTransforms(fixed=template, moving=x,
                        transformlist=trans$fwdtransforms)
}
)

a[[6]] = jac
a = mergeChannels(a)
antsImageWrite(a, file=paste0('norm/', gsub('.nii.gz', '_norm2unc.nii.gz', basename(ii))))
