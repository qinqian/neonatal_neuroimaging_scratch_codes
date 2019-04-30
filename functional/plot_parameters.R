library(ANTsR)
para.maps <- cbind(sort(Sys.glob("../data/fmri_nii/subj*/*b0*nii.gz")),
                  sort(Sys.glob("../data/fmri_nii/subj*/dti/*fsl*FA*nii.gz")),
                  sort(Sys.glob("../data/fmri_nii/subj*/dti/*fsl*MD*nii.gz")),
                  sort(Sys.glob("../data/fmri_nii/subj*/func/alff*nii.gz")),
                  sort(Sys.glob("../data/fmri_nii/subj*/func/falff*nii.gz")),
                  sort(Sys.glob("../data/fmri_nii/subj*/func/reho*nii.gz")))

i = 20
z1 <- antsImageRead(para.maps[i,1])
z2 <- antsImageRead(para.maps[i,2])
z3 <- antsImageRead(para.maps[i,3])
z4 <- antsImageRead(para.maps[i,4])
z5 <- antsImageRead(para.maps[i,5])
z6 <- antsImageRead(para.maps[i,6])
pdf("test.pdf", width=12, height=10)
par(mfrow=c(1,6), mar=c(0,0,0,0), mar=c(0,0,0,0))
plot(z1, ncolumn=1, axis=2, window.img=range(z1))
plot(z2, ncolumn=1, axis=2, window.img=range(z2))
plot(z3, ncolumn=1,axis=2, window.img=range(z3))
plot(z4, ncolumn=1, axis=2, window.img=range(z4))
plot(z5, ncolumn=1, axis=2, window.img=range(z5))
plot(z6, ncolumn=1, axis=2, window.img=range(z6))
dev.off()
