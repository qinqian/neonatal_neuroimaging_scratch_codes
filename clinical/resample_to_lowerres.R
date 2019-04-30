iso <- Sys.glob("nii/*iso_1mm*")
library(ANTsR)
for (i in iso) {
    a = antsImageRead(i)
    a = resampleImage(a, c(80,80,60), useVoxels = T)
    antsImageWrite(a, file=gsub('iso', 'lower', i))
    #a = resampleImage(a, c(128,128,80), useVoxels = T)
    #antsImageWrite(a, file=gsub('iso', '_136_', i))
}
