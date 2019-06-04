library(ANTsR)
#    mean        sd
#t2f 0.5872402   0.2024262
#t2: 0.5821896   0.1871165
#adc 0.4300238   0.2120626
#t1  0.4056657   0.1636801
#t1s 0.3866455   0.133584
get_mean_sd = function(x) {
    t2s = imageFileNames2ImageList(Sys.glob(paste0("/data1/qinqian/MRI/01_PVL_AI/scripts/models/pytorch/results/processed/*_", x, "_*nii.gz")))
    mask = getMask(t2s[[1]])
    t2s.mat = imageListToMatrix(t2s, mask)
    print(dim(t2s.mat))
    return(list(mean(t2s.mat), sd(t2s.mat)))
}

#get_mean_sd('t2')
#get_mean_sd('t2f')
#get_mean_sd('adc')
#get_mean_sd('t1')
get_mean_sd('t1s')

