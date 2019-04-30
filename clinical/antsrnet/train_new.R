library(ANTsR)

meta = read.table("data/Shanghai/pvl/source/participants.tsv")

length(unlist(sapply(matrix(meta[,1]), function(x) {
    at1 = Sys.glob(paste0("data/Shanghai/pvl/source/", x[1], "/anat/*run-01_T1w.nii.gz")) # axis T1w
    if (length(at1) == 0) {
        at1 = NA
    }

    at1 = Sys.glob("data/Shanghai/pvl/source/sub-*/anat/*run-01_T1w.nii.gz")

    st = Sys.glob("data/Shanghai/pvl/source/sub-*/anat/*run-02_T1w.nii.gz") # sagital T1w
    t2 = Sys.glob("data/Shanghai/pvl/source/sub-*/anat/*run-02_T2w.nii.gz") # T2w
    tf = Sys.glob("data/Shanghai/pvl/source/sub-*/anat/*run-02_T2w.nii.gz") # T2 flair

    adc = Sys.glob("data/Shanghai/pvl/source/sub-*/dwi/*run-02_dwi.nii.gz") # ADC
    dwi = Sys.glob("data/Shanghai/pvl/source/sub-*/dwi/*run-01_dwi.nii.gz") # DWI
    bval =  Sys.glob("data/Shanghai/pvl/source/sub-*/dwi/*run-01_dwi.bval") # b vals

})))

NonLocalSuperResolution
