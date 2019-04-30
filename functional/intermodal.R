library(ANTsR)
library(spm12r)
library(extrantsr)
library(neurobase)
library(rcamino)
library(animation)
library(fslr)
library(foreach)
library(doMC)
registerDoMC(2)

resample.read <- function(x) {
    f = antsImageRead(x)
    resampleImage(f, c(1,1,1), interpType=0)
}

### load template
### JHU template
fa <- resample.read('../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_fass.nii.gz')
t1 <- resample.read('../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_T1.nii.gz')
t1ss <- resample.read('../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_T1ss.nii.gz')
t2 <- resample.read('../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_T2.nii.gz')
b0.template <- resample.read('../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_b0ss.nii.gz')

labels <- resampleImage(antsImageRead('/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz'), c(1,1,1), interpType=1)

t1.mask <- getMask(t1ss, cleanup=2)
t1.a <- ants2oro(t1)

t1.all <- sort(c(Sys.glob("../data/fmri_nii/subj*/struct/?.nii.gz"), Sys.glob("../data/fmri_nii/subj*/struct/??.nii.gz")))
t1.all <- grep("st.nii.gz", t1.all, invert=T,value=T)
dti.all <- sort(c(Sys.glob("../data/fmri_nii/subj*/dti/?.nii.gz"), Sys.glob("../data/fmri_nii/subj*/dti/??.nii.gz")))
dti.all <- grep("dti.nii.gz", dti.all, invert=T,value=T)
rs.all <- sort(c(Sys.glob("../data/fmri_nii/subj*/func/?.nii.gz"), Sys.glob("../data/fmri_nii/subj*/func/??.nii.gz")))
rs.all <- grep("rs.nii.gz", rs.all, invert=T,value=T)

fs <- data.frame(t1=t1.all, rs=rs.all, dti=dti.all)
check.img <- function(x, ref, label) {
    z <- antsImageRead(x)
    arr = splitNDImageToList(z)
    ani.options(autobrowser=F)
    ## saveGIF({
    ##     for (i in 1:length(arr)) {
    ##         oro.nifti::image(ants2oro(arr[[i]]), z=16)
    ##     }
    ## },
    ## movie.name = paste0(x, "_", label, "_check1_img.gif"),
    ## img.name = paste0(basename(dirname(dirname(x))), label, basename(x), "_check1_img"))
    saveGIF({
        for (i in 1:length(arr)) {
            ortho2(ants2oro(arr[[i]]))
        }
    },
    movie.name = paste0(x, "_", label, "_check1_ortho.gif"),
    img.name = paste0(basename(dirname(dirname(x))), label, basename(x), "_check1_ortho"))
}

## check original data
## foreach(i = 1:nrow(fs)) %dopar% {
##     check.img(as.vector(fs[i,3]), t2, 'dti')
##     check.img(as.vector(fs[i,2]), t2, 'bold')
## }

## first denoise and bias correction of the data from bash
fs.denoise <- cbind(gsub(".nii.gz", "_bias.nii.gz", as.vector(fs[,1])),
                    gsub(".nii.gz", "_bias.nii.gz", as.vector(fs[,2])),
                    gsub(".nii.gz", "_denoised_bias.nii.gz", as.vector(fs[,3])))

## registerDoMC(32)
## foreach(i = 1:nrow(fs.denoise)) %dopar% {
##     check.img(as.vector(fs.denoise[i,3]), t1, 'dti')
##     check.img(as.vector(fs.denoise[i,2]), t1, 'bold')
## }


merge4D <- function(x, interval) {
    dimen <- dim(x[[1]])
    x.new <- array(0, dim=c(dimen, length(x)))
    for (i in 1:length(x)) {
        temp <- x[[i]][1:dimen[1], 1:dimen[2], 1:dimen[3]]
        x.new[1:dimen[1], 1:dimen[2], 1:dimen[3], i] <- temp
    }
    x.new <- as.antsImage(x.new)
    orig <- antsGetOrigin(x[[1]])
    direct <- antsGetDirection(x[[1]])
    direct <- cbind(direct, 0)
    direct <- rbind(direct, c(0,0,0,1))

    antsSetSpacing(x.new, c(antsGetSpacing(x[[1]]), interval))
    antsSetDirection(x.new, direct)
    antsSetOrigin(x.new, c(orig, 0))
    return(x.new)
}

## align T1 bravo to b0 average, then apply dwi2mask mask operation
intermodal.coregister <- function(x, maskornot=T, force=F) {
    out <- c(paste0(dirname(dirname(x[1])), "/brain_mask.nii.gz"), paste0(dirname(dirname(x[1])), "/struct/st.nii.gz"), paste0(dirname(dirname(x[1])), "/dti/dti.nii.gz"))
    judge <- file.exists(out)
    if (!force) {
        if (sum(judge) == 3) {
            print("Yes")
            return(T)
        }
    }
    print("Second")
    st <- antsImageRead(x[1])
    dti <- antsImageRead(x[3])
    dti.mask <- antsImageRead(gsub("_denoised_bias.nii.gz", "_mask.nii.gz", x[3]))

    ## use this as targets
    b0.ave <- getAverageOfTimeSeries(subset_4d(dti, 1:3))

    outprefix = paste0(x[3], "_st2b0")
    st.afffns = Sys.glob(file.path(paste(outprefix, 
                                          "*", "[0-9]GenericAffine.mat", sep = "")))
    if (force || length(dti.afffns) == 0) {
        st.mo <- antsRegistration(b0.ave, st, typeofTransform='Rigid', verbose=T, outprefix=outprefix)
        st.afffns = Sys.glob(file.path(paste(outprefix, 
                                             "*", "[0-9]GenericAffine.mat", sep = "")))
        st = st.mo$warpedmovout
    }

    ## apply transform
    mask <- getMask(b0.ave, cleanup=2) * dti.mask
    dti.new <- splitNDImageToList(dti)
    ## apply mask
    if (maskornot) {
        dti.list <- lapply(dti.new, function(x) {
            x * mask
        })
        st.out <- st * mask
    } else {
        dti.list <- lapply(dti.new, function(x) {
            x 
        })
        st.out <- st
    }
    print('merging...')
    dti.new <- merge4D(dti.list, antsGetSpacing(dti)[4])
    antsImageWrite(mask, paste0(dirname(dirname(x[1])), "/brain_mask.nii.gz"))
    antsImageWrite(st.out, paste0(dirname(dirname(x[1])), "/struct/st.nii.gz"))
    antsImageWrite(dti.new, paste0(dirname(dirname(x[1])), "/dti/dti.nii.gz"))
}

registerDoMC(2)
## intermodal.coregister(fs.denoise[1,], maskornot=T, force=T)

## for (i in 1:nrow(fs.denoise)) {
## foreach(i = 1:nrow(fs.denoise)) %dopar% {
##     intermodal.coregister(fs.denoise[i,], maskornot=T, force=T)
##     gc()
##     }

## when scanning through the lung and breast, rigid masked with b0 and average rsbold failed try to avoid this for 18th and 37th subject
## intermodal.coregister(fs.denoise[grep("18", fs.denoise[,1]),], maskornot=F, force=T)
## intermodal.coregister(fs.denoise[grep("37", fs.denoise[,1]),], maskornot=F, force=T)

check.register <- function(x, ref, trans="rigid", force=F) {
    st <- paste0(dirname(dirname(x[1])), "/struct/st.nii.gz")
    dti <- paste0(dirname(dirname(x[1])), "/dti/dti.nii.gz")
    if ((force) || !file.exists(paste0(x[3], "_dti_", trans, "_ortho.gif"))) {
        dti.list <- splitNDImageToList(antsImageRead(dti))
        rs.list <- splitNDImageToList(antsImageRead(func))
        st <- antsImageRead(st)
        saveGIF({
            for (i in 1:15) {
                print(i)
                double_ortho(ants2oro(dti.list[[i]]), ants2oro(st))
            }
        },
        movie.name = paste0(x[3], "_dti_", trans, "_ortho.gif"),
        img.name = paste0(basename(dirname(dirname(x[3]))), "_dti_", basename(x[3]), "_rigid_ortho"))
    }
}

## check.register(as.vector(fs.denoise[1,]), t1, 'rigid')
registerDoMC(6)
## foreach(i = 1:nrow(fs.denoise)) %dopar% {
##     check.register(as.vector(fs.denoise[i,]), t1, 'rigid', force=T)
##     gc()
## }

## check.register(as.vector(fs.denoise[grep("18", fs.denoise[,1]),]), t1, 'rigid', force=T)
## check.register(as.vector(fs.denoise[grep("37", fs.denoise[,1]),]), t1, 'rigid', force=T)
## check.register(as.vector(fs.denoise[grep("36", fs.denoise[,1]),]), t1, 'rigid', force=T)

## shi <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-withCerebellum.nii")
## shi.skull <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-withSkull.nii")
## reg <- antsRegistration(st, shi, typeofTransform="SyN", outprefix=outprefix)
## prior <- imageFileNames2ImageList(c("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-seg-csf.nii",
##                                     "/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-seg-gm.nii",
##                                     "/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-seg-wm.nii"))

## white matter mask
## segs <- eigSeg(getMask(mask.neobrain12, cleanup=2), prior.prob)
## segs[segs!=3] =  0
## segs <- getMask(segs, cleanup=2)

target <- cbind(data.frame(id=gsub("subj", "", gsub("_\\d", "",basename(dirname(dirname(fs.denoise[,1])))))), fs.denoise)
meta <- read.csv("Table1.csv")[,c(1, 7)]
meta <- merge(meta, target, by.x="X", by.y="id")
ga.indexes <- apply(matrix(meta[,2]), 1, function(x) {return(which.min(abs(29:44-x)))})

## a.skull strip: improve the dwi2mask with fLABEL method
## b.motion correction of dti and generate FA
## c.n4 correction b0 and t1 with weight by UWM
## d.segmentation with age-specific template prior to three classes
## masked T1, dti, and rsbold
multimodal.seg <- function(x, ref, prior) {
    st <- paste0(dirname(dirname(x[1])), "/struct/st.nii.gz")
    dti <- paste0(dirname(dirname(x[1])), "/dti/dti.nii.gz")

    ## original dti and mask by registration
    if (!file.exists(paste0(dirname(dirname(x[1])), "/dti/original_b0_avg-strip.nii")))
    {
        dti.o <- x[3]
        b0.ave.o <- antsImageRead(dti.o)
        b0.ave.o <- getAverageOfTimeSeries(subset_4d(b0.ave.o, 1:3))
        antsImageWrite(b0.ave.o, paste0(dirname(dirname(x[1])), "/dti/original_b0_avg.nii"))
        system(paste0("cd ", paste0(dirname(dirname(x[1])), "/dti/"), "; fLABEL original_b0_avg.nii -p 2"))
    }

    if (!file.exists(paste0(dirname(dirname(x[1])), "/brainmask2.nii.gz"))) {
        mask <- getMask(antsImageRead(paste0(dirname(dirname(x[1])), "/dti/original_b0_avg-strip.nii")), cleanup=2)
        antsImageWrite(mask, paste0(dirname(dirname(x[1])), "/brainmask2.nii.gz"))
    } else {
        mask <- antsImageRead(paste0(dirname(dirname(x[1])), "/brainmask2.nii.gz"))
    }

    outprefix=paste0(st, "_neo12tob0")
    st <- antsImageRead(st)
    ## mask t1
    t1.t <- st * mask

    ## motion correction and mask dti
    if (!file.exists(paste0(dirname(dirname(x[1])), "/dti/dti_moco.nii.gz"))) {
        dti <- antsImageRead(dti)
        print(dti)
        ## mask the average b0
        b0.ave <- getAverageOfTimeSeries(subset_4d(dti, 1:3))*mask
        ## mask the 4D DT images
        dti.mat <- timeseries2matrix(dti, mask)
        dti <- matrix2timeseries(dti, mask, dti.mat)
        dti.mo <- .motion_correction(dti, b0.ave, moreaccurate=1, txtype="Rigid", verbose=T)
        b0.ave <- getAverageOfTimeSeries(subset_4d(dti, 1:3))
        antsImageWrite(dti.mo$moco_img, filename=paste0(dirname(dirname(x[1])), "/dti/dti_moco.nii.gz"))
    } else {
        dti.mo <- antsImageRead(paste0(dirname(dirname(x[1])), "/dti/dti_moco.nii.gz"))
        b0.ave <- getAverageOfTimeSeries(subset_4d(dti.mo, 1:3))
    }

    # mask b0 as T2 image
    t2.t <- b0.ave*mask

    if (!file.exists(paste0(dirname(dirname(x[1])), "/dti/dti_moco_camino_fa.nii.gz"))) {
        ## restore in dipy reject so many outliers use camino instead
        ## system(paste0("python dti.py -dwi ", paste0(dirname(dirname(x[1])), "/dti/dti_moco.nii.gz"), " -vec ", gsub("_denoised_bias.nii.gz", ".bvec", x[3]), " -val ", gsub("_denoised_bias.nii.gz", ".bval", x[3]), " -mask ", paste0(dirname(dirname(x[1])), "/brainmask2.nii.gz")," -name ", paste0(dirname(dirname(x[1])), "/dti/dti_moco_"))) ## get FA and MD for segmentation

        system(paste0("fsl2scheme -bvecfile ", gsub("_denoised_bias.nii.gz", ".bvec", x[3]), " -bvalfile ",  gsub("_denoised_bias.nii.gz", ".bval", x[3]), " > ", gsub("_denoised_bias.nii.gz", ".scheme", x[3]))) ## get scheme

        if (!file.exists(paste0(dirname(dirname(x[1])), "/dti/dti_moco.Bfloat"))) {
            system(paste0("image2voxel -4dimage ", paste0(dirname(dirname(x[1])), "/dti/dti_moco.nii.gz "), " -outputfile ", paste0(dirname(dirname(x[1])), "/dti/dti_moco.Bfloat "))) # convert to camino format
        }

        if (!file.exists(paste0(dirname(dirname(x[1])), "/dti/dti_moco.Bdouble"))) {
            system(paste0("restore ", paste0(dirname(dirname(x[1])), "/dti/dti_moco.Bfloat "),  gsub("_denoised_bias.nii.gz", ".scheme 50.0", x[3]), " > ", paste0(dirname(dirname(x[1])), "/dti/dti_moco.Bdouble")))  ## restore algorithm
        }
        system(paste0("for prog in fa md; do cat " , paste0(dirname(dirname(x[1])), "/dti/dti_moco.Bdouble"), " | ${prog} | voxel2image -outputroot ", paste0(dirname(dirname(x[1])), "/dti/dti_moco_camino_"), "${prog}", " -header ", paste0(dirname(dirname(x[1])), "/dti/dti_moco.nii.gz; done ")))
    }
 
    fa = antsImageRead(paste0(dirname(dirname(x[1])), "/dti/dti_moco_camino_fa.nii.gz"))
    md = antsImageRead(paste0(dirname(dirname(x[1])), "/dti/dti_moco_camino_md.nii.gz"))

    alltx = Sys.glob(file.path(paste0(outprefix, "*", "[0-9]*")))
    if (length(alltx) == 0) {
        b0.reg <- antsRegistration(t2.t, ref, typeofTransform="SyNAggro", outprefix=outprefix)
        alltx = Sys.glob(file.path(paste0(outprefix, "*", "[0-9]*")))
    }
    findinv = which(grepl("[0-9]InverseWarp.nii.gz", alltx))
    findfwd = which(grepl("[0-9]Warp.nii.gz", alltx))
    fwdtransforms = rev(alltx[-findinv])
    invtransforms = alltx[-findfwd]

    prior.list <- lapply(prior, function(x) {
        antsApplyTransforms(t2.t, x, transformlist = fwdtransforms)
    })

    t2.t <- n4BiasFieldCorrection(t2.t, weight_mask=prior.list[[3]])
    t1.t <- n4BiasFieldCorrection(t1.t, weight_mask=prior.list[[3]])

    img <- list(t2.t, iMath(t2.t, "Laplacian"), t1.t, iMath(t1.t, "Laplacian"), fa, iMath(fa, "Laplacian")) # iMath(t2.t, "Grad"), t1.t, iMath(t1.t, "Grad")) 
    segs3 <- atropos(a = img, m = '[0.1,1x1x1]', k="HistogramParzenWindows",
                     c = '[5,0]',  i = prior.list, x = mask, priorweight=0.8,
                     r = 1, w="BoxPlot")
    ##                     w="GrubbsRosner[0.05,0.1]")

    antsImageWrite(t2.t, paste0(dirname(dirname(x[1])), "/dti_b0_clean.nii.gz"))
    antsImageWrite(t1.t, paste0(dirname(dirname(x[1])), "/t1_clean.nii.gz"))
    antsImageWrite(segs3$segmentation, paste0(dirname(dirname(x[1])), "/segs.nii.gz"))
    for (y in 1:length(segs3$probabilityimages)) {
        antsImageWrite(segs3$probabilityimages[[y]], paste0(dirname(dirname(x[1])), "/segs_prob_", y, ".nii.gz"))
    }
}

## segmentation based on T1/B0/FA and T1/B0 Grad/Laplacian
## through atropos, possibly with kmeans, rfSegmentation and eigSeg

t2.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/template_T2.nii.gz"))
mask.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/brainmask.nii.gz"))
wm.mask.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/wm.nii.gz"))
cortex.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/cortex.nii.gz"))
csf.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/csf.nii.gz"))
dgm.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/dgm.nii.gz"))
cere.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/cerebellum.nii.gz"))
stem.neobrain12 <- splitNDImageToList(antsImageRead("/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlasing_of_the_developing_human_brain/Consistent_High_definition_Spatio-temporal_Neonatal_Brain_Atlas/brainstem.nii.gz"))
prior.prob <- list(csf.neobrain12, cortex.neobrain12, wm.mask.neobrain12, dgm.neobrain12, cere.neobrain12, stem.neobrain12)

registerDoMC(6)
foreach(i = 1:nrow(fs.denoise)) %dopar% {
   prior.prob.i <- lapply(prior.prob, function(x) {
       x[[ga.indexes[i]]]
   })
   t2.neobrain12.i <- t2.neobrain12[[ga.indexes[i]]]
   multimodal.seg(as.vector(fs.denoise[i, ]), t2.neobrain12.i, prior.prob.i)
   gc()
}

check.seg <- function(x) {
    ## prior.prob <- list(csf.neobrain12, cortex.neobrain12, wm.mask.neobrain12, dgm.neobrain12, cere.neobrain12, stem.neobrain12)

    t2.t = antsImageRead(paste0(dirname(dirname(x[1])), "/dti_b0_clean.nii.gz"))
    t1.t = antsImageRead(paste0(dirname(dirname(x[1])), "/t1_clean.nii.gz"))
    segm = antsImageRead(paste0(dirname(dirname(x[1])), "/segs.nii.gz"))

    fa = antsImageRead(paste0(dirname(dirname(x[1])), "/dti/dti_moco_camino_fa.nii.gz"))

    label.listout <- list()
    print(dim(segm))
    print(dim(segm))
    for (i in sort(unique(c(as.array(segm)))[2:7])){
        label.listout <- lappend(label.listout, segm==i)
    }

    outprefix=paste0(paste0(dirname(dirname(x[1])), 'segs_clean'))
    plot(t2.t, slices=seq(50, 140, by=5), axis=2, outname=paste0(outprefix, '_0.png'))
    plot(fa, slices=seq(50, 140, by=5), axis=2, outname=paste0(outprefix, '_1.png'))
    plot(t2.t, label.listout, slices=seq(50, 140, by=5), color.overlay=c(rgb(253,234,79,maxColorValue=255), rgb(69,127,217, maxColorValue=255), rgb(140,253,113, maxColorValue=255), rgb(252,148,210, maxColorValue=255), rgb(159,69,251, maxColorValue=255), rgb(56,251,191, maxColorValue=255)), alpha=0.85, axis=2, outname=paste0(outprefix, "_2.png"), colorbar=T, window.overlay=c(0.25,1))
    plot(t1.t, label.listout, slices=seq(50, 140, by=5), color.overlay=c(rgb(253,234,79,maxColorValue=255), rgb(69,127,217, maxColorValue=255), rgb(140,253,113, maxColorValue=255), rgb(252,148,210, maxColorValue=255), rgb(159,69,251, maxColorValue=255), rgb(56,251,191, maxColorValue=255)), alpha=0.85, axis=2, outname=paste0(outprefix, "_3.png"), colorbar=T, window.overlay=c(0.25,1))
    plot(fa, label.listout[[3]], slices=seq(50, 140, by=4), color.overlay=rgb(253,234,79,maxColorValue=255), alpha=0.85, axis=2, outname=paste0(outprefix, "_4.png"), colorbar=T, window.overlay=c(0.25,1))
    pdf(paste0(paste0(dirname(dirname(x[1])), 'segs_clean5.pdf')))
    double_ortho(ants2oro(fa), ants2oro(segm), col.y=c("red", "blue", "yellow", "green", "purple", "gray"))
    double_ortho(ants2oro(t2.t), ants2oro(segm), col.y=c("red", "blue", "yellow", "green", "purple", "gray"))
    dev.off()
}

#registerDoMC(32)
#foreach(i = 1:nrow(fs.denoise)) %dopar% {
#    check.seg(as.vector(fs.denoise[i, ]))
#    gc()
#}

## output white matters
## registerDoMC(32)
## foreach(i = 1:nrow(fs.denoise)) %dopar% {
##     segs <- antsImageRead(paste0(dirname(dirname(as.vector(fs.denoise[i, 1]))), "/segs.nii.gz"))
##     segs[segs > 3] = 2    
##     wmmask.out <- segs == 3
##     antsImageWrite(wmmask.out, paste0(dirname(dirname(as.vector(fs.denoise[i, 1]))), "/wmmask.nii.gz"))
##     gc()
## }

## normalize to JHU neonate template
multimodal.normalize <- function(x, ref, atlas) {
    b0 <- antsImageRead(paste0(dirname(dirname(x[1])), "/dti_b0_clean.nii.gz"))
    print(dim(b0))

    outprefix = paste0(x[3], "_b02jhu_template_")
    alltx = Sys.glob(file.path(paste0(outprefix, "*", "[0-9]*")))

    if (length(alltx) == 0) {
        antsRegistration(ref, b0, typeofTransform="SyNAggro", outprefix=outprefix)
        alltx = Sys.glob(file.path(paste0(outprefix, "*", "[0-9]*")))
    }

    findinv = which(grepl("[0-9]InverseWarp.nii.gz", alltx))
    findfwd = which(grepl("[0-9]Warp.nii.gz", alltx))
    fwdtransforms = rev(alltx[-findinv])
    invtransforms = alltx[-findfwd]
    print(fwdtransforms)
    print(invtransforms)

    print(invtransforms)
    ## warp atlas back to native space
    label.trans <- antsApplyTransforms(b0, atlas, transformlist=invtransforms, interpolator="genericLabel", whichtoinvert=c(T, F))
    antsImageWrite(label.trans, paste0(dirname(dirname(x[1])), "/jhu_native_labels.nii.gz"))
    pdf(paste0(paste0(dirname(dirname(x[1])), '/warp_labels.pdf')))
    double_ortho(ants2oro(b0), ants2oro(label.trans))
    dev.off()
}

## registerDoMC(10)
## foreach(i = 1:nrow(fs.denoise)) %dopar% {
##     multimodal.normalize(as.vector(fs.denoise[i,]), b0.template, labels)
##    gc()
## }

## Important notes for resting bold
## ?fMRINormalization
## ?preprocessRestingBOLD
## ?preprocessfMRI
## ?filterfMRIforNetworkAnalysis
## ?alffmap
## ?reho

get_fd <- function(x) {
    mp <- apply(x[,1:3]*50, 2, diff)
    mp = rbind(rep(0, 3), mp)
    mp <- abs(mp)
    return (rowMeans(mp))
}

## preprocess the rsbold
rsbold.normalize <- function(x) {
    print(x)
    if (!file.exists(paste0(dirname(dirname(x[1])), "/func/rs_clean_with_lowpass.nii.gz")) || !file.exists(paste0(dirname(dirname(x[1])), "/func/rs_clean_without_lowpass.nii.gz"))) {
        print("run...")
        mask <- antsImageRead(paste0(dirname(dirname(x[1])), "/brainmask2.nii.gz"))
        b0 <- antsImageRead(paste0(dirname(dirname(x[1])), "/dti_b0_clean.nii.gz"))
        segs <- antsImageRead(paste0(dirname(dirname(x[1])), "/segs.nii.gz"))
        segs[segs > 3] = 2
        rs <- antsImageRead(x[2])
        tr = antsGetSpacing(rs)[4]

        ##  SliceTimingCorrection : Outputs a slice-timing corrected 4D time series
        ## Usage        : SliceTimingCorrection image.nii.gz sliceTiming [sinc / bspline] [sincRadius=4 / bsplineOrder=3]
        if (!file.exists(paste0(gsub(".nii.gz", "", x[2]), "_slicetimer.nii.gz"))) {
            first.scan = floor(10.0 / tr) + 1  # 10 seconds "stabilization of signal"
            rs.clean1 <- subset_4d(rs, first.scan:ntim(rs))
            fsl_slicetimer(file=rs.clean1, outfile=paste0(gsub(".nii.gz", "", x[2]), "_slicetimer"), tr=tr, direction="z", acq_order="interleaved", indexing="up")
        }

        if (!file.exists(paste0(gsub(".nii.gz", "", x[2]), "_moco.nii.gz"))) {
            rs.clean3 = mcflirt(
                file = paste0(gsub(".nii.gz", "", x[2]), "_slicetimer.nii.gz"),
                outfile = paste0(gsub(".nii.gz", "", x[2]), "_moco.nii.gz"),
                verbose = 2,
                opts=paste0(" -meanvol -plots ")
            )
        } 
        rs <- antsImageRead(paste0(gsub(".nii.gz", "", x[2]), "_moco.nii.gz"))
        rs.ave <- getAverageOfTimeSeries(rs)
        outprefix=paste0(x[2], "_rs2b0")
        rs.afffns = Sys.glob(file.path(paste(outprefix, 
                                             "*", "[0-9]GenericAffine.mat", sep = "")))
        if (length(rs.afffns) == 0) {
            rs.mo <- antsRegistration(b0, rs.ave, typeofTransform='Rigid', verbose=T, outprefix=outprefix, mask=mask)
            rs.afffns = Sys.glob(file.path(paste(outprefix, 
                                                 "*", "[0-9]GenericAffine.mat", sep = "")))
        }

        if (!file.exists(paste0(dirname(dirname(x[1])), "/func/rs.nii.gz"))) {
            rslist <- splitNDImageToList(rs)
            rs.new <- lapply(rslist, function(x) {
                antsApplyTransforms(b0, x, transformlist=rs.afffns, interpolator="linear")
            })
            rs.list <- lapply(rs.new, function(x) {
                x * mask
            })
            rs.new <- merge4D(rs.list, tr)
            antsImageWrite(rs.new, paste0(dirname(dirname(x[1])), "/func/rs.nii.gz"))
        } else {
            rs.new <- antsImageRead(paste0(dirname(dirname(x[1])), "/func/rs.nii.gz"))
        }

        if (!file.exists(paste0(x[2], "_rs_rigid_ortho.gif"))) {
            rs.list <- splitNDImageToList(rs.new)
            ani.options(autobrowser=F)
            saveGIF({
                for (i in 1:15) {
                    print(i)
                    double_ortho(ants2oro(rs.list[[i]]), b0)
                }
            },
            movie.name = paste0(x[2], "_rs_rigid_ortho.gif"),
            img.name = paste0(basename(dirname(dirname(x[2]))), "_rs_", basename(x[2]), "_rigid_ortho"))
        }

        moco.params <- as.matrix(read.table(paste0(gsub(".nii.gz", "", x[2]), "_moco.par")))
        fd <- get_fd(moco.params)

        boldMat <- timeseries2matrix(rs.new, mask)
        goodtimes <- 1:nrow(boldMat)
        badtimes = which(fd >= 0.2)

        havebad = F
        if (length(badtimes) > 0) {
            goodtimes = goodtimes[-badtimes]
            havebad = T
        }

        if (havebad) {
            ## interpolate bad time points
            for (v in 1:ncol(boldMat)) {
                boldMat[badtimes, v] = spline(goodtimes,
                                              boldMat[goodtimes, v], method="natural", xout=badtimes)$y
            }
        }

        ## run and polynomial nuisance
        runNuis <- rep(1, dim(rs)[4])
        linearNuis <- 1:dim(rs)[4] # detrend across times
        
        timevals = as.numeric(1:sum(runNuis == 1)) * tr
        polyNuis <- stats::poly(timevals, degree=3)

        ## kmeans 3 cluster is 
        ## white matter is 3
        ## gray matter is 2
        ## csf is 1
        mask <- antsImageRead(paste0(dirname(dirname(x[1])), "/brainmask2.nii.gz"))
        wmMask <- antsImageClone(mask)
        wmMask[segs!=3] = 0
        wmMask[segs==3] = 1
        wmMask <- iMath(wmMask, "ME", 1)
        wmVox = which(subset(wmMask, mask > 0) == 1)

        wmMean = rowMeans(boldMat[, wmVox])

        csfMask = antsImageClone(mask)
        csfMask[csfMask != 1] = 0
        csfVox = which(subset(csfMask, mask > 0) == 1)

        csfMean = rowMeans(boldMat[, csfVox])
        globalMean = rowMeans(boldMat)
        
        ## tissue nuisance
        tissueNuis = cbind(globalMean, wmMean, csfMean)

        if (havebad) {
            for (v in c(1:dim(tissueNuis)[2])) {
                tissueInterp = spline(goodtimes, tissueNuis[goodtimes, v], method="natural", xout=badtimes)$y
                tissueNuis[badtimes, v] = tissueInterp
            }
        }

        tissueDeriv = rbind(rep(0, dim(tissueNuis)[2]), diff(tissueNuis, 1))

        mocoNuis = cbind(moco.params, moco.params * moco.params)
        mocoDeriv = rbind(rep(0, dim(mocoNuis)[2]), diff(mocoNuis, 1))

        compcorNuis = compcor(boldMat, 6)
        nuisance = cbind(runs=runNuis, tissueNuis, tissueDeriv, mocoNuis, mocoDeriv, fd = fd, poly=polyNuis, compcor = compcorNuis, dvars = computeDVARS(boldMat), linearTrend=linearNuis)
        write.table(nuisance, file=paste0(dirname(dirname(x[1])), "/func/nuisance.txt"), sep="\t", quote=F)

        ## mean and std scale
        nuisance = cbind(runNuis, scale(nuisance[,-1]))
        ## boldMat <- scale(boldMat) # cause NA sometimes possibly caused by voxels with no variance across time
        boldMat <- apply(boldMat, 2, function(y) (y - mean(y)) / sd(y) ^ as.logical(sd(y))) # trick follows https://stackoverflow.com/questions/15363610/why-does-scale-return-nan-for-zero-variance-columns

        print(sum(complete.cases(boldMat)))

        ## nuisance regressor
        boldMat <- residuals(lm(boldMat ~ nuisance))

        clean.bold1 <- matrix2timeseries(rs.new, mask, boldMat)
        antsImageWrite(clean.bold1, paste0(dirname(dirname(x[1])), "/func/rs_clean_without_lowpass.nii.gz")) ## this is for alff and falff

        boldMat <- frequencyFilterfMRI(boldMat, tr = tr, freqLo = 0.009,
                                       freqHi = 0.08, opt = "trig") # for functional connectomes and 
        clean.bold2 <- matrix2timeseries(rs.new, mask, boldMat)
        antsImageWrite(clean.bold2, paste0(dirname(dirname(x[1])), "/func/rs_clean_with_lowpass.nii.gz"))
    }
}

## rsbold.normalize(as.vector(fs.denoise[38,]))
registerDoMC(2)
foreach(i = 1:nrow(fs.denoise)) %dopar% {
    rsbold.normalize(as.vector(fs.denoise[i,]))
}

## rsbold.parameter <- function(x) {
##     mat <- matrix(rnorm(3000),ncol=50)
##     fallf = apply( mat, FUN=alffmap, MARGIN=2 )
##     k = kernel("daniell", rep( 2, 3 ) )
##     fallf2 = apply( mat, FUN=alffmap, MARGIN=2, kernel=k )
##     return(fallf2)
## }

## for demo purpose Figure 1,
## compute ALFF, fALFF, ReHo, FA, MD for Figure 1 in template space
## rs.network <- function(x) {
##     return()
## }
