library(ANTsR)
args = commandArgs(T)
id = args[1]
cga = args[2]
images = args[3:6]
images = imageFileNames2ImageList(paste0('results/processed/', basename(images)))
bold   = antsImageRead(paste0('results/processed/', basename(args[7])))

index = ceiling(as.double(cga) - 28)
print(index)
if (index >= 17) {
    index = 17
}
if (index <= 0) {
    index = 1
}

load_template <- function(index) {
    prefix = '/data1/qinqian/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalAtlas2'
    setwd(prefix)
    t2.neobrain12 <- splitNDImageToList(antsImageRead("template_T2.nii.gz"))
    mask.neobrain12 <- splitNDImageToList(antsImageRead("brainmask.nii.gz"))
    wm.mask.neobrain12 <- splitNDImageToList(antsImageRead("wm.nii.gz"))
    cortex.neobrain12 <- splitNDImageToList(antsImageRead("cortex.nii.gz"))
    csf.neobrain12 <- splitNDImageToList(antsImageRead("csf.nii.gz"))
    dgm.neobrain12 <- splitNDImageToList(antsImageRead("dgm.nii.gz"))
    cere.neobrain12 <- splitNDImageToList(antsImageRead("cerebellum.nii.gz"))
    stem.neobrain12 <- splitNDImageToList(antsImageRead("brainstem.nii.gz"))
    prior.prob <- list(csf.neobrain12, cortex.neobrain12, wm.mask.neobrain12, dgm.neobrain12, cere.neobrain12, stem.neobrain12)
    prior.prob.i <- lapply(prior.prob, function(x) {
        x[[index]]
    })
    list(t2.neobrain12[[index]], prior.prob.i)
}
 
prior <- load_template(index)

## load a subject
#setwd('/cluster/home/qinqian/mri/hyperbilirubinemia')
setwd('/data1/qinqian/MRI/hyperbilirubinemia/phaseC_rsBOLD_filtered_analysis')

segmentation <- function(id) {
    print(images)
    fa = images[[1]]
    md = images[[2]]
    b0 = images[[3]]
    #t1 = images[[4]]  # skull stripping is too bad to continue

    print(lapply(images, dim))
    t2.template = prior[[1]]
    prior.roi   = prior[[2]]
    print(dim(t2.template))
    #t1.reg      = antsRegistration(b0, t1, type_of_registration='Rigid', outputPrefix=paste0("results/", id, '_t1'))
    #t1.reg      = antsRegistration(b0, t1.reg[['warpedmovout']], type_of_registration='QuickRigid', outputPrefix=paste0("results/", id, '_t1'))

    template.reg      = antsRegistration(b0, t2.template, type_of_registration='SyNAggro', outputPrefix=paste0("results/", id, '_template'))
    prior.roi = lapply(prior.roi, function(x) {
        antsApplyTransforms(b0, x, transformlist = template.reg[['fwdtransforms']])
    })
    #segs.data = list(t1.reg[['warpedmovout']], iMath(t1.reg[['warpedmovout']], "Grad"), fa)
    b0 = n4BiasFieldCorrection(b0, weight_mask=thresholdImage(prior.roi[[3]], 0.2, 1))
    segs.data = list(b0, iMath(b0, "Grad"), fa)
    print(segs.data)
    mask = getMask(b0, cleanup=2)
    segs = atropos(a = segs.data,
                   k="HistogramParzenWindows", c='[3,0]',m='[0.2, 1x1x1]',
                   i=prior.roi, x = mask,
                   priorweight=0.8, r=1, w="BoxPlot")
    hard.seg = eigSeg(mask, segs$probabilityimages)

    png(paste0('results/', id, '_fa.png'), width=800, height=800)
    plot(b0, thresholdImage(prior.roi[[3]], 0.2, 1) * thresholdImage(fa, 0.09, 1), alpha=0.5, 
         ncolumns=5,
         figsize=4, nslices=30, colorbar=T, window.overlay=c(0,6), axis=2)
    dev.off()

    png(paste0('results/', id, '_wm.png'), width=800, height=800)
    plot(b0, iMath(thresholdImage(hard.seg, 3, 3), 'ME', 2), alpha=0.5, 
         ncolumns=5,
         figsize=4, nslices=30, colorbar=T, window.overlay=c(0,6), axis=2)
    dev.off()

    png(paste0('results/', id, '_csf.png'), width=800, height=800)
    #plot(t1.reg[['warpedmovout']], iMath(thresholdImage(hard.seg, 1, 1), 'ME', 2), alpha=0.5, 
    plot(b0, iMath(thresholdImage(hard.seg, 1, 1), 'ME', 2), alpha=0.5, 
         ncolumns=5,
         figsize=4, nslices=30, colorbar=T, window.overlay=c(0,6), axis=2)
    dev.off()

    antsImageWrite(hard.seg, file=paste0('results/', id, '_segmentation.nii.gz'))
    for (i in seq(length(segs$probabilityimages))) {
        antsImageWrite(segs$probabilityimages[[i]], file=paste0('results/', id, '_segprob', i, '.nii.gz'))
    }
}

segmentation(id)

