library(ANTsR)
library(rgl)
library(plot3D)
library(OceanView)
library(misc3d)
library(plot3Drgl)

#setwd("~/Desktop/douma/")
tbss.f1 <- antsImageRead("anova/anova_tbss_tfce_corrp_filled0.05.nii.gz")
vbm1 <- antsImageRead("anova/anova_vbm_thr.nii.gz")
tbss1 <- antsImageRead("anova/anova_tbss_tfce_corrp_index0.05.nii.gz")

tbss.f2 <- antsImageRead("cs/cs_tbss_tfce_corrp_filled0.05.nii.gz")
vbm2 <- antsImageRead("cs/cs_vbm_thr.nii.gz")
tbss2 <- antsImageRead("cs/cs_tbss_tfce_corrp_index_cs0.05.nii.gz")

tbss.f3 <- antsImageRead("as/as_tbss_tfce_corrp_filled0.05.nii.gz")
vbm3 <- antsImageRead("as/as_vbm_thr.nii.gz")
tbss3 <- antsImageRead("as/as_tbss_tfce_corrp_index_as0.05.nii.gz")

template <- antsImageRead("Template_6.nii")
gm <- splitNDImageToList(template)[[1]]
#wm <- splitNDImageToList(template)[[2]]

gm <- getMask(gm, cleanup=1)
#wm <- getMask(wm, cleanup=2)

#mask = antsImageRead('fullbrain_mask.nii')

###gm.coord = imageDomainToSpatialMatrix(gm)
###tbss.f1.coords = imageDomainToSpatialMatrix(tbss.f1>0)

renderSurface <- function (surfimg, funcimg, surfval = 0.5, basefval, offsetfval,
                          smoothsval = 0, smoothfval = 0, blobrender = TRUE, alphasurf = 1,
                          alphafunc = 1, outdir = "./", outfn = NA, mycol, physical = TRUE,
                          movieDuration = 6, zoom = 1.1)
{
    if (missing(surfimg)) {
        stop("Check usage:  at minimum, you need to call \n renderSurfaceFunction( list(an_ants_image) ) \n ")
    }
    havemsc3d <- usePkg("misc3d")
    if (!havemsc3d) {
        print("Need misc3d for this")
        return(NA)
    }
    smoothsval <- rep(smoothsval, length.out = length(surfimg))
    for (i in 1:length(surfimg)) {
        if (smoothsval[i] > 0) {
            simg <- antsImageClone(surfimg[[i]])
            simg <- smoothImage(simg, smoothsval[i])
            surfimg[[i]] <- simg
        }
    }
    surfval <- rep(surfval, length.out = length(surfimg))
    if (length(alphasurf) != length(surfimg))
        alphasurf <- rep(alphasurf, length.out = length(surfimg))
    mylist <- list()
    if (missing(funcimg)) {
        for (i in 1:length(surfimg)) {
            surf <- as.array(surfimg[[i]])
            brain <- contour3d(surf, level = c(surfval[i]),
                alpha = alphasurf[i], draw = FALSE, smooth = T,
                material = "metal", depth = 0.6, color = "white")
            if (physical == TRUE) {
                brain$v1 <- antsTransformIndexToPhysicalPoint(surfimg[[i]],
                  brain$v1)
                brain$v2 <- antsTransformIndexToPhysicalPoint(surfimg[[i]],
                  brain$v2)
                brain$v3 <- antsTransformIndexToPhysicalPoint(surfimg[[i]],
                  brain$v3)
            }
            mylist[[i]] <- brain
        }
        misc3d::drawScene.rgl(mylist)
        return(mylist)
    }
    if (smoothfval > 0) {
        for (i in 1:length(funcimg)) {
            fimg <- antsImageClone(funcimg[[i]])
            fimg <- smoothImage(fimg, smoothfval)
            funcimg[[i]] <- fimg
        }
    }
    if (missing(mycol)) {
        mycol <- rainbow(length(funcimg))
    }
    if (length(alphafunc) != length(funcimg))
        alphafunc <- rep(alphafunc, length.out = length(funcimg))
    for (i in 1:length(surfimg)) {
        surf <- as.array(surfimg[[i]])
        brain <- misc3d::contour3d(surf, level = c(surfval[i]),
            alpha = alphasurf[i], draw = FALSE, smooth = F,
            material = "metal", depth = 0.6, color = "white")
        if (physical == TRUE) {
            brain$v1 <- antsTransformIndexToPhysicalPoint(surfimg[[i]],
                brain$v1)
            brain$v2 <- antsTransformIndexToPhysicalPoint(surfimg[[i]],
                brain$v2)
            brain$v3 <- antsTransformIndexToPhysicalPoint(surfimg[[i]],
                brain$v3)
        }
        mylist[[i]] <- brain
    }
    for (i in 1:length(funcimg)) {
        func <- as.array(funcimg[[i]])
        vals <- abs(funcimg[[i]][funcimg[[i]] > 0])
        if (missing(basefval)) {
            usefval <- mean(vals)
        }
        else usefval <- basefval
      
        if (missing(offsetfval))
            offsetfval <- sd(vals[vals > usefval])
        
        print(usefval, offsetfval)
        blob <- misc3d::contour3d(func, level = c(usefval), alpha = alphafunc[i],
            draw = FALSE, smooth = F, material = 'metal',
            depth = c(0.6,0.6)[i], color = mycol[[i]])
        if (physical == TRUE) {
            blob$v1 <- antsTransformIndexToPhysicalPoint(funcimg[[i]],
                blob$v1)
            blob$v2 <- antsTransformIndexToPhysicalPoint(funcimg[[i]],
                blob$v2)
            blob$v3 <- antsTransformIndexToPhysicalPoint(funcimg[[i]],
                blob$v3)
        }
        mylist <- lappend(mylist, list(blob))
    }
    misc3d::drawScene.rgl(mylist)
    rgl::par3d(windowRect = c(0, 0, 500, 500))
    rgl::par3d(zoom = zoom)
    misc3d::drawScene.rgl(mylist)
    if (!is.na(outfn))
        rgl::movie3d(rgl::spin3d(), duration = movieDuration,
            dir = outdir, movie = outfn, clean = T)
    return(mylist)
}


make4view = function (rotationView1, rotationView2, rotationView3, rotationView4, fnprefix) 
  {
    if (!usePkg("rgl") | !usePkg("grid") | !usePkg("png")) {
      print("Need rgl, grid and png")
      return(NULL)
    }
    if (nargs() == 0 | missing(fnprefix)) {
      print("Usage:  make3ViewPNG( x , fn = 'output.png') ")
      return(1)
    }
    id <- rgl::par3d("userMatrix")
    if (missing(rotationView1)) {
      rotationView1 <- rgl::rotate3d(id, -pi/2, 1, 0, 0)
    }
    if (missing(rotationView2)) {
      rotationView2 <- rgl::rotate3d(id, pi/2, 0, 0, 1)
    }
    if (missing(rotationView3)) {
      rotationView3 <- rgl::rotate3d(id, -pi/2, 0, 0, 1)
    }
    if (missing(rotationView4)) {
      rotationView3 <- rgl::rotate3d(id, pi/2, 0, 0, 1)
    }
    
    rgl::par3d(userMatrix = rotationView1)
    rgl::rgl.snapshot(paste(fnprefix, "a.png", sep = ""), fmt = "png", 
                      top = TRUE)
    rgl::par3d(userMatrix = rotationView2)
    rgl::rgl.snapshot(paste(fnprefix, "b.png", sep = ""), fmt = "png", 
                      top = TRUE)
    rgl::par3d(userMatrix = rotationView3)
    rgl::rgl.snapshot(paste(fnprefix, "c.png", sep = ""), fmt = "png", 
                      top = TRUE)
    rgl::par3d(userMatrix = rotationView4)
    rgl::rgl.snapshot(paste(fnprefix, "d.png", sep = ""), fmt = "png", 
                      top = TRUE)    
    
    aa <- png::readPNG(paste(fnprefix, "a.png", sep = ""))
    bb <- png::readPNG(paste(fnprefix, "b.png", sep = ""))
    cc <- png::readPNG(paste(fnprefix, "c.png", sep = ""))
    dd <- png::readPNG(paste(fnprefix, "d.png", sep = ""))
    
    aabbcc <- abind::abind(aa, bb, along = 2)
    aabbcc <- abind::abind(aabbcc, cc, along = 2)
    aabbccdd <- abind::abind(aabbcc, dd, along = 2)    
    
    png(paste(fnprefix, ".png", sep = ""), width = dim(aabbccdd)[2], 
        height = dim(aabbccdd)[1])
    grid::grid.raster(aabbccdd)
    dev.off()
    return(aabbccdd)
  }  

overlay.3d = function(vbm, tbss, name) {
    brain <- renderSurface(surfimg=list(gm),surfval=0.5,  blobrender=T, movieDuration = 12,
                                  funcimg=list(vbm, tbss),
                                  mycol=c("red", "blue"),
                                  alphasurf=0.15, alphafunc=c(0.25, 0.25), smoothsval = 2, smoothfval= 2, outfn=name)
    #writeSTL("test.STL", ascii=F, pointRadius=0.005, pointShape=icosahedron3d(), lineRadius=0.005, lineSides = 20, ids=NULL)
    #id<-rgl::par3d('userMatrix')
    #rid<-rotate3d( id , -pi/2, 0, 0, -1 )
    #rid3<-rotate3d( id , -pi/2, 0, 0, 1 )
    #dd<-make3ViewPNG( id, rotationMatrix(-pi/2, 1, 0, 0), rotationMatrix(, 1, 0, 0),  "tbss_fill_anova_1")
    #dd<-make3ViewPNG( id, rotate3d(id, pi/2, 1, 0, 0), rotate3d(id,pi/2, 0, 1, 0), 'tbss_anova')
    id<-rgl::par3d('userMatrix')
    rid<-rotate3d( id , -pi/2, 1, 0, 0 )
    rid3<-rotate3d( id , -pi/2, 0, 0, 1 )
    rid4<-rotate3d( id , pi/2, 0, 0, 1 )
    
    dd<-make4view( rid, id, rid3, rid4, paste0('tbss_anova_', name))
    rgl.close()
}

#plot(wm, list(vbm1, tbss1))
#
#overlay.3d(vbm1, tbss1, '1')
#
#overlay.3d(vbm1, tbss.f1, '1fill')
#
#overlay.3d(vbm2, tbss2, '2')
#overlay.3d(vbm2, tbss.f2, '2fill')
#
overlay.3d(vbm3, tbss3, '3')
overlay.3d(vbm3, tbss.f3, '3fill')


##  background colors 
##  bg3d(sphere = TRUE, color = c("black", "green"), lit = FALSE, back = "lines" )
##
##  # an environmental sphere with a nice texture.
##
##  bg3d(sphere = TRUE, texture = system.file("textures/sunsleep.png", package = "rgl"), 
##         back = "filled" )
