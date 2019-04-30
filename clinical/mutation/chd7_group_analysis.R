#### orientation: http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
setwd("~/data1/MRI/Genetics/phaseB_get_yanglin_positive_ones/")
library(ANTsR)
library(extrantsr)
control = read.table('control_merge_info_file.txt', sep='\t', header=T, stringsAsFactors = F)
#imgList = imageFileNames2ImageList(control$jac)
cont.jac = gsub("Warp_jac.nii.gz", "Warp_logj.nii.gz", control$jac)
#cont.jac = gsub("Warp_jac.nii.gz", "absj.nii.gz", control$jac)
imgList = imageFileNames2ImageList(gsub('remodel_mxnet', '', cont.jac))
# /data1/qinqian/MRI/01_PVL_AI/remodel_mxnet/cleaned_data/control/applytranformtissue/cont_74_2646185/*3/*nii.gz

#cont.adc = imageFileNames2ImageList(control$adc)
#tissue = Sys.glob(paste0(dirname(gsub("transform", "applytranformtissue", cont.jac)), '/*3/*nii.gz'))
#tissue = imageFileNames2ImageList(tissue)

mutation = read.table('yanglin_final_after_exclude.txt', sep='\t', header=T, stringsAsFactors = F)
mutation$Disease.category[is.na(mutation$Disease.category)] = 'syndromic'
mutation$Gene[is.na(mutation$Gene)] = 'CHD7'

table(mutation$Disease.category)

#imgList2 = imageFileNames2ImageList(mutation$jac)
mut.jac = gsub("Warp_jac.nii.gz", "Warp_logj.nii.gz", mutation$jac)
imgList2 = imageFileNames2ImageList(mut.jac)

#adc.mut = imageFileNames2ImageList(mutation$adc)

template = antsImageRead('../phaseB_get_yanglin_positive_ones/yanglin_genetics/fudan_neonate_cpvl/extract_template/template_T2_roi.nii.gz')
mask = getMask(template)

# validate the computation
#brain.tissue = sapply(tissue, function(x) {
#  sum(x >= 1) * prod(antsGetSpacing(x))
#})
#brain.tissue = brain.tissue/1000

imgMat = imageListToMatrix(imgList, mask)
imgMat2 = imageListToMatrix(imgList2, mask)

#imgADC = imageListToMatrix(cont.adc, mask)
#mutADC = imageListToMatrix(adc.mut, mask)

s = rowSums(imgMat)
s2 = rowSums(imgMat2)
s2 = s2 * prod(antsGetSpacing(imgList[[1]]))/1000
s = s * prod(antsGetSpacing(imgList[[1]]))/1000

ccga = control$cga
selection = (!is.na(ccga)) & (ccga <=50)
#adc = rowSums(imgADC)[selection]
ccga = ccga[selection]
s = s[selection]

#adc.mut = rowSums(mutADC)

model.plot = function(x, y, x2, y2, name = "log of Relative Brain volume" ) {
  m = lm(y~poly(x,3))
  print(summary(m))
  confint(m, level=0.95)
  predicted.intervals <- predict(m, data.frame(x=x), interval='confidence',
                                 level=0.90)
  

  plot(x, y, #     xlim=c(35,50), 
       xlab="Corrected Gestational age", pch=19, col='blue',
       ylab=name)
  
  cols = c("red", "purple", "yellow", "grey")
  disease = as.factor(mutation$Disease.category)
  
  points(x2, y2, pch=10, col=cols[disease])
  points(x, predicted.intervals[, 1], col='green', pch=8)
  points(x, predicted.intervals[, 2], col='black', lwd=1, pch=3)
  points(x, predicted.intervals[, 3], col='black', lwd=1, pch=3)
  legend('topleft', col=cols, legend=levels(disease), pch=10, cex=0.8, ncol=2)  
}

#pdf("ADC_volume.pdf", width=9, height=16)
#par(cex=1, font=2, mfrow=c(2, 1), mar=c(4, 4, 2, 1))
#model.plot(ccga, s, mutation$cga, s2)
##model.plot(ccga, log2(adc), mutation$cga, log2(adc.mut), name = 'ADC')
#dev.off()
#cor(control$cga[selection], s, use='complete.obs')

atlas87 = antsImageRead('/home/qinqian/data1/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalStructuralAtlas/atlas-87/44.nii.gz')
region87 = read.table('/home/qinqian/data1/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalStructuralAtlas/atlas-87/regions.csv', sep='\t')
atlas50 = antsImageRead('/home/qinqian/data1/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalStructuralAtlas/atlas-50/44.nii.gz')
region50 = read.table('/home/qinqian/data1/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalStructuralAtlas/atlas-50/regions.csv', sep='\t')

atlas7 = antsImageRead('/home/qinqian/data1/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalStructuralAtlas/atlas-7/44.nii.gz')
region7 = read.table('/home/qinqian/data1/MRI/data/Atlas/Regionalgrowth_and_atlas_Consistent_High_definition_Spatio-tempora/NeonatalStructuralAtlas/atlas-7/regions.csv', sep='\t')

atlas = imageListToMatrix(list(atlas7, atlas50, atlas87), mask = mask)

#syndrome='syndromic'

vbm_roi = function(syndrome, gene= "", cutoff=0.001, exclude="", adjust=F) {
  if (gene == "") {
    select_type = mutation$Disease.category==syndrome
  } else {
    select_type = mutation$Disease.category==syndrome & mutation$Gene == gene
  }
  
  if (exclude == "") {
    select_type = select_type 
  } else {
    select_type = select_type & mutation$Gene != exclude
  }
  
  type_length = sum(select_type)  
  print(type_length)
  syn.vol = imgMat2[select_type,]
  mi = min(mutation[select_type, 'cga'])
  mx = max(mutation[select_type, 'cga'])
  cga = c(mutation$cga[select_type],
          control[(mi<=control$cga & control$cga <= mx), 'cga'])
  sex = as.numeric(as.factor(c(mutation$sex[select_type],
                               control[(mi<=control$cga & control$cga <= mx), 'sex'])))
  
  syn.con.vol = imgMat[(mi<=control$cga & control$cga <= mx), ]

  syn.vol.con = rbind(syn.vol, syn.con.vol)
  total = nrow(syn.vol.con)
  
  classes = c(rep(1, type_length), 
              rep(0, total-type_length))
  print(table(classes))
  syn.vol.con.atlas7 = matrix(0, nrow=total, ncol=7)
  for (i in 1:7) {
    print(length(rowSums(syn.vol.con[, atlas[1, ] == i])))
    print(dim(syn.vol.con.atlas7))
    syn.vol.con.atlas7[,i] = rowSums(syn.vol.con[, atlas[1, ] == i])
  }
  colnames(syn.vol.con.atlas7) = paste0('atlas7_', region7[,1], '_', region7[,2])
  syn.vol.con.atlas50 = matrix(0, nrow=total, ncol=50)
  for (i in 1:50) {
    syn.vol.con.atlas50[,i] = rowSums(syn.vol.con[, atlas[2, ] == i])
  }
  colnames(syn.vol.con.atlas50) = paste0('atlas50_', region50[,1], '_', region50[,2])
  
  syn.vol.con.atlas87 = matrix(0, nrow=total, ncol=87)
  for (i in 1:87) {
    syn.vol.con.atlas87[,i] = rowSums(syn.vol.con[, atlas[3, ] == i])
  }
  colnames(syn.vol.con.atlas87) = paste0('atlas87_', region87[,1], '_', region87[,2])
  #syn.vol.con.atlas = syn.vol.con.atlas87
  
  syn.vol.con.atlas = cbind(syn.vol.con.atlas7, syn.vol.con.atlas50, syn.vol.con.atlas87)
  
  syn.vols = syn.vol.con.atlas[, !duplicated(gsub('atlas\\d+_\\d+_', '', 
                                                  colnames(syn.vol.con.atlas)))]
  syn.df = as.data.frame(cbind(class=classes, 
                               cga=cga,
                               sex=sex))
  syn.vols = scale(syn.vols[complete.cases(syn.df), ])
  syn.df = data.frame(scale(syn.df[complete.cases(syn.df), ]))
  #m = glm(class ~ ., family=binomial(link='logit'), data=syn.df)
  #anova(m, test="Chisq")
  
  globalsig=scale(rowSums(syn.vols))
  
  r = residuals(lm(syn.vols~globalsig))
  m = lm(syn.vols~class+cga+sex, data=syn.df)
  ms = bigLMStats(m)
  
  final = cbind(ms$beta['class', order(ms$beta.pval['class', ])], sort(ms$beta.pval['class', ]), 0)
  if (adjust) {
    final[,3] = p.adjust(final[,2], 'BH')    
  }
  
  final = final[final[,2]<=cutoff, ]
  print(dim(final))
  #final = final[1:20, ]
  print(dim(final))
  print(summary(final[,2]))
  print(summary(final[,3]))

  final = final[grep('CSF|background', final[,1], invert=T), ]

  statmask = antsImageClone(mask)
  statmask[statmask==1] = 0
  for (i in rownames(final)) {
    z = unlist(strsplit(i, '_'))
    if (z[1]=='atlas87') {
      if (as.numeric(z[2])==84 || as.numeric(z[2])==83) 
        next
      #statmask[atlas87==as.numeric(z[2])] = -log10(final[i, 3])*sign(final[i, 1])
      statmask[atlas87==as.numeric(z[2])] = -log10(final[i, 2])*sign(final[i, 1])
    }  else if (z[1] == 'atlas50') {
      #statmask[atlas50==as.numeric(z[2])] = -log10(final[i, 3])*sign(final[i, 1])
      statmask[atlas50==as.numeric(z[2])] = -log10(final[i, 2])*sign(final[i, 1])
    } else {
      if (as.numeric(z[2])==1 || as.numeric(z[2])==4)
        next      
      #statmask[atlas7==as.numeric(z[2])]  = -log10(final[i, 3])*sign(final[i, 1])
      statmask[atlas7==as.numeric(z[2])]  = -log10(final[i, 2])*sign(final[i, 1])
    }
  }
  
  return(list(statmask, final))
}

chdmask = vbm_roi('syndromic', 'CHD7', cutoff=0.05, exclude="", adjust=T)
library(ggplot2)
pdf('chd7_significant.pdf')
seg = seq(-3.4, 1.5, 0.1)
cols = colorRampPalette(c("blue", "white", "red"))(length(seg)-1)
ortho2(template, chdmask[[1]], crosshairs = F, text.cex=1.5, add.orient=T, 
       col.y=cols, mar=c(1,1,1,1), ycolorbar=T, ybreaks=seg, text='CHD7 mutation')
dev.off()
write.table(chdmask[[2]], file='CHD7_significant.txt', sep='\t', quote=F)

#smask = vbm_roi('syndromic', gene= "", cutoff=1, exclude='CHD7', adjust = T)
#emask = vbm_roi('epileptic', cutoff=1, adjust=T)
#mmask = vbm_roi('metabolic', cutoff=1, adjust=T)
#
#pdf(paste0('syndromic','.pdf'))
#seg = seq(-7.8, -0.05, 0.1)
#cols = colorRampPalette(c("blue", "white"))(length(seg)-1)
#ortho2(template, smask[[1]], crosshairs = F, col.y=cols, mar=c(2,2,2,2), #mfrow=c(1,3),
#       ycolorbar=T, ybreaks=seg, text='syndromic')
#
#seg = seq(-2.2, 2, 0.1)
#cols = colorRampPalette(c("blue", "white", "red"))(length(seg)-1)
#ortho2(template, emask[[1]], crosshairs = F, col.y=cols,  mar=c(2,2,2,2), #mfrow=c(1,3),
#       ycolorbar=T, ybreaks=seg, text='epileptic')
#
#seg = seq(-1.3, 1.3, 0.1)
#cols = colorRampPalette(c("blue", "white", "red"))(length(seg)-1)
#ortho2(template, mmask[[1]], crosshairs = F, col.y=cols,  mar=c(2,2,2,2), #mfrow=c(1,3),
#       ycolorbar=T, ybreaks=seg, text='metabolic')
#dev.off()
#
#png('epileptic.png')
#plot(template, emask[[1]], axis=3, nslices=30, ncolumns=6,
#     window.overlay=seq(-2.2, 2, 0.1), colorbar=T)
#dev.off()
#
#png('syndromic.png')
#plot(template, smask[[1]], axis=3, nslices=60, ncolumns=6,
#     window.overlay=seq(-7.8, -0.05, 0.1), colorbar=T)
#dev.off()
#png('metabolic.png')
#plot(template, mmask[[1]], axis=3, nslices=60, ncolumns=6,
#     window.overlay= seq(-1.3, 1.3, 0.1), colorbar=T)
#dev.off()
#
#write.table(smask[[2]], file='syndromic_significant_five.txt', sep='\t', quote=F)
#write.table(emask[[2]], file='epileptic_significant_five.txt', sep='\t', quote=F)
#write.table(mmask[[2]], file='metabolic_significant_five.txt', sep='\t', quote=F)
### conclusion
##plot(template, statmask, axis=3, nslices=100)
