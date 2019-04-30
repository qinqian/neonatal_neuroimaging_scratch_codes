library(ANTsR)
library(extrantsr)
library(foreach)
library(doMC)
library(visreg)
registerDoMC(4)

consistent.T2.list <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-withSkull.nii")
label.list <- read.table("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/labels.txt", sep='\t')
labels <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-aal.nii")
template <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-withCerebellum.nii")
mask = getMask(template)

cortical <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-seg-gm.nii")
wm <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-seg-wm.nii")
csf <- antsImageRead("/data1/qinqian/MRI/data/Atlas/UNCInfant012Atlas_20140325/infant-neo-seg-csf.nii")
segs = eigSeg(mask, list(cortical, wm, csf))

gmask = thresholdImage(segs, 1, 1)
antsImageWrite(gmask, file="gray_matter_mask.nii.gz")

mut <- Sys.glob('../data/fmri_nii/mut/*jacob.nii.gz')
mut.status <- rep(0, length(mut))
mut.status[grep('del', mut)] <- 'DEL'
mut.status[grep('UDP', mut)] <- 'UDP'

gender <- rep(0, length(mut))
gender[grep('__F__', mut)] <- 'F'
gender[grep('__M__', mut)] <- 'M'

cga <- apply(matrix(mut), 1, function(x) {
    y = unlist(strsplit(gsub("_jacob.nii.gz", "", x), "__"))
    if (length(grep("del", x))>0 || length(grep("UDP", x))>0) {
        as.numeric(y[length(y)-1])
    } else {
        as.numeric(gsub("a", "", y[length(y)]))
        }
    })

demog <- data.frame(mutation=as.factor(mut.status), lm_gender=as.factor(gender), log_cga=cga)
demog$mutation = as.numeric(demog$mutation)
demog$lm_gender = as.numeric(demog$lm_gender)
o = order(demog$mutation, demog$lm_gender, demog$log_cga)
demog <- demog[o, ]
mut = mut[o]

write.table(demog, file="mut_comparison_pws_t2_meta.xls", quote=F, sep="\t")
write.table(mut, file='mut_files.xls', quote=F, sep="\t")

jimg <- imageFileNames2ImageList(mut)
jimgs <- lapply(jimg, function(x) {return(smoothImage(x, 1.5))})

g1 = demog$mutation == 1
g2 = demog$mutation == 2
fe = demog$lm_gender == 1
ma = demog$lm_gender == 2
jmat = imageListToMatrix(jimgs, gmask)
total.vol = rowSums(jmat)

pdf("cga_volume.pdf", width=8, height=6)
plot(demog$log_cga[g1 & fe], (total.vol/1000)[g1 & fe], xlim=c(35, 150), ylim=c(260, 300), pch=19, col="red", xlab="Corrected Gestational age", ylab="Total brain volume")
points(demog$log_cga[g1 & ma], (total.vol/1000)[g1 & ma], pch=19, col="blue", xlab="Corrected Gestational age", ylab="Total brain volume")
points(demog$log_cga[g2 & fe], (total.vol/1000)[g2 & fe], pch=21, col="red", xlab="Corrected Gestational age", ylab="Total brain volume")
points(demog$log_cga[g2 & ma], (total.vol/1000)[g2 & ma], pch=21, col="blue", xlab="Corrected Gestational age", ylab="Total brain volume")
legend("bottomright", pch=c(19, 19, 21, 21), col=c("red", "blue", "red", "blue"),
       c("DEL Female", "DEL Male", "UDP Female", "UDP Male"))
dev.off()

boxplot(list(DEL=total.vol[1:17], UDP=total.vol[18:32]))

rois = lapply(jimgs, function(x) {
  stat = labelStats(x, labels*gmask)
  return(sum(stat[,8]))
})
rois.mat = Reduce(rbind, rois)
t.test(total.vol[1:17], total.vol[18:32])

rois = lapply(jimgs, function(x) {
  stat = labelStats(x, labels*gmask)
  return(stat[-1,8])
})
rois.mat = Reduce(rbind, rois)

prois = apply(rois.mat, 2, function(x) {t.test(x[1:17], x[18:31])$p.value})
z=data.frame(label=as.character(label.list[,2]), p=as.numeric(prois))
z[,2] = as.numeric(z[,2])
z = z[order(z[,2]), ]
  
scale.jmat = scale(jmat)
rjmeans = rowMeans(scale.jmat)
t.test(rjmeans[g1], rjmeans[g2], paired=F)

rjmat = residuals(lm(scale.jmat~rjmeans))
demog = data.frame(scale(demog))
lm.o <- lm(rjmat ~ mutation+log_cga+lm_gender, data=demog)
lm.out <- bigLMStats(lm.o)

vol.p = lm.out$beta.pval
vol.mut = vol.p["mutation", ]
pimg = makeImage(gmask, vol.mut)

vol.t = lm.out$beta.t
t.mut = vol.t["mutation", ]
timg = makeImage(gmask, t.mut)

timg = timg * thresholdImage(pimg, 1e-5, 0.05)
ortho2(template, timg, col.y=colorRampPalette(c("blue", "white", "red"))(7), 
       ybreaks=seq(-3.9, 3, length.out = 8), mfrow=c(2,2))

timg.cluster <- labelClusters(abs(timg), 50, 1e-05, 4, T)
timg.c = timg*thresholdImage(timg.cluster, 1, Inf)

library(RColorBrewer)
pdf("genotype_contrast.pdf", width=12, height=6)
ortho2(cortical, timg.c, col.y=colorRampPalette(c("blue", "white", "red"))(30), 
       ybreaks=seq(-3.96, 3.5, length.out = 31), text.cex=2.5, crosshairs = F, 
       add.orient=F, 
       mfrow=c(1,3), ycolorbar=T)
dev.off()

upd.down <- labels * thresholdImage(timg.cluster, 1, Inf)
final = list()

for (i in sort(unique(c(as.array(upd.down))))[-1]) {
  print(i)
  tmp.p = pimg * thresholdImage(upd.down, i, i)
  tmp.p = tmp.p * thresholdImage(tmp.p, 1e-05, 0.05)
  final[[i]] = c(i, sum(thresholdImage(tmp.p, 1e-05, Inf)), min(tmp.p[tmp.p>1e-05]), as.character(label.list[i, 2]))
  #cat(i, length(tmp), max(tmp), as.character(label.list[i, 2]), "\n", sep="\t", file="UPD_DOWN.txt", append=T)
}

final = data.frame(Reduce(rbind, final))

final[,2] = as.numeric(as.character(final$X2))
final[,3] = as.numeric(as.character(final$X3))
final = subset(final, X2>50 & X3 < 0.01)
colnames(final) = c("ClusterID", "VoxelNumber", "MinimumPvalue", "ROI")
write.table(final, file="genotype_contrast.txt", sep="\t", quote=F, row.names=F)

