library(ANTsR)
library(igraph)
library(reshape2)
library(foreach)
library(doMC)

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F)
pat <- read.csv("Table1.csv")
rs.lowpass <- Sys.glob('../data/fmri_nii/sub*/functional_connectome_meanroi_jhu.cor')

mr.con <- apply(matrix(rs.lowpass), 1, function(x) {
    y = read.table(x)
    y
})

func.connectome.z <- lapply(mr.con, function(x) {
    antsrimpute(psych::fisherz(x))
})

indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(rs.lowpass)))))+1
ann <- pat[indexes,]

mut.g <- ann$mutation=='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
disease.g <- ann$mutation!='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
cont.g <- ann$Hyperbilirubinemia.HP.0002904.==0

groups <- rep("", 50)
groups[mut.g] <- "Hyperbilirubinemia with UGT1A1"
groups[disease.g] <- "Hyperbilirubinemia without UGT1A1"
groups[cont.g] <- "Control"

groups <- factor(groups)

connectome.weight <- lapply(func.connectome.z, function(x) {
    x[upper.tri(x)]
})

connectome.weight <- Reduce(rbind, connectome.weight)

print(ann)
features <- data.frame(groups=scale(as.numeric(groups)), gender=scale(as.numeric(as.factor(ann$gender))), bw=scale(ann$bw_g))

connectome.weight <- connectome.weight[order(features$groups), ]
features <- features[order(features$groups),]

lm.obj <- lm(connectome.weight~groups+gender+bw, data=features)
lm.obj <- bigLMStats(lm.obj)

groups.stat <- lm.obj$beta.t['groups', ]
out.pmat <- matrix(0, nrow=122, ncol=122)

out.pmat[upper.tri(out.pmat)] = groups.stat
out.pmat[lower.tri(out.pmat)] = t(out.pmat)[lower.tri(out.pmat)]

out.pmat[abs(out.pmat)<=3.09] = NA
diag(out.pmat) <- NA
print(out.pmat)
out.pmat[is.na(out.pmat)] = 0
#write.table(out.pmat, file="functional_connectome_pval.txt", sep="\t", quote=F)

library(RColorBrewer)
cols <- colorRampPalette(c("blue", "white", "red"))(20)

template<-antsImageRead("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz")

template.surf <-thresholdImage( template, 1, max(template) )
template.surf<-iMath(template.surf,"FillHoles")
cnt<-getCentroids( template, clustparam = 0 )
coord.cnt<-cnt[1:122,1:3]
## testweights<- read.table('/data1/qinqian/MRI/UGT1A1/data/fmri_nii/subj0_0/functional_connectome_meanroi_jhu.cor')

brain<-renderSurfaceFunction(surfimg =list( template.surf ) , 
                             alphasurf=0.28, smoothsval = 1 )

pos.mat <- out.pmat
pos.mat[out.pmat<0] = 0
plotBasicNetwork(centroids=coord.cnt, brain, weights=pos.mat, edgecolors = c("red"), nodecolors=rgb(0, 148, 243, maxColorValue=255), minRadius=2, lwd=2, radius=2.2, showOnlyConnectedNodes=T)
neg.mat <- out.pmat
neg.mat[neg.mat>=0] = 0
plotBasicNetwork(centroids=coord.cnt, brain, weights=abs(neg.mat), edgecolors = c("blue"), nodecolors=rgb(0, 148, 243, maxColorValue=255), minRadius=2, lwd=2, radius=2.2, showOnlyConnectedNodes=T)
id<-rgl::par3d('userMatrix')
rid<-rotate3d( id , -pi/2, 0, 0, -1 )
rid3<-rotate3d( id , -pi/2, 0, 0, 1 )
dd<-make3ViewPNG(  rid, id, rid3,  "functional_network_ugt1a1")
rgl.close()

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F)
pat <- read.csv("Table1.csv")
dti.con <- Sys.glob('../data/fmri_nii/sub*/dti/structure_connectome.csv')

mr.con <- lapply(dti.con, function(x) {
    y = read.table(x, header=F, row.names=)
    print(dim(y))
    y[lower.tri(y)] = t(y)[lower.tri(y)]
    as.matrix(y)/sum(y)
})

indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(rs.lowpass)))))+1
ann <- pat[indexes,]

mut.g <- ann$mutation=='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
disease.g <- ann$mutation!='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
cont.g <- ann$Hyperbilirubinemia.HP.0002904.==0

groups <- rep("", 50)
groups[mut.g] <- "Hyperbilirubinemia with UGT1A1"
groups[disease.g] <- "Hyperbilirubinemia without UGT1A1"
groups[cont.g] <- "Control"

groups <- factor(groups)

connectome.weight <- lapply(mr.con, function(x) {
    x[upper.tri(x)]
})

connectome.weight <- Reduce(rbind, connectome.weight)

features <- data.frame(groups=scale(as.numeric(groups)), gender=scale(as.numeric(as.factor(ann$gender))), bw=scale(ann$bw_g))

connectome.weight <- connectome.weight[order(features$groups), ]
features <- features[order(features$groups),]

lm.obj <- lm(connectome.weight~groups+gender+bw, data=features)
lm.obj <- bigLMStats(lm.obj)

groups.stat <- lm.obj$beta.t['groups', ]
out.pmat <- matrix(0, nrow=122, ncol=122)

out.pmat[upper.tri(out.pmat)] = groups.stat
out.pmat[lower.tri(out.pmat)] = t(out.pmat)[lower.tri(out.pmat)]

out.pmat[abs(out.pmat)<=3.09] = NA
diag(out.pmat) <- NA
print(out.pmat)

out.pmat[is.na(out.pmat)] = 0
#write.table(out.pmat, file="structural_connectome_pval.txt", sep="\t", quote=F)

library(RColorBrewer)
cols <- colorRampPalette(c("blue", "white", "red"))(20)

template<-antsImageRead("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz")

template.surf <-thresholdImage( template, 1, max(template) )
template.surf<-iMath(template.surf,"FillHoles")
cnt<-getCentroids( template, clustparam = 0 )
coord.cnt<-cnt[1:122,1:3]
## testweights<- read.table('/data1/qinqian/MRI/UGT1A1/data/fmri_nii/subj0_0/functional_connectome_meanroi_jhu.cor')

brain<-renderSurfaceFunction(surfimg =list( template.surf ) , 
                            alphasurf=0.28, smoothsval = 1 )

pos.mat <- out.pmat
pos.mat[out.pmat<0] = 0
plotBasicNetwork(centroids=coord.cnt, brain, weights=pos.mat, edgecolors = c("red"), nodecolors=rgb(0, 148, 243, maxColorValue=255), minRadius=2, lwd=2, radius=2.2, showOnlyConnectedNodes=T)
neg.mat <- out.pmat
neg.mat[neg.mat>=0] = 0
plotBasicNetwork(centroids=coord.cnt, brain, weights=abs(neg.mat), edgecolors = c("blue"), nodecolors=rgb(0, 148, 243, maxColorValue=255), minRadius=2, lwd=2, radius=2.2, showOnlyConnectedNodes=T)
id<-rgl::par3d('userMatrix')
rid<-rotate3d( id , -pi/2, 0, 0, -1 )
rid3<-rotate3d( id , -pi/2, 0, 0, 1 )
dd<-make3ViewPNG(  rid, id, rid3,  "structural_network_ugt1a1")
rgl.close()

