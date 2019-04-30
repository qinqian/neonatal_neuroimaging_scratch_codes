library(ANTsR)
library(igraph)
library(reshape2)
library(foreach)
library(doMC)

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F)
pat <- read.csv("Table1.csv")
rs.lowpass <- Sys.glob('../data/fmri_nii/sub*/func/rs_clean_with_*')

functional.connectome <- function(x) {
    if (!file.exists(paste0(dirname(dirname(x)), "/functional_connectome_meanroi_jhu.cor"))) {
        atlas <- antsImageRead(paste0(dirname(dirname(x)), "/jhu_native_labels.nii.gz"))
        mask <- antsImageRead(paste0(dirname(dirname(x)), "/brainmask2.nii.gz"))

        bold <- antsImageRead(x)
        bold <- smoothImage(bold, sigma=c(6,6,6,1), FWHM=T)
        boldMat <- timeseries2matrix(bold, mask)

        ## current.labels <- atlas[mask > 0] # match the bold matrix
        current.labels <- subset(atlas, mask > 0) # match the bold matrix
        ulabels <- sort(unique(current.labels))
        if (ulabels[1] == 0)
            ulabels = ulabels[-1]

        print(length(current.labels))
        print(length(ulabels))

        ## time points x roi matrix filled with mean values
        labmat <- matrix(data = rep(NA, length(ulabels) * nrow(boldMat)), nrow=nrow(boldMat))
        for (mylab in ulabels) {
            dd <- current.labels == mylab
            ## labmat[, mylab] = svd(boldMat[, dd], nu=1, nv=0)$u[,1]
            labmat[, mylab] = rowMeans(boldMat[, dd])
        }

        cor.connectome <- cor(labmat)
        write.table(cor.connectome, sep="\t", quote=F, file=paste0(dirname(dirname(x)), "/functional_connectome_meanroi_jhu.cor"))
        return(list(cor.connectome, corpcor::cor2pcor(cor.connectome)))
    } else {
        cor.connectome = read.table(paste0(dirname(dirname(x)), "/functional_connectome_meanroi_jhu.cor"))
        return(list(cor.connectome, corpcor::cor2pcor(cor.connectome)))
    }
}

## test.con <- functional.connectome(rs.lowpass[1])
registerDoMC(2)
func.connectome <- foreach(i = 1:length(rs.lowpass)) %dopar% {
    functional.connectome(rs.lowpass[i])
}

library(gplots)
gm.labels <- labels$V4!="wm"
xlab <- paste0(labels[,2][gm.labels], " ", labels[,3][gm.labels])

for (i in 1:length(rs.lowpass)) {
    print(i)
    pdf(paste0(dirname(rs.lowpass[i]), "/connectome_mat.pdf"), width=15, height=15)
    par(font=2, cex=0.8, mar=c(18, 18, 8, 8), xpd=T)
    image(1:89, 1:89, as.matrix(func.connectome[[i]][[1]][gm.labels,gm.labels]), col=bluered(20), xlab="", ylab="", axes=F, main="Pearson Correlation across time for 89 Cortex ROIs")
    axis(1, at=1:89, labels=xlab, las=2)
    axis(2, at=1:89, labels=xlab, las=1)
    image(1:89, 1:89, as.matrix(func.connectome[[i]][[2]][gm.labels,gm.labels]), col=bluered(20), xlab="", ylab="", axes=F, main="Partial Correlation across time for 89 Cortex ROIs")
    axis(1, at=1:89, labels=xlab, las=2)
    axis(2, at=1:89, labels=xlab, las=1)
    dev.off()
    }

func.connectome.z <- lapply(func.connectome, function(x) {
    list(antsrimpute(psych::fisherz(x[[1]])),
         antsrimpute(psych::fisherz(x[[2]])))
})

indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(dirname(rs.lowpass))))))+1
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
    x[[1]][upper.tri(x[[1]])]
})

## connectome.weight <- lapply(func.connectome, function(x) {
##     x[[1]][upper.tri(x[[1]])]
## })

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

library(RColorBrewer)
cols <- colorRampPalette(c("blue", "white", "red"))(20)

pdf("connectome_mat.pdf", width=15, height=15)
par(font=2, cex=0.8, mar=c(18, 18, 8, 8), xpd=T)
image(1:89, 1:89, out.pmat[gm.labels,gm.labels], col=cols, xlab="", ylab="", axes=F, main="P value across three groups patient for 89 Cortex ROIs")
axis(1, at=1:89, labels=xlab, las=2)
axis(2, at=1:89, labels=xlab, las=1)
dev.off()

## connectome.weight <- lapply(func.connectome.z, function(x) {
##     x[[2]][upper.tri(x[[2]])]
## })

dti.con <- Sys.glob('../data/fmri_nii/sub*/dti/structure_connectome.csv')

mr.con <- lapply(dti.con, function(x) {
    y = read.table(x, header=F, row.names=)
    print(dim(y))
    y[lower.tri(y)] = t(y)[lower.tri(y)]
    as.matrix(y)/sum(y)
})

struct.connectome.weight <- lapply(mr.con, function(x) {
    x[upper.tri(x)]
})

struct.connectome.weight <- Reduce(rbind, struct.connectome.weight)

features <- data.frame(groups=scale(as.numeric(groups)), gender=scale(as.numeric(as.factor(ann$gender))), bw=scale(ann$bw_g))

struct.connectome.weight <- struct.connectome.weight[order(features$groups), ]
features <- features[order(features$groups),]

lm.obj <- lm(struct.connectome.weight~groups+gender+bw, data=features)
lm.obj <- bigLMStats(lm.obj)

groups.stat <- lm.obj$beta.t['groups', ]
out.pmat <- matrix(0, nrow=122, ncol=122)
out.pmat[upper.tri(out.pmat)] = groups.stat
out.pmat[lower.tri(out.pmat)] = t(out.pmat)[lower.tri(out.pmat)]

out.pmat[abs(out.pmat)<=3.09] = NA
diag(out.pmat) <- NA

library(RColorBrewer)
cols <- colorRampPalette(c("blue", "white", "red"))(20)

xlab <- paste0(labels[,2], " ", labels[,3])

pdf("struct_connectome_mat.pdf", width=18, height=18)
par(font=2, cex=0.7, mar=c(18, 18, 8, 8), xpd=T)
image(1:122, 1:122, out.pmat, col=cols, xlab="", ylab="", axes=F, main="linear regression t value across three groups patient for 89 Cortex ROIs")
axis(1, at=1:122, labels=xlab, las=2)
axis(2, at=1:122, labels=xlab, las=1)
dev.off()


