library(ANTsR)
library(doMC)
library(foreach)
library(reshape2)
library(data.table)
registerDoMC(22)

fs <- Sys.glob("/data1/qinqian/MRI/UGT1A1/data/fmri_nii/subj*/func/*_RSFC/run__fALFF+orig.nii")
atlas <- Sys.glob("/data1/qinqian/MRI/UGT1A1/data/fmri_nii/subj*/func/*_ants_warped_jhulabels.nii.gz")

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F)

fss <- imageFileNames2ImageList(fs)
atlass <- imageFileNames2ImageList(atlas)

stats <- list()
for (i in 1:length(fss)) {
    print(i)
    print(dim(fss[[i]]))
    print(dim(atlass[[i]]))
    a <- antsCopyImageInfo(atlass[[i]], fss[[i]])

    reg = antsRegistration(template, img, typeofTransform="SyNRA")
    x = reg$warpedmovout
    antsImageWrite(x, paste0(gsub("_fa.nii.gz", "", i), "_fa2jhufa.nii.gz"))
    reg.md = antsApplyTransforms(template, md, transformlist=reg$fwdtransforms, interpolator="bSpline")
    reg.rd = antsApplyTransforms(template, rd, transformlist=reg$fwdtransforms, interpolator="bSpline")
    antsImageWrite(reg.md, paste0(gsub("_fa.nii.gz", "", i), "_md2jhufa.nii.gz"))
    antsImageWrite(reg.rd, paste0(gsub("_fa.nii.gz", "", i), "_rd2jhufa.nii.gz"))
    label.mask = antsApplyTransforms(img, label.template, transformlist=reg$invtransforms, interpolator='genericLabel')

    stats[[i]] <- labelStats(a, atlass[[i]])
}

mean.stats <- Reduce(cbind, lapply(stats, function(x) {
    return(x$Mean)
    }))

pat <- read.csv("Table1.csv")

z <- function(x) {
    as.Date(paste0(substr(x,1, 4), "-", substr(x,5, 6), "-", substr(x,7, 8)))
    }

pat$cga <- as.numeric(pat$GA_week+((z(pat$MRI_checkdate)-z(pat$birthday))/7))

indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(dirname(dirname(fs)))))))+1
ann <- pat[indexes,]

mut.g <- ann$mutation=='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
disease.g <- ann$mutation!='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
cont.g <- ann$Hyperbilirubinemia.HP.0002904.==0

## pdf("UGT1A1_fa.pdf")
## par(mfrow=c(2,2), font=2, xpd=T)
## for (i in 1:nrow(mean.fa)) {
##     plot(pat$cga[mut.g], mean.fa[i, mut.g], col=rgb(239,19,38,max=255), pch=16, xlim=c(33,48), ylim=c(min(mean.fa[i, ]), 0.01+max(mean.fa[i, ])), xlab='corrected gestational age', ylab="Mean Fractional Anisotropy", main=rownames(mean.fa)[i])
##     points(pat$cga[disease.g], mean.fa[i, disease.g], col=rgb(28,161,95,max=255), pch=16)
##     points(pat$cga[cont.g], mean.fa[i, cont.g], col=rgb(44,89,157,max=255), pch=16)
##     legend(32.3, 1.05*max(mean.fa[i, ]),c("UGT1A1 mutation  ", "Hyperbilirubinemia  ", "control  "), col=c(rgb(239,19,38,max=255), rgb(28,161,95,max=255), rgb(44,89,157,max=255)), pch=c(16,16,16), ncol=3, bty="n", cex=0.6)
## }
## dev.off()

groups <- rep("", ncol(mean.stats))

groups[mut.g] <- "Hyperbilirubinemia with UGT1A1"
groups[disease.g] <- "Hyperbilirubinemia without UGT1A1"
groups[cont.g] <- "Control"

groups <- factor(groups, levels=c("Control", "Hyperbilirubinemia without UGT1A1", "Hyperbilirubinemia with UGT1A1"))

gd <- as.data.frame(t(mean.stats))
rownames(gd) <- 1:nrow(gd)
aov.data <- cbind(gd, group=as.factor(groups))

aov.dat <- aov.data[order(aov.data$group),]

write.table(aov.data, file="mean_falff_aov_data.xls", sep="\t", quote=F)

aov.p <- rep(0, nrow(mean.stats))
names(aov.p) <- rownames(mean.stats)

for (i in 1:(ncol(aov.data)-1)) {
    aov.obj <- aov(aov.data[,i]~aov.data$group)
    aov.p[i] <- summary(aov.obj)[[1]][,5][1]
}

t.p <- rep(0, nrow(mean.stats))
for (i in 1:(ncol(aov.data)-1)) {
    t.p[i] <- t.test(aov.data[aov.data$group=='Hyperbilirubinemia without UGT1A1',i], aov.data[aov.data$group=='Hyperbilirubinemia with UGT1A1',i], alternative='less')$p.value
}
