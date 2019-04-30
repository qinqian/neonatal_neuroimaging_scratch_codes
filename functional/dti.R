library(ANTsR)
library(doMC)
library(foreach)
library(reshape2)
library(data.table)
registerDoMC(22)

label.template <- antsImageRead("../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz")
template <- antsImageRead("../../data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_fass.nii.gz")

fs <- Sys.glob("../data/fmri_nii/subj*/dti/*_fa.nii.gz")

fa.stat <- foreach(i=fs) %dopar% {
    img = antsImageRead(i)
    md = antsImageRead(gsub("fa", "md", i))
    rd = antsImageRead(gsub("fa", "rd", i))
    reg = antsRegistration(template, img, typeofTransform="SyNRA")
    x = reg$warpedmovout
    antsImageWrite(x, paste0(gsub("_fa.nii.gz", "", i), "_fa2jhufa.nii.gz"))
    reg.md = antsApplyTransforms(template, md, transformlist=reg$fwdtransforms, interpolator="bSpline")
    reg.rd = antsApplyTransforms(template, rd, transformlist=reg$fwdtransforms, interpolator="bSpline")
    antsImageWrite(reg.md, paste0(gsub("_fa.nii.gz", "", i), "_md2jhufa.nii.gz"))
    antsImageWrite(reg.rd, paste0(gsub("_fa.nii.gz", "", i), "_rd2jhufa.nii.gz"))
    label.mask = antsApplyTransforms(img, label.template, transformlist=reg$invtransforms, interpolator='genericLabel')
    antsImageWrite(label.mask, paste0(gsub("_fa.nii.gz", "", i), "_fa_labeled.nii.gz"))
    return(list(labelStats(img, label.mask), labelStats(md, label.mask), labelStats(rd, label.mask)))
}

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F)
mean.fa <- Reduce(cbind, lapply(fa.stat, function(x) {x[[1]]$Mean}))[-1,]
rownames(mean.fa) <- paste0(labels[,2], " ", labels[,3], " ", labels[,4])
mean.rd <- Reduce(cbind, lapply(fa.stat, function(x) {x[[3]]$Mean}))[-1,]
rownames(mean.rd) <- paste0(labels[,2], " ", labels[,3], " ", labels[,4])
mean.md <- Reduce(cbind, lapply(fa.stat, function(x) {x[[2]]$Mean}))[-1,]
rownames(mean.md) <- paste0(labels[,2], " ", labels[,3], " ", labels[,4])
pat <- read.csv("Table1.csv")
z <- function(x) {
    as.Date(paste0(substr(x,1, 4), "-", substr(x,5, 6), "-", substr(x,7, 8)))
    }

pat$cga <- as.numeric(pat$GA_week+((z(pat$MRI_checkdate)-z(pat$birthday))/7))

indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(dirname(fs))))))+1
ann <- pat[indexes,]

mut.g <- ann$mutation=='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
disease.g <- ann$mutation!='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
cont.g <- ann$Hyperbilirubinemia.HP.0002904.==0

pdf("UGT1A1_fa.pdf")
par(mfrow=c(2,2), font=2, xpd=T)
for (i in 1:nrow(mean.fa)) {
    plot(pat$cga[mut.g], mean.fa[i, mut.g], col=rgb(239,19,38,max=255), pch=16, xlim=c(33,48), ylim=c(min(mean.fa[i, ]), 0.01+max(mean.fa[i, ])), xlab='corrected gestational age', ylab="Mean Fractional Anisotropy", main=rownames(mean.fa)[i])
    points(pat$cga[disease.g], mean.fa[i, disease.g], col=rgb(28,161,95,max=255), pch=16)
    points(pat$cga[cont.g], mean.fa[i, cont.g], col=rgb(44,89,157,max=255), pch=16)
    legend(32.3, 1.05*max(mean.fa[i, ]),c("UGT1A1 mutation  ", "Hyperbilirubinemia  ", "control  "), col=c(rgb(239,19,38,max=255), rgb(28,161,95,max=255), rgb(44,89,157,max=255)), pch=c(16,16,16), ncol=3, bty="n", cex=0.6)
}
dev.off()

mut.vs.cont <- apply(mean.fa, 1, function(x) {t.test(x[mut.g], x[cont.g], alternative='less')$p.value})
disease.vs.cont <- apply(mean.fa, 1, function(x) {t.test(x[disease.g], x[cont.g],  alternative='less')$p.value})
mut.vs.disease <- apply(mean.fa, 1, function(x) {t.test(x[mut.g], x[disease.g], alternative='less')$p.value})
all.p <- cbind(mut.vs.cont, disease.vs.cont, mut.vs.disease)
write.table(all.p, file="all_p_fa_map.txt", sep="\t", quote=F)
write.table(all.p[all.p[,1]<0.05&all.p[,2]>0.05,], file="all_p_fa_map_threegroups_comparison.txt", sep="\t", quote=F)

groups <- rep("", ncol(mean.fa))
groups[mut.g] <- "Hyperbilirubinemia with UGT1A1"
groups[disease.g] <- "Hyperbilirubinemia without UGT1A1"
groups[cont.g] <- "Control"
gd <- as.data.frame(t(mean.fa))
rownames(gd) <- 1:nrow(gd)
aov.data <- cbind(gd, group=as.factor(groups))
write.table(aov.data, file="mean_fa_aov_data.xls", sep="\t", quote=F)

aov.p <- rep(0, nrow(mean.fa))
names(aov.p) <- rownames(mean.fa)

for (i in 1:(ncol(aov.data)-1)) {
    aov.obj <- aov(aov.data[,i]~aov.data$group)
    aov.p[i] <- summary(aov.obj)[[1]][,5][1]
}

all.p <- cbind(all.p, aov.p)
write.table(all.p, file="all_p_fa_map_threegroups_aov.xls", sep="\t", quote=F)
write.table(all.p[all.p[,1]<0.05 | all.p[,3]<0.05 | all.p[,2]<0.05,], file="all_p_fa_map_threegroups_aov_focus.xls", sep="\t", quote=F)


pdf("UGT1A1_md.pdf")
par(mfrow=c(2,2), font=2, xpd=T)
for (i in 1:nrow(mean.md)) {
    plot(pat$cga[mut.g], mean.md[i, mut.g], col=rgb(239,19,38,max=255), pch=16, xlim=c(33,48), ylim=c(min(mean.md[i, ]), max(mean.md[i, ])), xlab='corrected gestational age', ylab="Mean Diffusivity", main=rownames(mean.md)[i])
    points(pat$cga[disease.g], mean.md[i, disease.g], col=rgb(28,161,95,max=255), pch=16)
    points(pat$cga[cont.g], mean.md[i, cont.g], col=rgb(44,89,157,max=255), pch=16)
    legend(32.3, 1.08*max(mean.md[i, ]),c("UGT1A1 mutation  ", "Hyperbilirubinemia  ", "control  "), col=c(rgb(239,19,38,max=255), rgb(28,161,95,max=255), rgb(44,89,157,max=255)), pch=c(16,16,16), ncol=3, bty="n", cex=0.6)
}
dev.off()

library(glmnet)
## VBM
