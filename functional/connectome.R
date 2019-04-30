library(ANTsR)
library(igraph)
library(reshape2)

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F)
pat <- read.csv("Table1.csv")

edgewise.test <- function(mr.conf, mr.con, type) {
    indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(dirname(mr.conf))))))+1
    ann <- pat[indexes,]
    mut.g <- ann$mutation=='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
    disease.g <- ann$mutation!='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
    cont.g <- ann$Hyperbilirubinemia.HP.0002904.==0
    groups <- rep("", length(mr.con))
    groups[mut.g] <- "Hyperbilirubinemia with UGT1A1"
    groups[disease.g] <- "Hyperbilirubinemia without UGT1A1"
    groups[cont.g] <- "Control"
    groups <- factor(groups)
    dfs <- matrix(0, nrow=nrow(mr.con[[1]]), ncol=nrow(mr.con[[1]]))
    rownames(dfs) <- paste0(labels[,2], "-", labels[,3])
    colnames(dfs) <- paste0(labels[,2], "-", labels[,3])

    dfs.hb2cont.less <- matrix(0, nrow=nrow(mr.con[[1]]), ncol=nrow(mr.con[[1]]))
    rownames(dfs.hb2cont.less) <- paste0(labels[,2], "-", labels[,3])
    colnames(dfs.hb2cont.less) <- paste0(labels[,2], "-", labels[,3])

    dfs.mut2cont.less <- matrix(0, nrow=nrow(mr.con[[1]]), ncol=nrow(mr.con[[1]]))
    rownames(dfs.mut2cont.less) <- paste0(labels[,2], "-", labels[,3])
    colnames(dfs.mut2cont.less) <- paste0(labels[,2], "-", labels[,3])

    dfs.mut2hb.less <- matrix(0, nrow=nrow(mr.con[[1]]), ncol=nrow(mr.con[[1]]))
    rownames(dfs.mut2hb.less) <- paste0(labels[,2], "-", labels[,3])
    colnames(dfs.mut2hb.less) <- paste0(labels[,2], "-", labels[,3])
    for (i in 1:nrow(mr.con[[1]])) {
        for (j in 1:ncol(mr.con[[1]])) {
            con <- unlist(lapply(mr.con, function(x) {return(x[i, j])}))
            df <- data.frame(groups=groups, connectivity=con)
            df <- df[order(df[,1]), ]
            aov.p <- summary(aov(connectivity~groups, df))
            ## boxplot(df$connectivity~df$groups, main=paste0(rownames(dfs)[i], " ", rownames(dfs)[j]))
            dfs[i,j] = aov.p[[1]][,5][1]
            mat <- pairwise.t.test(con, groups, p.adjust.method='none', alternative="less")$p.value
            vec <- c(mat[1,1], mat[2,1], mat[2,2])
            dfs.hb2cont.less[i,j] = vec[1]
            dfs.mut2cont.less[i,j] = vec[2]
            dfs.mut2hb.less[i,j] = vec[3]
        }
    }
    ## dfs <- matrix(p.adjust(c(dfs), "BH"), nrow=nrow(mr.con[[1]]))
    ## dfs.copy2 = dfs
    ## dfs.copy2[is.na(dfs)] = 1
    ## dfs.1minusp = 1-dfs.copy2
    ## dfs.1minusp[lower.tri(dfs.1minusp)] = dfs.1minusp[upper.tri(dfs.1minusp)] ## this is wrong!!! because lower.tri and upper.tri differs in col and row order, use followings
    ## dfs.1minusp[lower.tri(dfs.1minusp)] = t(dfs.1minusp)[lower.tri(dfs.1minusp)]
    ## dfs.1minusp[dfs.1minusp<=0.99] = 0
    ## write.csv(dfs.1minusp, file=paste0('ugt1a1_', type, '_connectome_1minusp_value.csv'), quote=F)
    ## dfs.final <- melt(dfs.1minusp)
    ## test <- dfs.final[dfs.final[,3]>0,]
    ## test <- test[order(test[,3]), ]
    ## write.csv(test, file=paste0('ugt1a1_', type, '_connectome.csv'), quote=F)
    ## list(dfs, dfs.1minusp)
}

mr.conf <- Sys.glob("../data/fmri_nii/*/func/*_fmri_connectivity_basics_correlation.csv")
mr.con <- lapply(mr.conf, function(x) {
    y = read.csv(x)
    y = y[,-1]
    z = as.matrix(y)
    diag(z) = 0
    rownames(z) = 1:nrow(z)
    colnames(z) = 1:nrow(z)
    z
    })

##pdf("anova_bold_cor_boxplot.pdf", width=8, height=5)
bold.p <- edgewise.test(mr.conf, mr.con, "rsbold_correlation")
##dev.off()

mr.conf <- Sys.glob("../data/fmri_nii/*/dti/*connectome*")
mr.con <- lapply(mr.conf, function(x) { y=read.table(x); return(as.matrix(y)/sum(y)) })

## pdf("anova_dti_mrtrix_boxplot.pdf", width=8, height=5)
dti.p <- edgewise.test(mr.conf, mr.con, "mrtrix3_dti_correlation")
## dev.off()

mr.conf <- Sys.glob("../data/fmri_nii/*/dti/*connectivity*")
mr.con <- lapply(mr.conf, function(x) {
    y = read.csv(x)
    y = y[,-1]
    y = y[-1,]
    y = y[,-1]
    as.matrix(y)/sum(y)
    })

dti.p2 <- edgewise.test(mr.conf, mr.con, "dipy_dti_correlation")

con.f <-
    apply(matrix(fs), 1, function(x) {
        y <- read.table(x)
        print(dim(y))
        makeGraph(y, communityMethod = "spinglass", getEfficiency=T, inverseValuesAsWeights = T)
    })
