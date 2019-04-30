setwd("/home/qinqian/data1/MRI/hyperbilirubinemia/scripts/")
library(ANTsR)
library(extrantsr)
bayley.metric <- c('Cognitive_12m', 'Fine Motor_12m', 'Gross Motor_12m', 'Receptive Communication_12m', 'Expressive Communication_12m', 'Cognitive_18m', 'Fine Motor_18m', 'Gross Motor_18m', 'Receptive Communication_18m', 'Expressive Communication_18m')

a <- read.table("info_T1_DTI.csv", sep='\t', header=T)
colnames(a)[27:36] = bayley.metric

a <- a[a$participant_id!=159 & a$participant_id!=156 & a$DTI_quality==1, ]
ids <- a$participant_id

mds <- Sys.glob(paste0('/home/qinqian/data1/MRI/hyperbilirubinemia/output/', ids, '/dti_results/subject_id*/*1/*nii.gz'))
fas <- Sys.glob(paste0('/home/qinqian/data1/MRI/hyperbilirubinemia/output/', ids, '/dti_results/subject_id*/*0/*nii.gz'))
fwdwarp <- Sys.glob(paste0('/home/qinqian/data1/MRI/hyperbilirubinemia/output/', ids, '/fwd_b0/subject_id*/*nii.gz'))
template <- antsImageRead('/home/qinqian/data1/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_b0ss.nii.gz')
## for (i in seq_along(fwdwarp)) {
##     r <- antsImageRead(fwdwarp[i])
##     z <- createJacobianDeterminantImage(template, r, doLog=T, geom=T)
##     antsImageWrite(z, paste0(ids[i], '_jac.nii.gz'))
## }
reljac <- Sys.glob(paste0(ids, '_jac.nii.gz'))
atlas <-  antsImageRead('/home/qinqian/data1/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz')

reljac.l = imageFileNames2ImageList(reljac)
reljac.stat = lapply(reljac.l, function(x) {labelStats(x, atlas)})

fa.l = imageFileNames2ImageList(fas)
fa.stat = lapply(fa.l, function(x) {labelStats(x, atlas)})

md.l = imageFileNames2ImageList(mds)
md.stat = lapply(md.l, function(x) {labelStats(x, atlas)})

library(glmnet)

get.mean.var <- function(x) {
    y <- lapply(x, function(x) {
        return(c(x[, 2], x[, 5]))
    })
    return(t(Reduce(cbind, y)))
}

fa <- get.mean.var(fa.stat)
colnames(fa) <- c(paste0(seq(123), '.meanfa.'), paste0(seq(123), '.varfa.'))

md <- get.mean.var(md.stat)
colnames(md) <- c(paste0(seq(123), '.meanmd.'), paste0(seq(123), '.varmd.'))

jac <- get.mean.var(reljac.stat)
colnames(jac) <- c(paste0(seq(123), '.meanjac.'), paste0(seq(123), '.varjac.'))

feature <- cbind(fa, md, jac)
rownames(feature) <- ids

set.seed(999)

clinical <- a[,c(3, 4, 19, 37)] # 22, 
grid <- 10^seq(1, -5, length=200)

x <- cbind(feature, clinical)
x$sex = as.numeric(x$sex)
x$bw = as.numeric(sub('g', '', as.vector(x$bw)))

## apgar = as.vector(x$apgar1)
## apgar <- as.numeric(apgar)
## apgar[97] = 7

x <- as.matrix(x)
y <- scale(as.matrix(a[,27:36]))

z.cor <- t(cor(x, y, method='spearman'))
z.cor2 <- t(cor(x, y))

out <- matrix(0, ncol=ncol(x), nrow= ncol(y))
out2 <- matrix(0, ncol=ncol(x), nrow= ncol(y))

for (i in 1:ncol(x)) {
    for (j in 1:ncol(y)) {
        z = cor.test(x[,i], y[,j])
        out[j, i] = z$estimate
        out2[j, i] = z$p.value
    }
}

out2 <- p.adjust(c(out2), 'BH')
out2 <- matrix(out2, nrow=ncol(y))

write.table(cbind(x,y), 'fa_md_volume_with_bayley.xls', sep='\t', quote=F)
write.table(z.cor, 'spearman_correlation_between_roi_and_bayley.xls', sep='\t', quote=F)
write.table(z.cor2, 'correlation_between_roi_and_bayley.xls', sep='\t', quote=F)
write.table(out2, 'correlation_bhfdr_between_roi_and_bayley.xls', sep='\t', quote=F)

train <- sample(1:nrow(x), nrow(a)*2/3, replace=F)
## train <- 1:nrow(x)

lasso.bayley <- function(i, alpha=1) {
    if (alpha==1)
        label = 'lasso'
    else
        label = 'ridge'
    lasso.mod = glmnet(scale(x[train,]), y[train, i], alpha=alpha, lambda=grid, standardize = T, intercept=TRUE, thresh = 1e-06)

    pdf(paste0(colnames(y)[i], '_', label, '.pdf'))
    plot(lasso.mod, xvar='lambda')
    dev.off()

    set.seed(1)
    cv.out <- cv.glmnet(scale(x[train,]), y[train, i], alpha=1, lambda=grid, type.measure='mse', nfolds=5)

    pdf(paste0(colnames(y)[i], '_', label, 'cv.pdf'))
    plot(cv.out)
    dev.off()

    bestlam=cv.out$lambda.min
    lasso.pred = predict(lasso.mod, s=bestlam, newx=scale(x[-train,]))
    ## lasso.pred = predict(lasso.mod, s=bestlam, newx=scale(x))

    lasso.coef = predict(lasso.mod, type='coefficients', s=bestlam)[1:ncol(x),]
    return(list(lasso.coef[lasso.coef!=0], mean((lasso.pred-y[-train, i])),   mean((lasso.pred-y[-train, i])^2), cor(lasso.pred, y[-train, i])))
    ## return(list(lasso.coef[lasso.coef!=0], mean((lasso.pred-y[, i])^2),     cor(lasso.pred, y[, i], method='spearman')))
}

results <- apply(matrix(1:10), 1, lasso.bayley, alpha=1)
sapply(results, function(x) {x[[4]]})
sapply(results, function(x) {length(x[[1]])})

results <- apply(matrix(1:10), 1, lasso.bayley, alpha=0)
sapply(results, function(x) {x[[4]]})
sapply(results, function(x) {length(x[[1]])})
