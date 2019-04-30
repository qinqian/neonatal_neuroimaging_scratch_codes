library(ANTsR)
library(gdata)
library(foreach)
library(doMC)

## control <- read.table("/data1/qinqian/MRI/01_PVL_AI/data/Shanghai/control_patient_id.txt")
## control <- cbind(control, dir=paste0("/data1/qinqian/MRI/01_PVL_AI/data/Shanghai/cont/source/sub-", control[,1]))
## pvl <- read.table("/data1/qinqian/MRI/01_PVL_AI/data/Shanghai/pvl_patient_id.txt")
## pvl <- cbind(pvl, dir=paste0("/data1/qinqian/MRI/01_PVL_AI/data/Shanghai/pvl/source/sub-", pvl[,1]))

getnii <- function(x) {
    t1 <- Sys.glob(paste0(x[1], '/anat/*run-01_T1w.nii.gz'))
    if (length(t1)==0)
        t1=NA
    t1s <- Sys.glob(paste0(x[1], '/anat/*run-02_T1w.nii.gz'))
    if (length(t1s)==0)
        t1s=NA
    t2f <- Sys.glob(paste0(x[1], '/anat/*run-01_T2w.nii.gz'))
    if (length(t2f)==0)
        t2f=NA
    t2 <- Sys.glob(paste0(x[1], '/anat/*run-02_T2w.nii.gz'))
    if (length(t2)==0)
        t2=NA
    dwi <- Sys.glob(paste0(x[1], '/dwi/*run-01_dwi.nii.gz'))
    if (length(dwi)==0)
        dwi=NA
    return(c(t1, t1s, t2f, t2, dwi))
}

genetics <- read.table("../genetics/genetics_patient_id.txt", stringsAsFactors=F, sep='\t')
new.seq <- read.table("../new_sequence/new_sequence_patient_id.txt", sep='\t', stringsAsFactors=F)
shanghai <- read.table("../data/shanghai_patient_id.txt", stringsAsFactors=F, sep='\t')
xiamen <- read.table("../data/xiamen_patient_id.txt", stringsAsFactors=F, sep='\t')

bids <- rbind(genetics, new.seq, shanghai, xiamen)
print(head(bids))
## remove redundandancy datasets
bids <- bids[!duplicated(cbind(bids$V4, bids$V5, bids$V6)), ]
table(paste0(bids[,10]," ", bids[, 11], " ", bids[, 12]))

files <- t(apply(matrix(bids[,1]), 1, getnii))

bids <- cbind(bids,files)

dim(cbind(bids$V3, bids$V4, bids$V5))
dim(unique(cbind(bids$V3, bids$V4, bids$V5)))
sum(!duplicated(cbind(bids$V3, bids$V4, bids$V5)))

get_age <- function(x) {
    if (gsub("M", "", x)!=x)
        return(as.numeric(gsub("M", "", x)) * 4)
    if (gsub("W", "", x)!=x)
        return(as.numeric(gsub("W", "", x)))
    if (gsub("D", "", x)!=x)
        return(as.numeric(gsub("D", "", x)) / 7.0)
    if (gsub("Y", "", x)!=x)
        return(as.numeric(gsub("Y", "", x)) * 52)
}

age <- apply(matrix(bids[,7]), 1, get_age)
bids <- bids[age<53,]

clinical.report <- read.xls("/data1/qinqian/MRI/Annotations/patient_information/新生儿颅脑MR.xls", header=T, stringsAsFactors=F)
xiamen.report <- read.delim("/data1/qinqian/MRI/01_PVL_AI/data/Xiamen/meta/MRI_xiamen_autoclean_newline.tsv", header=T, stringsAsFactors=F, as.is=T, quote="")

reports <- rep("", nrow(bids))
xiamen.reports <- xiamen.report[match(bids[,8], xiamen.report[,9]), 6]
reports[!is.na(xiamen.reports)] = xiamen.reports[!is.na(xiamen.reports)]

sh.reports <- clinical.report[match(bids[,4], clinical.report[,2]), ncol(clinical.report)]
reports[!is.na(sh.reports)] = sh.reports[!is.na(sh.reports)]

select <- reports!=""
reports <- reports[select]

bids = bids[select,]

niis <- bids[,14:ncol(bids)]
colnames(niis) <-  c('t1', 't1s', 't2f', 't2', 'dwi')

apply(niis, 2, function(x) {return(sum(!is.na(x)))})
sum(rowSums(!is.na(niis))>=5)

cooccurence.mat <- function(symptom) {
    mat <- matrix(0, nrow=length(symptom), ncol=length(symptom))
    rownames(mat) = symptom
    colnames(mat) = symptom
    for (i in 1:length(symptom)) {
        for (j in 1:(length(symptom))) {
            mat[i, j] = length(intersect(grep(symptom[i], reports),
                                         grep(symptom[j], reports)))
        }
    }
    mat
}

#labelsi <- c("囊", "软化", "出血|积血", "血肿", "髓鞘|发育", "扩张", "损伤")
labelsi <- c("囊", "软化", "出血|积血", "髓鞘|发育", "扩张", "损伤")
result.comat <- cooccurence.mat(labelsi)

colnames(result.comat) <- c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin or Develop", "Expansion", "Lesion")
rownames(result.comat) <- c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin or Develop", "Expansion", "Lesion")

library(circlize)
library(gplots)
pdf('results/cooccur.pdf', width=8, height=8, family="GB1")
par(cex=1.7, font=1)
set.seed(999)
#chordDiagram(result.comat, transparency = 0.5, row.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink', 'orange'), grid.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink', 'orange')) # col=bluered(36),
## chordDiagram(result.comat, transparency = 0.5, row.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink'), grid.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink')) # col=bluered(36),
dev.off()

#write.table(bids, file='results/bids_ready_datasets_annotation.xls', sep='\t', quote=F)
#
labelmat <- function(labels, r) {
   mat <- matrix(0, nrow=length(r), ncol=length(labels))
   n=1
   for (i in labels) {
       print(i)
       mat[grep(i, r),n] = 1
       n <- n+1
   }
   mat
}

niisall <- niis[rowSums(!is.na(niis))>=5,]
reportsall <- reports[rowSums(!is.na(niis))>=5]
labelsall <- labelmat(labelsi, reportsall)
platform <- bids[rowSums(!is.na(niis))>=5, 11]
print(dim(niisall))

colnames(labelsall) <- labelsi
apply(labelsall, 2, table)

### df <- cbind(niisall, reportsall, labelsall)
#df <- cbind(id=1:nrow(niisall), platform=platform, niisall, labelsall)
#print(dim(df))
#write.table(df, file="results/data_for_model.xls", quote=F, sep="\t")
