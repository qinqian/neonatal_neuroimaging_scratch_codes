library(ANTsR)
library(ggplot2)
library(gdata)
library(foreach)
library(doMC)
setwd("/home/qinqian/data1/MRI/01_PVL_AI/scripts/models/pytorch")

getnii <- function(x, directory) {
    x[1] = paste0(directory, '/', x[1])
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
    adc <- Sys.glob(paste0(x[1], '/dwi/*run-02_dwi.nii.gz'))
    if (length(adc)==0)
        adc=NA
    return(c(t1, t1s, t2f, t2, dwi, adc))
}

genetics <- read.table("../../../genetics/genetics_patient_id.txt", stringsAsFactors=F, sep='\t')
new.seq <- read.table("../../../new_sequence/new_sequence_patient_id.txt", sep='\t', stringsAsFactors=F)
shanghai <- read.table("../../../data/shanghai_patient_id.txt", stringsAsFactors=F, sep='\t')
xiamen <- read.table("../../../data/xiamen_patient_id.txt", stringsAsFactors=F, sep='\t')

bids <- rbind(genetics, new.seq, shanghai, xiamen)
print(head(bids))

## remove redundandancy datasets
bids <- bids[!duplicated(cbind(bids$V4, bids$V5, bids$V6)), ]
table(paste0(bids[,10]," ", bids[, 11], " ", bids[, 12]))

files <- t(apply(matrix(bids[,1]), 1, getnii, '../..'))
bids <- cbind(bids, files)

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

bids <- bids[complete.cases(bids),]

bids[,7] = apply(matrix(bids[,7]), 1, get_age)
bids <- bids[bids[,7]<=52,]

clinical.report <- read.xls("/data1/qinqian/MRI/Annotations/patient_information/新生儿颅脑MR.xls", header=T, stringsAsFactors=F)
xiamen.report <- read.delim("/data1/qinqian/MRI/01_PVL_AI/data/Xiamen/meta/MRI_xiamen_autoclean_newline.tsv", header=T, stringsAsFactors=F, as.is=T, quote="")

reports <- rep("", nrow(bids))
xiamen.reports <- xiamen.report[match(bids[,8], xiamen.report[,9]), 6]
reports[!is.na(xiamen.reports)] = xiamen.reports[!is.na(xiamen.reports)]

sh.reports <- clinical.report[match(bids[,4], clinical.report[,2]), ncol(clinical.report)]
reports[!is.na(sh.reports)] = sh.reports[!is.na(sh.reports)]

gender <- rep("", nrow(bids))
xiamen.gender <- xiamen.report[match(bids[,8], xiamen.report[,9]), 3]
gender[!is.na(xiamen.gender)] = xiamen.gender[!is.na(xiamen.gender)]

sh.gender <- clinical.report[match(bids[,4], clinical.report[,2]), 4]
gender[!is.na(sh.gender)] = sh.gender[!is.na(sh.gender)]

select <- reports!=""
reports <- reports[select]
gender <- gender[select]
bids = bids[select,]

bids <- cbind(bids, gender)

colnames(bids)[14:(ncol(bids)-1)] <-  c('t1', 't1s', 't2f', 't2', 'dwi', 'adc')
gender = as.vector(bids$gender)
gender[gender=='男'] = 'Male'
gender[gender=='女'] = 'Female'
bids$gender = gender

## apply(niis, 2, function(x) {return(sum(!is.na(x)))})
## sum(rowSums(!is.na(niis))>=5)

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

## labelsi <- c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin or Develop", "Expansion", "Lesion")
result.comat <- cooccurence.mat(labelsi)

colnames(result.comat) <- c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin or Develop", "Expansion", "Lesion")
rownames(result.comat) <- c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin or Develop", "Expansion", "Lesion")

#library(circlize)
#library(gplots)
#pdf('results/cooccur.pdf', width=8, height=8, family="GB1")
#par(cex=1.7, font=1)
#set.seed(999)
#chordDiagram(result.comat, transparency = 0.5, row.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink', 'orange'), grid.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink', 'orange')) # col=bluered(36),
## chordDiagram(result.comat, transparency = 0.5, row.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink'), grid.col=c('red', 'purple', 'blue', 'yellow', 'green', 'pink')) # col=bluered(36),
#dev.off()

# we need more direct way of showing all data labels
# install.packages('corrplot')
library(corrplot)
library(UpSetR)
system("mkdir -p results")
pdf('results/Figure1.pdf')
corrplot(as.matrix(result.comat), add=F, method='number',
         col=colorRampPalette(c("white", "blue"))(20), cl.pos='n',
         is.corr = F, type='lower')
dev.off()


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

labelsall <- labelmat(labelsi, reports)

colnames(labelsall) <- c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin_or_Develop", "Expansion", "Lesion")

apply(labelsall, 2, table)

final <- bids[,c(4,7,10,11,12,14:20)]
final <- cbind(final, labelsall)

colnames(final)[1:5] <- c('ID', 'age', 'machine', 'platform', 'tesla')

pdf('results/Figure1_version2.pdf', width=10, height=5.5)
upset_plot = upset(subset(final, age<=8), 
                   sets=c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin_or_Develop", "Expansion", "Lesion"), 
                   sets.bar.color = "#56B4E9", order.by = "freq", empty.intersections = "on", text.scale=1.5)
print(upset_plot)
dev.off()

write.table(final, file='results/bids_ready_datasets_annotation.xls', sep='\t', quote=F)

# Part 2
# PVL lesion loci 
library(gdata)
pvl_delineated = read.xls('../../../clinical/final_pvl_with_mask_with_control.xlsx')
## pvl <- with(pvl_delineated, group.1=='pvl'&LABELED=='1-3')
## ids = as.vector(pvl_delineated$cleaned_file_id)
## ids = ids!='#N/A' & (!is.na(ids)) & pvl

pvl.ga <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 11]
pvl.ga <- as.numeric(as.vector(pvl.ga))

pvl.birth <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 9]
pvl.birth <- as.vector(pvl.birth)
pvl.birth[grep("顺", pvl.birth)] = 1
pvl.birth[(pvl.birth != "1") & (!is.na(pvl.birth))] = 0

pvl.weig <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 11]
pvl.weig <- as.numeric(as.vector(pvl.weig))

pvl.head <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 12]
pvl.head <- as.numeric(as.vector(pvl.head))

pvl.draw <- pvl.draw[!is.na(pvl.draw$clean_file_id), ]
dim(pvl_delineate_ones)
