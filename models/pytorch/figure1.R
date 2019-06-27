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
bids <- bids[bids$V12!='None', ]
table(bids$V11)

## remove redundandancy datasets
bids <- bids[!duplicated(cbind(bids$V4, bids$V5, bids$V6)), ]
table(paste0(bids[,10]," ", bids[, 11], " ", bids[, 12]))

files <- t(apply(matrix(bids[,1]), 1, getnii, '../..'))
bids <- cbind(bids, files)

dim(cbind(bids$V4, bids$V5, bids$V6))
dim(unique(cbind(bids$V4, bids$V5, bids$V6)))
sum(!duplicated(cbind(bids$V4, bids$V5, bids$V6)))

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
bids <- bids[bids[,7]<=8, ]

clinical.report <- read.xls("/data1/qinqian/MRI/Annotations/patient_information/新生儿颅脑MR.xls", header=T, stringsAsFactors=F)
clinical.report <- subset(clinical.report, !grepl('SWI|MRA|MRV|MRS', clinical.report[,8]))
xiamen.report <- read.delim("/data1/qinqian/MRI/01_PVL_AI/data/Xiamen/meta/MRI_xiamen_autoclean_newline.tsv", header=T, stringsAsFactors=F, as.is=T, quote="")

reports <- rep("", nrow(bids))
xiamen.reports <- xiamen.report[match(bids[,8], xiamen.report[,9]), 6]
reports[!is.na(xiamen.reports)] = xiamen.reports[!is.na(xiamen.reports)]

sh.reports <- clinical.report[match(paste0(bids[,4], bids[,6]), paste0(clinical.report[,2], gsub('-', '', clinical.report[,11]))), ncol(clinical.report)]
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
labelsi <- c("囊肿", "软化", "出血|积血", "髓鞘|发育", "扩张", "损伤")

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
bad.ids <- scan("check_datasets/bad_ids_thrown_ids_for_antspy2.txt", what="")
sum(rownames(final)%in%bad.ids)
final <- final[!(rownames(final)%in%bad.ids), ]

#write.table(final, file='results/bids_ready_datasets_annotation.xls', sep='\t', quote=F) ## < 53 weeks
write.table(final, file='results/bids_ready_datasets_annotation_updated.xls', sep='\t', quote=F)

final$Control=1
final$Control[rowSums(final[,13:18])>0]=0

## pdf('results/Figure1_version2.pdf', width=10, height=5.5)
## upset_plot = upset(final,
##                    sets=c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin_or_Develop", "Expansion", "Lesion", "Control"),
##                    sets.bar.color = "#56B4E9", order.by = "freq", empty.intersections = "on", text.scale=1.5)
## print(upset_plot)
## dev.off()

# Part 2
# PVL lesion loci
library(gdata)
pvl_delineated = read.xls('../../../clinical/final_pvl_with_mask_with_control.xlsx')
pvl.ga <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 11]
pvl.ga <- as.numeric(as.vector(pvl.ga))

pvl.birth <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 9]
pvl.birth <- as.vector(pvl.birth)
pvl.birth[grep("顺", pvl.birth)] = 1
pvl.birth[(pvl.birth != "1") & (!is.na(pvl.birth))] = 0
pvl.birth <- as.numeric(pvl.birth)

pvl.length <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 15]
pvl.length <- as.numeric(as.vector(pvl.length))

pvl.weig <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 12]
pvl.weig <- as.numeric(as.vector(pvl.weig))/1000
pvl.head <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 14]
pvl.head <- as.numeric(as.vector(pvl.head))
pvl.label <- pvl_delineated[match(final$ID, pvl_delineated[,2]), 20]
pvl.label <- as.vector(pvl.label)
pvl.label[pvl.label=="#N/A"]=NA

table(pvl_delineated$group.1, pvl_delineated$LABELED) ## 3 has been duplicated

pvl.only <- subset(pvl_delineated, group.1=='pvl')
pvl.cleanid <- pvl.only[match(final$ID, pvl.only[,2]), 23]
pvl.cleanid <- as.vector(pvl.cleanid)
pvl.cleanid[pvl.cleanid=="#N/A"]=NA

pvl.draw <- cbind(paste0("results/mask/all/chen_labels/1-3/", pvl.cleanid, ".nii.gz"),
                 paste0("results/mask/all/qq_labels/1-3/", pvl.cleanid, ".nii.gz"))
pvl.draw[!file.exists(pvl.draw)] = NA
colnames(pvl.draw) <- c("chen_label", "qin_label")

pvl.manual <- cbind(pvl.ga, pvl.birth, pvl.weig, pvl.head, pvl.label, pvl.length, pvl.draw)

final.manual <- cbind(final, pvl.manual)

table(final.manual$Leukomalacia, final.manual$Control)
table(final.manual$Leukomalacia, final.manual$pvl.label) ## false positive from reports

which((final.manual$pvl.label=='1-3')&(final.manual$Leukomalacia==1))

which((final.manual$pvl.label=='4')&(final.manual$Leukomalacia==1))
final.manual[494, ]

final.manual$Leukomalacia.manual <- final.manual$Leukomalacia
## turn label 5 (false positive) to negative manually, and redraw the figure
final.manual$Leukomalacia.manual[final.manual$Leukomalacia==1 & final.manual$pvl.label=='5'] <- 0
## turn 1 cyst patient from label 5 to cyst later
final.manual[which(pvl.cleanid=='40'),]$Cyst = 1

final.manual$Leukomalacia <- final.manual$Leukomalacia.manual

bayleyIII <- read.xls('../../../../Annotations/patient_information/Bayley III follow-up_2018.9.5.xlsx', stringsAsFactors=F)
bayley.scores <- bayleyIII[match(final.manual[, 1], bayleyIII[,2]), 3:12]
colnames(bayley.scores) <- c('Cognitive_12m', 'Fine Motor_12m', 'Gross Motor_12m', 'Receptive Communication_12m', 'Expressive Communication_12m', 'Cognitive_18m', 'Fine Motor_18m', 'Gross Motor_18m', 'Receptive Communication_18m', 'Expressive Communication_18m')
bayley.scores = apply(bayley.scores,2,as.numeric)

final.manual.bayley <- cbind(final.manual, bayley.scores)

write.table(final.manual.bayley, file='results/bids_delineated_updated.xls', sep='\t', quote=F)

pdf('results/Figure1_version2.pdf', width=10, height=5.5)
upset_plot = upset(final.manual.bayley,
                   sets=c("Cyst", "Leukomalacia", "Hemorrhage", "Myelin_or_Develop", "Expansion", "Lesion", "Control"),
                   sets.bar.color = "#56B4E9", order.by = "freq", empty.intersections = "on", text.scale=1.5)
print(upset_plot)
dev.off()
