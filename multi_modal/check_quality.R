library(gdata)

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

all.images <- read.table('../phaseA_clean_dicom/output/participants.tsv', stringsAsFactors = F, header=T)
patient.info <- read.table('../../01_PVL_AI/new_sequence/new_sequence_patient_id.txt')
patient.info$ID = gsub('../new_sequence/source/', '', patient.info$V1)
all.images = cbind(all.images, patient.info[match(all.images$participant_id, patient.info$ID), ])
sum(apply(matrix(all.images$age), 1, get_age) == apply(matrix(all.images$V7), 1, get_age)) == nrow(all.images)

my.quality <- read.table('../phaseA_clean_dicom/quality_check.list', header=T)
my.quality$id <- paste0("sub-", my.quality$id)
yan.bolds <- as.numeric(gsub('_run-01_bold.nii.gz', '', gsub('sub-', '', basename(Sys.glob("../rsBOLD_filtered/*nii.gz")))))
yan.bolds <- paste0("sub-", yan.bolds)

quality <- cbind(my.quality[match(all.images$ID, my.quality$id), 2:3],
                BOLD_quality=as.numeric(all.images$ID %in% yan.bolds))
all.images <- cbind(all.images, quality)

library(gdata)
## raw.bayley <- read.xls('../../Annotations/patient_information/Bayley III follow-up_2018.9.5.xlsx')
raw.bayley <- read.xls('Bayley III follow-up_2019.7.2_corrected.xlsx')
bayley.score <- raw.bayley[match(all.images$V4, raw.bayley[, 2]), 3:12]
colnames(bayley.score) <- c('Cognitive_12m', 'Fine Motor_12m', 'Gross Motor_12m', 'Receptive Communication_12m', 'Expressive Communication_12m', 'Cognitive_18m', 'Fine Motor_18m', 'Gross Motor_18m', 'Receptive Communication_18m', 'Expressive Communication_18m')

all.images <- cbind(all.images, bayley.score)

baep <- read.xls('../../Annotations/patient_information/BEAP.xlsx', sheet=4)
baep2 <- read.xls('../../Annotations/patient_information/BEAP.xlsx')
procs <- paste0('sub-', na.omit(as.numeric(list.files('../output'))))
beap <- data.frame(BEAP=baep[,"BEAP"], ID=baep2[match(baep$id, baep2$id), 'X3T_id'])
beap <- beap[match(all.images$V4, beap$ID), ]
all.images <- cbind(all.images, beap, processed=as.numeric(all.images$ID %in% procs))
write.table(all.images, 'kai_2017_dti_t1_bold.xls', quote=F, sep='\t')

table(!is.na(all.images[, "Expressive Communication_18m"]), all.images$BEAP)

both.visit.quality <- na.omit(all.images[!is.na(all.images[, "Expressive Communication_18m"]) & (all.images$BEAP==1),
                                        grepl('quality', colnames(all.images))])
sum(rowSums(both.visit.quality)>=2)
table(both.visit.quality[,3])

bayley.quality <- na.omit(all.images[!is.na(all.images[, "Expressive Communication_18m"]), grepl('quality', colnames(all.images))])
sum(rowSums(bayley.quality[,2:3])>=2)

## strict quality control
## final.involved.images <-  which((rowSums(all.images[, grepl('quality', colnames(all.images))])>=2) & !is.na(all.images[, "Expressive Communication_18m"]))

## loose quality controls
final.involved.images <-  which((all.images[, 'BOLD_quality']==1 | all.images[, 'DTI_quality']==1) & (!is.na(all.images[, "Expressive Communication_18m"]) | (!is.na(all.images[, 'BEAP']))))

final <- all.images[final.involved.images, ]

files <- cbind(paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/dwi/", final$ID, "_run-1_dti.nii.gz"),
              paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/dwi/", final$ID, "_run-1_dti.bvec"),
              paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/dwi/", final$ID, "_run-1_dti.bval"),
              paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/func/", final$ID, "_run-1_bold.nii.gz"),
              paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/anat/", final$ID, "_T1w.nii.gz")
              ## paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/dwi/", final$ID, "_run-1_dki.nii.gz"),
              ## paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/dwi/", final$ID, "_run-1_dki.bval"),
              ## paste0('/data1/qinqian/MRI/hyperbilirubinemia/phaseA_clean_dicom/output/', final$ID, "/dwi/", final$ID, "_run-1_dki.bvec")
              )

other.info <- read.xls('../../Annotations/patient_information/Copy of all new seqs_2017.9.14.xlsx')

file.select <- rowSums(apply(files, 2, file.exists))==5
final <- final[file.select,]
final$age = apply(matrix(final$age), 1, get_age)

final$ga <- as.vector(other.info[match(final$V4, other.info[,4]), 2])


final$ga[final$ga=='.'] = NA
final$ga[final$ga==''] = NA

final$ga[grep("\\+", final$ga)] <- apply(matrix(final$ga[grep("\\+", final$ga)]), 1, function(x) {
    y=as.numeric(unlist(strsplit(x, '\\+')))
    y[1] + (y[2] / 7)
})

final$ga = as.numeric(final$ga)

final$bw <- as.vector(other.info[match(final$V4, other.info[,4]), 'bw'])
final$bw <- gsub('g|克', '', final$bw)
final$bw[final$bw==""] = NA
final$bw[final$bw=="."] = NA
final$bw = as.numeric(final$bw)

write.table(final, 'kai_2017_dti_t1_bold_to_be_processed.xls', quote=F, sep='\t')
