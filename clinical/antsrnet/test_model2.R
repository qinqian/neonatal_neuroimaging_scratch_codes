library(tfestimators)
require(tensorflow)
require(keras)
require(ANTsR)
require(jsonlite)
library(reticulate)
library(abind)
library(rhdf5)
require(fslr)
library(foreach)
library(doMC)
library(extrantsr)
library(ggplot2)
refine = read.table('control_list_refine.txt',sep='\t', header=T)
refine.ids = refine[,3]

setwd("/home/qinqian/data1/MRI/01_PVL_AI")
pvl <- read.table("output_pvl/participants.tsv", header=T, stringsAsFactors = F)
pvl[,4] <- 'pvl'
pvl <- pvl[file.exists(paste0('output_pvl/', pvl[,1], '/anat/', pvl[,1], "_run-2_T2w.nii.gz")),]
pvl[,1] <- paste0('output_pvl/', pvl[,1])
cont <- read.table("output_control/participants.tsv", header=T, sep='\t', stringsAsFactors = F)
cont <- cont[!(cont[,1]%in%basename(pvl[,1])), ]
cont <- cont[file.exists(paste0('output_control/', cont[,1], '/anat/', cont[,1], "_run-2_T2w.nii.gz")), ]
cont.ids = as.integer(gsub('_\\d+', '', gsub('sub-', '', cont[,1])))
cont <- cont[cont.ids%in%refine.ids, ]

cont[,1] <- paste0('output_control/', cont[,1])
meta <- rbind(pvl, cont)
group = meta[,4]
labels = rep(0, length(group))
labels[group=='pvl'] = 1
meta[,4] = labels
keras = import("keras")

fn <- keras$wrappers$scikit_learn$KerasClassifier
metrics <- import('sklearn.metrics')

roc_auc <- metrics$roc_auc_score
auc_curve <- metrics$roc_curve
recall <- metrics$recall_score
precision <- metrics$precision_score
f1 <- metrics$f1_score
prauc <- metrics$average_precision_score
pr_curve <- metrics$precision_recall_curve

seeds <- c(100, 111, 1234, 5000, 9999, 1111, 6666, 8000, 9000, 10000, 2000, 1989)
results <- rep(0, length(seeds))

h=H5Fopen('all_images_refin.h5')
allArray = h$pvl$images
labelss = h$pvl$labels
H5Fclose(h)

m = mean(allArray)
s = sd(allArray)
allArray = (allArray-m)/s

#cat(c(m, s), '\n', file='mean_sd.txt')

accs = c()
aucs = c()
praucs = c()
f1s = c()
rr = c()
pr = c()
for (i in seq_along(seeds)) {
    set.seed(seeds[i])
    trainProportion = 0.8
    subindex = sample(1:nrow(meta), nrow(meta)*trainProportion, replace = F)
    backend()$clear_session()
    source("models.R")
    vggModel <- createVggModel3D(dim(allArray)[2:5],
                                numberOfClassificationLabels = 2,
                                layers = c(1, 1),
                                lowestResolution = 8,
                                convolutionKernelSize = c(3, 3, 3),
                                poolSize = c( 2, 2, 2 ),
                                strides = c( 2, 2, 2 ),
                                        #denseUnits = 64,  ## 100, 0.6 very well,
                                denseUnits = 50,
                                dropoutRate = 0.0,  ## try 0.1
                                weightDecay = 0.0,
                                style=19)
    vggModel %>% compile(
                     loss = 'binary_crossentropy',
                                        #optimizer = optimizer_rmsprop(lr=0.0001),
                     optimizer = optimizer_adam(lr = 0.001),
                                        #optimizer =  optimizer_sgd(lr = 0.001, momentum = 0.9, nesterov = TRUE),
                                        #optimizer = optimizer_adadelta(lr = 0.001, decay=1e-4),
                                        #optimizer = optimizer_sgd(lr = 0.01),
                     metrics=c('accuracy'))
    ## 4__vgg_refine_Weights_all.h5
    load_model_weights_hdf5(vggModel, paste0(i, "__vgg_refine_Weights_all.h5"))
    #z = read.table(paste0(i, "__vgg", "_refine_accuracy.txt"))
    #d=subset(z, data=='validation')
    #d.acc= subset(d, metric=='acc')
    #d.loss= subset(d, metric=='loss')
    #epoc = d.acc[which.max(d.acc$value), 1]
    #epoc = d.loss[which.min(d.loss$value), 1]

    #load_model_weights_hdf5(vggModel, paste0("weights_noaug_refine/", i, "vgg_lr001_", epoc, ".h5"))
    ##vggModel <- load_model_hdf5(paste0(i, "__vggWeights_all.h5"))
    print(labels[-subindex])
    print(mean(labels[-subindex]))

    testc <- vggModel %>% predict_classes(allArray[-subindex,,,,])
    print(testc)
    accs[i] = mean(testc==labels[-subindex])
    testc <- vggModel %>% predict_classes(allArray)
    print(mean(testc==labels))

#    print(subindex)
    print(accs)
#    test <- vggModel %>% predict(allArray[-subindex,,,,])
#    print(test)
    print(vggModel %>% evaluate(allArray[-subindex,,,,], labelss[-subindex,]))
#    print(dim(test))
#    print('auc')
#    aucs[i] = roc_auc(labels[-subindex], test[,2])
#    print(aucs)
#    print('pr auc')
#    praucs[i] = prauc(labels[-subindex], test[,2])
#    print('f1')
#    f1s[i] = f1(labels[-subindex], testc)
#    print('recall')
#    rr[i] = recall(labels[-subindex], testc)
#    print('precision')
#    pr[i] = precision(labels[-subindex], testc)
#    ## break
#    print(accs)
#    print(aucs)
#    print(praucs)
#    print(f1s)
#    print(rr)
#    print(pr)
}

df = cbind(accs, aucs, praucs, f1s, rr, pr)
#write.table(df, file="performance_highest_performance.txt", quote=F, sep="\t")
H5close()
