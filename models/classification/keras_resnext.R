library(ANTsR)
library(ANTsRNet)
library(reticulate)
library(keras)
library(tensorflow)
library(gdata)
library(foreach)
library(doMC)

model <- commandArgs(T)
model <- 'res'
meta <- read.table('results/data_for_model_manual_fixed_resample_failure.xls', sep='\t')
source('models.R')
source('data_generator.R')

load.tfrecord <- function(x) {
    ## shape=as.integer(c(192,192,128,5))
    shape=as.integer(c(80,80,60,5))
    featdef <- dict(image=tf$FixedLenFeature(shape=list(), dtype=tf$string), # list() == []
                   label=tf$FixedLenFeature(shape=list(), dtype=tf$string))
    example <- tf$parse_single_example(x, featdef)
    im = tf$reshape(tf$decode_raw(example[['image']], tf$int16), shape)
    label = tf$decode_raw(example$label, tf$int16)
    return(list(im, label))
}

filenames <- tf$placeholder(tf$string, shape=list(NULL)) # [None] == list(NULL)
tr <- tf$data$TFRecordDataset(filenames, num_parallel_reads=as.integer(12))
tr <- tr$map(load.tfrecord)
tr <- tr$shuffle(as.integer(50))$batch(as.integer(12))$prefetch(as.integer(12))

# $repeat(as.integer(10))

iterator <- tr$make_initializable_iterator()
next_elem <- iterator$get_next()
sess <- k_get_session()

files <- Sys.glob('nii/*tfrecord')
sess$run(iterator$initializer, feed_dict=dict(filenames=files))

#x <- sess$run(next_elem)

K <- backend()
grad <- k_gradients

if (model == 'dense') {
    model <- createDenseNetModel3D(
        c(80, 80, 80, 1),
        numberOfClassificationLabels = 2,
        numberOfFilters = 8,
        depth = 7,
        numberOfDenseBlocks = 2,
        growthRate = 12,
        dropoutRate = 0.2,
        weightDecay = 0.001)
} else if (model == 'res') {
    model = createResNetModel3D(as.integer(c(80, 80, 60, 5)),
                                numberOfClassificationLabels=as.integer(2),
                                layers=as.integer(1:2), residualBlockSchedule=c(3,4), lowestResolution = as.integer(64), cardinality=as.integer(16))
}


train.iterator <- py_iterator(tfrecord_generator(sess, next_elem))
## valid.iterator <- py_iterator(training_generator(valindex, batch_size=batch.size)) # no augmentation
## test.iterator <- py_iterator(training_generator(testindex, batch_size=batch.size))

model %>% compile(
              loss = 'binary_crossentropy',
              optimizer = optimizer_adadelta(lr = 0.0002),
              metrics=c('accuracy')
          )

lr_schedule <- function(epoch, lr) {
   if (epoch %% 3 == 0) {
       lr = lr * 0.9
       if (lr < 1e-7) {
          lr = 1e-7
       }
   }
   lr
}
## lr_callback=callback_reduce_lr_on_plateau(monitor = "val_loss",
##                                           factor = 0.1,
##                                           patience = 10,
##                                           min_delta = 1e-3,
##                                           verbose = 1)

callbacks <- list(
    callback_tensorboard('tf_logs'),
    callback_learning_rate_scheduler(lr_schedule),
    callback_model_checkpoint(filepath = paste0("weights_r_", 'tf', "/", 'checkpoint', "_{epoch:0004d}.h5"),
                              monitor = "val_loss",
                              save_best_only = TRUE,
                              save_weights_only = TRUE))

track = model %>% fit_generator(
                      train.iterator,
                      steps_per_epoch=10,
                      epochs=50,
                      callbacks=callbacks,
                      ## validation_data=valid.iterator,
                      ## validation_steps=length(valindex)/batch.size,
                      verbose=1)

k_clear_session()
