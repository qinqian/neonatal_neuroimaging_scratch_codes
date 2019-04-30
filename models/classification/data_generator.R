tfrecord_generator <- function(sess, iterator) {
    function() {
        x <- sess$run(iterator)
        print(dim(x[[1]]))
        print(dim(x[[2]]))
        result <- list(keras_array(x[[1]]), keras_array(to_categorical(x[[2]][,1])))
    }
}

training_generator <- function(data, label, sample_index, batch_size, aug=F) {
    imgs.cur <- data[sample_index, ]
    label.cur <- labels[sample_index, ]
    imgs.next <- all[sample_index, ]
    label.next <- labels[sample_index,]

    function() {
        ## start new epoch
        if (nrow(imgs.cur) < batch_size) {
            imgs.cur <<- imgs.next
            label.cur <<- label.next
        }
        batch_ind <- sample(1:nrow(imgs.cur), batch_size, replace=F) # shuffle current batch
        batch_images_list <- imgs.cur[batch_ind, ]
        imgs.cur <<- imgs.cur[-batch_ind, ]
        batch_label_list <- label.cur[batch_ind,]
        label.cur <<- label.cur[-batch_ind,]
        ## data augmentation by random inverse transformation to each of the inviduals
        x_y.batch <- foreach(i=1:batch_size) %do% {
            load_one_modal <- function(j) {
                if (aug) {
                    x = imageFileNames2ImageList(batch_images_list[j,1:4])
                    for (z in seq_along(x)) {
                    if (runif(1)<=0.3) {
                        if (runif(1)<=0.5) {
                            rand = randomImageTransformAugmentation( x[[z]], list(list(x[[z]])), list(x[[z]]), n=1, typeOfTransform='AffineAndDeformation')
                            x[[z]] = rand$outputPredictorList[[1]][[1]]
                        } else {
                            if (runif(1) <= 0.5) {
                            x[[z]] = x[[z]] + array(rnorm(prod(dim(x[[z]]))), dim=dim(x[[z]]))
                            }
                        }
                    }}
                } else {
                    x = imageFileNames2ImageList(batch_images_list[j,1:4])
                }
                y <- lapply(x, function(y0) {
                    # 107x144x111
                     z = iMath(cropImage(y0, mask), 'Normalize')
                    ## z = as.array(cropIndices(z, c(20, 20, 20), c(99, 99, 99)))
                    #z = as.array(resampleImage(y0, c(100, 100, 100), useVoxels=1))
                    #z = as.array(y0)[10:99, 10:99, 10:99]
                    -(z-mean(z))/sd(z)
                })
                y <- abind(y, along=4) # 1 * x * y * z * channels
                y
            }
            #x0 = future({ load_one_modal(i) }) %plan% multiprocess
            x0 = load_one_modal(i)
            #list(x0=value(x0), y=batch_label_list[i, ])
            list(x0=x0, y=batch_label_list[i, ])
        }
        x_y.batch <- purrr::transpose(x_y.batch)
        x.batch <- do.call(abind, c(x_y.batch$x0, list(along=0))) # batch * x * y * z * channels
        y.batch <-  do.call(rbind, x_y.batch$y)
        result <- list(keras_array(x.batch), keras_array(y.batch))
    }
}
