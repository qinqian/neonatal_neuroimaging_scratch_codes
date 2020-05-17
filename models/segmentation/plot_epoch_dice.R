test <- read.csv('64_999_patch.csv')

library(ggplot2)
p <- ggplot(data=test, aes(x=epoch, y=dice_score, group=dataset, shape=dataset, color=dataset)) +
    geom_line() +
    geom_point()
pdf('eddnet_dice.pdf')
print(p)
dev.off()


library(ggplot2)
p <- ggplot(data=test, aes(x=epoch, y=detection_rate, group=dataset, shape=dataset, color=dataset)) +
    geom_line() +
    geom_point()
pdf('eddnet_detection.pdf')
print(p)
dev.off()

#test <- read.csv('npy_dense_node13/denseunet_999_64_patch.csv')
#library(ggplot2)
#p <- ggplot(data=test, aes(x=epoch, y=dice_score, group=dataset, shape=dataset, color=dataset)) +
#    geom_line() +
#    geom_point()
#pdf('denseunet_dice.pdf')
#print(p)
#dev.off()
#
#p <- ggplot(data=test, aes(x=epoch, y=detection_rate, group=dataset, shape=dataset, color=dataset)) +
#    geom_line() +
#    geom_point()
#pdf('denseunet_detection.pdf')
#print(p)
#dev.off()
