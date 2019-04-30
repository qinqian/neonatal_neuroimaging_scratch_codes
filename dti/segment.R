
library(rgl)
open3d(scale=c(1/5,1,1))
segments3d(x=as.vector(t(markers[,c(1,4)])),
           y=as.vector(t(markers[,c(2,5)])),
           z=as.vector(t(markers[,c(3,6)])))
axes3d()
title3d(xlab="X",ylab="Y",zlab="Z")

