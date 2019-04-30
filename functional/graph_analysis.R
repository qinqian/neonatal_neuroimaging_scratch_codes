library(igraph)
library(ANTsR)
## library(poweRlaw)
library(sand)
## library(Rgraphviz)
library(doMC)
library(foreach)
## library(tnet)
## library(brainwaver) # not work ... yet
## library(brainGraph) # use with X11 only!!! borrow small.world from here
library(data.table)

labels <- read.csv("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/label_JHU-neonate-SS_seg.csv", header=F)
pat <- read.csv("Table1.csv")
rs.lowpass <- Sys.glob('../data/fmri_nii/sub*/functional_connectome_meanroi_jhu.cor')
dti.con <- Sys.glob('../data/fmri_nii/sub*/dti/structure_connectome.csv')

## pdf("correlation_distribution.pdf")
## lapply(rs.lowpass, function(x) {
##     y = read.table(x)
##     plot(density(c(as.matrix(y))))
## }) 
## dev.off()

## library(RColorBrewer)
## pdf("fiber_distribution.pdf")
## lapply(dti.con, function(x) {
##     y = read.table(x)
##     y[y <= 1e-06] = 0 # simple threshold
##     y[lower.tri(y)] = t(y)[lower.tri(y)]
##     ## plot(density(c(as.matrix(y))))
##     image(as.matrix(y), col=colorRampPalette(c("white", "red"))(20))
## })
## dev.off()

network.stat <- function(x, graphdensity, corr=T) {
    y = read.table(x)
    y = data.matrix(y)
    diag(y) <- 0
    if (corr) {
        ## y <- antsrimpute(psych::fisherz(y))
        ## y <- (y+1)/2
        y <- abs(y)
    } else {
        y <- y/sum(y)
        y <- y/max(y)
    }
    y[y <= 1e-06] = 0 # simple threshold

    y[lower.tri(y)] = t(y)[lower.tri(y)]
    myord <- order(y)
    whichto0 <- round((1 - graphdensity) * length(myord))
    y[myord[1:whichto0]] <- 0
    y[is.infinite(y)] <- 0
    n <- nrow(y)
    g1 <- graph_from_adjacency_matrix(y, mode=c("undirected"), weighted=T, diag=F)
    st <- graph.strength(g1, weights=E(g1)$weight)
    deg <- degree(g1)
    hub <- hub_score(g1, weights=E(g1)$weight)$vector
    sps <- shortest.paths(g1, weights=E(g1)$weight)
    spath = 1/shortest.paths(g1, weights=E(g1)$weight)
    spath[!is.finite(spath)] <- 0
    geff <- apply(spath, 1, function(x) {sum(x, na.rm=T)})/(n-1)
    geff <- sum(geff)/n
    ## goodnodes <- which(deg>1)
    ## leff <- apply(matrix(goodnodes), 1, function(x) {
    ##     index <- y[x, ]>0
    ##     w.exclude <- y[index, index] # exclude nodes
    ##     g.sub <- graph_from_adjacency_matrix(w.exclude, mode=c("undirected"), weighted=T, diag=F)
    ##     paths <- 1/shortest.paths(g.sub)
    ##     paths[!is.finite(paths)] = 0
    ##     w.ij <- y[x, index]
    ##     ## paths[upper.tri(paths)] <- 0
    ##     o <- outer(w.ij, w.ij)
    ##     o <- o * paths
    ##     weighted.eff <- sum((o) ^ (1/3), na.rm=T)
    ##     ki <- st[x]
    ##     weighted.eff/(ki*(ki-1))
    ## })
    ## leff <- sum(leff)/(2*n)
    ## Cp <- transitivity(g1, weights=E(g1)$weight, type = "global", isolates = "zero")
    ## Lp <- mean(sps)
    ## ## generate a weighted random network by shuffling the weights for 10 times
    ## r = 10
    ## randCp = 0
    ## randsps = 0
    ## for (i in seq(r)) {
    ##     randadj <- matrix(sample(y, 122*122, replace=T), nrow=122, ncol=122)
    ##     randg <- graph_from_adjacency_matrix(randadj, mode=c("undirected"), weighted=T, diag=F)
    ##     randCp <- randCp + transitivity(randg, weights=E(randg)$weight, type = "global", isolates = "zero")
    ##     randsps <- randsps + mean(shortest.paths(randg), na.rm=T)
    ## }
    ## randCp = randCp / r
    ## randsps = randsps / r
    ## sigma <- (Cp/randCp)/(Lp/randsps)
    return(list(## local_eff = leff,
        global_eff = geff,
        betweeness = igraph::betweenness(g1),
        closeness = igraph::closeness(g1),
        Cp = igraph::transitivity(g1, isolates = c("zero"), type = c("barrat"), weights = E(g1)$weights),
                                        #small_world = sigma,
        hub=hub,
        strength=st,
        degree=deg
    ))
}

set.seed(999)
## y = network.stat(rs.lowpass[1], 0.1)

indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(rs.lowpass)))))+1
ann <- pat[indexes,]
mut.g <- ann$mutation=='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
disease.g <- ann$mutation!='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
cont.g <- ann$Hyperbilirubinemia.HP.0002904.==0
groups <- rep("", 50)
groups[mut.g] <- "Hyperbilirubinemia with UGT1A1"
groups[disease.g] <- "Hyperbilirubinemia without UGT1A1"
groups[cont.g] <- "Control"
groups <- as.factor(groups)
features=data.frame(groups=scale(as.numeric(groups)), gender=scale(as.numeric(as.factor(ann$gender))), bw=scale(ann$bw_g))
order.correct <- order(features$groups)
features <- features[order.correct,]
groups <- groups[order.correct]

##density = c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7)
##density = c(0.1, 0.2, 0.3, 0.4, 0.5)
density = seq(0.6, 0.9, 0.05)

registerDoMC(10)
out <- foreach(i=density) %dopar% {
    test <- apply(matrix(rs.lowpass), 1, function(x) {
        network.stat(x, i)
    })
}

plist <- list()
plist.out <- list()
for (i in seq_along(density)) {
    y <- unlist(lapply(out[[i]], function(x) {
        x$global_eff
    }))
    y <- y[order.correct]
    lm.obj <- lm(scale(y)~groups+gender+bw, data=features)
    print(summary(lm.obj))
    plist.out[[i]] <- data.frame(density=vecs[i], group=groups, efficiency=y)
}

plist.out <- Reduce(rbind, plist.out)

library(ggplot2)
p <- ggplot(plist.out)
p <- p + geom_boxplot(aes(fill=group, y=efficiency, x=as.factor(density)))
p <- p + theme_classic() + xlab("Graph Density") + ylab("Functional Connectome Global Efficiency") + theme(legend.justification = "top")
ggsave(p, file="rsbold_efficiency.pdf", width=16)

plist <- list()
plist.out <- list()
for (i in seq_along(density)) {
    y <- unlist(lapply(out[[i]], function(x) {
        x$global_eff
    }))
    y <- y[order.correct]
    lm.obj <- lm(scale(y)~groups+gender+bw, data=features)
    print(summary(lm.obj))
    plist.out[[i]] <- data.frame(density=vecs[i], group=ann$bw_g[order.correct], efficiency=y)
}

plist.out <- Reduce(rbind, plist.out)
plist.out$groups <- cut(plist.out$group, 6)

library(ggplot2)
library(dplyr)
p <- ggplot(plist.out)
p <- p + geom_boxplot(aes(fill=groups, y=efficiency, x=as.factor(density)))
p <- p + theme_classic() + xlab("Graph Density") + ylab("Functional Connectome Global Efficiency") + theme(legend.justification = "top")
ggsave(p, file="rsbold_bw_efficiency.pdf", width=16)

template<-antsImageRead("/data1/qinqian/MRI/data/Atlas/jhu_atlas_neonate/nii/JHU_neonate_SS_122parcellation.nii.gz")
template.surf <-thresholdImage( template, 1, max(template) )
template.surf<-iMath(template.surf,"FillHoles")
cnt<-getCentroids( template, clustparam = 0 )
coord.cnt<-cnt[1:122,1:3]

plist <- list()
for (i in seq_along(density)) {
    y <- lapply(out[[i]], function(z) {
        z$strength
##          z$closeness
##        z$Cp
##        z$betweeness
    })
    y <- Reduce(rbind, y)
    y <- y[order.correct, ]
    ## which hubs node
    z=data.table(y, groups=groups)
    median.hubs <- as.data.frame(z[, lapply(.SD, median), by=groups])   # get rows *and* sum columns 'v' and 'y' by group
    ## significant across density
    lm.obj <- lm(scale(y)~groups+gender+bw, data=features)
    lm.obj <- bigLMStats(lm.obj)
    ## print(i)
    sign.nodes <- lm.obj$beta.p['groups',] <= 0.05
    nodes.p <- lm.obj$beta.p['groups',][sign.nodes]
    order.o <- order(lm.obj$beta.p['groups',])
    ##plist[[i]] <- paste0(as.character(labels[order.o, 2]), as.character(labels[order.o, 3]))
    plist[[i]] <- paste0(as.character(labels[order.o, 1]))
    coord.cnt.tmp <- coord.cnt[sign.nodes,]
    print(labels[sign.nodes,])
    ## apply(matrix(1:3), 1, function(x) {
    ##     node.order <- order(median.hubs[,-1][1,])[1:10]
    ##     coord.cnt.tmp <- coord.cnt[node.order, ]
    ##     weight.tmp <- median.hubs[,-1][1,][node.order]
    ##     print(labels[node.order,])
}

library(RColorBrewer)
library(metap)
## https://brainder.org/2012/05/11/the-logic-of-the-fisher-method-to-combine-p-values/
## integration by rank robust method
library(RobustRankAggreg)
rr <- aggregateRanks(plist)
rr$Name <- as.numeric(as.character(rr$Name))

test <- merge(rr,labels, by.x='Name', by.y='V1', sort=F)
test <- test[order(test$Score), ]
test$Score <- p.adjust(test$Score, 'BH')
write.table(test, file="strength_rsbold_connectome_integrated_ranks.txt", sep="\t")
out.nodes <- subset(test, Score < 1e-4)
coord.cnt.tmp <- coord.cnt[out.nodes[,1],]

brain<-renderSurfaceFunction(surfimg =list( template.surf ) , 
                             alphasurf=0.23, smoothsval = 1 )

plotBasicNetwork(centroids=coord.cnt.tmp, brain, nodecolors=rev(colorRampPalette(c("white", rgb(230, 27,28, maxColorValue=255)))(nrow(coord.cnt.tmp))), lwd=0, radius=-log10(out.nodes$Score)/3, showOnlyConnectedNodes=F)
id<-rgl::par3d('userMatrix')
rid<-rotate3d( id , -pi/2, 0, 0, -1 )
rid3<-rotate3d( id , -pi/2, 0, 0, 1 )
dd<-make3ViewPNG(  rid, id, rid3,  paste0("diff_rsbold_network_hub_nodes_", "_density_all"))

indexes <- as.numeric(gsub("subj", "", gsub("_\\d", "", basename(dirname(dirname(dti.con))))))+1
ann <- pat[indexes,]
mut.g <- ann$mutation=='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
disease.g <- ann$mutation!='UGT1A1' & ann$Hyperbilirubinemia.HP.0002904.==1
cont.g <- ann$Hyperbilirubinemia.HP.0002904.==0
groups <- rep("", 50)
groups[mut.g] <- "Hyperbilirubinemia with UGT1A1"
groups[disease.g] <- "Hyperbilirubinemia without UGT1A1"
groups[cont.g] <- "Control"
groups <- as.factor(groups)
features=data.frame(groups=scale(as.numeric(groups)), gender=scale(as.numeric(as.factor(ann$gender))), bw=scale(ann$bw_g))
order.correct <- order(features$groups)
features <- features[order.correct,]

registerDoMC(10)
density <- c(0.05, 0.1, 0.15, 0.2, 0.25)
out2 <- foreach(i=density) %dopar% {
    test <- apply(matrix(dti.con), 1, function(x) {
        network.stat(x, i, corr=F)
    })
    test
}

plist <- list()
plist.out <- list()
for (i in seq_along(density)) {
    y <- unlist(lapply(out2[[i]], function(x) {
        x$global_eff
    }))
    y <- y[order.correct]
    lm.obj <- lm(scale(y)~groups+gender+bw, data=features)
    print(summary(lm.obj))
    plist.out[[i]] <- data.frame(density=vecs[i], group=groups, efficiency=y)
}
plist.out <- Reduce(rbind, plist.out)

library(ggplot2)
p <- ggplot(plist.out)
p <- p + geom_boxplot(aes(fill=group, y=efficiency, x=as.factor(density)))
p <- p + theme_classic() + xlab("Graph Density") + ylab("Structural Connectome Global Efficiency") + theme(legend.justification = "top")
ggsave(p, file="dti_bw_efficiency.pdf", width=16)

plist <- list()
for (i in seq_along(density)) {
    y <- lapply(out2[[i]], function(z) {
        z$strength
##        z$closeness
##        z$Cp
##        z$betweeness
    })
    y <- Reduce(rbind, y)
    y <- y[order.correct, ]
    ## which hubs node
    z=data.table(y, groups=groups)
    median.hubs <- as.data.frame(z[, lapply(.SD, median), by=groups])   # get rows *and* sum columns 'v' and 'y' by group
    ## significant across density
    lm.obj <- lm(y~groups+gender+bw, data=features)
    lm.obj <- bigLMStats(lm.obj)
    ## print(i)
    sign.nodes <- lm.obj$beta.p['groups',] <= 0.05
    nodes.p <- lm.obj$beta.p['groups',][sign.nodes]
    order.o <- order(lm.obj$beta.p['groups',])
    plist[[i]] <- paste0(as.character(labels[order.o, 1]))
    coord.cnt.tmp <- coord.cnt[sign.nodes,]
    print(labels[sign.nodes,])
    ## apply(matrix(1:3), 1, function(x) {
    ##     node.order <- order(median.hubs[,-1][1,])[1:10]
    ##     coord.cnt.tmp <- coord.cnt[node.order, ]
    ##     weight.tmp <- median.hubs[,-1][1,][node.order]
    ##     print(labels[node.order,])
}

library(RColorBrewer)
library(metap)
## https://brainder.org/2012/05/11/the-logic-of-the-fisher-method-to-combine-p-values/
## integration by rank robust method
library(RobustRankAggreg)
rr <- aggregateRanks(plist)
rr$Name <- as.numeric(as.character(rr$Name))

test <- merge(rr,labels, by.x='Name', by.y='V1', sort=F)
test <- test[order(test$Score), ]
test$Score <- p.adjust(test$Score, 'BH')
write.table(test, file="strength_dti_connectome_integrated_ranks.txt", sep="\t")
out.nodes <- subset(test, Score < 1e-2)
coord.cnt.tmp <- coord.cnt[out.nodes[,1],]

brain<-renderSurfaceFunction(surfimg =list( template.surf ) , 
                             alphasurf=0.23, smoothsval = 1 )

plotBasicNetwork(centroids=coord.cnt.tmp, brain, nodecolors=rev(colorRampPalette(c("white", rgb(230, 27,28, maxColorValue=255)))(nrow(coord.cnt.tmp))), lwd=0, radius=-log10(out.nodes$Score)/3, showOnlyConnectedNodes=F)
id<-rgl::par3d('userMatrix')
rid<-rotate3d( id , -pi/2, 0, 0, -1 )
rid3<-rotate3d( id , -pi/2, 0, 0, 1 )
dd<-make3ViewPNG(  rid, id, rid3,  paste0("diff_dti_network_hub_nodes_", "_density_all"))

