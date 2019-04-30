
# https://stats.stackexchange.com/questions/86535/bootstrap-to-evaluate-variance-of-auc-roc
library(boot)
library(pROC)
set.seed(999) # you should not set.seed unless you are debugging
test <- c(rnorm(70, 40, 10), rnorm(30, 55, 10))
dx <- c(rep(0, 65), rep(1,35) )
df <- data.frame(cbind(test, dx))
print(df)
r <- 1000
boot.f <- function(d, i){
    data <- d[i,]
    auc(roc(data$dx, data$test))
}
roc.boot<-boot(df, boot.f, r)
print(roc.boot)
print(boot.ci(roc.boot, type="bca"))
