## compare TBS levels 
pat <- read.csv("Table1.csv")

pdf("tsb_comparison.pdf",width=5.5)
par(cex=1, font=2)
mut.tsb <- na.omit(pat$TSB[as.vector(pat$mutation)=="UGT1A1"])
cont.tsb <- na.omit(pat$TSB[as.vector(pat$mutation)=="negative"])
a <- boxplot(list(UGT1A1.mutation=mut.tsb, control=cont.tsb), ylab="TSB peak level", main="4 patients with UGT1A1 mutation vs 3 control patients", ylim=c(0,35), width=c(0.5,0.5))
lines(c(1, 2), c(34,34), col='black')
text(1.5, 34.5, round(t.test(mut.tsb, cont.tsb, alternative='greater')$p.value, 3))
dev.off()

mut <- rep(0, nrow(pat))
mut[1:5] = 1
df <- data.frame(UGT1A1.mutation=mut, Hyperbilirubinemia=pat$Hyperbilirubinemia.HP.0002904)

dim(pat)

df <- data.frame(Mutation=as.numeric(pat$mutation),pat[,22:40])
table(df)
