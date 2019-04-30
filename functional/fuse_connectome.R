
a=read.table('functional_connectome_pval.txt')
b=read.table('structural_connectome_pval.txt')

sum((abs(a) > 0) & (abs(b) > 0))

