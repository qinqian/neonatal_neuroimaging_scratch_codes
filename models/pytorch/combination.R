
cat(c("1", "2", "3", "4", apply(combn(seq(0, 4), 2), 2, function(x) paste(x, collapse=",")), apply(combn(seq(0, 4), 3), 2, function(x) paste(x, collapse=",")), apply(combn(seq(0, 4), 4), 2, function(x) paste(x, collapse=","))), sep=' ', '\n')

print(length(c("1", "2", "3", "4", apply(combn(seq(0, 4), 2), 2, function(x) paste(x, collapse=",")), apply(combn(seq(0, 4), 3), 2, function(x) paste(x, collapse=",")), apply(combn(seq(0, 4), 4), 2, function(x) paste(x, collapse=",")))))

print(length(c("0,1,2,3,4", "0", "1", "2", "3", "4", apply(combn(seq(0, 4), 2), 2, function(x) paste(x, collapse=",")), apply(combn(seq(0, 4), 3), 2, function(x) paste(x, collapse=",")), apply(combn(seq(0, 4), 4), 2, function(x) paste(x, collapse=",")))))
