import json
import easydict
import sys

a = sys.argv[1]
b = sys.argv[2]
d = sys.argv[3]
c = sys.argv[4]

with open(a) as inf:
    a = json.load(inf)
with open(b) as inf:
    b = json.load(inf)
with open(d) as inf:
    d = json.load(inf)

with open(c) as inf:
    c = json.load(inf)

for i in c.keys():
    if i in a:
        c[i] = a[i]
    if i in b:
        c[i] = b[i]
    if i in d:
        c[i] = d[i]

with open('test.json', 'w') as outf:
    outf.write(json.dumps(c, indent=4))
