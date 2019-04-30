#!/usr/bin/env python
import sys
import pandas as pd

cleaner = lambda y: ''.join([x for x in y if x.isalnum()])

in_id = sys.argv[1]
z=pd.read_table('output_meta3/.heudiconv/{}/info/dicominfo.tsv'.format(cleaner(in_id)))
out_id = str(z.patient_id[0])
out_date = str(z.date[0])
sys.stdout.write("%s_%s\n" % (out_id, out_date))

