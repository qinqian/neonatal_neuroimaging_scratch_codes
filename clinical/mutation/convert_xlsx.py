#!/usr/bin/env python
import pandas as pd

df = pd.read_excel("脑病-MRI信息.xlsx")
df.to_csv('yanglin_MRI.txt', sep='\t')
