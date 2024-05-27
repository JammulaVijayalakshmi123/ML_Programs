import pandas as pd
import numpy as np
p = pd.read_csv('en.csv')
print(p)
#p.drop(p.columns[0], axis=1, inplace=True)
n = np.array(p)
h = n[0]
G = []
l1=[]
for i in n:
 if i[len(h) - 1] == 1:
 for j in range(len(i)):
 if h[j] != i[j]:
 h[j] = '?'
 elif i[len(h) - 1] == 0:
 for j in range(len(i)-1):
 l = ['?'] * len(h)
 if h[j] != i[j] and h[j] != '?':
 l[j] = h[j]
 print(l)
 l1.append(l)
 G.append(l1)
print("Specific hypothesis",h)
print("\n")
print("General Hypothesis",G
