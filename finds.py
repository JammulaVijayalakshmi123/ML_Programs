import pandas as pd
import numpy as np
df= pd.read_csv('en.csv')
df=df[df[df.columns[-1]]==1]
print(df)
d=np.array(df)
print(d)
def FindS(df):
 h=['pi']*(len(df.columns)-1)
 print(h)
 for i in d:
 if 'pi' in h:
 for j in range(len(h)):
 h[j]=i[j]
 else:
 for j in range(len(h)):
 if h[j]!=i[j]:
 h[j]='?'
 print(h)
FindS(df)
