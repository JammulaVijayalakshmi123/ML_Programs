import pandas as pd
import numpy as np
from sklearn import metrics
df=pd.read_csv("pt.csv")
print(df)
from sklearn import preprocessing
string_int=preprocessing.LabelEncoder()
df=df.apply(string_int.fit_transform)
print(df)
x=df[['outlook','temp','humidity','wind']]
y=df['play']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=1)
from sklearn.naive_bayes import GaussianNB
k=GaussianNB()
k.fit(xtrain,ytrain)
p=k.predict(xtest)
#initial accuracy
from sklearn.metrics import accuracy_score
print("accuracy is:",accuracy_score(p,ytest))
fp=k.predict([[1,1,0,1]])
print("the final label is:",fp)
