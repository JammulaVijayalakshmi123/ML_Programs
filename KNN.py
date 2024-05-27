from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
i=pd.read_csv("Iris.csv")
print(i)
x=i[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y=i["Species"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
k=KNeighborsClassifier(n_neighbors=1)
k.fit(xtrain,ytrain)
p=k.predict(xtest)
ytestt=np.array(ytest)
for i in range(len(ytestt)):
 print("the actual is:",ytestt[i]," ","the predicted is:",p[i])
