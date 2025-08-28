import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('placement.csv')
print(df.shape)
print(df.head(101))

# preprocesing ...
# unnecesary data ko remove karne ke liye .....
df = df.iloc[:,1:]
df.info()
print(df)
 
# color code
plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
plt.show()

# row And column ko seprate karne ke liye use karte hain
x = df.iloc[:,0:2]
y = df.iloc[:,-1]
print(x,y)

x_train , x_test , y_train , y_test =train_test_split(x , y, test_size = 0.1)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

scaler = StandardScaler()
scaler1 =scaler.fit_transform(x_train)
print("X_train scaler :",scaler1)
x_test = scaler.transform(x_test)
print("x_test Scaler :",x_test)

# Model train :
clf = LogisticRegression()
ModelTrain=clf.fit(x_train,y_train)
print("Model train data:",ModelTrain)

y_pred =clf.predict(x_test)
print(y_pred)
print(y_test)

Accuracy = accuracy_score(y_test,y_pred)
print(":Accuracy:",Accuracy)
#pickle.dump(clf,open('model.pkl','wb'))