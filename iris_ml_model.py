#importing the libraries
import pandas as pd # data preprossesing
import numpy as np # mathematical computation
import matplotlib.pyplot as plt # visualisation

# ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Read the dataset
df = pd.read_csv('iris.csv')
#print(df.shape)
#print(type(df))
#print(df.columns)

# handling null values
nv=df.isnull().sum()
#print(nv)

# check duplicates
#print(df.duplicated().sum())
# 3 cuplicates have been noticed

# Remove the duplicates
df.drop_duplicates(inplace=True)

# Check the target variable
#print(df['label'].value_counts())   # Versicolour-50, virginica-49, detosa 47

#Select the dependent and independent features 
x = df.drop('label',axis=1)
y = df['label']
# print(x.shape) # (146,4)
# print(y.shape) #(146,)
# print(type(x)) # dataframe
# print(type(y)) # series

#split the data into train and test data
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=42)
# print(x_train.shape) # (109,4)
# print(y_train.shape) # (109)
# print(type(x_test)) # dataframe
# print(type(y_test)) # series

#Train the data model
lr = LogisticRegression(max_iter=1000)
dt = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_split=8)
knn = KNeighborsClassifier(n_neighbors=11)
rf = RandomForestClassifier(n_estimators=70,criterion='gini',max_depth=5,
                          min_samples_split=8)

lr.fit(x_train,y_train)
dt.fit(x_train,y_train)
knn.fit(x_train,y_train)
rf.fit(x_train,y_train)

# save the model - pickle
#pickle is uused to serialize the ml model - comversion of ml models into binary files

pickle.dump(lr,open('lr,model.pk1','wb'))
pickle.dump(dt,open('dt,model.pk1','wb'))
pickle.dump(knn,open('knn,model.pk1','wb'))
pickle.dump(rf,open('rf,model.pk1','wb'))

# wb - web binary  

# to save this file in terminal  - write the following
# python iris_ml_model.py
# To stop the server - write the following
# ctrl + c
# To clear the screen 
# cls