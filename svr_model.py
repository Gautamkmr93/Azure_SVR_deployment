# -*- coding: utf-8 -*-
"""
Created on Mon May 25 19:52:33 2020

@author: Gautam
"""
#importing the library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import r2_score
from sklearn.svm import SVR
#importing the data
RawData=pd.read_csv('C:\\Users\\Gautam\\Desktop\\SVR\\house_data.csv')
RawData['Location']=lb.fit_transform(RawData['Location'])
RawData['Swimming Pool']=lb.fit_transform(RawData['Swimming Pool'])
RawData.isnull().sum()
RawData['No_Of_Bathrooms'].fillna((RawData['No_Of_Bathrooms'].mean()), inplace=True)
#RawData.isnull().sum()
plt.scatter(RawData['Living_Space(sqft)'],RawData['Price'])
#plt.scatter(RawData['No_Of_Bathrooms'],RawData['Price'])
#plt.scatter(RawData['No_Of_Bedrooms'],RawData['Price'])
#plt.scatter(RawData['Condition(on scale of 5)'],RawData['Price'])
#plt.scatter(RawData['Location'],RawData['Price'])
#x=RawData[['Living_Space(sqft)','No_Of_Bedrooms','No_Of_Bathrooms','Condition(on scale of 5)','Year_Build','Swimming Pool','Location']].values          
x=RawData.iloc[:,0:7].values
y=RawData.iloc[:,7].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25, random_state=0)
svr_model=SVR(kernel='rbf',epsilon=1.0,degree=3,gamma='auto')
svr_model.fit(x_train,y_train)
#pred=svr_model.predict(x_test)
#print(svr_model.score(x_test,y_test))
#print(r2_score(y_test,pred))
pickle.dump(svr_model, open('svrmodel.pkl','wb'))
model = pickle.load(open('svrmodel.pkl','rb'))
