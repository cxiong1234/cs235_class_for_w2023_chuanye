#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 21:36:01 2022

@author: cxiong
"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.linear_model import Perceptron

from xiong_preceptron import LinearPerceptron

from numpy import genfromtxt






my_data = genfromtxt("/Users/cxiong/Desktop/235_project/diabetes.csv", delimiter = ',')


my_data_no_header = my_data[1:,:]



X =  my_data_no_header[:, :-1]

y = my_data_no_header[:, -1]

#print(X)

#print(y)

scalar_dist = MinMaxScaler()
X = scalar_dist.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100, stratify=y)
f1_score_list = []

for i  in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
    



    mylove = LinearPerceptron(LearningRate = i)

    mylove.fitting(X_train, y_train)

    predictions, accuracy_iterations = mylove.predicting(X_test)


    f1 = metrics.f1_score(y_test, predictions, labels= None, pos_label=1, average='binary', sample_weight=None)
       
    f1_score_list.append(f1)
   
   

print('f1 score list:', f1_score_list)
   


   

# p_reference = Perceptron(random_state= 50)
# p_reference.fit(X_train, y_train)
# predictions_test =  p_reference.predict(X_test)

# f1 = metrics.f1_score(y_test, predictions_test, labels= None, pos_label=1, average='binary', sample_weight=None)

   
#print("Perceptron classification f1 score by library", f1)
    
#print("Perceptron classification f1 score", accuracy(y_test, predictions))

##print(accuracy_iterations)

#print("f1 score list:",  f1_score_list)



##print(len(accuracy_iterations))







    