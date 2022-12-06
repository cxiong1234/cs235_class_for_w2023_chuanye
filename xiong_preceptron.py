#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 13:12:37 2022

@author: cxiong
"""

import numpy as np
import math

class LinearPerceptron:
    
    
    def __init__(self, LearningRate = 0.02, IterationNum=800):
        
        self.IterationNum = IterationNum
        self.LearningRate = LearningRate
        
        
        self.Weights = 0  ##its actually arrays ut initialized as 0 number 
        self.Bias = 0   ##Its actually arrays but initialized as 0 number 
        
        self.ActFunction_for_single = self._single_data_act_function
        
        self.ActFunction_for_multi = self._multi_data_act_function
        self.CS_accuracy = np.array([0])


    # def _single_data_act_function(self, data): ##only accept one data
    #     if data >= 0:
    #         return 1
    #     else:
    #         return 0
    
    
    # def _multi_data_act_function(self, data_list): ## accept datalist
    #     for i in range(len(data_list)):
    #         if data_list[i] >= 0:
    #             data_list[i] = 1
    #         else:
    #             data_list[i] = 0
        
    #     return data_list
       
    
    def _single_data_act_function(self, data):
        data = 1 / (1 + math.exp(-data))
        if data >= 0.5:
            return 1
        else:
            return 0
    
    def _multi_data_act_function(self, data_list):
        for i in range((len(data_list))):
            data_list[i] = 1 / (1 + math.exp(-data_list[i]))
            if data_list[i] >= 0.5:
                data_list[i] = 1
            else:
                data_list[i] = 0
        
        return data_list
    
    
    
    
    
    def fitting(self, X, y):
        self.Bias = 0
        feature_amount= X.shape[1]
        self.Weights= np.zeros(feature_amount)
        
        
        y_normalized = np.array([])
        for i in y:
            if i > 0:
                y_normalized = np.append(y_normalized, 1)
            else:
                y_normalized = np.append(y_normalized, 0)
        
        
        
        difference_each_iteration_list = np.array([])
        for i in range(self.IterationNum):
            difference = np.array([0])
            
            
           # for index, x_i in enumerate(X):
            for i in range(X.shape[0]):
                ith_data_of_X = X[i,:]   
                x_w_multiple_single = self.Bias + np.dot(ith_data_of_X, self.Weights)
               # print(linear_function_output)
                #print(type(linear_function_output))
                y_predict_intraining = self.ActFunction_for_single(x_w_multiple_single)
                
                delta_weight = self.LearningRate * (y_normalized[i] - y_predict_intraining)
                
                if i == 0:
                    difference[0] = y_normalized[i] - y_predict_intraining
                
                difference = np.append(difference,[y_normalized[i] - y_predict_intraining])
                
                self.Weights = delta_weight * ith_data_of_X + self.Weights
               # print(self.Weights)
                self.Bias = delta_weight + self.Bias
                
            difference_each_iteration_list = np.append(difference_each_iteration_list, np.linalg.norm(difference))
            
            #print(difference_each_iteration_list)
           
            
            if i == 0:
                self.CS_accuracy[0] = np.sqrt((difference * difference).sum())
            
            
            self.CS_accuracy = np.append(self.CS_accuracy,[np.sqrt((difference * difference).sum())])
                
        
    
        
    
    def predicting(self, X_predict):
        x_w_multiple_list =  self.Bias + np.dot(X_predict, self.Weights) 
        
        y_predict_intesting = self.ActFunction_for_multi(x_w_multiple_list)
       
        return y_predict_intesting, self.CS_accuracy
    
    
    
        