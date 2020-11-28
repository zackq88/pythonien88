# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 00:45:30 2020

@author: Zaki
"""

import numpy as np

class SimplePerceptron:
    def __init__(self,no_of_inputs,learning_rate=0.1):
       self. learning_rate=learning_rate
       self.w=np.random.randn(no_of_inputs+1)/np.sqrt(no_of_inputs) #this " /np.sqrt(no_of_inputs) " is an (initialisation technique) optimization to get reach the global min point  
    def step(self,input_x):
        return 1 if input_x>0 else 0
    def fit_update_train(self,X,y,no_of_epochs=10):
        #add 1 to X
        X=np.c_[X,np.ones((X.shape[0]))]
        for single_epoch in range(0,no_of_epochs):
            #Loop through every data point:
            for (training_input,expected_output)in zip(X,y):
                prediction=np.dot(training_input,self.w)
                prediction=self.step(prediction)
                if prediction != expected_output:
                    error=prediction-expected_output
                    self.w=self.w - self.learning_rate*error*training_input
    def predict_eval(self,X):
        X=np.atleast_2d(X)
        X=np.c_[X,np.ones((X.shape[0]))]
        return self.step(np.dot(X,self.w))