# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:41:12 2020

@author: Zaki
"""

import numpy as np
from simplePerceptron import SimplePerceptron

#AND dataset
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

perceptron=SimplePerceptron(X.shape[1],learning_rate=0.1)
#train
perceptron.fit_update_train(X,y,no_of_epochs=20)



#eval

for (input_train,_output)in zip(X,y):
    predicted_output=perceptron.predict_eval(input_train)
    print("input: {} predicted output: {} and the real output:{} ".format(input_train,predicted_output,_output))
   

    

