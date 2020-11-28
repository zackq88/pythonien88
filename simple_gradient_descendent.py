# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 17:49:46 2020

@author: Zaki
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs #to create our dataset 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#define learning rate and epochs
learning_rate=0.001 #we should try first 0.1 then 0.01 , 0.001 ..
no_of_epochs=100

#1#generate a sample data set with 2 features columns and one class comuln
#features with random numbers and class with either 0 or 1
#total number of samples will be 100
(X,y) = make_blobs(n_samples=100,n_features=2,centers=2,cluster_std=1.5,random_state=1) #cluster_std is how much varitation each data should have 
#convert y from(100,) to (100,1)
y=y.reshape((y.shape[0],1))
#reshape X, add one column [1] due to the new W
X= np.c_[X,np.ones((X.shape[0]))]



#2#splitting the dataset 
(trainX,testX,trainY,testY)=train_test_split(X,y,test_size=0.5,random_state=50)



#initialize a 3x1 weight matrix
w= np.random.randn(X.shape[1],1)
#initialize a list for storing loss values during epochs
losses_value=[]




#3#Evaluation
#define sigmoid function
def sigmoid_function(x):
    return 1.0/(1+np.exp(-x))
#define predict function#This is for evaluation not training
def predict_function(X,W):
    prediction=sigmoid_function(X.dot(W))
    #use step function to convert the prediction to class labels 1 or 0
    prediction[prediction<=0.5]=0
    prediction[prediction>0.5]=1
    return prediction




#4#start training epochs#TO HAVE THE W*
for epoch in np.arange(0,no_of_epochs):
    prediction = sigmoid_function(trainX.dot(w))
    #find error
    error=prediction-trainY
    #find the loss value and append it to the losses_value list
    loss_value=np.sum(error ** 2)
    losses_value.append(loss_value)
    #find the gradient , dot product of training input (transposed) and current error
    gradient= trainX.T.dot(error)
    #add to the existing value of weight W,the new variation
    #using the negative gradient (the descending gradient)
    w=w - (learning_rate) * gradient
    print("Epoch Number : {}, loss :{:.7f}".format(int(epoch),loss_value))
    
    

#test and evaluation of our model
print("Starting evaluation")
predictions = predict_function(testX,w)
print(classification_report(testY,predictions))


#plotting the data set as scatter plot
plt.style.use("ggplot")
plt.figure()
plt.title("Scatter plot of dataset")
plt.scatter(testX[:,0],testX[:,1])


#plotting the error vs epochs graph
plt.style.use("ggplot")
plt.figure()
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(np.arange(0,no_of_epochs),losses_value)