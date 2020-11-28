# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:07:39 2020

@author: Zaki
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from imutils import paths
from dsloader import DsLoader
from dspreprocessor import DsPreprocessor
from keras.preprocessing.image import load_img

neighbors=3
jobs=-1

#get the list of images from the dataset path
image_paths=list(paths.list_images('C:\\Users\\zakar\\Desktop\\python_computervision_and_dl\\code\\dataset\\animals'))


print("INFO: loading and preprocessing")
#1#loading and preprocessing images using created classes 
dp=DsPreprocessor(32,32)
dl=DsLoader(preprocessors=[dp]) 
(data,labels)=dl.load(image_paths)

#reshape from(3000,32,32,3)to(3000,32*32*3=3072)#convertion data
data= data.reshape((data.shape[0],3072))
print("INFO:MEMORY size of feature matrix {:.1f}MB".format(data.nbytes/(1024*1000.0)))

#encode the string labels as integers like 0,1,2.. #convertion labels
le= LabelEncoder()
labels=le.fit_transform(labels)



#2#split 25 percentage for testing and the rest fo training
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=40)
print("INFO:splitting the dataset")


#3#traning the KNN classifier using the 75% of training data
model= KNeighborsClassifier(n_neighbors=neighbors,n_jobs=jobs) #K=1 and n_jobs : the number of parallel jobs to run for neighbours search , 1 means none and -1 means all pc processors
model.fit(trainX,trainY) #to pass the informations to train with (X,Y) TO the model KNN
print("INFO:trainning the model")

#4#evaluating the model
print(classification_report(testY,model.predict(testX),target_names=le.classes_))
print("INFO:evaluating the model")



