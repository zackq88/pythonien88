# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 23:41:30 2020

@author: Zaki
"""
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 
from imutils import paths
from dsloader import DsLoader
from dspreprocessor import DsPreprocessor
from keras.preprocessing.image import load_img , img_to_array
import numpy as np
import matplotlib.pyplot as plt

neighbors=3 #3NN algo #K=3
jobs=-1

#get the list of images from the dataset path
image_paths=list(paths.list_images('C:\\Users\\zakar\\Desktop\\python_computervision_and_dl\\code\\dataset\\animals'))


print("INFO: loading and preprocessing")
#1#loading and preprocessing images using created classes 
dp=DsPreprocessor(32,32)
dl=DsLoader(preprocessors=[dp]) 
(data,labels)=dl.load(image_paths) #this is our tupple of information 

#reshape from(3000,32,32,3)to(3000,32*32*3=3072)   #convertion data
data= data.reshape((data.shape[0],3072))
print("INFO:MEMORY size of feature matrix {:.1f}MB".format(data.nbytes/(1024*1000.0)))

#encode the string labels as integers like 0,1,2.. #convertion labels
le= LabelEncoder()
labels=le.fit_transform(labels)



#2#split 25 percentage for testing and the rest fo training
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.25,random_state=40)
print("INFO:splitting the dataset")




#3#traning
print("INFO:trainning the model")
#looping trough L1,L2 and without regularization
for regularization in ("l1","l2",None):
    print("INFO:training the model with {} regularization ".format(regularization))
    model = SGDClassifier(loss="log",penalty=regularization,max_iter=20,
                          learning_rate="constant",eta0=0.01,random_state=42)
    
    model.fit(trainX,trainY)
    print("INFO:Evaluating the model")
    #evaluationg the accuracy 
    accuracy=model.score(testX,testY)
    print("INFO:evaluation accuracy the model with {} regularization is {:.2f}%".format(regularization,accuracy*100))
   

 


















'''
#predict
animals_classes = ['cat','dog','panda']
unknown_image = load_img('C:\\Users\\zakar\\Desktop\\python_computervision_and_dl\\code\\images\\img.jpg')
unknown_image = unknown_image.resize((32,32))
unknown_image_array=img_to_array(unknown_image) #this is a 3D array
unknown_image_array=unknown_image_array.reshape(1,-1) #reshape our 3D array to 1 row , 3072 columns ..

prediction = model.predict(unknown_image_array) #model.predict   [input is:unknown_image_array(the image=the data=the 1D array) ]   //   output is:[prediction (0 , 1 , 2 = cat , dog ,panda )=output=labels]
print("the predicted animal is")
print(str([animals_classes[int(prediction)]]))

'''
"""
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    model2 = KNeighborsClassifier(n_neighbors=i)
    model2.fit(trainX, trainY)
    prediction2 = model2.predict(testX)
    error.append(np.mean(prediction2 != testY))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
"""