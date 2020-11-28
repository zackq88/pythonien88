# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:24:19 2020

@author: Zaki
"""
import cv2
import numpy as np
import os #os is recording to going to the folder

class DsLoader:
    def __init__(self, preprocessors=None):
        #save the preprocessor passed in (if we passed a proccessor deja )
        self.preprocessors=preprocessors
        #initialize an empty list if the passed preprocessor is empty
        if self.preprocessors is None:
            self.preprocessors = []
    def load(self, imagePaths):
        #initialize the lists for data and labels
        data=[]
        labels=[]
        for(i,imagePath) in enumerate(imagePaths):
            image=cv2.imread(imagePath)
            label=imagePath.split(os.path.sep)[-2]
            
    
            #ex path\code\datasets\animals\cats\cat1.jpg-->"cats"
            
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image=p.preprocess(image)
            #displayin the progress in format
            #processe 500/3000
            if i>500 and (i+1)%500==0:
                print("processed {}/{}".format(i+1,len(imagePaths)))
            
            data.append(image)
            labels.append(label)
            
        return(np.array(data),np.array(labels))
    
        
