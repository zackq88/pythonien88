# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:48:00 2020

@author: Zaki
"""
import cv2
class DsPreprocessor:
    def __init__(self,width,height):
    #save the width and height into the attributes of the class
        self.width=width
        self.height=height
    def preprocess(self,image):
    #ignore the aspect ratio and resize the image
        return cv2.resize(image,(self.width,self.height),interpolation=cv2.INTER_AREA)