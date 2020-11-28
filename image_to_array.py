# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:59:25 2020

@author: Zaki
"""

from keras.preprocessing.image import img_to_array

class ImageToArray:
    def __init__(self,dataFormat=None):
        self.dataFormat=dataFormat
    def preprocess(self,image):
        return img_to_array(image,data_format=self.dataFormat)