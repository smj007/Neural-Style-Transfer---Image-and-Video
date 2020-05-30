# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:31:49 2020

@author: saimi
"""

from os import listdir
from os.path import join, isfile

import cv2

model_file_path = './models'
model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]

img = cv2.imread('./images/XYZ.jpg')

model = ('la_muse.t7')

for i in model_file_paths:
    style = cv2.imread('./art/' + str(model)[:-3] + '.jpg')
    
    neural_style_model = cv2.dnn.readNetFromTorch(model_file_path + model)
    
    height, width = int(img.shape[0]), int(img.shape[1])
    new_width = ((640/height) * width)
    resized_image = cv2.resize(img,
                              (new_width, 640),
                              interpolation = cv2.INTER_AREA)
    
    input_blob = cv2.dnn.blobFromImage(resized_image,
                                       1.0,
                                       (new_width, 640),
                                       (103.93, 116.77, 123.68),
                                       swapRB = False,
                                       crop = False)
    
    neural_style_model.setInput(input_blob)
    output = neural_style_model.forward()
    
    output = output.reshape(3, output.shape[2], output.shape[3])
    output[0] = output[0] + 103.93
    output[1] = output[1] + 116.77
    output[2] = output[2] + 123.68
    output /= 255
    output = output.transpose(1, 2, 0)
    
    cv2.imshow('Original', img)
    cv2.imshow('Style', style)
    cv2.imshow('Neural Style Transfer', output)
    cv2.waitKey(0)
    
    
    if cv2.waitKey(0) & 0xFF == 27:
        break
    
cv2.destroyAllWindows()    
    
    
    
    
