# -*- coding: utf-8 -*-
"""
Created on Sat May 30 13:19:24 2020

@author: saimi
"""

import cv2
import numpy as np

def neural_style_transfer(img, model, size = 320, upscale = 1):
    
    model_file_path = 'models/'
    style = cv2.imread('art/' + str(model)[:-3] + '.jpg')
    neural_style_model = cv2.dnn.readNetFromTorch(model_file_path + model + '.t7')
    
    height, width = int(img.shape[0]), int(img.shape[1])
    new_width = ((size/height) * width)
    resized_img = cv2.resize(img, 
                              (new_width, 640),
                              interpolation = cv2.INTER_AREA)
    
    input_blob = cv2.dnn.blobFromImage(resized_img,
                                       1.0,
                                       (new_width, size),
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
    output = cv2.resize(output, None, fx=upscale, fy=upscale interpolation = INTER_LINEAR)
    
    return output

cap = cv2.VideoCapture('./video/run.mp4') #Any example video

while True:
    ret, frame = cap.read()
    cv2.imshow('Neural Style transfer Video',
               neural_style_transfer(frame,
                                     'starry_night',
                                     320,
                                     2))
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()    
    
    
    
