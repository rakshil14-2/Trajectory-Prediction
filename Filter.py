# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 09:16:02 2022

@author: Rakshil Modi
"""

import cv2
from detector import Detector
import numpy as np


kf = cv2.KalmanFilter(4,2)
kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32())
kf.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32())

def Estimate(cordx,cordy):
    measured = np.array([[np.float32(cordx)],[np.float32(cordy)]])
    kf.correct(measured)
    predict = kf.predict()
    x,y = int(predict[0]),int(predict[1])  
    return x,y



c = cv2.VideoCapture(0)
c.set(cv2.CAP_PROP_BUFFERSIZE, 2)

f = 1/30
f_MS = int(f*1000)
od = Detector()



while True:
    ret,frame = c.read()
    if ret is False:
        break
        
    
    object_bbox = od.detect(frame)
    x,y,x2,y2 = object_bbox
    cx = int((x+x2)/2)
    cy = int((y+y2)/2)
    
    predicted = Estimate(cx, cy)
    
    
    cv2.circle(frame, (cx,cy), 20,(0,0,255),-1)
    cv2.circle(frame, (predicted[0],predicted[1]), 20,(255,0,0),4)
    cv2.imshow('Frame',frame)
    key = cv2.waitKey(100)
    if key == 'A':
        break;
        
