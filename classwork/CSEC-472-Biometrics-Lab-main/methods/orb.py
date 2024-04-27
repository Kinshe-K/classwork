# Alex Rosse
# CSEC 472 - Lab 3
# Group 2
# Method 1

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image

def detect(detector, image):
    finger = cv2.imread(image)
    finger = cv2.cvtColor(finger, cv2.COLOR_BGR2RGB)
    key_points, des = detector.detectAndCompute(finger, None)
    return finger, key_points, des

def compare_prints(image1, image2):
    detector = cv2.ORB_create() 
    fng1, kp1, des1 = detect(detector, image1)
    fng2, kp2, des2 = detect(detector, image2)
  
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
   
    count = 0 
    for i in matches:
        count += 1

    return count/500