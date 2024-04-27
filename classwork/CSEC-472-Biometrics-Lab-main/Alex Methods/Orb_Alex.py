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

def compare(detector, image1, image2,plot):
   fng1, kp1, des1 = detect(detector, image1)
   fng2, kp2, des2 = detect(detector, image2)
  
   bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
   matches = bf.match(des1, des2)
   
   count = 0 
   for i in matches:
      count += 1
   
   if plot == 1:
      matches = sorted(matches, key = lambda x: x.distance) 
      fng_matches = cv2.drawMatches(fng1, kp1, fng2, kp2, matches[:50], fng2, flags=2)
      plt.figure(figsize=(16, 16))
      plt.imshow(fng_matches)
      plt.show()

   if count > 300:
      print('Finger Prints', image1, '&', image2,'Match with', count , 'Matched Points')
      return True
   else:
      return False

def train_test():
   dTest = r'/home/student/Downloads/test/'
   dTrain = r'/home/student/Downloads/train/'
   tested = 0
   pos = 0
   neg = 0
   falsePos = 0
   falseNeg = 0
   for trainFile in os.listdir(dTrain):
      if trainFile.endswith('.png'):
         og = os.path.join(dTrain, trainFile)
         new = os.path.join(dTest, trainFile)
         shutil.copyfile(og,new)
         
   for trainFile in os.listdir(dTrain):
      if trainFile.endswith('.png'):
         ogTrainFinger = Image.open(os.path.join(dTrain,trainFile))
         trainFinger = os.path.join(dTrain,trainFile)
         for testFile in os.listdir(dTest):
            if testFile.endswith('.png'):
               testFinger = os.path.join(dTest,testFile)
               orb = cv2.ORB_create()
               results = compare(orb, trainFinger, testFinger,0)
               if results == True:
                  pos += 1
                  os.remove(testFinger)
                  if trainFile != testFile:
                     falsePos += 1
                     print('FALSE POSITIVE!!!!!!!!!')
                  break
               else:
                  neg += 1        
                  if trainFile == testFile:
                     falseNeg += 1       
                     print('FALSE NEGATIVE!!!!!!!!!!!!!')
               tested += 1
   print('Total Tested: ',tested)
   print('Positive Matches: ',pos)
   print('Negative: ',neg)
   print('Total False Positives: ',falsePos)
   print('Total False Negatives: ' , falseNeg)
               
if __name__ == '__main__':
   train_test()
   
