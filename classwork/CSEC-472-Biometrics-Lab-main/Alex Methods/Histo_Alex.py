# Alex Rosse
# CSEC 472 - Lab 3
# Group 2
# Method 3

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil
from PIL import Image

def compute(img):
   finger = cv2.imread(img,0)
   hist = cv2.calcHist([finger],[0],None,[256],[0,256]) 
   return hist

def compare(img1, img2):
   img1_hist = compute(img1)
   img2_hist = compute(img2)
   corr = cv2.compareHist(img1_hist, img2_hist,cv2.HISTCMP_CORREL)
   
   if corr == 1.0:
      print('Finger Prints', img1, '&', img2,'Match')
      return True
   else:
      return False  

def train_test():
   dTest = r'NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_0'
   dTrain = r'NISTSpecialDatabase4GrayScaleImagesofFIGS/sd04/png_txt/figs_1'
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
               results = compare(trainFinger, testFinger)
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
