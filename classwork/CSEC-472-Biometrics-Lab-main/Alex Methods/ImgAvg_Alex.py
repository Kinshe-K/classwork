# Alex Rosse
# CSEC 472 - Lab 3
# Group 2
# Method 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import shutil
from PIL import Image, ImageStat

def compute(img):
   img1 = Image.open(img)
   widith, height = img1.size

   total = 0
   for i in range(0,widith):
      for j in range(0,height):
         total += img1.getpixel((i,j))

   mean = total / (widith * height)
   return mean

def compare(img1, img2):
   img1_avg = round(compute(img1),4)
   img2_avg = round(compute(img2),4)

   if img1_avg == img2_avg:
      print('Finger Prints', img1, '&', img2,'Match')
      return True
   else:
      return False    
      
def train_test():
   dTest = r'test'
   dTrain = r'train'
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
