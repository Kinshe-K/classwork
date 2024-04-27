import cv2
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance
import numpy as np

import pandas as pd
import glob 
import progressbar

def cleanup_img(path):
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    return(cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 181, 11))

from sklearn.decomposition import PCA
import pickle

with (open("primary_component_model.pcl", "rb")) as pca_pickle:
    print_pca = pickle.load(pca_pickle)

def printbitstring(pimg):
    pca_arr = print_pca.transform(cleanup_img(pimg).flatten().reshape(1, -1))
    return(''.join(format(abs(int(b)), 'b').zfill(16) for b in pca_arr[0,:8]))

def hamming_distance(print1, print2):
    return sum(c1 != c2 for c1, c2 in zip(print1, print2))

def compare_prints(print1, print2):
    p1 = printbitstring(print1)
    p2 = printbitstring(print2)
    return 1 - (hamming_distance(p1, p2) / len(p1))

