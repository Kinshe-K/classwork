import pandas as pd
import glob 
import progressbar
import cv2
from matplotlib import pyplot as plt
from PIL import Image,ImageEnhance
import numpy as np

def cleanup_img(path):
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    return(cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 181, 11))

def find_minutae(path, n=150, disp=False):
    img = cleanup_img(path)
    dst = cv2.cornerHarris(img,6,5,0.04)
    thresh = sorted(dst.flatten(), reverse=True)[n-1]
    minutae = np.array(np.where(dst > thresh))
    if disp: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img[dst>thresh]=[255, 127, 0]
        plt.figure(figsize=(12,12))
        plt.imshow(img)
    return(minutae)

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def compare_prints(path_a, path_b, thresh = 110, debug=False):
    m_a = find_minutae(path_a, disp=debug)
    m_b = find_minutae(path_b, disp=debug)
    c_a = np.expand_dims(centroidnp(m_a), 0)
    c_b = np.expand_dims(centroidnp(m_b), 0)
    ca_test = c_a[:, None]
    dists_a = np.linalg.norm(np.transpose(m_a) - c_a[:, None], axis=2)
    dists_b = np.linalg.norm(np.transpose(m_b) - c_b[:, None], axis=2)
    sort_dists = np.array(sorted(dists_a[0])) - np.array(sorted(dists_b[0]))
    #print(sort_dists)
    similarity = len(sort_dists[np.where(abs(sort_dists) < thresh)]) / len(sort_dists)
    return(similarity)


