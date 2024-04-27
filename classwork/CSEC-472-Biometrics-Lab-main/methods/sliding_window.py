import pandas as pd
import glob 
import progressbar
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import skeletonize
from PIL import Image,ImageEnhance
from scipy import ndimage
import numpy as np

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return(mask)

def cleanup_img(path):
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    equ = cv2.equalizeHist(img)
    img = cv2.adaptiveThreshold(equ, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 181, 11)
    return(img)
    

def find_minutae(path, disp=False):
    img = cleanup_img(path)
    # Apply circular masks
    timg = img//255
    img = skeletonize(timg, method='lee')
    com = ndimage.measurements.center_of_mass(img)
    cmask = create_circular_mask(512,512,com,224)
    #result = img.copy()
    img[cmask==0] = 0
    if disp:
        plt.imshow(255-img, 'gray')
        plt.show()
    stepSize = 3
    (w_width, w_height) = (3, 3) # window size
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    coords = []
    for x in range(0, img.shape[1] - w_width , stepSize):
        for y in range(0, img.shape[0] - w_height, stepSize):
            window = img[x:x + w_width, y:y + w_height]
            winmean = np.mean(window)
            if winmean in (8/9, 1/9):
                #print(winmean)
                coords.append((x,y))
    coords = np.array(coords)
    coords_centr = centroidnp(coords)
    sort_coords = sorted(coords, key = lambda coord: np.linalg.norm(coord - coords_centr))
    return np.array(sort_coords[1:100])

def centroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def compare_prints(path_a, path_b, thresh = 7, debug=False):
    m_a = find_minutae(path_a, disp=debug)
    m_b = find_minutae(path_b, disp=debug)
    c_a = np.expand_dims(centroidnp(m_a), 0)
    c_b = np.expand_dims(centroidnp(m_b), 0)
    dists_a = np.linalg.norm(m_a - c_a[:, ], axis=1)
    dists_b = np.linalg.norm(m_b - c_b[:, ], axis=1)
    sort_dists = np.array(dists_a) - np.array(dists_b)
    #print(sort_dists)
    similarity = len(sort_dists[np.where(abs(sort_dists) < thresh)]) / len(sort_dists)
    return(similarity)

