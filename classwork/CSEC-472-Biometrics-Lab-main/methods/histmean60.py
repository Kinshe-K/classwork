import skimage

import numpy as np # computations in the mse command
from PIL import Image, ImageChops, ImageStat  # Image Library

import glob # Import file paths
import mahotas
import pylab
import random # shuffle array
import math

def compare_prints(imageA, imageB):
    imA = Image.open(imageA)
    imB = Image.open(imageB)
    h = ImageChops.difference(imA, imB).histogram()
    val = math.sqrt(sum(h*(i**2) for i, h in enumerate(h)) / (float(imA.size[0]) * imB.size[1]))
    difference = val/100
    return(1-difference)