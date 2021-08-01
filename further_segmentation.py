import time
s = time.time()
import networkx as nx
import pandas as pd
import cv2
import numpy as np
from numpy.linalg import norm
import re
import math
import pytesseract
import matplotlib.pyplot as plt
import itertools
import sklearn.metrics
from numberExtraction import *
import tkinter as tk
import os
print(f"Imports completed in {time.time() - s}s")

APP_FOLDER = r'C:\Users\prana\OneDrive\Desktop\Projects and Coursework\Anki4Textbook\segmentation_output'

totalFiles = 0

for base, dirs, files in os.walk(APP_FOLDER):
    for Files in files:
        totalFiles += 1


print('Total number of files: ',totalFiles)


def segment(path, current_index):
    img = load(path)
    questionCoords = select_numbers(img, current_index)
    h, w = img.shape
    imgNum = 0
    for quad in cropCoords(questionCoords, w, h):
        temp = img[quad[0]:quad[1], quad[2]:quad[3]]
        cv2.imshow(str(imgNum), temp)
        cv2.imwrite('segmentation_output/output{}.png'.format(str(imgNum)), temp)
        imgNum += 1
    cv2.waitKey(0)

# iterations = int(input("Segmentation Interations: ")) - 1
# heirarchy = str(input("Giu heirarchy da")) #format 1,a,i,I
segment('images/test.png', 'a')

# for i in range(iterations):

#     for l in range(totalFiles):
#         segment('segmentation_output/output{}'.format(l))
