import cv2
import pytesseract
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from scipy.cluster.vq import kmeans2
import numpy as np
from numberExtraction import *

# pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe' (Windoge moment)


# Returns an array in the format [bottom right x, bottom right y, top left x, top left y]
def create_boxes(img):
    Height, Width = img.shape
    boxes = pytesseract.image_to_boxes(img)
    L = []
    for b in boxes.splitlines():
        b = b.split(' ')
        try:
            # Channge to b[0:5] if you need the actual character in the bounding box
            L.append([int(x) for x in b[1:5]])
        except:
            print("lol")
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img, (x, Height - y), (w, Height - h), (0, 0, 225), 1)
    return L

image = load('images/mathtest2.png')

# Some code to visualize each box and the clusters
avg_coords = []
for coord in create_boxes(image):
    # Average the x and y values to get a single point (easier to visualize)
    x = (coord[0] + coord[2]) / 2
    y = (coord[1] + coord[3]) / 2
    avg_coords.append([x, y])
    # Plot character coordinates
    plt.scatter(x, y, color='gold')

avg_coords = np.array(avg_coords)
centroids = np.array(clusterCoords(image))

# Kmeans 2
centroid, label = kmeans2(avg_coords, centroids, minit='matrix')
# plt.subplot(121)
plt.title('Centroids after initializing based on questions')
plt.scatter(avg_coords[:,0], avg_coords[:,1])
plt.plot(centroid[:, 0], centroid[:, 1], 'k*')
plt.show()
# # Kmeans 3
# centroid, label = kmeans2(avg_coords, 5, minit='points')
# plt.subplot(122)
# plt.title('Centroids after random allocation')
# plt.scatter(avg_coords[:,0], avg_coords[:,1])
# plt.plot(centroid[:, 0], centroid[:, 1], 'k*')
# plt.show()
