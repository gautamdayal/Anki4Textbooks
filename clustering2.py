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
        print(b)
        b = b.split(' ')
        print(b)
        try:
            # Channge to b[0:5] if you need the actual character in the bounding box
            L.append([int(x) for x in b[1:5]])
        except:
            print("lol")
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img, (x, Height - y), (w, Height - h), (0, 0, 225), 1)
    return L

filename = 'images/econtest1.png'
image = load(filename)
# Some code to visualize each box and the clusters
avg_coords = []
for coord in create_boxes(image):
    # Average the x and y values to get a single point (easier to visualize)
    x = (coord[0] + coord[2]) / 2
    y = (coord[1] + coord[3]) / 2
    avg_coords.append([x, y])
    # Plot character coordinates
#    plt.scatter(x, y, color='red')

# print(avg_coords)

avg_coords = np.array(avg_coords)


centroids = clusterCoords(load(filename))
print(centroids)
# centroids = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
centroids = np.array(centroids)

""" Cluster Visualization
km = KMeans(init=centroids, n_clusters=5)
km_clustering = km.fit(avg_coords)
plt.figure(1)
plt.subplot(221)
plt.title('K-Means Visualization')
plt.scatter(avg_coords[:,0], avg_coords[:,1], c=km_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
"""

# Kmeans 2
centroid, label = kmeans2(avg_coords, centroids, minit='matrix')
plt.subplot(121)
plt.title('Centroids after manual')
plt.scatter(avg_coords[:,0], avg_coords[:,1])
plt.plot(centroid[:, 0], centroid[:, 1], 'k*')

# Kmeans 3
centroid, label = kmeans2(avg_coords, 5, minit='points')
plt.subplot(122)
plt.title('Centroids after random allocation')
plt.scatter(avg_coords[:,0], avg_coords[:,1])
plt.plot(centroid[:, 0], centroid[:, 1], 'k*')


""" Spectral Clustering
sc = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=0)
sc_clustering = sc.fit(avg_coords)
plt.subplot(224)
plt.title("Spectral")
plt.scatter(avg_coords[:,0], avg_coords[:,1], c=sc_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
"""

plt.show()
