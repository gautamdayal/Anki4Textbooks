import cv2
import pytesseract
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Loads image file with specified path
def load(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# Returns an array in the format [bottom right x, bottom right y, top left x, top left y]
def create_boxes(img):
    Height, Width, null = img.shape
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
            print("lol fuck you")
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img, (x, Height - y), (w, Height - h), (0, 0, 225), 1)
    return L

# Some code to visualize each box and the clusters
avg_coords = []
for coord in create_boxes(load('images/naive_test.png')):
    # Average the x and y values to get a single point (easier to visualize)
    x = (coord[0] + coord[2]) / 2
    y = (coord[1] + coord[3]) / 2
    avg_coords.append([x, y])
    # Plot character coordinates
    plt.scatter(x, y, color='red')

print(avg_coords)

kmean=KMeans(n_clusters=3)
kmean.fit(avg_coords)

# centers is an np array with the coordinates of each centroid
centers = kmean.cluster_centers_

# Plot centroids
for center in centers:
    plt.scatter(center[0], center[1], marker='*', color='black')
plt.show()
