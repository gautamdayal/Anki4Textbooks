import cv2
import pytesseract
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def load(somethinglol):
    img = cv2.imread(somethinglol)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def create_boxes(img):
    Height, Width, null = img.shape
    boxes = pytesseract.image_to_boxes(img)
    L = []
    for b in boxes.splitlines():
        print(b)
        b = b.split(' ')
        print(b)
        try:
            L.append([int(x) for x in b[1:5]])
        except:
            print("lol fuck you")
        x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        cv2.rectangle(img, (x, Height - y), (w, Height - h), (0, 0, 225), 1)
    return L


print(create_boxes(load('biotest.png')))

avg_coords = []
for coord in create_boxes(load('epictest.png')):
    x = (coord[0] + coord[2]) / 2
    y = (coord[1] + coord[3]) / 2
    avg_coords.append([x, y])
    plt.scatter(x, y, color='red')

print(avg_coords)

kmean=KMeans(n_clusters=3)
kmean.fit(avg_coords)
centers = kmean.cluster_centers_

for center in centers:
    plt.scatter(center[0], center[1], marker='*', color='black')
plt.show()
