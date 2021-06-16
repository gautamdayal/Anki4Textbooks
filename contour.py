import cv2

image = cv2.imread('images/numexample1.jpg', 0)
thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[0]
print(thresh)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c, num in zip(cnts, range(len(cnts))):
    x,y,w,h = cv2.boundingRect(c)
    ROI = 255 - thresh[y:y+h, x:x+w]
    cv2.imwrite('ROI_{}.png'.format(num), ROI)

cv2.imshow('thresh', 255 - thresh)
cv2.waitKey()
