import numpy as np
import cv2 as cv

img = cv.imread('pills3.jpg', 0)
img = cv.medianBlur(img, 5)
cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
ret, thresh = cv.threshold(cimg, 50, 255, cv.THRESH_BINARY)

mask = np.zeros(cimg.shape, dtype=np.uint8)

orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(cimg,None)
# compute the descriptors with ORB
kp, des = orb.compute(cimg, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(cimg, kp, None, color=(0,255,0), flags=0)
cv.imshow('features', img2)
cv.waitKey(0)

circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, 1000, param1=50, param2=30, minRadius=500, maxRadius=600)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    # # draw the outer circle
    # cv.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # # draw the center of the circle
    # cv.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv.circle(mask,(i[0],i[1]),i[2]-200,(255,255,255),thickness=-1)


ROI = cv.bitwise_and(cimg, mask)

_, thresh = cv.threshold(mask, 1, 255, cv.THRESH_BINARY)

mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
x,y,w,h = cv.boundingRect(mask)

result = ROI[y:y+h,x:x+w]
mask = mask[y:y+h,x:x+w]
result[mask==0] = (255,255,255)

cv.imshow('cropped image', result)
var = np.var(result)
print(var)
cv.waitKey(0)
cv.destroyAllWindows()
