import numpy as np
import cv2
img=cv2.imread("path_img.png",0)
img_real=cv2.imread("path_img.png")
t_lower = 20  # Lower Threshold
t_upper = 60  # Upper threshold
  
# Applying the Canny Edge filter
thresh1 = cv2.inRange(img, 20, 60)
thresh2 = cv2.inRange(img, 0, 40)
kernel = np.array([[1,1,1,1,0,1,1,1,1,1,1],
                   [1,1,1,2,1,1,1,1,1,1,1],
                   [1,1,1,0,1,0,1,1,1,1,1],
                   [1,1,1,0,1,0,1,1,2,1,1],
                   [1,1,1,1,1,0,1,1,1,1,1],
                   [1,1,1,1,1,0,0,1,1,1,1],
                   [1,1,1,1,1,1,1,1,1,1,1]],np.uint8)
kernel1=np.ones((10,10))
opening = cv2.dilate(thresh2, kernel1,iterations = 1)
opening = cv2.erode(opening, kernel,iterations = 4)
#opening = cv2.erode(opening,np.ones((18,18),np.uint8),iterations = 1)
thresh2=cv2.bitwise_not(thresh2)
detected_lanes=cv2.bitwise_and(thresh2,opening)
detected_lanes=cv2.bitwise_and(detected_lanes,thresh2)
img_real[detected_lanes==255]=(0,255,0)
"""edge_şerit = cv2.Canny(img, 100, 700)
edge = cv2.Canny(img, t_lower, t_upper)
"""

cv2.imshow('original', img_real)
cv2.imshow('s', opening)
cv2.imshow('edge', detected_lanes)
cv2.waitKey(0)
cv2.destroyAllWindows()


