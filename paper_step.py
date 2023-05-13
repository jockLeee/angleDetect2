import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# img = cv.imread('bin_1103.png', 0)
oimg = cv.imread('white_black_black_white/3.jpg', 1)
image = cv.imread('white_black_black_white/3.jpg', 0)
# cv.imwrite('oimg.jpg', oimg)
# cv.imshow('img', img)
lines = cv.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=100, minLineLength=200, maxLineGap=200)
inf_x1 = []
inf_y1 = []
inf_x2 = []
inf_y2 = []
x1_sum = []
y1_sum = []
for line in lines:
    for x1, y1, x2, y2 in line:
        print(x1, y1, x2, y2)
        inf_x1.append(x1)
        inf_y1.append(y1)
        inf_x2.append(x2)
        inf_y2.append(y2)
        # x1_sum.append(x1,x2)
        # y1_sum.append(y1,y2)
        # cv.line(oimg, (x1, y1), (x2, y2), (255, 0, 0), 1)
        # cv.circle(oimg, (x1, y1), 3, (255, 0, 0), 1)
# cv.line(oimg, (min(inf_x1), min(inf_y1)), (max(inf_x2), max(inf_y2)), (250, 0, 0), 1)
print(int(max(inf_x1)), inf_x1, int(np.mean(inf_y1)), inf_y1)

cv.line(oimg, (lines[0][0][0], 0), (lines[0][0][2], image.shape[0]), (250, 0, 0), 1)
# cv.line(oimg, (min(inf_x1), min(inf_y1)), (max(inf_x2), max(inf_y2)), (250, 0, 0), 1)
cv.imshow('oimg', oimg)
plt.imshow(oimg, cmap=plt.cm.gray)
plt.show()
cv.waitKey(0)
