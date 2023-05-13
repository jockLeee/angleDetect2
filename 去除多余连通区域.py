import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

str1 = './liantongquyu/14.jpg'
un7 = cv.imread(str1, 0)
un9 = cv.imread(str1, 1)
plt.subplot(1, 3, 1)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(un9, cmap=plt.cm.gray)
lst = []
un7 = cv.erode(un7, kernel=np.ones((3, 3), np.uint8))
contours, hierarchy = cv.findContours(un7, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for x in range(len(contours)):
    area = cv.contourArea(contours[x])
    lst.append(area)
print(contours)
for i in range(len(contours)):
    con = contours[i]
    x, y, w, h = cv.boundingRect(con)
    cv.rectangle(un9, (x, y), (x + w, y + h), (255, 225, 0), 1)
toupl = contours[:lst.index(max(lst))] + contours[lst.index(max(lst)) + 1:]
for l in range(len(toupl)):
    x, y, w, h = cv.boundingRect(toupl[l])
    for k in range(x, x + w):
        for j in range(y, y + h):
            un7[j, k] = 0
un7 = cv.dilate(un7, kernel=np.ones((3, 3), np.uint8))

# cv.drawContours(un7,contours,-1,(255,255,0),1)  # 轮廓标出

# max_rect = contours[lst.index(max(lst))]
# x, y, w, h = cv.boundingRect(max_rect)
# cv.rectangle(un9, (x, y), (x + w, y + h), (255, 225, 0), 1)


plt.subplot(1, 3, 2)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(un9, cmap=plt.cm.gray)
# plt.title('un9')
plt.subplot(1, 3, 3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(un7, cmap=plt.cm.gray)
# plt.title('un7')
plt.show()
