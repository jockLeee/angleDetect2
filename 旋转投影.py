import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage import morphology
import time


def get_skeleton(binary):
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)  # 骨架提取
    # skel, distance = morphology.medial_axis(binary, return_distance=True)
    # skeleton0 = distance * skel
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton


def get_vertical_project(img):  # 竖直
    lst = []
    h = img.shape[0]
    w = img.shape[1]
    #
    project_img = np.zeros(shape=(img.shape), dtype=np.uint8) + 255
    for j in range(w):
        num = 0
        for i in range(h):
            if img[i][j] < 255:
                num += 1
        lst.append(num)
        for k in range(num):
            project_img[h - 1 - k][j] = 0
    return project_img, max(lst)


def skeleton_extraction(un7):
    skel = np.zeros(un7.shape, np.uint8)
    erode = np.zeros(un7.shape, np.uint8)
    temp = np.zeros(un7.shape, np.uint8)
    i = 0
    while (cv.countNonZero(un7) != 0):
        erode = cv.erode(un7, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        temp = cv.dilate(erode, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        temp = cv.subtract(un7, temp)
        skel = cv.bitwise_or(skel, temp)
        un7 = erode.copy()
        i += 1
    return skel


def detect_direction(img1):
    flag = 0
    for j in range(img1.shape[1]):
        for i in range(img1.shape[0]):
            if img1[i][j] >= 150:
                flag += 1
    return flag


str1 = './liantongquyu/14.jpg'
t1 = time.time()
un7 = cv.imread(str1, 0)
ori_img = cv.imread(str1, 0)
plt.subplot(2, 4, 1)
plt.imshow(ori_img, cmap=plt.cm.gray)
plt.title('ori_img')
# -----------------------------------------------------------------------------------#
lst = []
un7 = cv.erode(un7, kernel=np.ones((3, 3), np.uint8))
contours, hierarchy = cv.findContours(un7, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
for x in range(len(contours)):
    area = cv.contourArea(contours[x])
    lst.append(area)
toupl = contours[:lst.index(max(lst))] + contours[lst.index(max(lst)) + 1:]
for l in range(len(toupl)):
    x, y, w, h = cv.boundingRect(toupl[l])
    for k in range(x, x + w):
        for j in range(y, y + h):
            un7[j, k] = 0
un7 = cv.dilate(un7, kernel=np.ones((3, 3), np.uint8))
# -----------------------------------------------------------------------------------#
plt.subplot(2, 4, 2)
plt.imshow(un7, cmap=plt.cm.gray)
plt.title('un7')

for j in range(un7.shape[1]):
    for i in range(un7.shape[0]):
        if un7[i][j] >= 200:
            un7[i][j] = 255
        else:
            un7[i][j] = 0
skel = get_skeleton(un7)
skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))
# skel = skeleton_extraction(un7)
H, W = skel.shape
# print(W, H) # 48 480

skel = np.pad(skel, ((int(W / 2), int(W / 2)), (int(H / 2), int(H / 2))), 'constant', constant_values=((0, 0), (0, 0)))
plt.subplot(2, 4, 3)
plt.title('skel')
plt.imshow(skel, cmap=plt.cm.gray)
rows, cols = skel.shape  # 高度 宽度
print(rows, cols)
img1 = skel[0:int(rows / 2), 0:int(cols / 2)]
img2 = skel[0:int(rows / 2), int(cols / 2):cols]
img3 = skel[int(rows / 2):rows, 0:int(cols / 2)]
img4 = skel[int(rows / 2):rows, int(cols / 2):cols]
# plt.subplot(2, 4, 5)
# plt.title('img1')
# plt.imshow(img1, cmap=plt.cm.gray)
# plt.subplot(2, 4, 6)
# plt.title('img2')
# plt.imshow(img2, cmap=plt.cm.gray)
# plt.subplot(2, 4, 7)
# plt.title('img3')
# plt.imshow(img3, cmap=plt.cm.gray)
img1_flag = detect_direction(img1)
img2_flag = detect_direction(img2)
img3_flag = detect_direction(img3)
img4_flag = detect_direction(img4)
# print(img1_flag, img2_flag, img3_flag, img4_flag)
if (img1_flag + img4_flag) > (img2_flag + img3_flag):
    print('left')
else:
    print('right')

m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), -2.1, 1)
dst = cv.warpAffine(skel, m, (cols, rows))
dst = 255 - dst
plt.subplot(2, 4, 4)
plt.title('dst')
dst = cv.erode(dst, kernel=np.ones((2, 2), np.uint8))
plt.imshow(dst, cmap=plt.cm.gray)

vel, max_y = get_vertical_project(dst)
plt.subplot(2, 4, 8)
plt.title(str(max_y))
plt.imshow(vel, plt.cm.gray)
t2 = time.time()
print(t2 - t1)
plt.show()
