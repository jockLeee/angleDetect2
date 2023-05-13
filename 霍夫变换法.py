# import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology
import math
def get_skeleton(binary):
    binary[binary == 255] = 1
    skeleton0 = morphology.skeletonize(binary)  # 骨架提取
    # skel, distance = morphology.medial_axis(binary, return_distance=True)
    # skeleton0 = distance * skel
    skeleton = skeleton0.astype(np.uint8) * 255
    return skeleton

def skeleton_extraction(un8):
    skel = np.zeros(un8.shape, np.uint8)
    erode = np.zeros(un8.shape, np.uint8)
    temp = np.zeros(un8.shape, np.uint8)
    i = 0
    while (cv.countNonZero(un8) != 0):
        erode = cv.erode(un8, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        temp = cv.dilate(erode, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
        temp = cv.subtract(un8, temp)
        skel = cv.bitwise_or(skel, temp)
        un8 = erode.copy()
        i += 1
    return skel

def get_vertical_project(img):  # 竖直投影
    h = img.shape[0]
    w = img.shape[1]
    project_img = np.zeros(shape=(img.shape), dtype=np.uint8) + 255
    for j in range(w):
        num = 0
        for i in range(h):
            if img[i][j] == 0:
                num += 1
        for k in range(num):
            project_img[h - 1 - k][j] = 0
    return project_img


def get_vertical_project_2(img):  # 竖直
    lst = []
    h = img.shape[0]
    w = img.shape[1]
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


def image_delete(image, save_dir, mid_point):  # 图像多余区域去除并拆分
    every_line_angle = []

    for i in range(0, len(mid_point) - 1):
        print("epoch %d ---------------------------------------------------" % (i+1))
        un7 = image[0:image.shape[0], mid_point[i]:mid_point[i + 1]]
        lst = []
        un7 = cv.erode(un7, kernel=np.ones((3, 3), np.uint8))
        _, thresh = cv.threshold(un7, 150, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for x in range(len(contours)):
            area = cv.contourArea(contours[x])
            lst.append(area)
        toupl = contours[:lst.index(max(lst))] + contours[lst.index(max(lst)) + 1:]
        for l in range(len(toupl)):
            x, y, w, h = cv.boundingRect(toupl[l])
            for k in range(x, x + w):
                for j in range(y, y + h):
                    un7[j, k] = 0
        un7 = cv.dilate(un7, kernel=np.ones((3, 5), np.uint8))
        # un7 = get_skeleton(un7)
        # cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un7)
        skel = get_skeleton(un7)
        # # # # # # # # # # # # # # # # 霍 夫 变 换 法 # # # # # # # # # # # # # # # # # # # # # #
        lines = cv.HoughLinesP(skel, rho=1, theta=np.pi / 180, threshold=100, minLineLength=200, maxLineGap=100)
        for line in lines:
            for x1, y1, x2, y2 in line:
                if abs(y2 - y1) > skel.shape[0] * 3 / 4:
                    angle = math.degrees(math.atan(((y2 - y1)/(x2 - x1))))
                    every_line_angle.append(round(90 + angle, 2))
    return  every_line_angle    # 最大投影的角度


def image_combin(path, mid_point):  # 图像拼接
    # path = './black_white_black_white/'
    img_out = cv.imread(path + str(0) + ".jpg")
    num = len(mid_point) - 1
    for i in range(1, num):
        img_tmp = cv.imread(path + str(i) + ".jpg")
        img_out = np.concatenate((img_out, img_tmp), axis=1)
    return img_out


def black_white_black_white(image, white_to_black, black_to_white):
    mid_point = []
    save_dir = "./black_white_black_white"
    path = './black_white_black_white/'
    mid_point.append(0)
    for i in range(0, len(white_to_black)):
        temp = (black_to_white[i + 1] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)    # 调用
    print(angle)
    # img_out = image_combin(path, mid_point)    # 调用
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
    return mid_point

def black_white_white_black(image, white_to_black, black_to_white):
    mid_point = []
    save_dir = "./black_white_white_black"
    path = './black_white_white_black/'
    mid_point.append(0)
    for i in range(0, len(white_to_black) - 1):
        temp = (black_to_white[i + 1] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)    # 调用
    print(angle)
    img_out = image_combin(path, mid_point)    # 调用
    cv.imshow('com', img_out)
    cv.imshow('img', image)
    cv.waitKey(0)
    return mid_point

def white_black_white_black(image, white_to_black, black_to_white):
    mid_point = []
    save_dir = "./white_black_white_black"
    path = './white_black_white_black/'
    mid_point.append(0)
    for i in range(0, len(black_to_white)):
        temp = (black_to_white[i] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)    # 调用
    print(angle)
    # img_out = image_combin(path, mid_point)     # 调用
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
    return mid_point

def white_black_black_white(image, white_to_black, black_to_white):
    mid_point = []
    save_dir = "./white_black_black_white"
    path = './white_black_black_white/'
    mid_point.append(0)
    for i in range(0, len(black_to_white)):
        temp = (black_to_white[i] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)  # 调用
    print(angle)
    img_out = image_combin(path, mid_point)  # 调用
    cv.imshow('com', img_out)
    cv.imshow('img', image)
    cv.waitKey(0)
    return mid_point


img = cv.imread('bin_3-3.png', 0)
oimg = cv.imread('../Wind_angle/test/3-3.jpg', 1)
image1 = get_vertical_project(img)
cut_num = int(img.shape[0] / 5 * 4)
image1[cut_num - 1:cut_num, 0:640] = 0
white_to_black = []
black_to_white = []
for i in range(0, img.shape[1] - 1):
    if i == image1.shape[1] - 1:
        break
    else:
        if image1[cut_num, i + 1] < image1[cut_num, i]:
            white_to_black.append(i)
        elif image1[cut_num, i + 1] > image1[cut_num, i]:
            black_to_white.append(i)
print('white_to_black:', white_to_black)
print('black_to_white:', black_to_white)

lst = []
if len(white_to_black) > len(black_to_white):
    lst = white_black_white_black(img, white_to_black, black_to_white)  # 1
elif len(white_to_black) < len(black_to_white):
    lst = black_white_black_white(img, white_to_black, black_to_white)  # 6
else:
    if white_to_black[0] < black_to_white[0] and white_to_black[len(white_to_black) - 1] < black_to_white[
        len(black_to_white) - 1]:
        lst = white_black_black_white(img, white_to_black, black_to_white)  # 3-3

    elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
        len(black_to_white) - 1]:
        lst = black_white_white_black(img, white_to_black, black_to_white)  # 1103
    else:
        print('the technology has some problem, please check it and try again!')
