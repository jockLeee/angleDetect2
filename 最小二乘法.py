import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology
import math
import time


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

def image_leastsquare(image):
    set_x = []
    set_y = []
    point = []
    for j in range(image.shape[1]):
        for i in range(image.shape[0]):
            if image[i][j] >= 200:
                set_x.append(j)
                set_y.append(i)
                point.append([j, i])
            else:
                continue

    x_1 = np.array(set_x)
    y_1 = np.array(set_y)
    x_sum = np.zeros(1)
    x_squ = np.zeros(1)
    y_sum = np.zeros(1)
    xy_sum = np.zeros(1)
    for i in np.arange(len(x_1)):
        x_squ += x_1[i] ** 2
        x_sum += x_1[i]
        y_sum += y_1[i]
        xy_sum += x_1[i] * y_1[i]
    fenmu = (len(x_1) * x_squ - x_sum ** 2)
    k = (len(x_1) * xy_sum - x_sum * y_sum) / fenmu
    b = (x_squ * y_sum - x_sum * xy_sum) / fenmu
    return k[0], b[0], point

def point_to_line(k, b, key_points):
    point_lst = key_points.copy()
    for i in (key_points):
        temp = round(abs(k * i[0] - i[1] + b) / math.sqrt(1 + k ** 2), 2)
        if temp > 7:
            point_lst.remove(i)
    return point_lst

def decorate(image):
    un9 = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    k_1, b_1, point_1 = image_leastsquare(image)
    changed_point = point_to_line(k_1, b_1, point_1)  # 一次优化后的点
    for i in changed_point:
        un9[i[1]][i[0]] = 255
    k_2, b_2, point_2 = image_leastsquare(un9)
    return k_2, b_2

def detect_direction(img1):
    flag = 0
    for j in range(img1.shape[1]):
        for i in range(img1.shape[0]):
            if img1[i][j] >= 150:
                flag += 1
    return flag

def image_delete(image, save_dir, mid_point):  # 图像多余区域去除并拆分
    every_line_angle = []
    for i in range(0, len(mid_point) - 1):
        print("epoch %d ---------------------------------------------------" % (i + 1))
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
        un7 = get_skeleton(un7)
        # cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un7)
        # 最 小 二 乘 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        skel = get_skeleton(un7)
        skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))

        # set_x = []
        # set_y = []
        # for j in range(skel.shape[1]):
        #     for i in range(skel.shape[0]):
        #         if skel[i][j] >= 200:
        #             set_x.append(j)
        #             set_y.append(i)
        #         else:
        #             continue
        # x = np.array(set_x)
        # y = np.array(set_y)
        # data_len = len(x)
        # k, b = Least_squares(x, y, data_len)

        # [2.45, 2.36, 2.9, 2.29, 3.22, 3.3, 2.21, 2.29, 2.93, 1.96, 1.95, 3.13]
        # [2.45, 2.36, 2.9, 2.29, 3.22, 3.3, 2.21, 2.29, 2.93, 1.96, 1.95, 3.13]
        k, b, points = image_leastsquare(skel)  # 算法优化前的结果

        # k, b = decorate(skel)   # 算法优化，2次拟合
        # [2.45, 2.36, 2.39, 2.29, 1.97, 2.09, 2.21, 2.29, 2.05, 1.96, 1.95, 1.9]
        angle2 = math.degrees(math.atan(k))
        every_line_angle.append(round(90 + angle2, 2))

    return every_line_angle


def leastsquare(lstx, lsty, lens):
    data_len = lens
    x_sum = np.zeros(1)
    x_squ = np.zeros(1)
    y_sum = np.zeros(1)
    xy_sum = np.zeros(1)
    for i in np.arange(data_len):
        x_squ += lstx[i] ** 2
        x_sum += lstx[i]
        y_sum += lsty[i]
        xy_sum += lstx[i] * lsty[i]
    fenmu = (data_len * x_squ - x_sum ** 2)
    k = (data_len * xy_sum - x_sum * y_sum) / fenmu
    b = (x_squ * y_sum - x_sum * xy_sum) / fenmu
    return k, b


def Least_squares(x, y, lens):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(lens):
        k = (x[i] - x_) * (y[i] - y_)
        m += k
        p = np.square(x[i] - x_)
        n = n + p
    a = m / n
    b = y_ - a * x_
    return a, b


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
    angle = image_delete(image, save_dir, mid_point)  # 调用
    print(angle)
    # img_out = image_combin(path, mid_point)  # 调用
    # # cv.imwrite(save_dir + '6.jpg', img_out)
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
    angle = image_delete(image, save_dir, mid_point)  # 调用
    print(angle)
    # img_out = image_combin(path, mid_point)  # 调用
    # # cv.imwrite(save_dir + '1103.jpg', img_out)
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
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
    angle = image_delete(image, save_dir, mid_point)  # 调用
    print(angle)
    # img_out = image_combin(path, mid_point)  # 调用
    # # cv.imwrite(save_dir + '6.jpg', img_out)
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
    # img_out = image_combin(path, mid_point)  # 调用
    # # cv.imwrite(save_dir + '6.jpg', img_out)
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
    return mid_point

def average_Grayval(img):
    sum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum += img[i, j]
    average = sum / img.shape[0] / img.shape[1] / 2
    return average

t1 = time.time()
img = cv.imread('bin_1103.png', 0)
# img = cv.imread('../pap2/5.jpg', 0)
# average = average_Grayval(img)
# _, temp = cv.threshold(img, average, 255, cv.THRESH_BINARY)
# oimg = cv.imread('../Wind_angle/test/3-3.jpg', 1)
image1 = get_vertical_project(img)
cut_num = int(img.shape[0] / 5 * 4)
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
        # [1.54, 0.5, 0.79, 0.83, 0.87, 0.95, 1.31, 1.33, 1.07, 1.19, 0.92, 1.17, 1.54, 1.64, 2.23, 1.03]
        # [1.54, 0.5, 0.79, 0.83, 0.87, 0.95, 1.31, 1.33, 1.07, 1.19, 0.92, 1.17, 1.54, 1.64, 2.23, 1.03]
        # [0.36, 0.5, 0.79, 0.83, 0.87, 0.95, 1.25, 1.33, 1.07, 1.19, 0.92, 1.15, 1.54, 1.64, 2.23, 1.03]
    elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
        len(black_to_white) - 1]:
        lst = black_white_white_black(img, white_to_black, black_to_white)  # 1103

    else:
        print('the technology has some problem, please check it and try again!')
print('time used:',time.time() - t1)
