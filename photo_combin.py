import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def get_vertical_project(img):
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


def white_black_black_white(image, white_to_black, black_to_white):
    mid_point = []
    save_dir = "./photocombin"
    path = './photocombin/'
    mid_point.append(0)
    for i in range(0, len(black_to_white)):
        temp = (black_to_white[i] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    for i in range(0, len(mid_point) - 1):
        un7 = image[0:image.shape[0], mid_point[i]:mid_point[i + 1]]
        un9 = cv.erode(un7, kernel=np.ones((3, 5), np.uint8))

        _, thresh = cv.threshold(un9, 150, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for l in range(len(contours) - 1):
            min_rect = contours[l]
            x, y, w, h = cv.boundingRect(min_rect)
            print(x, y, w, h)
            for k in range(x, x + w):
                for j in range(y, y + h):
                    un9[j, k] = 0
        un9 = cv.dilate(un9, kernel=np.ones((3, 5), np.uint8))
        cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un9)

    img_out = cv.imread(path + str(0) + ".jpg")
    num = len(mid_point)-1
    for i in range(1, num):
        img_tmp = cv.imread(path + str(i) + ".jpg")
        img_out = np.concatenate((img_out, img_tmp), axis=1)
    cv.imwrite("%d.jpg" % (num), img_out)
    cv.imshow('com',img_out)
    cv.imshow('img',image)
    cv.waitKey(0)
    return mid_point

def black_white_white_black(image, white_to_black, black_to_white):
    # white_to_black: [52, 111, 159, 211, 262, 314, 365, 416, 464, 519, 568, 621]
    # black_to_white: [32, 90, 135, 191, 241, 292, 342, 393, 443, 497, 546, 597]
    mid_point = []
    save_dir = "./black_white_white_black"
    path = './black_white_white_black/'
    mid_point.append(0)
    for i in range(0, len(white_to_black) - 1):
        temp = (black_to_white[i + 1] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    for i in range(0, len(mid_point) - 1):
        un7 = image[0:image.shape[0], mid_point[i]:mid_point[i + 1]]
        un9 = cv.erode(un7, kernel=np.ones((3, 5), np.uint8))
        # ##
        _, thresh = cv.threshold(un9, 150, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        for l in range(len(contours) - 1):
            min_rect = contours[l]
            x, y, w, h = cv.boundingRect(min_rect)
            for k in range(x, x + w):
                for j in range(y, y + h):
                    un9[j, k] = 0
        un9 = cv.dilate(un9, kernel=np.ones((3, 5), np.uint8))
        ##
        cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un9)
    img_out = cv.imread(path + str(0) + ".jpg")
    num = len(mid_point) - 1
    for i in range(1, num):
        img_tmp = cv.imread(path + str(i) + ".jpg")
        img_out = np.concatenate((img_out, img_tmp), axis=1)
    # cv.imwrite("%d.jpg" % (num), img_out)
    cv.imshow('com', img_out)
    cv.imshow('img', image)
    cv.waitKey(0)
    return mid_point

img = cv.imread('bin_1103.png', 0)
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
# white_black_black_white(img, white_to_black, black_to_white)
black_white_white_black(img, white_to_black, black_to_white)  # 1103