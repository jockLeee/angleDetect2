import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology
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

def image_delete(image, save_dir, mid_point):  # 图像多余区域去除并拆分
    for i in range(0, len(mid_point) - 1):
        un7 = image[0:image.shape[0], mid_point[i]:mid_point[i + 1]]
        lst = []
        un7 = cv.erode(un7, kernel=np.ones((3, 1), np.uint8))
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
        un7 = cv.dilate(un7, kernel=np.ones((3, 1), np.uint8))
        un7 = get_skeleton(un7)
        un7 = cv.erode(un7, kernel=np.ones((2, 1), np.uint8))
        cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un7)

def image_combin(path, mid_point):  # 图像拼接
    # path = './black_white_black_white/'
    img_out = cv.imread(path + str(0) + ".jpg")
    num = len(mid_point) - 1
    for i in range(1, num):
        img_tmp = cv.imread(path + str(i) + ".jpg")
        img_out = np.concatenate((img_out, img_tmp), axis=1)
    return img_out

def black_white_black_white(image, white_to_black, black_to_white):
    # white_to_black: [39, 79, 123, 163, 210, 254, 296, 339, 380, 428, 469, 513, 554, 596]
    # black_to_white: [15, 59, 101, 142, 187, 230, 274, 316, 358, 404, 446, 487, 532, 574, 615]
    mid_point = []
    save_dir = "./black_white_black_white"
    path = './black_white_black_white/'
    mid_point.append(0)
    for i in range(0, len(white_to_black)):
        temp = (black_to_white[i + 1] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    image_delete(image, save_dir, mid_point)  # 调用
    img_out = image_combin(path, mid_point)  # 调用
    # cv.imwrite(save_dir + '6.jpg', img_out)
    cv.imshow('com', img_out)
    # cv.imshow('image', image)
    # cv.waitKey(0)
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
    image_delete(image, save_dir, mid_point)  # 调用
    img_out = image_combin(path, mid_point)  # 调用
    # cv.imwrite(save_dir + '1103.jpg', img_out)
    cv.imshow('com', img_out)
    cv.imshow('img', image)
    cv.waitKey(0)
    return mid_point

def white_black_white_black(image, white_to_black, black_to_white):
    # white_to_black: [16, 53, 92, 129, 166, 205, 242, 281, 318, 357, 394, 434, 473, 511, 549, 586, 628]
    # black_to_white: [22, 61, 99, 138, 175, 214, 252, 290, 329, 367, 407, 444, 481, 521, 559, 599]
    mid_point = []
    save_dir = "./white_black_white_black"
    path = './white_black_white_black/'
    mid_point.append(0)
    for i in range(0, len(black_to_white)):
        temp = (black_to_white[i] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    image_delete(image, save_dir, mid_point)  # 调用
    img_out = image_combin(path, mid_point)  # 调用
    # cv.imwrite(save_dir + '1.jpg', img_out)
    cv.imshow('com', img_out)
    cv.imshow('img', image)
    cv.waitKey(0)
    return mid_point

def white_black_black_white(image, white_to_black, black_to_white):
    # white_to_black= [10, 54, 93, 134, 176, 217, 258, 299, 340, 382, 422, 466, 507, 554, 594]
    # black_to_white= [30, 73, 115, 155, 197, 238, 281, 322, 363, 404, 447, 492, 534, 582, 635]
    mid_point = []
    save_dir = "./white_black_black_white"
    path = './white_black_black_white/'
    mid_point.append(0)
    for i in range(0, len(black_to_white)):
        temp = (black_to_white[i] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    image_delete(image, save_dir, mid_point)  # 调用
    img_out = image_combin(path, mid_point)  # 调用
    # cv.imwrite(save_dir + '3-3.jpg', img_out)
    cv.imshow('image', image)
    cv.imshow('com', img_out)
    cv.waitKey(0)
    return mid_point

def average_Grayval(img):
    sum = 0
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            sum += img[i, j]
    average = sum / img.shape[0] / img.shape[1] / 2
    return average

def get_strengthen_gray(image, precision, low_value):
    save_dir = ".\\save"
    path = "./save/"
    x = int(image.shape[0] / precision)  # 分块长度
    for i in range(0, precision + 1):
        un8 = np.zeros((x, image.shape[1], 3), np.uint8)
        un9 = image[i * x:(i + 1) * x, 0:image.shape[1]]
        sum = 0
        for k in range(0, un9.shape[0]):
            for j in range(0, un9.shape[1]):
                sum += un9[k][j]
        average_gray = sum / (un9.shape[0] * un9.shape[1])
        if average_gray < low_value:
            un7 = cv.equalizeHist(un9)
        else:
            un7 = un9
        cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un7)
    save_path = path + str(0) + ".jpg"
    img_out = cv.imread(save_path)
    num = precision + 1
    for i in range(1, num):
        save_path = path + str(i) + ".jpg"
        img_tmp = cv.imread(save_path)
        img_out = np.concatenate((img_out, img_tmp), axis=0)
    img_out = cv.cvtColor(img_out, cv.COLOR_BGR2GRAY)
    # cv.imwrite("%d.jpg" % (num), img_out)
    return img_out

# oimg = cv.imread('../Wind_angle/test/3-3.jpg', 1)
# img = cv.imread('bin_1103.png', 0)
# image1 = get_vertical_project(img)

oimg = cv.imread('../pap2/2.jpg', 1)
img = cv.imread('../pap2/2.jpg', 0)
# img = cv.imread('../Wind_angle/test/0402.jpg', 0)
img = get_strengthen_gray(img, 79, 200)
_, temp = cv.threshold(img, average_Grayval(img), 255, cv.THRESH_BINARY)

image1 = get_vertical_project(temp)
cv.imshow('temp1', temp)

cut_num = int(img.shape[0] / 10 * 9)
# image1[cut_num - 1:cut_num, 0:img.shape[1]] = 0
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
    print(1)
    lst = white_black_white_black(temp, white_to_black, black_to_white)  # 1
elif len(white_to_black) < len(black_to_white):
    print(2)
    lst = black_white_black_white(temp, white_to_black, black_to_white)  # 6
else:
    if white_to_black[0] < black_to_white[0] and white_to_black[len(white_to_black) - 1] < black_to_white[
        len(black_to_white) - 1]:
        print(3)
        lst = white_black_black_white(temp, white_to_black, black_to_white)  # 3-3

    elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
        len(black_to_white) - 1]:
        print(4)
        lst = black_white_white_black(temp, white_to_black, black_to_white)  # 1103

    else:
        print('the technology has some problem, please check it and try again!')

for i in lst:
    cv.line(oimg, (i, 0), (i, oimg.shape[0]), (255, 255, 0), 2)
cv.imshow('oimg', oimg)
# cv.imwrite('oimg.jpg',oimg)
cv.imshow('img', img)
cv.imshow('shutou', image1)
# cv.imwrite('shutou.jpg',image1)
cv.waitKey(0)
