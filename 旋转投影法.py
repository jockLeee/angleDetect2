# import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology
import time

# class image_process(object):
#     def get_skeleton(self, binary):
#         binary[binary == 255] = 1
#         skeleton0 = morphology.skeletonize(binary)  # 骨架提取
#         skeleton = skeleton0.astype(np.uint8) * 255
#         time.sleep(0.001)
#         return skeleton
#
#     def get_vertical_project(self, img):  # 竖直投影
#         h = img.shape[0]
#         w = img.shape[1]
#         project_img = np.zeros(shape=(img.shape), dtype=np.uint8) + 255
#         for j in range(w):
#             num = 0
#             for i in range(h):
#                 if img[i][j] == 0:
#                     num += 1
#             for k in range(num):
#                 project_img[h - 1 - k][j] = 0
#         time.sleep(0.001)
#         return project_img
#
#     def get_vertical_project_2(self, img):  # 竖直
#         lst = []
#         h = img.shape[0]
#         w = img.shape[1]
#         project_img = np.zeros(shape=(img.shape), dtype=np.uint8) + 255
#         for j in range(w):
#             num = 0
#             for i in range(h):
#                 if img[i][j] < 255:
#                     num += 1
#             lst.append(num)
#             for k in range(num):
#                 project_img[h - 1 - k][j] = 0
#         time.sleep(0.001)
#         return project_img, max(lst)
#
#     def detect_direction(self, img1):
#         flag = 0
#         for j in range(img1.shape[1]):
#             for i in range(img1.shape[0]):
#                 if img1[i][j] >= 150:
#                     flag += 1
#         return flag
#
#     def plot_data(self, lst, t1):
#         x1 = list(range(len(lst)))
#         plt.Figure()
#         plt.ylim(-5, 5)  # 显示y轴范围
#         plt.plot(x1, lst, color='r', marker='o', linestyle='-', linewidth=2)
#         plt.scatter(x1, lst, s=1, c='r')  # 散点图点大小
#         plt.xticks(fontsize=20)  # x轴刻度字体大小
#         plt.yticks(fontsize=20)  # y轴刻度字体大小
#         t2 = time.time()
#         print('time used:', round(t2 - t1, 1), 's')
#         time.sleep(0.001)
#         plt.show()
#
#     def image_delete(self, image, save_dir, mid_point):  # 图像多余区域去除并拆分
#         every_line_angle = []
#         for i in range(0, len(mid_point) - 1):
#             print("epoch %d ---------------------------------------------------" % (i + 1))
#             un7 = image[0:image.shape[0], mid_point[i]:mid_point[i + 1]]
#             lst = []
#             un7 = cv.erode(un7, kernel=np.ones((3, 3), np.uint8))
#             _, thresh = cv.threshold(un7, 150, 255, 0)
#             contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#             for x in range(len(contours)):
#                 area = cv.contourArea(contours[x])
#                 lst.append(area)
#             toupl = contours[:lst.index(max(lst))] + contours[lst.index(max(lst)) + 1:]
#             for l in range(len(toupl)):
#                 x, y, w, h = cv.boundingRect(toupl[l])
#                 for k in range(x, x + w):
#                     for j in range(y, y + h):
#                         un7[j, k] = 0
#             un7 = cv.dilate(un7, kernel=np.ones((3, 5), np.uint8))
#             skel = self.get_skeleton(un7)
#             # # # # # # # # # # # # # # 旋 转 投 影 # # # # # # # # # # # # # # # # # # # #
#             skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))
#             # H, W = skel.shape
#             # skel = np.pad(skel, ((int(W / 2), int(W / 2)), (int(H / 2), int(H / 2))), 'constant',
#             #               constant_values=((0, 0), (0, 0)))
#             rows, cols = skel.shape  # 高度 宽度
#             img1_flag = self.detect_direction(skel[0:int(rows / 2), 0:int(cols / 2)])
#             img2_flag = self.detect_direction(skel[0:int(rows / 2), int(cols / 2):cols])
#             img3_flag = self.detect_direction(skel[int(rows / 2):rows, 0:int(cols / 2)])
#             img4_flag = self.detect_direction(skel[int(rows / 2):rows, int(cols / 2):cols])
#             lst_ax = []
#             lst_angle = []
#             if (img1_flag + img4_flag) > (img2_flag + img3_flag):
#                 print('left')
#                 for ii in range(-40, 0, 1):  # -10,11,1 i/2
#                     print('angle:', (ii / 10))
#                     lst_ax.append(ii / 10)
#                     m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii / 10, 1)
#                     dst = cv.warpAffine(skel, m, (cols, rows))
#                     dst = 255 - dst
#                     vel, max_y = self.get_vertical_project_2(dst)
#                     lst_angle.append(max_y)
#                     print("max black value: ", (max_y))
#                     i += 1
#                 print(lst_ax[lst_angle.index(max(lst_angle))])
#                 every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])
#             else:
#                 print('right')
#                 for ii in range(0, 31, 1):  # -10,11,1 i/2
#                     print('angle:', (ii / 10))
#                     lst_ax.append(ii / 10)
#                     m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii / 10, 1)
#                     dst = cv.warpAffine(skel, m, (cols, rows))
#                     dst = 255 - dst
#                     vel, max_y = self.get_vertical_project_2(dst)
#                     lst_angle.append(max_y)
#                     print("max black value: ", (max_y))
#                     i += 1
#                 print(lst_ax[lst_angle.index(max(lst_angle))])
#                 every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])
#         time.sleep(0.001)
#         return every_line_angle  # 最大投影的角度
#
#     def black_white_black_white(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         save_dir = "./black_white_black_white"
#         path = './black_white_black_white/'
#         mid_point.append(0)
#         for i in range(0, len(white_to_black)):
#             temp = (black_to_white[i + 1] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, save_dir, mid_point)  # 调用
#         print(angle)
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#
#
#     def black_white_white_black(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         save_dir = "./black_white_white_black"
#         path = './black_white_white_black/'
#         mid_point.append(0)
#         for i in range(0, len(white_to_black) - 1):
#             temp = (black_to_white[i + 1] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, save_dir, mid_point)  # 调用
#         print(angle)
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#
#
#     def white_black_white_black(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         save_dir = "./white_black_white_black"
#         path = './white_black_white_black/'
#         mid_point.append(0)
#         for i in range(0, len(black_to_white)):
#             temp = (black_to_white[i] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, save_dir, mid_point)  # 调用
#         print(angle)
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#
#
#     def white_black_black_white(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         save_dir = "./white_black_black_white"
#         path = './white_black_black_white/'
#         mid_point.append(0)
#         for i in range(0, len(black_to_white)):
#             temp = (black_to_white[i] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, save_dir, mid_point)  # 调用
#         print(angle)
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#
#
#     def main_process(self, img):
#         t1 = time.time()
#         time.sleep(0.001)
#         image1 = self.get_vertical_project(img)
#         cut_num = int(img.shape[0] / 5 * 4)
#         white_to_black = []
#         black_to_white = []
#         for i in range(0, img.shape[1] - 1):
#             if i == image1.shape[1] - 1:
#                 break
#             else:
#                 if image1[cut_num, i + 1] < image1[cut_num, i]:
#                     white_to_black.append(i)
#                 elif image1[cut_num, i + 1] > image1[cut_num, i]:
#                     black_to_white.append(i)
#         if len(white_to_black) > len(black_to_white):
#             self.white_black_white_black(img, white_to_black, black_to_white, t1)  # 1
#         elif len(white_to_black) < len(black_to_white):
#             self.black_white_black_white(img, white_to_black, black_to_white, t1)  # 6
#         else:
#             if white_to_black[0] < black_to_white[0] and white_to_black[len(white_to_black) - 1] < black_to_white[
#                 len(black_to_white) - 1]:
#                 self.white_black_black_white(img, white_to_black, black_to_white, t1)  # 3-3
#
#             elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
#                 len(black_to_white) - 1]:
#                 self.black_white_white_black(img, white_to_black, black_to_white, t1)  # 1103
#             else:
#                 print('the technology has some problem, please check it and try again!')
#
# if __name__ == '__main__':
#     img = cv.imread('bin_1103.png', 0)
#     process = image_process()
#     process.main_process(img)
#
#
#
#
#
#
#
#
#
#
#
#
#

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
        skel = get_skeleton(un7)
        # cv.imwrite(save_dir + '/' + '%d.jpg' % (i), un7)
        skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))
        # H, W = skel.shape
        # skel = np.pad(skel, ((int(W / 2), int(W / 2)), (int(H / 2), int(H / 2))), 'constant', constant_values=((0, 0), (0, 0)))
        rows, cols = skel.shape  # 高度 宽度
        lst_ax = []
        lst_angle = []

        # 判断初始倾斜方向   ##################    ##################    ##################    ##################
        img1_flag = detect_direction(skel[0:int(rows / 2), 0:int(cols / 2)])
        img2_flag = detect_direction(skel[0:int(rows / 2), int(cols / 2):cols])
        img3_flag = detect_direction(skel[int(rows / 2):rows, 0:int(cols / 2)])
        img4_flag = detect_direction(skel[int(rows / 2):rows, int(cols / 2):cols])
        if (img1_flag + img4_flag) > (img2_flag + img3_flag):
            print('left')
            for ii in range(-40, 0, 1):  # -10,11,1 i/2
                print('angle:', (ii / 10))
                lst_ax.append(ii / 10)
                m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii / 10, 1)
                dst = cv.warpAffine(skel, m, (cols, rows))
                dst = 255 - dst
                vel, max_y = get_vertical_project_2(dst)
                lst_angle.append(max_y)
                print("max black value: ", (max_y))
                i += 1
            print(lst_ax[lst_angle.index(max(lst_angle))])
            every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])
        else:
            print('right')
            for ii in range(0, 31, 1):  # -10,11,1 i/2
                print('angle:', (ii / 10))
                lst_ax.append(ii / 10)
                m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii / 10, 1)
                dst = cv.warpAffine(skel, m, (cols, rows))
                dst = 255 - dst
                vel, max_y = get_vertical_project_2(dst)
                lst_angle.append(max_y)
                print("max black value: ", (max_y))
                i += 1
            print(lst_ax[lst_angle.index(max(lst_angle))])
            every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])
        # ##################    ##################    ##################    ##################    ##################

        # for ii in range(-30, 31, 1):  # -10,11,1 i/2
        #     print('angle:', (ii / 10))
        #     lst_ax.append(ii / 10)
        #     m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii / 10, 1)
        #     dst = cv.warpAffine(skel, m, (cols, rows))
        #     dst = 255 - dst
        #     vel, max_y = get_vertical_project_2(dst)
        #     lst_angle.append(max_y)
        #     print("max black value: ", (max_y))
        #     i += 1
        # print(lst_ax[lst_angle.index(max(lst_angle))])
        # every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])

    return every_line_angle  # 最大投影的角度


def image_combin(path, mid_point):  # 图像拼接
    # path = './black_white_black_white/'
    img_out = cv.imread(path + str(0) + ".jpg")
    num = len(mid_point) - 1
    for i in range(1, num):
        img_tmp = cv.imread(path + str(i) + ".jpg")
        img_out = np.concatenate((img_out, img_tmp), axis=1)
    return img_out


def black_white_black_white(image, white_to_black, black_to_white, t1):
    mid_point = []
    save_dir = "./black_white_black_white"
    path = './black_white_black_white/'
    mid_point.append(0)
    for i in range(0, len(white_to_black)):
        temp = (black_to_white[i + 1] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)  # 调用
    t2 = time.time()
    print('angle:', angle, t2-t1)
    # img_out = image_combin(path, mid_point)    # 调用
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
    return mid_point

def black_white_white_black(image, white_to_black, black_to_white, t1):
    mid_point = []
    save_dir = "./black_white_white_black"
    path = './black_white_white_black/'
    mid_point.append(0)
    for i in range(0, len(white_to_black) - 1):
        temp = (black_to_white[i + 1] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)  # 调用
    t2 = time.time()
    print('angle:', angle, t2 - t1)
    # img_out = image_combin(path, mid_point)    # 调用
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
    return mid_point

def white_black_white_black(image, white_to_black, black_to_white, t1):
    mid_point = []
    save_dir = "./white_black_white_black"
    path = './white_black_white_black/'
    mid_point.append(0)
    for i in range(0, len(black_to_white)):
        temp = (black_to_white[i] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)  # 调用
    t2 = time.time()
    print('angle:', angle, t2 - t1)
    # img_out = image_combin(path, mid_point)     # 调用
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
    return mid_point

def white_black_black_white(image, white_to_black, black_to_white, t1):
    mid_point = []
    save_dir = "./white_black_black_white"
    path = './white_black_black_white/'
    mid_point.append(0)
    for i in range(0, len(black_to_white)):
        temp = (black_to_white[i] + white_to_black[i]) / 2
        mid_point.append(int(temp))
    mid_point.append(image.shape[1])
    angle = image_delete(image, save_dir, mid_point)  # 调用
    t2 = time.time()
    print('angle:', angle, t2 - t1)
    # img_out = image_combin(path, mid_point)  # 调用
    # cv.imshow('com', img_out)
    # cv.imshow('img', image)
    # cv.waitKey(0)
    return mid_point


if __name__ == '__main__':
    t1 = time.time()
    img = cv.imread('bin_1103.png', 0)
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
        lst = white_black_white_black(img, white_to_black, black_to_white, t1)  # 1
    elif len(white_to_black) < len(black_to_white):
        lst = black_white_black_white(img, white_to_black, black_to_white, t1)  # 6
    else:
        if white_to_black[0] < black_to_white[0] and white_to_black[len(white_to_black) - 1] < black_to_white[
            len(black_to_white) - 1]:
            lst = white_black_black_white(img, white_to_black, black_to_white, t1)  # 3-3

        elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
            len(black_to_white) - 1]:
            lst = black_white_white_black(img, white_to_black, black_to_white, t1)  # 1103

        else:
            print('the technology has some problem, please check it and try again!')

# for i in lst:
#     cv.line(oimg, (i, 0), (i, oimg.shape[0]), (255, 255, 0), 2)
# cv.imshow('oimg', oimg)
# cv.imshow('img', img)
# cv.imshow('shutou', image1)
# cv.waitKey(0)
# [2.4, 2.4, 2.4, 2.1, 1.9, 2.1, 2.2, 2.5, 2.0, 1.6, 1.8, 1.8] 非扩张
# [2.4, 2.4, 2.4, 2.1, 1.9, 2.1, 2.2, 2.5, 2.0, 1.6, 1.8, 1.8] 扩张


# [0.7, 0.9, 1.2, 0.9, 1.0, 0.6, 1.1, 0.8, 0.8, 0.3, 1.0, 0.4, 1.0, 0.5, 0.2] # 图6 扩充  不判断方向 374.4s
# [0.7, 0.9, 1.2, 0.9, 1.0, 0.6, 1.1, 0.8, 0.8, 0.3, 1.0, 0.4, 1.0, 0.5, 0.2] # 图6 非扩充 不判断方向 58.09s
# [0.7, 0.9, 1.2, 0.9, 1.0, 0.6, 1.1, 0.8, 0.8, -0.2, 1.0, 0.4, 1.0, 0.5, 0.2] # 图6 非扩充 判断方向  29s / 23.4s

# [2.4, 2.3, 2.4, 2.1, 1.9, 2.1, 2.2, 2.5, 2.0, 1.5, 1.8, 1.8] # 图1103 扩充 不判断方向  356.44s
# [2.4, 2.3, 2.4, 2.1, 1.9, 2.1, 2.2, 2.5, 2.0, 1.5, 1.8, 1.8] # 图1103 不扩充 不判断方向  45.02s
# [2.4, 2.3, 2.4, 2.1, 1.9, 2.1, 2.2, 2.5, 2.0, 1.5, 1.8, 1.8] # 图1103 不扩充 判断方向  28.68s / 29.03s / 18.27s