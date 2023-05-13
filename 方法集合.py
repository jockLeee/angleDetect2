# import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology
import math
import time

class image_process(object):
    def __init__(self):
        self.method_flag = 1
    def get_skeleton(self, binary):
        binary[binary == 255] = 1
        skeleton0 = morphology.skeletonize(binary)  # 骨架提取
        skeleton = skeleton0.astype(np.uint8) * 255
        time.sleep(0.001)
        return skeleton

    def get_vertical_project(self, img):  # 竖直投影
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
        time.sleep(0.001)
        return project_img

    def get_vertical_project_2(self, img):  # 竖直
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
        time.sleep(0.001)
        return project_img, max(lst)

    def leastsquare(self, lstx, lsty, lens):
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
        time.sleep(0.001)
        return k, b

    def Least_squares(self, x, y, lens):
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
        time.sleep(0.001)
        return a, b

    def plot_data(self, lst, t1):
        x1 = list(range(len(lst)))
        plt.Figure()
        plt.ylim(-5, 5)  # 显示y轴范围
        plt.plot(x1, lst, color='r', marker='o', linestyle='-', linewidth=2)
        plt.scatter(x1, lst, s=1, c='r')  # 散点图点大小
        plt.xticks(fontsize=20)  # x轴刻度字体大小
        plt.yticks(fontsize=20)  # y轴刻度字体大小
        t2 = time.time()
        print('time used:', round(t2 - t1, 1), 's')
        time.sleep(0.001)
        plt.show()

    def image_delete(self, image, mid_point):  # 图像多余区域去除并拆分
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
            skel = self.get_skeleton(un7)
            if self.method_flag == 1:
                print("hough")
                lines = cv.HoughLinesP(skel, rho=1, theta=np.pi / 180, threshold=100, minLineLength=200, maxLineGap=100)
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        if abs(y2 - y1) > skel.shape[0] * 3 / 4:
                            angle = math.degrees(math.atan(((y2 - y1) / (x2 - x1))))
                            every_line_angle.append(round(90 + angle, 2))
            elif self.method_flag == 2:
                print('lesqu')
                skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))
                set_x = []
                set_y = []
                for j in range(skel.shape[1]):
                    for i in range(skel.shape[0]):
                        if skel[i][j] >= 200:
                            set_x.append(j)
                            set_y.append(i)
                        else:
                            continue
                x = np.array(set_x)
                y = np.array(set_y)
                data_len = len(x)
                k, b = self.Least_squares(x, y, data_len)
                angle2 = math.degrees(math.atan(k))
                every_line_angle.append(round(90 + angle2, 1))
                time.sleep(0.001)
            else:
                print("rot")
                H, W = skel.shape
                skel = np.pad(skel, ((int(W / 2), int(W / 2)), (int(H / 2), int(H / 2))), 'constant',
                              constant_values=((0, 0), (0, 0)))
                rows, cols = skel.shape  # 高度 宽度
                lst_ax = []
                lst_angle = []
                for ii in range(-3, 3, 1):  # -10,11,1 i/2
                    print('angle:' + str(ii))
                    lst_ax.append(ii)
                    m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii, 1)
                    dst = cv.warpAffine(skel, m, (cols, rows))
                    dst = 255 - dst
                    vel, max_y = self.get_vertical_project_2(dst)
                    lst_angle.append(max_y)
                    print("max black value: " + str(max_y))
                    i += 1
                print(lst_ax[lst_angle.index(max(lst_angle))])
                every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])
        return every_line_angle  # 最大投影的角度

    def black_white_black_white(self, image, white_to_black, black_to_white, t1):
        mid_point = []
        mid_point.append(0)
        for i in range(0, len(white_to_black)):
            temp = (black_to_white[i + 1] + white_to_black[i]) / 2
            mid_point.append(int(temp))
        mid_point.append(image.shape[1])
        angle = self.image_delete(image, mid_point)  # 调用
        print(angle)
        self.plot_data(angle, t1)
        time.sleep(0.001)

    def black_white_white_black(self, image, white_to_black, black_to_white, t1):
        mid_point = []
        mid_point.append(0)
        for i in range(0, len(white_to_black) - 1):
            temp = (black_to_white[i + 1] + white_to_black[i]) / 2
            mid_point.append(int(temp))
        mid_point.append(image.shape[1])
        angle = self.image_delete(image, mid_point)  # 调用
        print(angle)
        print(str(angle))
        self.plot_data(angle, t1)
        time.sleep(0.001)
        return mid_point

    def white_black_white_black(self, image, white_to_black, black_to_white, t1):
        mid_point = []
        mid_point.append(0)
        for i in range(0, len(black_to_white)):
            temp = (black_to_white[i] + white_to_black[i]) / 2
            mid_point.append(int(temp))
        mid_point.append(image.shape[1])
        angle = self.image_delete(image, mid_point)  # 调用
        print(angle)
        self.plot_data(angle, t1)
        time.sleep(0.001)

    def white_black_black_white(self, image, white_to_black, black_to_white, t1):
        mid_point = []
        mid_point.append(0)
        for i in range(0, len(black_to_white)):
            temp = (black_to_white[i] + white_to_black[i]) / 2
            mid_point.append(int(temp))
        mid_point.append(image.shape[1])
        angle = self.image_delete(image, mid_point)  # 调用
        print(angle)
        self.plot_data(angle, t1)
        time.sleep(0.001)

    def main_process(self, img):
        t1 = time.time()
        time.sleep(0.001)
        image1 = self.get_vertical_project(img)
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
        time.sleep(0.001)
        if len(white_to_black) > len(black_to_white):
            self.white_black_white_black(img, white_to_black, black_to_white, t1)  # 1
        elif len(white_to_black) < len(black_to_white):
            self.black_white_black_white(img, white_to_black, black_to_white, t1)  # 6
        else:
            if white_to_black[0] < black_to_white[0] and white_to_black[len(white_to_black) - 1] < black_to_white[
                len(black_to_white) - 1]:
                self.white_black_black_white(img, white_to_black, black_to_white, t1)  # 3-3
            elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
                len(black_to_white) - 1]:
                self.black_white_white_black(img, white_to_black, black_to_white, t1)  # 1103
            else:
                print('the method has some problem, please check it and try again!')


if __name__ == '__main__':
    img = cv.imread('bin_1103.png', 0)
    process = image_process()
    process.main_process(img)