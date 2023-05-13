import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology
import math
import time


class image_process(object):
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

    def leastsquare(self, lstx, lsty, lens):
        x_sum = np.zeros(1)
        x_squ = np.zeros(1)
        y_sum = np.zeros(1)
        xy_sum = np.zeros(1)
        for i in np.arange(lens):
            x_squ += lstx[i] ** 2
            x_sum += lstx[i]
            y_sum += lsty[i]
            xy_sum += lstx[i] * lsty[i]
        fenmu = (lens * x_squ - x_sum ** 2)
        k = (lens * xy_sum - x_sum * y_sum) / fenmu
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
        x1 = range(0, len(lst), 1)
        # plt.Figure()
        plt.ylim(-5, 5)  # 显示y轴范围
        plt.plot(x1, lst, color='r', marker='o', linestyle='-', linewidth=2)
        plt.scatter(x1, lst, s=1, c='r')  # 散点图点大小
        plt.xticks(fontsize=20)  # x轴刻度字体大小
        plt.yticks(fontsize=20)  # y轴刻度字体大小
        t2 = time.time()
        print('time used:', round(t2 - t1, 1), 's')
        time.sleep(0.001)
        plt.show()

    def image_leastsquare(self, image):
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

    def point_to_line(self, k, b, key_points):
        point_lst = key_points.copy()
        for i in (key_points):
            temp = round(abs(k * i[0] - i[1] + b) / math.sqrt(1 + k ** 2), 2)
            if temp > 7:
                point_lst.remove(i)
        return point_lst

    def decorate(self, image):
        un9 = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        k_1, b_1, point_1 = self.image_leastsquare(image)
        changed_point = self.point_to_line(k_1, b_1, point_1)  # 一次优化后的点
        for i in changed_point:
            un9[i[1]][i[0]] = 255
        k_2, b_2, point_2 = self.image_leastsquare(un9)
        return k_2, b_2

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
            # # # # # # # # # # # # # # 最 小 二 乘 # # # # # # # # # # # # # # # # # # # # # # # # #
            skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))

            # k, b, points = self.image_leastsquare(skel)  # 算法优化前的结果
            k, b = self.decorate(skel)  # 算法优化，2次拟合

            angle2 = math.degrees(math.atan(k))
            every_line_angle.append(round(90 + angle2, 2))
            time.sleep(0.001)
        return every_line_angle

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
        self.plot_data(angle, t1)
        time.sleep(0.001)

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

    def average_Grayval(self, img):
        sum = 0
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                sum += img[i, j]
        average = sum / img.shape[0] / img.shape[1] / 2
        return average

    def main_process(self, img):
        t1 = time.time()
        time.sleep(0.001)
        average = self.average_Grayval(img)
        _, temp = cv.threshold(img, average, 255, cv.THRESH_BINARY)
        image1 = self.get_vertical_project(temp)
        cut_num = int(img.shape[0] / 10 * 9)
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

        if len(white_to_black) > len(black_to_white):
            self.white_black_white_black(temp, white_to_black, black_to_white, t1)  # 1
        elif len(white_to_black) < len(black_to_white):
            self.black_white_black_white(temp, white_to_black, black_to_white, t1)  # 6
        else:
            if white_to_black[0] < black_to_white[0] and white_to_black[len(white_to_black) - 1] < black_to_white[
                len(black_to_white) - 1]:
                self.white_black_black_white(temp, white_to_black, black_to_white, t1)  # 3-3

            elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
                len(black_to_white) - 1]:
                self.black_white_white_black(temp, white_to_black, black_to_white, t1)  # 1103
            else:
                print('the technology has some problem, please check it and try again!')


if __name__ == '__main__':
    # img = cv.imread('bin_3-3.png', 0)
    img = cv.imread('../pap2/2.jpg', 0)
    process = image_process()
    process.main_process(img)
