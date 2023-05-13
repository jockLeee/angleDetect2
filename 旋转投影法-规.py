import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import morphology
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

    def detect_direction(self, img1):
        flag = 0
        for j in range(img1.shape[1]):
            for i in range(img1.shape[0]):
                if img1[i][j] >= 150:
                    flag += 1
        return flag

    def plot_data(self, lst, t1):
        x1 = list(range(len(lst)))
        plt.Figure()
        plt.ylim(-5, 5)
        plt.plot(x1, lst, color='r', marker='o', linestyle='-', linewidth=2)
        plt.scatter(x1, lst, s=1, c='r')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
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
            # # # # # # # # # # # # # # 旋 转 投 影 # # # # # # # # # # # # # # # # # # # #
            skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))
            # H, W = skel.shape
            # skel = np.pad(skel, ((int(W / 2), int(W / 2)), (int(H / 2), int(H / 2))), 'constant',
            #               constant_values=((0, 0), (0, 0)))
            rows, cols = skel.shape  # 高度 宽度
            lst_ax = []
            lst_angle = []

            img1_flag = self.detect_direction(skel[0:int(rows / 2), 0:int(cols / 2)])
            img2_flag = self.detect_direction(skel[0:int(rows / 2), int(cols / 2):cols])
            img3_flag = self.detect_direction(skel[int(rows / 2):rows, 0:int(cols / 2)])
            img4_flag = self.detect_direction(skel[int(rows / 2):rows, int(cols / 2):cols])

            if (img1_flag + img4_flag) > (img2_flag + img3_flag):
                print('left')
                for ii in range(-40, 0, 1):  # -10,11,1 i/2
                    print('angle:', (ii / 10))
                    lst_ax.append(ii / 10)
                    m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii / 10, 1)
                    dst = cv.warpAffine(skel, m, (cols, rows))
                    dst = 255 - dst
                    vel, max_y = self.get_vertical_project_2(dst)
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
                    vel, max_y = self.get_vertical_project_2(dst)
                    lst_angle.append(max_y)
                    print("max black value: ", (max_y))
                    i += 1
                print(lst_ax[lst_angle.index(max(lst_angle))])
                every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])
        time.sleep(0.001)
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
                print('the photo has some problem, please check it and try again!')


if __name__ == '__main__':
    # img = cv.imread('bin_1103.png', 0)
    img = cv.imread('../pap2/2.jpg', 0)
    process = image_process()
    process.main_process(img)
