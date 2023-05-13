from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2 as cv
import numpy as np
import math
from skimage import morphology
from PyQt5.QtWidgets import *
import time
from matplotlib import pyplot as plt
import PIL.Image as Image

# class image_process(object):
#     def __init__(self):
#         self.method_flag = 1
#
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
#         print(1)
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
#     def leastsquare(self, lstx, lsty, lens):
#         data_len = lens
#         x_sum = np.zeros(1)
#         x_squ = np.zeros(1)
#         y_sum = np.zeros(1)
#         xy_sum = np.zeros(1)
#         for i in np.arange(data_len):
#             x_squ += lstx[i] ** 2
#             x_sum += lstx[i]
#             y_sum += lsty[i]
#             xy_sum += lstx[i] * lsty[i]
#         fenmu = (data_len * x_squ - x_sum ** 2)
#         k = (data_len * xy_sum - x_sum * y_sum) / fenmu
#         b = (x_squ * y_sum - x_sum * xy_sum) / fenmu
#         time.sleep(0.001)
#         return k, b
#
#     def Least_squares(self, x, y, lens):
#         x_ = x.mean()
#         y_ = y.mean()
#         m = np.zeros(1)
#         n = np.zeros(1)
#         k = np.zeros(1)
#         p = np.zeros(1)
#         for i in np.arange(lens):
#             k = (x[i] - x_) * (y[i] - y_)
#             m += k
#             p = np.square(x[i] - x_)
#             n = n + p
#         a = m / n
#         b = y_ - a * x_
#         time.sleep(0.001)
#         return a, b
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
#     def image_delete(self, image, mid_point):  # 图像多余区域去除并拆分
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
#             if self.method_flag == 1:
#                 print("hough")
#                 lines = cv.HoughLinesP(skel, rho=1, theta=np.pi / 180, threshold=100, minLineLength=200, maxLineGap=100)
#                 for line in lines:
#                     for x1, y1, x2, y2 in line:
#                         if abs(y2 - y1) > skel.shape[0] * 3 / 4:
#                             angle = math.degrees(math.atan(((y2 - y1) / (x2 - x1))))
#                             every_line_angle.append(round(90 + angle, 2))
#             elif self.method_flag == 2:
#                 print('lesqu')
#                 skel = cv.erode(skel, kernel=np.ones((2, 1), np.uint8))
#                 set_x = []
#                 set_y = []
#                 for j in range(skel.shape[1]):
#                     for i in range(skel.shape[0]):
#                         if skel[i][j] >= 200:
#                             set_x.append(j)
#                             set_y.append(i)
#                         else:
#                             continue
#                 x = np.array(set_x)
#                 y = np.array(set_y)
#                 data_len = len(x)
#                 k, b = self.Least_squares(x, y, data_len)
#                 angle2 = math.degrees(math.atan(k))
#                 every_line_angle.append(round(90 + angle2, 1))
#                 time.sleep(0.001)
#             else:
#                 print("rot")
#                 H, W = skel.shape
#                 skel = np.pad(skel, ((int(W / 2), int(W / 2)), (int(H / 2), int(H / 2))), 'constant',
#                               constant_values=((0, 0), (0, 0)))
#                 rows, cols = skel.shape  # 高度 宽度
#                 lst_ax = []
#                 lst_angle = []
#                 for ii in range(-3, 3, 1):  # -10,11,1 i/2
#                     print('angle:' + str(ii))
#                     lst_ax.append(ii)
#                     m = cv.getRotationMatrix2D(((cols - 1) / 2, (rows - 1) / 2), ii, 1)
#                     dst = cv.warpAffine(skel, m, (cols, rows))
#                     dst = 255 - dst
#                     vel, max_y = self.get_vertical_project_2(dst)
#                     lst_angle.append(max_y)
#                     print("max black value: " + str(max_y))
#                     i += 1
#                 print(lst_ax[lst_angle.index(max(lst_angle))])
#                 every_line_angle.append(lst_ax[lst_angle.index(max(lst_angle))])
#         return every_line_angle  # 最大投影的角度
#
#     def black_white_black_white(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         mid_point.append(0)
#         for i in range(0, len(white_to_black)):
#             temp = (black_to_white[i + 1] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, mid_point)  # 调用
#         print(angle)
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#
#     def black_white_white_black(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         mid_point.append(0)
#         for i in range(0, len(white_to_black) - 1):
#             temp = (black_to_white[i + 1] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, mid_point)  # 调用
#         print(angle)
#         # self.mainwindow.print_info(str(angle))
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#         return mid_point
#
#     def white_black_white_black(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         mid_point.append(0)
#         for i in range(0, len(black_to_white)):
#             temp = (black_to_white[i] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, mid_point)  # 调用
#         print(angle)
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#
#     def white_black_black_white(self, image, white_to_black, black_to_white, t1):
#         mid_point = []
#         mid_point.append(0)
#         for i in range(0, len(black_to_white)):
#             temp = (black_to_white[i] + white_to_black[i]) / 2
#             mid_point.append(int(temp))
#         mid_point.append(image.shape[1])
#         angle = self.image_delete(image, mid_point)  # 调用
#         print(angle)
#         self.plot_data(angle, t1)
#         time.sleep(0.001)
#
#     def main_process(self, img):
#         t1 = time.time()
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
#         time.sleep(0.001)
#         if len(white_to_black) > len(black_to_white):
#             self.white_black_white_black(img, white_to_black, black_to_white, t1)  # 1
#         elif len(white_to_black) < len(black_to_white):
#             self.black_white_black_white(img, white_to_black, black_to_white, t1)  # 6
#         else:
#             if white_to_black[0] < black_to_white[0] and white_to_black[len(white_to_black) - 1] < black_to_white[
#                 len(black_to_white) - 1]:
#                 self.white_black_black_white(img, white_to_black, black_to_white, t1)  # 3-3
#             elif white_to_black[0] > black_to_white[0] and white_to_black[len(white_to_black) - 1] > black_to_white[
#                 len(black_to_white) - 1]:
#                 self.black_white_white_black(img, white_to_black, black_to_white, t1)  # 1103
#             else:
#                 print('the method has some problem, please check it and try again!')


class Ui_MainWindow(QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv.VideoCapture()
        self.CAM_NUM1 = 0
        self.CAM_NUM2 = 'http://admin:admin@10.61.117.7:8081'
        # self.image_process = image_process()
        self.method_flag = 1


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1380, 740)
        MainWindow.setMinimumSize(QtCore.QSize(1380, 740))
        MainWindow.setMaximumSize(QtCore.QSize(1380, 740))
        MainWindow.setStyleSheet("#MainWindow{background-color: rgb(235, 235, 235);}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(910, 320, 181, 61))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(20)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("#pushButton\n"
                                      "{background-color: rgb(214, 214, 214);}\n"
                                      "#pushButton:pressed\n"
                                      "{background-color: rgb(0, 190, 0);}")
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 0, 821, 691))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(48)
        self.label.setFont(font)
        self.label.setStyleSheet("#label{border: 2px dashed;\n"
                                 "background-color: rgb(235, 235, 235);\n"
                                 "border-color: rgb(255, 0, 0);}")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(850, 0, 511, 301))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(28)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("#label_2{border: 2px dashed;\n"
                                   "background-color: rgb(255, 255, 255);\n"
                                   "border-color: rgb(255, 0, 0);}")
        self.label_2.setObjectName("label_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1120, 320, 181, 61))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(20)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("#pushButton_2\n"
                                        "{background-color: rgb(214, 214, 214);}\n"
                                        "#pushButton_2:pressed\n"
                                        "{background-color: rgb(255, 0, 0);}")
        self.pushButton_2.setObjectName("pushButton_2")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(850, 550, 511, 141))
        self.textEdit_2.setObjectName("textEdit_2")
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(15)
        self.textEdit_2.setFont(font)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(870, 410, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(14)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setStyleSheet("#pushButton_3\n"
                                        "{background-color: rgb(214, 214, 214);}")
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1050, 410, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(14)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setStyleSheet("#pushButton_4\n"
                                        "{background-color: rgb(214, 214, 214);}")
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(1220, 410, 131, 41))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(14)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setStyleSheet("#pushButton_5\n"
                                        "{background-color: rgb(214, 214, 214);}")
        self.pushButton_5.setObjectName("pushButton_5")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(850, 310, 511, 81))
        self.label_3.setStyleSheet("#label_3\n"
                                   "{border: 1px solid;\n"
                                   "border-color: rgb(180,180 , 180);}")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(850, 400, 511, 71))
        self.label_4.setStyleSheet("#label_4\n"
                                   "{border: 1px solid;\n"
                                   "border-color: rgb(180,180 , 180);}")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(1170, 480, 191, 61))
        font = QtGui.QFont()
        font.setFamily("Palatino Linotype")
        font.setPointSize(18)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setStyleSheet("#pushButton_6\n"
                                        "{background-color: rgb(214, 214, 214);}")
        self.pushButton_6.setObjectName("pushButton_6")
        self.label_3.raise_()
        self.label_4.raise_()
        self.pushButton.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.pushButton_2.raise_()
        self.textEdit_2.raise_()
        self.pushButton_3.raise_()
        self.pushButton_4.raise_()
        self.pushButton_5.raise_()
        self.pushButton_6.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1380, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.textEdit_2.setReadOnly(1)
        self.pushButton.clicked.connect(self.button_open_camera1_clicked)
        self.pushButton_2.clicked.connect(self.button_close_camera1_clicked)
        self.pushButton_3.clicked.connect(self.method_Rotation)
        self.pushButton_4.clicked.connect(self.method_square)
        self.pushButton_5.clicked.connect(self.method_hough)
        self.pushButton_6.clicked.connect(self.main_)
        self.timer_camera.timeout.connect(self.show_camera1)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Open Camera"))
        self.label.setText(_translate("MainWindow", "Camrea"))
        self.label_2.setText(_translate("MainWindow", "Result show"))
        self.pushButton_2.setText(_translate("MainWindow", "Close Camera"))
        self.pushButton_3.setText(_translate("MainWindow", "Rot - method"))
        self.pushButton_4.setText(_translate("MainWindow", "squ - method"))
        self.pushButton_5.setText(_translate("MainWindow", "hough-method"))
        self.pushButton_6.setText(_translate("MainWindow", "Begin Detection"))

    def button_open_camera1_clicked(self):
        self.flag1 = self.cap.open(self.CAM_NUM1)
        if self.flag1 == False:
            msg = QtWidgets.QMessageBox.warning(self, 'warning',
                                                "Please check whether the camera is connected to the computer correctly.",
                                                buttons=QtWidgets.QMessageBox.Ok)
        else:
            self.pushButton.setEnabled(False)
            self.timer_camera.start(30)
            self.pushButton.setText('Running')
            ntm = time.localtime(time.time())
            self.textEdit_2.append("{}-{}-{}, {}:{}:{}   ".
                                   format(ntm[0], ntm[1], ntm[2], ntm[3], ntm[4], ntm[5])
                                   + 'Camera is open...')

    def button_close_camera1_clicked(self):
        self.pushButton.setEnabled(True)
        self.timer_camera.stop()
        self.cap.release()
        self.label_2.clear()
        self.label.setText('Camera')
        self.label_2.setText('Result show')
        self.pushButton.setText('Open Camera')
        ntm = time.localtime(time.time())
        self.textEdit_2.append("{}-{}-{}, {}:{}:{}   ".
                               format(ntm[0], ntm[1], ntm[2], ntm[3], ntm[4], ntm[5])
                               + 'Camera is close...')

    def show_camera1(self):
        flag1, image = self.cap.read()
        show = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # self.label.setScaledContents(True)  # 自适应大小

    def method_Rotation(self):
        self.method_flag = 3
        print(str(self.method_flag))
        self.print_info('Current method is rot...')

    def method_square(self):
        self.method_flag = 2
        print(str(self.method_flag))
        self.print_info('Current method is squ...')

    def method_hough(self):
        self.method_flag = 1
        print(str(self.method_flag))
        self.print_info('Current method is hough...')

    def print_info(self, text):
        ntm = time.localtime(time.time())
        self.textEdit_2.append("{}-{}-{}, {}:{}:{}   ".
                               format(ntm[0], ntm[1], ntm[2], ntm[3], ntm[4], ntm[5]) + str(text))

    def get_skeleton(self, binary):
        binary[binary == 255] = 1
        skeleton0 = morphology.skeletonize(binary)  # 骨架提取
        skeleton = skeleton0.astype(np.uint8) * 255
        time.sleep(0.001)
        return skeleton

    def get_vertical_project(self, img):  # 竖直投影
        h = img.shape[0]
        w = img.shape[1]
        print(1)
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

    def fig2data(self, fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tobytes())
        image = np.asarray(image)
        return image

    def plot_data(self, lst, t1):
        x1 = list(range(len(lst)))
        figure = plt.figure()
        plt.ylim(-5, 5)  # 显示y轴范围
        plt.plot(x1, lst, color='r', marker='o', linestyle='-', linewidth=2)
        plt.scatter(x1, lst, s=1, c='r')  # 散点图点大小
        plt.xticks(fontsize=20)  # x轴刻度字体大小
        plt.yticks(fontsize=20)  # y轴刻度字体大小

        time.sleep(0.001)
        # plt.show()
        image = self.fig2data(figure)
        # cv.imshow("image", image)
        # cv.waitKey(0)

        show = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label_2.setScaledContents(True)
        t2 = time.time()
        print('time used:', round(t2 - t1, 1), 's')

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
            cv.imwrite('mid_step_photo' + '/' + '%d.jpg' % (i), un7)
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
                # skel = np.pad(skel, ((int(W / 2), int(W / 2)), (int(H / 2), int(H / 2))), 'constant',
                #               constant_values=((0, 0), (0, 0)))
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
        self.print_info(every_line_angle)
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
        # self.mainwindow.print_info(str(angle))
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
        average = self.average_Grayval(img) / 2
        _, img = cv.threshold(img, average, 255, cv.THRESH_BINARY)
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

    def average_Grayval(self, img):
        sum = 0
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                sum += img[i, j]
        average = sum / img.shape[0] / img.shape[1]
        return average

    def main_(self):
        try:
            img_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开图片", "data/图片名字测试",
                                                                "All Files(*);;*.jpg;;*.png")
        except OSError as reason:
            print('文件打开出错！核对路径是否正确' + str(reason))
        else:
            # 判断图片是否为空

            if not img_name:
                QtWidgets.QMessageBox.warning(self, u"Warning", u"打开图片失败", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                img = cv.imread(img_name)
                print("img_name:", img_name)
                self.result = cv.cvtColor(img, cv.COLOR_BGR2BGRA)
                self.QtImg = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0], QtGui.QImage.Format_RGB32)
                self.label.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                # self.label.setScaledContents(True)
                time.sleep(0.1)
                self.main_process(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        time.sleep(0.1)
        self.print_info('image is loaded.')



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
