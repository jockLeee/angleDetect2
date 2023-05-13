# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import leastsq
# plt.rcParams['font.sans-serif'] = ['SimHei']
#
# Xi = np.array([1, 2, 3])
# Yi = np.array([2, 3, 3])
#
# def error(p, x, y):
#     k, b = p
#     return k * x + b - y
# p0 = [1, 1]
#
# Para = leastsq(error, p0, args=(Xi, Yi))
# k, b = Para[0]
# print("k=", k, "b=", b)
# print("cost：" + str(Para[1]))
# print("求解的拟合直线为:")
# print("y=" + str(round(k, 2)) + "x+" + str(round(b, 2)))
#
# plt.figure(figsize=(8, 6))
# plt.scatter(Xi, Yi, color="green", label="样本数据", linewidth=2)
# x = np.array([1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
# y = k * x + b
# plt.plot(x, y, color="red", label="拟合直线", linewidth=2)
# plt.title('y={}+{}x'.format(b, k))
# plt.legend(loc='lower right')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import time
import math
t1 = time.time()

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
    return k[0], b[0]

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

def point_to_line(key_points, k, b):
    temp_list = []
    point_lst = key_points.copy()
    for i in (key_points):
        temp = round(abs(k * i[0] - i[1] + b) / math.sqrt(1 + k ** 2), 2)
        temp_list.append(temp)
        if temp > 7:
            # print(i)
            point_lst.remove(i)
    # print('points', point_lst)
    print('templist',temp_list)
    # print(np.median(temp_list), np.mean(temp_list))
    return point_lst

image = cv.imread('white_black_black_white/0.jpg', 0)
# image = cv.imread('black_white_white_black/2.jpg', 0)
# image = cv.imread('black_white_white_black/8.jpg', 0)
image = cv.erode(image, kernel=np.ones((2, 1), np.uint8))
un8 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
un9 = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
un7 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
# print(image.shape[1])  # 46
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

print('points_1:', point)

data_len = len(x_1)
k_1, b_1 = leastsquare(x_1, y_1, data_len)
# k, b = Least_squares(x, y, len(x))

print('第一次结果: ', 'k=', k_1, 'b=', b_1)

# def detect_distance_point_to_line(x0, y0, k, b):
#     return abs(k * x0 - y0 + b) / math.sqrt(1 + k ** 2)
# print(detect_distance_point_to_line(13, 17, 37.134, 15))

angle1 = math.degrees(math.atan(k_1))
print('第一次拟合角度 =', angle1)

x1 = (0 - b_1) / k_1
x2 = (image.shape[0] - b_1) / k_1
# print(int(x1), 0, int(x2), image.shape[0])

plt.subplot(1, 6, 1)
plt.imshow(image, plt.cm.gray)

plt.subplot(1, 6, 2)
cv.line(un8, (int(x1), 0), (int(x2), image.shape[0]), (0, 255, 0), 1)
plt.imshow(un8, plt.cm.gray)

plt.subplot(1, 6, 3)
cv.line(image, (int(x1), 0), (int(x2), image.shape[0]), (255, 0, 0), 1)
plt.imshow(image, plt.cm.gray)

# plt.plot(x, y, 'ro', lw=1, markersize=2)
# plt.plot(x1, 0, 'ro', linewidth=2, markersize=2)
# plt.plot(x2, image.shape[0], 'ro', linewidth=2, markersize=2)

changed_point = point_to_line(point, k_1, b_1)  # 一次优化后的点
for i in changed_point:
    un9[i[1]][i[0]] = 255

plt.subplot(1, 6, 4)
plt.imshow(un9, plt.cm.gray)

# print(type(un9),un9.shape)
set_x.clear()
set_y.clear()
point.clear()
for j in range(un9.shape[1]):
    for i in range(un9.shape[0]):
        if un9[i][j] >= 250:
            set_x.append(j)
            set_y.append(i)
            point.append([j, i])
        else:
            continue
x_2 = np.array(set_x)
y_2 = np.array(set_y)
print('points_2:', point)
data_len = len(x_2)
k_2, b_2 = leastsquare(x_2, y_2, data_len)
print('第2次结果: ', 'k=', k_2, 'b=', b_2)
angle2 = math.degrees(math.atan(k_2))
print('第2次拟合角度 =', angle2)
x1 = (0 - b_2) / k_2
x2 = (image.shape[0] - b_2) / k_2

plt.subplot(1, 6, 5)
cv.line(un7, (int(x1), 0), (int(x2), image.shape[0]), (0, 255, 0), 1)
plt.imshow(un7, plt.cm.gray)

plt.subplot(1, 6, 6)
cv.line(un9, (int(x1), 0), (int(x2), image.shape[0]), (255, 255, 255), 1)
plt.imshow(un9, plt.cm.gray)



plt.show()
