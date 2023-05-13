import numpy as np
import cv2


def nothing(x):
    pass


cv2.namedWindow('image')
cv2.createTrackbar('Binar', 'image', -1, 1, nothing)
cv2.createTrackbar('Binar2', 'image', -4, 1, nothing)
cv2.createTrackbar('Binar3', 'image', -1, 1, nothing)
# img = cv2.imread('../Opencv基础/Calibration_ZhangZhengyou_Method-master/jibian.jpg', 1)
img = cv2.imread('../Wind_gap/big.jpg', 1)
# mat_inter = np.array([[473.0621, 0, 322.243],[0, 472.490, 246.847],[0, 0, 1]])
# coff_dis = np.array([-0.107, -0.126, -0.0000954, 0.00189, 0.1719])
while(1):
    b = cv2.getTrackbarPos('Binar', 'image')
    g = cv2.getTrackbarPos('Binar2', 'image')
    r = cv2.getTrackbarPos('Binar3', 'image')
    mat_inter = np.array([[3234.789, 0, 2012], [0, 3221.795, 1518], [0, 0, 1]])
    coff_dis = np.array([b, r, g, 0.0075, -0.1121])
    w, h = (9, 9)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis, (w, h), 0, (w, h))  # 自由比例参数

    image_test = cv2.undistort(img, mat_inter, coff_dis, None, newcameramtx)
    cv2.imshow('image', cv2.resize(image_test, None, fx=0.3, fy=0.3))
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
