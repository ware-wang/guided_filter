# -*- coding: utf-8 -*-
# @Time    : 2022/5/24 16:47
# @Author  : ware
# @File    : guided_filter_3.py

import cv2
import datetime


def guideFilter(I, p, r, epsilon):
    start = datetime.datetime.now()    # 计算运行时间
    I = I / 255
    p = p / 255
    win_size = (2 * r + 1, 2 * r + 1)
    mean_I = cv2.blur(I, win_size)  # 引导图像的均值
    mean_p = cv2.blur(p, win_size)  # 输入图像的均值
    mean_II = cv2.blur(I * I, win_size)  # 引导图像的方均
    mean_Ip = cv2.blur(I * p, win_size)  # 求协方差
    var_I = mean_II - mean_I * mean_I  # 输入图像的方差
    cov_Ip = mean_Ip - mean_I * mean_p  # 协方差
    a = cov_Ip / (var_I + epsilon)  # 系数a
    b = mean_p - a * mean_I  # 系数b
    mean_a = cv2.blur(a, win_size)
    mean_b = cv2.blur(b, win_size)
    q = mean_a * I + mean_b  # 求出输出图像
    end = datetime.datetime.now()
    print('guided_filter time cost is: %s seconds' % (end - start))
    return q


def fast_guided_filter(I, p, r, epsilon, s):
    start = datetime.datetime.now()          # 用于计算运行时间
    H, W, C = I.shape
    I_copy = I.copy()
    I = cv2.resize(I, [int(W / s), int(H / s)])     # 降采样
    p = cv2.resize(p, [int(W / s), int(H / s)])
    I = I / 255
    I_copy = I_copy / 255             # 先做归一化
    p = p / 255
    win_size = (2 * r + 1, 2 * r + 1)  # 卷积窗口大小
    mean_I = cv2.blur(I, win_size)  # 引导图像的均值
    mean_p = cv2.blur(p, win_size)  # 输入图像的均值
    mean_II = cv2.blur(I * I, win_size)  # 引导图像的方均
    mean_Ip = cv2.blur(I * p, win_size)  # 求协方差
    var_I = mean_II - mean_I * mean_I  # 输入图像的方差
    cov_Ip = mean_Ip - mean_I * mean_p  # 协方差
    a = cov_Ip / (var_I + epsilon)  # 系数a
    b = mean_p - a * mean_I  # 系数b
    mean_a = cv2.blur(a, win_size)  # 均值
    mean_b = cv2.blur(b, win_size)
    mean_a = cv2.resize(mean_a, [W, H])  # 上采样
    mean_b = cv2.resize(mean_b, [W, H])
    q = mean_a * I_copy + mean_b  # 求出输出图像
    end = datetime.datetime.now()
    print('fast_guided_filter time cost is: %s seconds' % (end - start))  # 输出运行时间
    return q


guide = cv2.imread('target.jpg')       # 引导图像
filter = cv2.imread('target.jpg')       # 需要卷积的图像
guide = cv2.resize(guide, (512, 512))
filter = cv2.resize(filter, (512, 512))
out_1 = fast_guided_filter(guide, filter, 11, 0.01, 4)    # r epsilon s
out_2 = guideFilter(guide, filter, 5, 0.001)       # r epsilon

# cv2.imshow("out_1", out_1)
cv2.imshow("out_2", out_2)
out_1 *= 255
out_2 *= 255
cv2.imwrite('target_out1.jpg', out_1)
cv2.imwrite('target_out2.jpg', out_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
