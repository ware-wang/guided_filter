# -*- coding: utf-8 -*-
# @Time    : 2022/5/22 22:28
# @Author  : ware
# @File    : bilateral_filter().py

import cv2
import numpy as np


# bilateral_filter filter

def bilateral_filter(img, K_size, sigmad, sigmac):
    H, W, C = img.shape  # 获取图像高和宽
    pad = K_size // 2  # 给图像周围填充0，以便卷积核移动

    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float64)  # 定义输出，初始化为0
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float64)  # 将图像拷贝到输出矩阵
    # prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float64)  # 卷积核

    tmp = out.copy()  # 复制带pad的图像，用于随时调用

    for y in range(H):  # 遍历高
        for x in range(W):  # 遍历宽
            for c in range(C):  # 遍历rgb通道

                for m in range(-pad, -pad + K_size):  # 用于改变卷积核
                    for n in range(-pad, -pad + K_size):
                        cd = tmp[y + m + pad, x + n + pad, c] - tmp[y + pad, x + pad, c]  # 颜色差异
                        K[m + pad, n + pad] = np.exp(
                            -(m ** 2 + n ** 2) / (sigmad ** 2) - cd * cd / (sigmac ** 2))  # 综合权重
                K /= K.sum()  # 归一化
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])  # 卷积

    out = np.clip(out, 0, 255)  # 保证颜色区间位于0~255
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)  # 将输出图像裁剪至与输入相同
    return out


img = cv2.imread("target.jpg")             # 读图像
img = cv2.resize(img, (512, 512))
# out = bilateral_filter(img, 5, 16, 0.1)  # 卷积核大小，距离，颜色
# cv2.imwrite("flower_out.jpg", out)       # 输出
# cv2.imshow("result", out)


out = cv2.bilateralFilter(img, 9, 64, 10)
cv2.imshow("result",  out)
cv2.imwrite("flower_out.jpg", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
