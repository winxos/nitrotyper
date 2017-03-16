# coding:utf8
# https://www.nitrotype.com/race auto typer
# python 3.5 + opencv 3.2 + pyautogui
# MIT LICENSE
# winxos
# since:2017-03-17
import cv2
import pyautogui as pg
import numpy as np
import time


# todo 实现较强的哈希算法
def img_hash(src):
    return cv2.countNonZero(src)


imgs = []

last_cursor = 0
if __name__ == "__main__":
    r = (1200, 835, 1700, 868)  # todo 可以实现坐标自动采集
    # sub.show()
    while True:
        im = pg.screenshot()
        sub = im.crop(r)
        cvs = np.array(sub)
        cvs = cv2.cvtColor(cvs, cv2.COLOR_BGR2RGB)
        # cvs = cv2.cvtColor(cvs, cv2.COLOR_BGR2GRAY)
        roi_color = np.array([160, 234, 172])
        roi_color_lower = np.array([150, 214, 152])
        roi_color_upper = np.array([180, 244, 182])  # todo 添加对输入错误的颜色采集
        roi_mask = cv2.inRange(cvs.copy(), roi_color_lower, roi_color_upper)
        roi_mask = cv2.erode(roi_mask, None, iterations=2)
        # cv2.imshow("mask", roi_mask)
        x, y, w, h = cv2.boundingRect(roi_mask)  # x,y,w,h
        if w * h == 10 * 24:  # w*h
            im_char = cvs[y:y + h, x:x + w, :]
            im_char = cv2.cvtColor(im_char, cv2.COLOR_RGB2GRAY)
            t, im_char = cv2.threshold(im_char, 0, 255, cv2.THRESH_OTSU)
            cv2.imshow("im", im_char)
            img_h = img_hash(im_char)
            if img_h not in imgs:
                imgs.append(img_h)
                cv2.imwrite("./datas/%s.png" % str(img_h), im_char)  # todo 自动采集样本，通过配置文件来打标签
                # todo 自动录入实现
                # cv2.imshow("raw", cvs)
        time.sleep(0.1)
        key = cv2.waitKey(1)
        if key == 27:  # 按空格切换窗体
            break  # esc退出程序
        pass
