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
import json


def img_hash(src):
    return cv2.countNonZero(src)


#
# def img_dhash(src):
#     h, w = src.shape[:2]
#     hash = ""
#     for r in range(h - 1):
#         for c in range(w):
#             if src[r][c] != src[r + 1][c]:
#                 hash += "1"
#             else:
#                 hash += "0"
#     return '%0*X' % ((len(hash) + 3) // 4, int(hash, 2))
def img_dhash(src):
    h, w = src.shape[:2]
    hash = ""
    for r in range(h - 1):
        for c in range(w):
            hash += "%d" % (src[r][c] == 0)
    return '%0*X' % ((len(hash) + 3) // 4, int(hash, 2))


def hash_diff(h1, h2):
    if len(h1) != len(h2):
        print("%d %d" % (len(h1), len(h2)))
        print("h1 %s" % h1)
        print("h2 %s" % h2)
        return None
    diff = 0
    h1 = bin(int(h1, 16))[2:]
    h2 = bin(int(h2, 16))[2:]
    h1 = '0' * (300 - len(h1)) + h1
    h2 = '0' * (300 - len(h2)) + h2
    # print(h1)
    for i, _ in enumerate(h1):
        if h1[i] != h2[i]:
            diff += 1
    return diff


def image_decode(img):
    mh = img_dhash(img)
    res = []
    for i in CONFIG.keys():
        if CONFIG[i] != "":
            res.append((hash_diff(mh, CONFIG[i]), i))
    return sorted(res)


imgs = []


def found_sub():
    im = pg.screenshot()
    im_raw = np.array(im)
    roi_color = np.array([172, 234, 160])
    roi_mask = cv2.inRange(im_raw.copy(), roi_color, roi_color)
    roi_mask = cv2.erode(roi_mask, None, iterations=1)
    x, y, w, h = cv2.boundingRect(roi_mask)  # x,y,w,h
    if w * h > 0:
        return x, y, w, h
    else:
        time.sleep(0.1)
        return found_sub()


def all_press():
    s = []
    s += [chr(k) for k in range(97, 123)]
    pg.press(s)


def get_roi_bounding_rect(img, color_min, color_max):
    roi_mask = cv2.inRange(img.copy(), color_min, color_max)
    roi_mask = cv2.erode(roi_mask, None, iterations=1)  # todo 准确度测试
    # cv2.imshow("mask", roi_mask)
    return cv2.boundingRect(roi_mask)  # x,y,w,h


def main_loop():
    bx, by, bw, bh = found_sub()
    r = (bx - 5, by, bx + bw * 42, by + bh)
    enter_counter = 0
    while True:
        im = pg.screenshot()
        sub = im.crop(r)
        cvs = np.array(sub)
        cvs = cv2.cvtColor(cvs, cv2.COLOR_BGR2RGB)
        # cvs = cv2.cvtColor(cvs, cv2.COLOR_BGR2GRAY)
        roi_color = np.array([160, 234, 172]) #BGR of normal green box
        x, y, w, h = get_roi_bounding_rect(cvs, roi_color, roi_color)

        if w * h != bw * bh:  # err
            roi_color_error = np.array([160, 170, 234]) #error red box
            x, y, w, h = get_roi_bounding_rect(cvs, roi_color_error, roi_color_error)
            if w * h != bw * bh:
                enter_counter += 1
                if enter_counter > 50:
                    pg.press("enter")
                    print(enter_counter)
                    print("w,h: %d %d" % (w, h))
                    if enter_counter > 500:
                        # print("[debug] not found.")
                        print((x, y, w, h))
                        pg.press("f5")
                        time.sleep(2)
                        pg.press("enter")
                        enter_counter = 0
                    time.sleep(0.1)
                    continue
        if w * h == bw * bh:  # w*h
            im_char = cvs[y:y + h, x:x + w, :]
            im_char = cv2.cvtColor(im_char, cv2.COLOR_RGB2GRAY)
            t, im_char = cv2.threshold(im_char, 0, 255, cv2.THRESH_OTSU)
            # im_char = cv2.resize(im_char, (8, 9), interpolation=cv2.INTER_AREA)
            # cv2.imshow("im", im_char)
            ch = image_decode(im_char)
            print(ch[:3])
            if ch[0][0] < 20:  # auto press
                pg.press(ch[0][1])
                enter_counter = 0
            else:
                all_press()
                time.sleep(1)
            img_h = img_dhash(im_char)
            if img_h not in imgs:
                imgs.append(img_h)
                cv2.imwrite("./data/%s.png" % str(img_h), im_char)
                # todo 自动录入实现
                # cv2.imshow("raw", cvs)
        time.sleep(0.05)

        key = cv2.waitKey(1)
        if key == 27:  # 按空格切换窗体
            break  # esc退出程序


try:
    with open('data/lib.json', 'r', encoding='utf-8') as fg:
        CONFIG = json.load(fg)
        # log("[debug] config loaded.")
except IOError as e:
    print("[error] %s" % e)
    exit()


def test():
    m = cv2.imread(
        "./data/0000000000000000000000000000F81640101800F01702000002100721EE000000000000000.png",
        cv2.IMREAD_GRAYSCALE)
    t, m = cv2.threshold(m, 0, 255, cv2.THRESH_OTSU)
    print(image_decode(m)[:3])
    # time.sleep(1)
    # while True:
    #     s = [' ', '.', ',', '\'','?','!']
    #     s += [chr(k) for k in range(65, 123)]
    #     pg.press(s)


if __name__ == "__main__":
    # test()
    main_loop()
