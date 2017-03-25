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

TYPE_SPEED_DELAY = 0.05
IS_SAMPLER = False
NORMAL_CHAR_BOX_COLOR = np.array([160, 234, 172])
ERROR_CHAR_BOX_COLOR = np.array([160, 170, 234])


def img_dhash(src):
    h, w = src.shape[:2]
    hash_str = ""
    for r in range(h - 1):
        for c in range(w):
            hash_str += "%d" % (src[r][c] == 0)
    return '%0*X' % ((len(hash_str) + 3) // 4, int(hash_str, 2))


def hash_diff(h1, h2):
    if len(h1) != len(h2):
        print("[error] hash length not equal, %s ,%s" % (h1, h2))
        return None
    diff = 0
    h1 = bin(int(h1, 16))[2:]
    h2 = bin(int(h2, 16))[2:]
    h1 = '0' * (300 - len(h1)) + h1  # make same length
    h2 = '0' * (300 - len(h2)) + h2
    # print(h1)
    for i, _ in enumerate(h1):
        if h1[i] != h2[i]:
            diff += 1
    return diff


def image_recognize(img):
    mh = img_dhash(img)
    res = []
    for i in CONFIG.keys():
        if CONFIG[i] != "":
            res.append((hash_diff(mh, CONFIG[i]), i))
    return sorted(res)


def all_press():
    s = ['f', 'c', 'y', 'r', 'o', 'k']
    s += ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    pg.press(s)
    print("[debug] all pressed.")


def found_bound():
    timer = 0
    while True:
        im = pg.screenshot()
        im_raw = np.array(im)
        im_raw = cv2.cvtColor(im_raw, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("a.png", im_raw)
        roi_mask = cv2.inRange(im_raw.copy(), NORMAL_CHAR_BOX_COLOR, NORMAL_CHAR_BOX_COLOR)
        # cv2.imwrite("b.png", roi_mask)
        roi_mask = cv2.erode(roi_mask, None, iterations=1)
        x, y, w, h = cv2.boundingRect(roi_mask)  # x,y,w,h
        if w * h > 0:
            break
        time.sleep(0.1)
        timer += 1
        print("[debug] try found bound. %d" % timer)
        if timer > 30:
            pg.press("f5")
            print("[debug] time out, refresh")
            time.sleep(3)
            timer = 0
    return x, y, w, h


def get_roi_bounding_rect(img, color_min, color_max):
    roi_mask = cv2.inRange(img.copy(), color_min, color_max)
    roi_mask = cv2.erode(roi_mask, None, iterations=1)
    # cv2.imshow("mask", roi_mask)
    return cv2.boundingRect(roi_mask)  # x,y,w,h


def get_bound():
    bx, by, bw, bh = found_bound()
    r = (bx - 5, by, bx + bw * 42, by + bh)
    return r, bw, bh


imgs = []


def sampler(im_char):
    img_h = img_dhash(im_char)
    if img_h not in imgs:
        imgs.append(img_h)
        cv2.imwrite("./data/%s.png" % str(img_h), im_char)


def main_loop():
    r, bw, bh = get_bound()
    enter_counter = 0
    last_x = 0
    miss = 0
    while True:
        if 300 > miss > 200:  # nothing found state
            print("[debug] enter.")
            pg.press("enter")
            time.sleep(1)
            r, _, _ = get_bound()
        elif miss > 500:
            pg.press("f5")
            time.sleep(1)
            miss = 0
        im = pg.screenshot()
        sub = im.crop(r)
        cvs = np.array(sub)
        cvs = cv2.cvtColor(cvs, cv2.COLOR_BGR2RGB)
        x, y, w, h = get_roi_bounding_rect(cvs, NORMAL_CHAR_BOX_COLOR, NORMAL_CHAR_BOX_COLOR)
        if w * h != bw * bh:  # err
            x, y, w, h = get_roi_bounding_rect(cvs, ERROR_CHAR_BOX_COLOR, ERROR_CHAR_BOX_COLOR)
            if w * h != bw * bh:  # nothing found.
                print("[debug] miss %d." % miss)
                miss += 1
                continue
        else:
            miss = 0
        if last_x == x:  # type miss match error state
            enter_counter += 1
            print("[debug] retry times %d" % enter_counter)
            if enter_counter == 6:
                pg.press("enter")
                print("[debug] press enter")
            elif enter_counter == 10:
                pg.press("enter")
                print("[debug] press enter")
        else:
            enter_counter = 0
        last_x = x
        im_char = cvs[y:y + h, x:x + w, :]
        im_char = cv2.cvtColor(im_char, cv2.COLOR_RGB2GRAY)
        t, im_char = cv2.threshold(im_char, 0, 255, cv2.THRESH_OTSU)
        ch = image_recognize(im_char)
        print("[debug] recognize %s" % str(ch[:3]))
        if ch[0][0] < 20:  # auto press
            pg.press(ch[0][1])
        if IS_SAMPLER:
            sampler(im_char)
        # cv2.imshow("raw", cvs)
        time.sleep(TYPE_SPEED_DELAY)

        key = cv2.waitKey(1)
        if key == 27:
            break  # esc退出程序


try:
    with open('data/lib.json', 'r', encoding='utf-8') as fg:
        CONFIG = json.load(fg)
        # log("[debug] config loaded.")
except IOError as e:
    print("[error] %s" % e)
    exit()


def test():
    # while True:
    #     s = [' ', '.', ',', '\'','?','!']
    #     s += [chr(k) for k in range(65, 123)]
    #     pg.press(s)
    pass


if __name__ == "__main__":
    # test()
    main_loop()
