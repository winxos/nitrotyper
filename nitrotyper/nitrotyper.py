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


def img_dhash_to_hex(src):
    h, w = src.shape[:2]
    hash_str = ""
    for r in range(h - 1):
        for c in range(w):
            hash_str += "%d" % (src[r][c] == 0)
    return '%0*X' % ((len(hash_str) + 3) // 4, int(hash_str, 2))


def hash_diff_for_hex(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def image_recognize(img):
    mh = img_dhash_to_hex(img)
    res = []
    for i in char_data.keys():
        if char_data[i] != "":
            res.append((hash_diff_for_hex(mh, char_data[i]), i))
    return sorted(res)


def get_roi_bounding_rect(img, color_min, color_max):
    roi_mask = cv2.inRange(img.copy(), color_min, color_max)
    roi_mask = cv2.erode(roi_mask, None, iterations=1)
    # cv2.imshow("mask", roi_mask)
    return cv2.boundingRect(roi_mask)  # x,y,w,h


def sampler(im_char, imgs=[]):
    img_h = img_dhash_to_hex(im_char)
    if img_h not in imgs:
        imgs.append(img_h)
        cv2.imwrite("./data/%s.png" % str(img_h), im_char)


def run(delay=0.01, is_sampler=False):
    NORMAL_CHAR_BOX_COLOR = np.array([160, 234, 172])
    ERROR_CHAR_BOX_COLOR = np.array([160, 170, 234])

    def get_current_line_rect():
        def found_current_rect():
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
                if timer > 60:
                    pg.press("f5")
                    print("[debug] time out, refresh")
                    time.sleep(3)
                    timer = 0
            return x, y, w, h

        bx, by, bw, bh = found_current_rect()
        r = (bx - 5, by, bx + bw * 42, by + bh)
        return r, bw, bh

    r, bw, bh = get_current_line_rect()
    enter_counter = 0
    last_x = 0
    miss = 0
    while True:
        if 300 > miss > 200:  # nothing found state
            print("[debug] enter.")
            pg.press("enter")
            time.sleep(1)
            r, _, _ = get_current_line_rect()
        elif miss > 500:
            pg.press("f5")
            time.sleep(1)
            miss = 0
        st = time.clock()
        sub = pg.screenshot(region=r)  # todo 截屏速度太慢，考虑提高截屏速度或者一次打多个
        # print("[debug]sreenshot %f" % (time.clock() - st))
        cvs = np.array(sub)
        cvs = cv2.cvtColor(cvs, cv2.COLOR_BGR2RGB)
        # print("[debug]convert %f" % (time.clock() - st))
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
        # im_char=cv2.resize(im_char,(10,30))
        # print("[debug]exact %f" % (time.clock() - st))
        ch = image_recognize(im_char)
        print("[debug] recognize %s time used:%f" % (str(ch[:3]), time.clock() - st))
        if ch[0][0] < 20:  # auto press
            pg.press(ch[0][1])
        if is_sampler:
            sampler(im_char)
        # cv2.imshow("raw", cvs)
        time.sleep(delay)

        key = cv2.waitKey(1)
        if key == 27:
            break  # esc退出程序


import pkgutil  # 必须采用pkgutil.get_data才能读取egg格式包中的数据

try:
    f = pkgutil.get_data("nitrotyper", 'data/chars.json').decode('utf-8')  #
    char_data = json.loads(f)
except IOError as e:
    print("[error] %s" % e)
    exit()

if __name__ == "__main__":
    run(delay=0)
