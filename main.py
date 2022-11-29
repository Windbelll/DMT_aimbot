import cv2
import numpy as np
import keyboard
import win32api
import math
import win32gui
from PyQt5.QtWidgets import QApplication
import sys
import win32con
from pymouse_pyhook3 import PyMouse

hwnd_title = dict()
flag = 1


def get_all_hwnd(hwnd, mouse):
    if win32gui.IsWindow(hwnd) and win32gui.IsWindowEnabled(hwnd) and win32gui.IsWindowVisible(hwnd):
        hwnd_title.update({hwnd: win32gui.GetWindowText(hwnd)})


def qtpixmap_to_cvimg(qtpixmap):
    qimg = qtpixmap
    temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
    temp_shape += (4,)
    ptr = qimg.bits()
    ptr.setsize(qimg.byteCount())
    result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
    result = result[..., :3]

    return result


def filter_out_red(src_frame):
    if src_frame is not None:
        hsv = cv2.cvtColor(src_frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([156, 43, 46])
        upper_red = np.array([180, 255, 255])
        # inRange()方法返回的矩阵只包含0,255 (CV_8U) 0表示不在区间内
        mask = cv2.inRange(hsv, lower_red, upper_red)
        return cv2.bitwise_and(src_frame, src_frame, mask=mask)


def out():
    print("end")
    global flag
    flag = 0


keyboard.add_hotkey('q', out)
mouse = PyMouse()

# 查询所有程序的句柄
win32gui.EnumWindows(get_all_hwnd, 0)
for h, t in hwnd_title.items():
    if t != "":
        print(h, t)
hwnd = win32gui.FindWindow(None, 'aimlab_tb')

print("获取到目标句柄: " + str(hwnd))

# cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)

print("等待热键:enter")
keyboard.wait('enter')
print("程序启动")
app = QApplication(sys.argv)
while flag:
    # start_time = time.perf_counter()
    # current_time = time.perf_counter()
    # print("time test:start @" + str(0) + "ms")
    screen = QApplication.primaryScreen()
    img_src = qtpixmap_to_cvimg(screen.grabWindow(0).toImage())
    # current_time_temp = time.perf_counter()
    # print("time test:grab screen @" + str((current_time_temp - current_time) * 1000) + "ms")
    # current_time = time.perf_counter()
    # if cv2.waitKey(1) == ord('q'):
    #     exit()
    # img = filter_out_red(img_src)

    # current_time_temp = time.perf_counter()
    # print("time test:cv get red object @" + str((current_time_temp - current_time) * 1000) + "ms")
    # current_time = time.perf_counter()
    img = img_src[200:880, 300:1620]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    src = np.zeros(img.shape, np.uint8)
    cv2.threshold(img, 200, 255, cv2.THRESH_BINARY, src)
    # cv2.imshow("test",src)
    # current_time_temp = time.perf_counter()
    # print("time test:after threshold @" + str((current_time_temp - current_time) * 1000) + "ms")
    # current_time = time.perf_counter()

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src)

    # current_time_temp = time.perf_counter()
    # print("time test:components calculate @" + str((current_time_temp - current_time) * 1000) + "ms")
    # current_time = time.perf_counter()

    # current = current + 1
    # if current == 6:
    #     current = 1
    distance = []
    locs = []
    for i in range(1, num_labels):
        if stats[i][4] < 20:
            continue
        else:
            src_x = int(stats[i][0] + stats[i][2] / 2) - 660
            src_y = int(stats[i][1] + stats[i][3] / 2) - 339
            distance.append(math.sqrt(src_x * src_x + src_y * src_y))
            locs.append([src_x, src_y])

    # current_time_temp = time.perf_counter()
    # print("time test:distances calculate @" + str((current_time_temp - current_time) * 1000) + "ms")
    # current_time = time.perf_counter()

    index = distance.index(min(distance))
    x = locs[index][0]
    y = locs[index][1]
    move_x = x if x < 300 else int((1 - (x - 300) / 1620 * 0.05) * x)
    move_y = y if y < 300 else int((1 - (y - 300) / 780 * 0.08) * y)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, move_y)
    mouse.click(0, 0)

    # current_time_temp = time.perf_counter()
    # print("time test:circle end once @" + str((current_time_temp - current_time) * 1000) + "ms")
    # current_time = time.perf_counter()
    #
    # print("time test:total @ " + str((current_time - start_time) * 1000) + "ms")
    # cv2.waitKey(0)
    # keyboard.wait('enter')
