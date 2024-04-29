import logging
import sys
import time
import cv2
import numpy
import win32gui
import win32api
import win32con
import pytesseract

from ctypes import windll
from PyQt6.QtWidgets import QApplication


pytesseract.pytesseract.tesseract_cmd = r'.\Tesseract-OCR\tesseract.exe'


class BasicFunctional:
    def __init__(self):
        self.primaryScreen = None

    @staticmethod
    def get_handle():
        """
        获取鼠标下窗口句柄
        :return:
        """
        time.sleep(0.5)
        mouse_x, mouse_y = win32gui.GetCursorPos()
        child_handle = win32gui.WindowFromPoint((mouse_x, mouse_y))

        if child_handle:
            window_title = win32gui.GetWindowText(child_handle)
            if window_title == '一梦江湖':
                return child_handle
        return None

    @staticmethod
    def set_window(handle, win):

        widget, height = win

        for _ in range(300):
            image = basic_functional.screen_shot(handle)
            image_height, image_width, _ = image.shape
            if image_width == 1334 and image_height == 750:
                time.sleep(0.8)
                break

            if image_height < 750:
                height += 1
            elif image_height > 750:
                height -= 1
            if image_width < 1334:
                widget += 1
            elif image_width > 1334:
                widget -= 1

            try:
                rect = win32gui.GetWindowRect(handle)
                win32gui.MoveWindow(handle, rect[0], rect[1], widget, height, True)
            except Exception as e:
                logging.error(f'设置游戏窗口大小: {e}')

    @staticmethod
    def DisableTheWindow(handle):
        """
                禁用窗口
                :param handle: 窗口句柄
                :return:
                """
        # 设置窗口样式，禁用鼠标移动输入
        win32gui.SetWindowLong(handle, win32con.GWL_EXSTYLE,
                               win32gui.GetWindowLong(handle, win32con.GWL_EXSTYLE) | win32con.WS_EX_NOACTIVATE)

    @staticmethod
    def UnDisableTheWindow(handle):
        """
        解除禁用窗口
        :param handle: 窗口句柄
        :return:
        """
        # 解除禁用
        win32gui.SetWindowLong(handle, win32con.GWL_EXSTYLE,
                               win32gui.GetWindowLong(handle, win32con.GWL_EXSTYLE) & ~win32con.WS_EX_NOACTIVATE)

    @staticmethod
    def cv_imread(file_path):
        """
        读取中文图片
        :param file_path:
        :return:
        """
        try:
            cv_img = cv2.imdecode(numpy.fromfile(file_path, dtype=numpy.uint8), cv2.IMREAD_UNCHANGED)
            return cv_img
        except Exception as e:
            logging.error(f'读取图片: {e}')

    @staticmethod
    def mouse_down(handle, tap_x, tap_y):
        """
        鼠标按下
        :param handle: 窗口句柄
        :param tap_x: 坐标x
        :param tap_y: 坐标y
        :return:
        """
        # 模拟鼠标指针， 传送到指定坐标
        long_position = win32api.MAKELONG(tap_x, tap_y)
        # 模拟鼠标按下
        win32api.PostMessage(handle, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, long_position)

    @staticmethod
    def mouse_up(handle, tap_x, tap_y):
        """
        鼠标释放
        :param handle: 窗口句柄
        :param tap_x: 坐标x
        :param tap_y: 坐标y
        :return:
        """
        # 模拟鼠标指针， 传送到指定坐标
        long_position = win32api.MAKELONG(tap_x, tap_y)
        # 模拟鼠标弹起
        win32api.PostMessage(handle, win32con.WM_LBUTTONUP, win32con.MK_LBUTTON, long_position)

    @staticmethod
    def mouse_move(handle, tap_x, tap_y):
        """
        鼠标移动
        :param handle: 窗口句柄
        :param tap_x: 坐标x
        :param tap_y: 坐标y
        :return:
        """
        # 模拟鼠标指针， 传送到指定坐标
        long_position = win32api.MAKELONG(tap_x, tap_y)
        # 模拟鼠标移动
        win32api.PostMessage(handle, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, long_position)

    @staticmethod
    # 文本输入
    def input_text(handle, text):
        for char in text:
            win32api.SendMessage(handle, win32con.WM_CHAR, ord(char), 0)
            time.sleep(0.02)

    @staticmethod
    def get_vk_code(key):
        if len(key) == 1:
            return windll.user32.VkKeyScanA(ord(key)) & 0xFF
        else:
            return VkCode[key]

    @staticmethod
    def key_down(handle, key):
        vk_code = BasicFunctional.get_vk_code(key)
        scan_code = windll.user32.MapVirtualKeyW(vk_code, 0)
        wparam = vk_code
        lparam = (scan_code << 16) | 1
        windll.user32.PostMessageW(handle, win32con.WM_KEYDOWN, wparam, lparam)

    @staticmethod
    def key_up(handle, key):
        vk_code = BasicFunctional.get_vk_code(key)
        scan_code = windll.user32.MapVirtualKeyW(vk_code, 0)
        wparam = vk_code
        lparam = (scan_code << 16) | 0xC0000001
        windll.user32.PostMessageW(handle, win32con.WM_KEYUP, wparam, lparam)

    @staticmethod
    def set_game_size(handle):
        try:
            rect = win32gui.GetWindowRect(handle)
            scale_factor = QApplication.primaryScreen().devicePixelRatio()
            print(scale_factor)
            if scale_factor == 1.0:
                win32gui.MoveWindow(handle, rect[0], rect[1], 1350, 789, True)
            elif scale_factor == 1.25:
                win32gui.MoveWindow(handle, rect[0], rect[1], 1352, 797, True)
            elif scale_factor == 1.5:
                win32gui.MoveWindow(handle, rect[0], rect[1], 1356, 806, True)
            elif scale_factor == 1.75:
                win32gui.MoveWindow(handle, rect[0], rect[1], 1360, 814, True)
            time.sleep(0.5)
        except Exception as e:
            logging.error(f'设置游戏窗口大小: {e}')

    def screen_shot(self, handle):
        """
        获取屏幕指定句柄对象
        :param handle: 句柄
        :return:
        """
        try:
            if self.primaryScreen is None:
                self.primaryScreen = QApplication.primaryScreen()

            image = self.primaryScreen.grabWindow(handle).toImage()
            # 获取图像数据，转换为 NumPy 数组
            width, height = image.width(), image.height()
            buffer = image.bits().asstring(width * height * 4)
            image = numpy.frombuffer(buffer, dtype=numpy.uint8).reshape((height, width, 4))
            # cv2.imshow('Image', image)
            # cv2.waitKey(0)
            return image
        except AttributeError:
            return None

    @staticmethod
    def img_ocr(handle, search_area):
        image = basic_functional.screen_shot(handle)
        x1, y1, x2, y2 = search_area
        image = image[y1:y2, x1:x2]
        return pytesseract.image_to_string(image)


basic_functional = BasicFunctional()


VkCode = {
    "back": 0x08,
    "tab": 0x09,
    "return": 0x0D,
    "shift": 0x10,
    "control": 0x11,
    "menu": 0x12,
    "pause": 0x13,
    "capital": 0x14,
    "ESC": 0x1B,
    "space": 0x20,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "print": 0x2A,
    "snapshot": 0x2C,
    "insert": 0x2D,
    "delete": 0x2E,
    "lwin": 0x5B,
    "rwin": 0x5C,
    "Num0": 0x60,
    "Num1": 0x61,
    "Num2": 0x62,
    "Num3": 0x63,
    "Num4": 0x64,
    "Num5": 0x65,
    "Num6": 0x66,
    "Num7": 0x67,
    "Num8": 0x68,
    "Num9": 0x69,
    "multiply": 0x6A,
    "add": 0x6B,
    "separator": 0x6C,
    "subtract": 0x6D,
    "decimal": 0x6E,
    "divide": 0x6F,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
    "numlock": 0x90,
    "scroll": 0x91,
    "lshift": 0xA0,
    "rshift": 0xA1,
    "lcontrol": 0xA2,
    "rcontrol": 0xA3,
    "lmenu": 0xA4,
    "rmenu": 0XA5
}


if __name__ == "__main__":
    app = QApplication(sys.argv)
    image = basic_functional.screen_shot(198148)
    # rect = win32gui.GetWindowRect(basic_functional.get_handle())
    cv2.imwrite(fr"D:\Desktop\test_img\{time.time()}.bmp", image)
