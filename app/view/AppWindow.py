import ctypes.wintypes
import json
import os
import shutil
import tempfile
import threading
import configparser
import time

import cv2
import logging

import win32con
import win32gui
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, QDateTime, QRegularExpression, QUrl, QMimeData, QEvent
from PyQt6.QtGui import QIcon, QPainter, QColor, QFont, QPixmap, QRegularExpressionValidator, QImage, QDesktopServices, \
    QFontDatabase, QDrag, QTextCursor
from PyQt6.QtWidgets import QWidget, QPushButton, QApplication, QDialog, QTableWidget, \
    QTableWidgetItem, QMessageBox, QListWidgetItem, QListWidget, QLineEdit, QCompleter, QLabel, QVBoxLayout
from win32gui import GetWindowRect
from threading import Thread

from app.view.Ui.MainWindow import Ui_MainWindow
from app.view.Ui.HomeWindow import Ui_Home
from app.view.Ui.ScriptWindow import Ui_Script
from app.view.Ui.RunWindow import Ui_Run
from app.view.Ui.LoginWindow import Ui_Login
from app.view.Ui.SettingWindow import Ui_Setting
from app.view.Ui.EditorWindow import Ui_Editor
from app.Script.BasicFunctional import basic_functional
from app.Script.Task import StartTask, TASK_MAPPING, TASK_SHOW, EXCLUDE_TASK_MAPPING, EXCLUDE_ADD_TASK
from app.view.Public import publicSingle, TABLE_WINDOW, DPI_MAPP, ConfigDialog, DelConfigDialog, CustomLineEdit, \
    TimingQMessageBox, Mask, TextEdit, VERSION, ShortCutLineEdit, START_ID
from app.view.ClientServices import services


class MainWindow(QWidget, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        start = time.time()
        # 加载字体
        QFontDatabase.addApplicationFont('app/images/font/Xingxingniannian-Bold-2.ttf')
        self.username = None
        self.user_quit = True
        self.setupUi(self)
        self.login = LoginWindow()
        publicSingle.login.connect(self.initWindow)
        # print(int(self.winId()))
        # basic_functional.register_hotkey(self.winId(), START_ID, 'Shift+W')
        # basic_functional.register_hotkey(self.winId(), 2, win32con.MOD_ALT | win32con.MOD_SHIFT, 'W')

        self.index = 0
        # self.menu_layout = QVBoxLayout(self.menu_widget)

        # create window
        self.home = HomeWindow()
        self.script = ScriptWindow()
        self.run = RunWindow()
        self.setting = SettingWindow(self)
        # self.editor = EditorWindow()
        # self.main_layout = QVBoxLayout(self.main_widget)

        self.initNavigation()
        self.script.pushButton_11.clicked.connect(self.save_config)
        self.script.pushButton_12.clicked.connect(self.delete_config)

        self.script.comboBox.currentTextChanged.connect(self.load_config)
        self.load_config('默认配置')
        self.load_system_config()
        self.show_page(0)
        print(time.time() - start)

    # add window
    def initNavigation(self):
        self.addSubInterface(self.home, '主页')
        self.addSubInterface(self.script, '脚本')
        self.addSubInterface(self.run, '运行')
        self.addSubInterface(self.setting, '设置')
        # self.addSubInterface(self.editor, '编辑器')
        # self.main_widget.addWidget(self.home)
        # self.main_widget.addWidget(self.run)

    # Add menu main
    def addSubInterface(self, interface: QWidget, text: str):
        """
        主布局挂载子窗口 菜单添加按钮绑定子窗口
        :param interface: 子窗口
        :param text: 按钮文字
        :return:
        """
        index = self.index
        self.index += 1
        self.main_widget.addWidget(interface)
        button = QPushButton(text)
        # icon = QIcon("app/images/svg/Home_black.svg")  # 替换为您的图标文件路径
        # button.setIcon(icon)
        # button.setIconSize(QSize(16, 20))
        # button.setStyleSheet("background-color: transparent;")
        self.verticalLayout.insertWidget(index, button)
        button.clicked.connect(lambda: self.show_page(index))

    def nativeEvent(self, eventType, message):
        msg = ctypes.wintypes.MSG.from_address(message.__int__())
        if msg.message == win32con.WM_HOTKEY:
            print("成功了吧~~")
            print(msg.wParam)
            if msg.wParam == 1 and self.username is not None:
                publicSingle.start.emit('_')
        return False, message

    # SwitchPages
    def show_page(self, index):
        """
        show page for index
        :param index:
        :return:
        """
        self.main_widget.setCurrentIndex(index)
        for i in range(self.verticalLayout.count()):
            widget = self.verticalLayout.itemAt(i).widget()
            if isinstance(widget, QPushButton):
                if i == index:
                    # 当前点击的按钮，修改样式
                    widget.setStyleSheet("""
                            QPushButton {

                                border-color: #555; /* 被点击时的边框颜色 */
                                border-radius: 15px; /* 圆角 */
                                padding: 10px 20px;
                            }
                            QPushButton:hover {
                                border-color: red; /* 鼠标悬停时边框颜色 */
                            }
                            QPushButton:pressed {
                                border-color: #555; /* 鼠标点击时边框颜色 */
                            }
                        """)
                else:
                    # 其他按钮，恢复初始样式
                    widget.setStyleSheet("""
                            QPushButton {
                                background-color: transparent;
                                border: 2px solid transparent; /* 初始时透明边框 */
                                border-radius: 15px; /* 圆角 */
                                padding: 10px 20px;
                            }
                            QPushButton:hover {
                                border-color: red; /* 鼠标悬停时边框颜色 */
                            }
                            QPushButton:pressed {
                                border-color: #555; /* 鼠标点击时边框颜色 */
                            }
                        """)

    def return_data(self):
        return {
            "执行列表": [self.script.listWidget.item(i).text() for i in range(self.script.listWidget.count())],
            "世界喊话内容": self.script.lineEdit_2.text(),
            "世界喊话次数": self.script.spinBox_4.value(),
            "江湖英雄榜次数": self.script.spinBox_7.value(),
            "江湖英雄榜秒退": self.script.checkBox_11.isChecked(),
            "副本人数": self.script.comboBox_2.currentIndex(),
            "副本自动匹配": self.script.checkBox.isChecked(),
            "副本喊话内容": self.script.lineEdit.text(),
            "侠缘昵称": self.script.lineEdit_3.text(),
            # "侠缘喊话内容": self.script.textEdit_2.toPlainText(),
            "山河器": self.script.checkBox_2.isChecked(),
            "帮派铜钱捐献": self.script.checkBox_8.isChecked(),
            "帮派银两捐献": self.script.checkBox_7.isChecked(),
            "银票礼盒": self.script.checkBox_3.isChecked(),
            "商会鸡蛋": self.script.checkBox_5.isChecked(),
            "吴越剑坯": self.script.checkBox_33.isChecked(),
            "白公鼎坯": self.script.checkBox_34.isChecked(),
            "碧铜马坯": self.script.checkBox_35.isChecked(),
            "天幕雅苑": self.script.checkBox_32.isChecked(),
            "榫头卯眼": self.script.checkBox_6.isChecked(),
            "锦芳绣残片": self.script.checkBox_4.isChecked(),
            "摇钱树": self.script.checkBox_9.isChecked(),
            "摇钱树目标": self.script.comboBox_3.currentIndex(),
            "生活技能艾草": self.script.checkBox_30.isChecked(),
            "生活技能莲子": self.script.checkBox_31.isChecked(),
            "精制面粉": self.script.checkBox_36.isChecked(),
            "土鸡蛋": self.script.checkBox_37.isChecked(),
            "鲜笋": self.script.checkBox_38.isChecked(),
            "猪肉": self.script.checkBox_39.isChecked(),
            "糯米": self.script.checkBox_40.isChecked(),
            "扫摆摊延迟1": self.script.spinBox.value(),
            "扫摆摊延迟2": self.script.spinBox_2.value(),
            "扫摆摊延迟3": self.script.spinBox_3.value(),
            "关注1": self.script.checkBox_21.isChecked(),
            "关注2": self.script.checkBox_25.isChecked(),
            "关注3": self.script.checkBox_22.isChecked(),
            "关注4": self.script.checkBox_26.isChecked(),
            "关注5": self.script.checkBox_24.isChecked(),
            "关注6": self.script.checkBox_27.isChecked(),
            "关注7": self.script.checkBox_28.isChecked(),
            "关注8": self.script.checkBox_29.isChecked(),
            "优先级1": self.script.comboBox_9.currentIndex(),
            "优先级2": self.script.comboBox_10.currentIndex(),
            "优先级3": self.script.comboBox_11.currentIndex(),
            "优先级4": self.script.comboBox_12.currentIndex(),
            "优先级5": self.script.comboBox_13.currentIndex(),
            "优先级6": self.script.comboBox_14.currentIndex(),
            "优先级7": self.script.comboBox_15.currentIndex(),
            "优先级8": self.script.comboBox_16.currentIndex(),
            "华山论剑次数": self.script.spinBox_6.value(),
            "华山论剑秒退": self.script.checkBox_10.isChecked(),
            "背包": self.script.lineEdit_14.text(),
            "好友": self.script.lineEdit_15.text(),
            "队伍": self.script.lineEdit_16.text(),
            "地图": self.script.lineEdit_17.text(),
            "设置": self.script.lineEdit_18.text(),
            "帮派": self.script.lineEdit_55.text(),
            "技能逻辑": self.script.textEdit.toPlainText(),
            "采集线数": self.script.comboBox_7.currentIndex(),
            "指定地图": self.script.comboBox_8.currentText(),
            "自动吃鸡蛋": self.script.checkBox_19.isChecked(),
            "吃鸡蛋数量": self.script.spinBox_8.value(),
            "切角色1": self.script.checkBox_14.isChecked(),
            "切角色2": self.script.checkBox_15.isChecked(),
            "切角色3": self.script.checkBox_16.isChecked(),
            "切角色4": self.script.checkBox_17.isChecked(),
            "切角色5": self.script.checkBox_18.isChecked(),
            "江湖行商次数": self.script.spinBox_9.value(),
            "江湖行商喊话内容": self.script.lineEdit_49.text(),
            "商票上缴": self.script.checkBox_20.isChecked(),
            "队伍模式": self.script.comboBox_17.currentText(),
            "地图搜索": self.script.radioButton_2.isChecked(),
            "定点采集": self.script.radioButton_6.isChecked(),
            "自定义坐标采集": self.script.radioButton.isChecked(),
            "采草": self.script.radioButton_4.isChecked(),
            "伐木": self.script.radioButton_3.isChecked(),
            "挖矿": self.script.radioButton_5.isChecked(),
            "采草目标": self.script.comboBox_4.currentText(),
            "伐木目标": self.script.comboBox_5.currentText(),
            "挖矿目标": self.script.comboBox_6.currentText(),
            "坐标1": [self.script.lineEdit_19.text(), self.script.lineEdit_20.text()],
            "坐标2": [self.script.lineEdit_21.text(), self.script.lineEdit_22.text()],
            "坐标3": [self.script.lineEdit_23.text(), self.script.lineEdit_24.text()],
            "坐标4": [self.script.lineEdit_25.text(), self.script.lineEdit_26.text()],
            "坐标5": [self.script.lineEdit_27.text(), self.script.lineEdit_28.text()],
            "坐标6": [self.script.lineEdit_29.text(), self.script.lineEdit_30.text()],
            "坐标7": [self.script.lineEdit_31.text(), self.script.lineEdit_32.text()],
            "坐标8": [self.script.lineEdit_33.text(), self.script.lineEdit_34.text()],
            "坐标9": [self.script.lineEdit_35.text(), self.script.lineEdit_36.text()],
            "坐标10": [self.script.lineEdit_37.text(), self.script.lineEdit_38.text()],
            "坐标11": [self.script.lineEdit_39.text(), self.script.lineEdit_40.text()],
            "坐标12": [self.script.lineEdit_41.text(), self.script.lineEdit_42.text()],
            "坐标13": [self.script.lineEdit_43.text(), self.script.lineEdit_44.text()],
            "坐标14": [self.script.lineEdit_45.text(), self.script.lineEdit_46.text()],
            "坐标15": [self.script.lineEdit_47.text(), self.script.lineEdit_48.text()],
            "普攻": self.script.lineEdit_4.text(),
            "技能1": self.script.lineEdit_9.text(),
            "技能2": self.script.lineEdit_5.text(),
            "技能3": self.script.lineEdit_6.text(),
            "技能4": self.script.lineEdit_7.text(),
            "技能5": self.script.lineEdit_8.text(),
            "技能6": self.script.lineEdit_10.text(),
            "技能7": self.script.lineEdit_11.text(),
            "技能8": self.script.lineEdit_12.text(),
            "绝学": self.script.lineEdit_13.text(),
            "闪避": self.script.lineEdit_50.text(),
            "关山": self.script.lineEdit_51.text(),
            "自创1": self.script.lineEdit_52.text(),
            "自创2": self.script.lineEdit_53.text(),
            "自创3": self.script.lineEdit_54.text(),
            "自创4": self.script.lineEdit_56.text(),

        }

    # 保存任务配置信息
    def write_task_json(self, row):
        data = self.return_data()
        # 检查文件是否存在
        temp_dir = tempfile.gettempdir()
        temp_img_path = os.path.join(temp_dir, f'config{row}.json')
        if os.path.exists(temp_img_path):
            # 如果存在，清空文件内容
            with open(temp_img_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False)
        else:
            # 如果不存在，创建文件
            with open(temp_img_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False)

    # initialization
    def initWindow(self, username):
        self.login.close()
        self.username = username
        publicSingle.write_json.connect(self.write_task_json)
        publicSingle.offline.connect(self.offline)
        self.resize(1100, 610)
        self.setMinimumWidth(1000)
        self.setWindowIcon(QIcon('app/images/icon/favicon.ico'))
        self.setWindowTitle(f'时雪{VERSION}')

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.show()
        QApplication.processEvents()

    def offline(self):
        self.user_quit = False
        self.close()
        self.login.show()

    def save_config(self, _):
        # 获取当前用户的路径
        user_path = os.path.expanduser('~')

        # 拼接文件路径
        config_path = os.path.join(user_path, 'ChronoSnow', 'TaskConfig')

        # 递归创建目录和文件
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        dialog = ConfigDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            file_name = dialog.file_name_input.text() if dialog.file_name_input.text() else dialog.get_selected_items()[
                1] if dialog.get_selected_items() else False
            if file_name:
                config = configparser.ConfigParser()
                # 添加一些配置项
                config['日常任务'] = self.return_data()

                with open(f'{config_path}\\{file_name}.ini', 'w', encoding='utf-8') as configfile:
                    config.write(configfile)

                # 配置文件加入了列表
                # 判断 file_name 是否已经存在于 config_selector 中
                if file_name in [self.script.comboBox.itemText(i) for i in range(self.script.comboBox.count())]:
                    # 如果存在，则移除它
                    index_to_remove = self.script.comboBox.findText(file_name)
                    self.script.comboBox.removeItem(index_to_remove)

                # 添加新的 file_name
                self.script.comboBox.addItem(file_name)

                # 设置当前选中项的索引为新添加的项
                index_of_new_item = self.script.comboBox.findText(file_name)
                self.script.comboBox.setCurrentIndex(index_of_new_item)

    # 读取配置文件
    def load_config(self, file_name):
        # 获取当前用户的路径
        user_path = os.path.expanduser('~')
        logging.info(f'This is a debug message{user_path}')
        # 拼接文件路径
        config_path = os.path.join(user_path, 'ChronoSnow', 'TaskConfig')
        # self.update_log(f'加载{file_name}配置')
        # 拼接完整的文件路径和文件名
        config_path = f'{config_path}\\{file_name}.ini'
        config = configparser.ConfigParser()
        # 读取配置文件
        config.read(config_path, encoding='utf-8')
        if file_name == '默认配置':
            self.script.listWidget.clear()
            self.script.lineEdit_2.setText('')
            self.script.spinBox_4.setValue(30)
            self.script.spinBox_7.setValue(1)
            self.script.comboBox_2.setCurrentIndex(0)
            self.script.checkBox.setChecked(False)
            self.script.lineEdit_3.setText('')
            self.script.checkBox_2.setChecked(False)
            self.script.checkBox_8.setChecked(False)
            self.script.checkBox_7.setChecked(False)
            self.script.checkBox_3.setChecked(False)
            self.script.checkBox_5.setChecked(False)
            self.script.checkBox_6.setChecked(False)
            self.script.checkBox_4.setChecked(False)
            self.script.checkBox_9.setChecked(False)
            self.script.comboBox_3.setCurrentIndex(0)
            self.script.spinBox.setValue(100)
            self.script.spinBox_2.setValue(0)
            self.script.spinBox_3.setValue(0)
            self.script.spinBox_6.setValue(1)
            self.script.checkBox_10.setChecked(False)
            self.script.lineEdit_14.setText('B')
            self.script.lineEdit_15.setText('H')
            self.script.lineEdit_16.setText('T')
            self.script.lineEdit_17.setText('M')
            self.script.lineEdit_18.setText('ESC')
            self.script.comboBox_7.setCurrentIndex(0)
            self.script.comboBox_8.setCurrentText('江南')

            self.script.radioButton_2.setChecked(True)
            self.script.radioButton_6.setChecked(False)
            self.script.radioButton.setChecked(False)

            self.script.radioButton_4.setChecked(True)
            self.script.radioButton_3.setChecked(False)
            self.script.radioButton_5.setChecked(False)

            self.script.comboBox_4.setCurrentText('野草')
            self.script.comboBox_5.setCurrentText('枯木')
            self.script.comboBox_6.setCurrentText('碎石')

            coord = ['', '']
            self.script.lineEdit_19.setText(coord[0])
            self.script.lineEdit_20.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_21.setText(coord[0])
            self.script.lineEdit_22.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_23.setText(coord[0])
            self.script.lineEdit_24.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_25.setText(coord[0])
            self.script.lineEdit_26.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_27.setText(coord[0])
            self.script.lineEdit_28.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_29.setText(coord[0])
            self.script.lineEdit_30.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_31.setText(coord[0])
            self.script.lineEdit_32.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_33.setText(coord[0])
            self.script.lineEdit_34.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_35.setText(coord[0])
            self.script.lineEdit_36.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_37.setText(coord[0])
            self.script.lineEdit_38.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_39.setText(coord[0])
            self.script.lineEdit_40.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_41.setText(coord[0])
            self.script.lineEdit_42.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_43.setText(coord[0])
            self.script.lineEdit_44.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_45.setText(coord[0])
            self.script.lineEdit_46.setText(coord[1])
            coord = ['', '']
            self.script.lineEdit_47.setText(coord[0])
            self.script.lineEdit_48.setText(coord[1])

            self.script.spinBox_9.setValue(5)
            self.script.checkBox_19.setChecked(False)
            self.script.spinBox_8.setValue(1)
            self.script.lineEdit_49.setText('')
            self.script.checkBox_20.setChecked(False)

            self.script.checkBox_21.setChecked(False)
            self.script.checkBox_25.setChecked(False)
            self.script.checkBox_22.setChecked(False)
            self.script.checkBox_26.setChecked(False)
            self.script.checkBox_24.setChecked(False)
            self.script.checkBox_27.setChecked(False)
            self.script.checkBox_28.setChecked(False)
            self.script.checkBox_29.setChecked(False)

            self.script.comboBox_9.setCurrentIndex(0)
            self.script.comboBox_10.setCurrentIndex(1)
            self.script.comboBox_11.setCurrentIndex(2)
            self.script.comboBox_12.setCurrentIndex(3)
            self.script.comboBox_13.setCurrentIndex(4)
            self.script.comboBox_14.setCurrentIndex(5)
            self.script.comboBox_15.setCurrentIndex(6)
            self.script.comboBox_16.setCurrentIndex(7)

            self.script.checkBox_30.setChecked(False)
            self.script.checkBox_31.setChecked(False)

            self.script.lineEdit.setText('悬赏副本来人!!!')

            self.script.lineEdit_50.setText('F'),
            self.script.lineEdit_51.setText('G'),
            self.script.lineEdit_52.setText('K'),
            self.script.lineEdit_53.setText('L'),
            self.script.lineEdit_54.setText('5'),

            self.script.textEdit.setText('技能[技能1] 延迟[2000]ms <>\n'
                                         '技能[技能5] 延迟[2000]ms <>\n'
                                         '技能[技能6] 延迟[2000]ms <>\n'
                                         '技能[技能2] 延迟[2000]ms <>\n'
                                         '技能[技能8] 延迟[2000]ms <>\n'
                                         '技能[技能2] 延迟[2000]ms <>\n'
                                         '技能[技能3] 延迟[2000]ms <>\n'
                                         '技能[技能4] 延迟[6000]ms <>\n')

            self.script.lineEdit_4.setText('1')
            self.script.lineEdit_9.setText('2')
            self.script.lineEdit_5.setText('3')
            self.script.lineEdit_6.setText('4')
            self.script.lineEdit_7.setText('5')
            self.script.lineEdit_8.setText('6')
            self.script.lineEdit_10.setText('7')
            self.script.lineEdit_11.setText('8')
            self.script.lineEdit_12.setText('9')
            self.script.lineEdit_13.setText('R')

            self.script.checkBox_33.setChecked(False)
            self.script.checkBox_34.setChecked(False)
            self.script.checkBox_35.setChecked(False)
            self.script.checkBox_32.setChecked(False)

            self.script.checkBox_36.setChecked(False)
            self.script.checkBox_37.setChecked(False)
            self.script.checkBox_38.setChecked(False)
            self.script.checkBox_39.setChecked(False)
            self.script.checkBox_40.setChecked(False)

            self.script.checkBox_14.setChecked(False)
            self.script.checkBox_15.setChecked(False)
            self.script.checkBox_16.setChecked(False)
            self.script.checkBox_17.setChecked(False)
            self.script.checkBox_18.setChecked(False)

            self.script.lineEdit_55.setText('O'),
            self.script.lineEdit_56.setText('4')

        try:
            self.script.listWidget.clear()
            for item in eval(config.get('日常任务', '执行列表')):
                self.script.listWidget.addItem(item)

            self.script.lineEdit_2.setText(config.get('日常任务', '世界喊话内容'))
            self.script.spinBox_4.setValue(config.getint('日常任务', '世界喊话次数'))
            self.script.spinBox_7.setValue(config.getint('日常任务', '江湖英雄榜次数'))
            self.script.comboBox_2.setCurrentIndex(config.getint('日常任务', '副本人数'))
            self.script.checkBox.setChecked(config.getboolean('日常任务', '副本自动匹配'))
            self.script.lineEdit_3.setText(config.get('日常任务', '侠缘昵称'))
            self.script.checkBox_2.setChecked(config.getboolean('日常任务', '山河器'))
            self.script.checkBox_8.setChecked(config.getboolean('日常任务', '帮派铜钱捐献'))
            self.script.checkBox_7.setChecked(config.getboolean('日常任务', '帮派银两捐献'))
            self.script.checkBox_3.setChecked(config.getboolean('日常任务', '银票礼盒'))
            self.script.checkBox_5.setChecked(config.getboolean('日常任务', '商会鸡蛋'))
            self.script.checkBox_6.setChecked(config.getboolean('日常任务', '榫头卯眼'))
            self.script.checkBox_4.setChecked(config.getboolean('日常任务', '锦芳绣残片'))
            self.script.checkBox_9.setChecked(config.getboolean('日常任务', '摇钱树'))
            self.script.comboBox_3.setCurrentIndex(config.getint('日常任务', '摇钱树目标'))
            self.script.spinBox.setValue(config.getint('日常任务', '扫摆摊延迟1'))
            self.script.spinBox_2.setValue(config.getint('日常任务', '扫摆摊延迟2'))
            self.script.spinBox_3.setValue(config.getint('日常任务', '扫摆摊延迟3'))
            self.script.spinBox_6.setValue(config.getint('日常任务', '华山论剑次数'))
            self.script.checkBox_10.setChecked(config.getboolean('日常任务', '华山论剑秒退'))
            self.script.lineEdit_14.setText(config.get('日常任务', '背包'))
            self.script.lineEdit_15.setText(config.get('日常任务', '好友'))
            self.script.lineEdit_16.setText(config.get('日常任务', '队伍'))
            self.script.lineEdit_17.setText(config.get('日常任务', '地图'))
            self.script.lineEdit_18.setText(config.get('日常任务', '设置'))
            self.script.comboBox_7.setCurrentIndex(config.getint('日常任务', '采集线数'))
            self.script.comboBox_8.setCurrentText(config.get('日常任务', '指定地图'))

            self.script.radioButton_2.setChecked(config.getboolean('日常任务', '地图搜索'))
            self.script.radioButton_6.setChecked(config.getboolean('日常任务', '定点采集'))
            self.script.radioButton.setChecked(config.getboolean('日常任务', '自定义坐标采集'))

            self.script.radioButton_4.setChecked(config.getboolean('日常任务', '采草'))
            self.script.radioButton_3.setChecked(config.getboolean('日常任务', '伐木'))
            self.script.radioButton_5.setChecked(config.getboolean('日常任务', '挖矿'))

            self.script.comboBox_4.setCurrentText(config.get('日常任务', '采草目标'))
            self.script.comboBox_5.setCurrentText(config.get('日常任务', '伐木目标'))
            self.script.comboBox_6.setCurrentText(config.get('日常任务', '挖矿目标'))

            coord = eval(config.get('日常任务', '坐标1'))
            self.script.lineEdit_19.setText(coord[0])
            self.script.lineEdit_20.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标2'))
            self.script.lineEdit_21.setText(coord[0])
            self.script.lineEdit_22.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标3'))
            self.script.lineEdit_23.setText(coord[0])
            self.script.lineEdit_24.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标4'))
            self.script.lineEdit_25.setText(coord[0])
            self.script.lineEdit_26.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标5'))
            self.script.lineEdit_27.setText(coord[0])
            self.script.lineEdit_28.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标6'))
            self.script.lineEdit_29.setText(coord[0])
            self.script.lineEdit_30.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标7'))
            self.script.lineEdit_31.setText(coord[0])
            self.script.lineEdit_32.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标8'))
            self.script.lineEdit_33.setText(coord[0])
            self.script.lineEdit_34.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标9'))
            self.script.lineEdit_35.setText(coord[0])
            self.script.lineEdit_36.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标10'))
            self.script.lineEdit_37.setText(coord[0])
            self.script.lineEdit_38.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标11'))
            self.script.lineEdit_39.setText(coord[0])
            self.script.lineEdit_40.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标12'))
            self.script.lineEdit_41.setText(coord[0])
            self.script.lineEdit_42.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标13'))
            self.script.lineEdit_43.setText(coord[0])
            self.script.lineEdit_44.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标14'))
            self.script.lineEdit_45.setText(coord[0])
            self.script.lineEdit_46.setText(coord[1])
            coord = eval(config.get('日常任务', '坐标15'))
            self.script.lineEdit_47.setText(coord[0])
            self.script.lineEdit_48.setText(coord[1])

            self.script.spinBox_9.setValue(config.getint('日常任务', '江湖行商次数'))
            self.script.checkBox_19.setChecked(config.getboolean('日常任务', '自动吃鸡蛋'))
            self.script.spinBox_8.setValue(config.getint('日常任务', '吃鸡蛋数量'))
            self.script.lineEdit_49.setText(config.get('日常任务', '江湖行商喊话内容'))
            self.script.checkBox_20.setChecked(config.getboolean('日常任务', '商票上缴'))

            self.script.checkBox_21.setChecked(config.getboolean('日常任务', '关注1'))
            self.script.checkBox_25.setChecked(config.getboolean('日常任务', '关注2'))
            self.script.checkBox_22.setChecked(config.getboolean('日常任务', '关注3'))
            self.script.checkBox_26.setChecked(config.getboolean('日常任务', '关注4'))
            self.script.checkBox_24.setChecked(config.getboolean('日常任务', '关注5'))
            self.script.checkBox_27.setChecked(config.getboolean('日常任务', '关注6'))
            self.script.checkBox_28.setChecked(config.getboolean('日常任务', '关注7'))
            self.script.checkBox_29.setChecked(config.getboolean('日常任务', '关注8'))

            self.script.comboBox_9.setCurrentIndex(config.getint('日常任务', '优先级1'))
            self.script.comboBox_10.setCurrentIndex(config.getint('日常任务', '优先级2'))
            self.script.comboBox_11.setCurrentIndex(config.getint('日常任务', '优先级3'))
            self.script.comboBox_12.setCurrentIndex(config.getint('日常任务', '优先级4'))
            self.script.comboBox_13.setCurrentIndex(config.getint('日常任务', '优先级5'))
            self.script.comboBox_14.setCurrentIndex(config.getint('日常任务', '优先级6'))
            self.script.comboBox_15.setCurrentIndex(config.getint('日常任务', '优先级7'))
            self.script.comboBox_16.setCurrentIndex(config.getint('日常任务', '优先级8'))

            self.script.checkBox_30.setChecked(config.getboolean('日常任务', '生活技能艾草'))
            self.script.checkBox_31.setChecked(config.getboolean('日常任务', '生活技能莲子'))

            self.script.lineEdit.setText(config.get('日常任务', '副本喊话内容'))

            self.script.lineEdit_50.setText(config.get('日常任务', '闪避')),
            self.script.lineEdit_51.setText(config.get('日常任务', '关山')),
            self.script.lineEdit_52.setText(config.get('日常任务', '自创1')),
            self.script.lineEdit_53.setText(config.get('日常任务', '自创2')),
            self.script.lineEdit_54.setText(config.get('日常任务', '自创3')),

            self.script.textEdit.setText(config.get('日常任务', '技能逻辑'))

            self.script.lineEdit_4.setText(config.get('日常任务', '普攻'))
            self.script.lineEdit_9.setText(config.get('日常任务', '技能1'))
            self.script.lineEdit_5.setText(config.get('日常任务', '技能2'))
            self.script.lineEdit_6.setText(config.get('日常任务', '技能3'))
            self.script.lineEdit_7.setText(config.get('日常任务', '技能4'))
            self.script.lineEdit_8.setText(config.get('日常任务', '技能5'))
            self.script.lineEdit_10.setText(config.get('日常任务', '技能6'))
            self.script.lineEdit_11.setText(config.get('日常任务', '技能7'))
            self.script.lineEdit_12.setText(config.get('日常任务', '技能8'))
            self.script.lineEdit_13.setText(config.get('日常任务', '绝学'))

            self.script.checkBox_33.setChecked(config.getboolean('日常任务', '吴越剑坯'))
            self.script.checkBox_34.setChecked(config.getboolean('日常任务', '白公鼎坯'))
            self.script.checkBox_35.setChecked(config.getboolean('日常任务', '碧铜马坯'))
            self.script.checkBox_32.setChecked(config.getboolean('日常任务', '天幕雅苑'))

            self.script.checkBox_36.setChecked(config.getboolean('日常任务', '精制面粉'))
            self.script.checkBox_37.setChecked(config.getboolean('日常任务', '土鸡蛋'))
            self.script.checkBox_38.setChecked(config.getboolean('日常任务', '鲜笋'))
            self.script.checkBox_39.setChecked(config.getboolean('日常任务', '猪肉'))
            self.script.checkBox_40.setChecked(config.getboolean('日常任务', '糯米'))

            self.script.checkBox_14.setChecked(config.getboolean('日常任务', '切角色1'))
            self.script.checkBox_15.setChecked(config.getboolean('日常任务', '切角色2'))
            self.script.checkBox_16.setChecked(config.getboolean('日常任务', '切角色3'))
            self.script.checkBox_17.setChecked(config.getboolean('日常任务', '切角色4'))
            self.script.checkBox_18.setChecked(config.getboolean('日常任务', '切角色5'))

            self.script.lineEdit_55.setText(config.get('日常任务', '帮派')),
            self.script.lineEdit_56.setText(config.get('日常任务', '自创4'))

        except configparser.NoOptionError:
            pass
        except configparser.NoSectionError:
            pass
        except ValueError:
            pass
        except TypeError:
            pass
        except StopIteration:
            pass

    # 删除配置文件
    def delete_config(self, _):
        dialog = DelConfigDialog()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            file_name = dialog.get_selected_items()[1]

            user_path = os.path.expanduser('~')
            # 拼接文件路径
            config_path = os.path.join(user_path, 'ChronoSnow', 'TaskConfig')
            # self.update_log(f'加载{file_name}配置')
            # 拼接完整的文件路径和文件名
            config_path = f'{config_path}\\{file_name}.ini'

            # 删除配置文件
            os.remove(config_path)

            # 配置文件列表移除
            index = self.script.comboBox.findText(file_name)
            if index != -1:
                self.script.comboBox.removeItem(index)

    def save_system_config(self):
        # 获取当前用户的路径
        user_path = os.path.expanduser('~')

        # 拼接文件路径
        config_path = os.path.join(user_path, 'ChronoSnow', 'SystemConfig', 'System.ini')

        # 递归创建目录和文件
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        config = configparser.ConfigParser()
        # 添加一些配置项
        config['界面设置'] = {
            '当前配置': self.script.comboBox.currentText(),
            '开始快捷键': self.setting.lineEdit.text(),
        }

        with open(config_path, 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    # 读取system配置文件
    def load_system_config(self):
        # 获取当前用户的路径
        user_path = os.path.expanduser('~')

        # 拼接文件路径
        config_path = os.path.join(user_path, 'ChronoSnow', 'SystemConfig')

        os.makedirs(config_path, exist_ok=True)

        # 拼接文件路径
        task_config_path = os.path.join(user_path, 'ChronoSnow', 'TaskConfig')

        os.makedirs(task_config_path, exist_ok=True)

        # 获取config文件下文件名字
        ini_files = [os.path.splitext(f)[0] for f in os.listdir(task_config_path) if f.endswith('.ini')]

        self.script.comboBox.addItem('默认配置')
        # 加载配置文件
        for text in ini_files:
            self.script.comboBox.addItem(text)
        config = configparser.ConfigParser()
        # 读取配置文件
        config.read(f'{config_path}\\System.ini', encoding='utf-8')
        try:
            # 加载当前配置信息
            current_text = config.get('界面设置', '当前配置')
            if current_text in ini_files:
                self.script.comboBox.setCurrentText(current_text)

            self.setting.lineEdit.setText(config.get('界面设置', '开始快捷键'))
        except configparser.NoOptionError as e:
            logging.error(e)
        except configparser.NoSectionError as e:
            logging.error(e)

    # 重写关闭事件
    def closeEvent(self, event):
        if self.user_quit:
            reply = QMessageBox.question(self, '确认退出',
                                         "你确定要退出吗?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.No:
                event.ignore()
            else:
                self.save_system_config()
                self.run.unbind_all(None)
                self.user_quit = True
                services.stop.set()
                event.accept()
        else:
            self.save_system_config()
            self.run.unbind_all(None)
            self.user_quit = True
            services.stop.set()
            event.accept()


class LoginWindow(QWidget, Ui_Login):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.lineEdit.setValidator(QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9]{16}"), self.lineEdit))
        self.lineEdit_3.setValidator(
            QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9]{16}"), self.lineEdit_3))
        self.lineEdit_2.setValidator(
            QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9.@#%*+*/]{16}"), self.lineEdit_2))
        self.lineEdit_4.setValidator(
            QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9.@#%*+*/]{16}"), self.lineEdit_4))
        self.lineEdit_6.setValidator(
            QRegularExpressionValidator(QRegularExpression("[a-zA-Z0-9.@#%*+*/]{16}"), self.lineEdit_6))

        self.lineEdit_2.setEchoMode(QLineEdit.EchoMode.Password)
        self.lineEdit_4.setEchoMode(QLineEdit.EchoMode.Password)
        self.lineEdit_6.setEchoMode(QLineEdit.EchoMode.Password)
        self.load_user_config()
        self.initWindow()
        self.auto_login()

    def initWindow(self):
        self.setWindowIcon(QIcon('app/images/icon/favicon.ico'))
        self.setWindowTitle(f'时雪{VERSION}')
        self.login_button.clicked.connect(self.start_login)
        self.signup_button.clicked.connect(self.start_signup)
        self.button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.button_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.show()

    def auto_login(self):
        if self.checkBox_2.isChecked():
            self.login_button.click()

    def start_login(self, _):
        self.login_button.setText('登录中')
        threading.Thread(target=self.login).start()

    def start_signup(self, _):
        self.signup_button.setText('注册中')
        threading.Thread(target=self.signup).start()

    def signup(self):
        username = self.lineEdit_3.text()
        password = self.lineEdit_4.text()
        again_password = self.lineEdit_6.text()
        if len(username) < 4:
            self.label_2.setText('账号长度不能小于4位')
            self.signup_button.setText('注册')
            return 0
        elif len(password) < 8:
            self.label_2.setText('密码长度不能小于8位')
            self.signup_button.setText('注册')
            return 0
        elif password != again_password:
            self.label_2.setText('两次密码不一致')
            self.signup_button.setText('注册')
            return 0
        success, message = services.signup(username, password)
        if not success:
            print('注册失败')
            self.label_2.setText(message)
        elif success:
            print('注册成功')
            self.stackedWidget.setCurrentIndex(0)
            self.lineEdit.setText(username)
            self.lineEdit_2.setText(password)
        self.signup_button.setText('注册')

    def login(self):
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()
        success, message = services.login(username, password)
        if success:
            publicSingle.login.emit(username)
            threading.Thread(target=services.heartbeat, args=(username,)).start()
        else:
            self.label.setText(message)
        self.login_button.setText('登录')

    def save_user_config(self):
        # 获取当前用户的路径
        user_path = os.path.expanduser('~')

        # 拼接文件路径
        config_path = os.path.join(user_path, 'ChronoSnow', 'SystemConfig', 'User.ini')

        # 递归创建目录和文件
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        config = configparser.ConfigParser()
        # 添加一些配置项
        config['User'] = {
            'username': self.lineEdit.text(),
            'password': self.lineEdit_2.text(),
            '记住密码': self.checkBox.isChecked(),
            '自动登录': self.checkBox_2.isChecked()
        }

        with open(config_path, 'w', encoding='utf-8') as configfile:
            config.write(configfile)

    def load_user_config(self):
        # 获取当前用户的路径
        user_path = os.path.expanduser('~')

        # 拼接文件路径
        user_config_path = os.path.join(user_path, 'ChronoSnow', 'SystemConfig')

        os.makedirs(user_config_path, exist_ok=True)
        config = configparser.ConfigParser()
        # 读取配置文件
        config.read(f'{user_config_path}\\User.ini', encoding='utf-8')
        try:
            # 加载当前配置信息
            username = config.get('User', 'username')
            password = config.get('User', 'password')
            if config.getboolean('User', '记住密码'):
                self.checkBox.setChecked(config.getboolean('User', '记住密码'))
                self.lineEdit.setText(username)
                self.lineEdit_2.setText(password)
            if config.getboolean('User', '自动登录'):
                self.checkBox_2.setChecked(config.getboolean('User', '自动登录'))
        except configparser.NoSectionError as e:
            print(e)

    def closeEvent(self, event):
        self.save_user_config()


class HomeWindow(QWidget, Ui_Home):
    def __init__(self):
        super().__init__()

        self.setupUi(self)
        self.link_show()
        self.text_show()

    def text_show(self):
        # 创建一个QLabel用于显示文本
        text = """
        <p>tips：</p>
        <h3>[成员注意事项]</h3>
        <p>- 为了更好的使用体验，脚本使用过程中，遇到问题请及时反馈给群主。</p>
        <p>- 群人数有限，当本群人数达到四百人开始月末清理两个月以上不活跃成员，请适当冒泡。</p>
        <p>- 反馈问题需有明确的逻辑关系，具体说明某某任务卡在哪里。为了更快定位问题不要只说什么任务卡住了。</p>
        <p>- 需要解决问题,到群文件下todesk的免安装工具,官网自己下也行,不要给我发QQ远程。</p>
    
        <h3>[脚本基本使用教程]</h3>
        <p>- 脚本只可以在pc端下运行。套壳pc需找到游戏安装路径启动。不懂群里问。</p>
        <p>- 游戏设置夜泊模式</p>
        <p>- 游戏窗口可以被遮挡/重叠。但游戏窗口不能出现在屏幕外。</p>
        <p>- 脚本通过侧边栏切换页面，双击添加/移除 任务。</p>
        <p>- 任务准备完成。请切换到运行页面。点击开始后，脚本将会在1秒后绑定鼠标下面的窗口，请注意鼠标移动。</p>
        <p>- 双击角色信息 对应窗口会在一秒后展示在最上方。</p>
        <p>- 绑定前手动通过键盘打开背包,确认键盘是否可用</p>
        """
        # 创建一个QLabel用于显示文本
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)  # 启用自动换行

        # 将文本添加到布局中
        layout = QVBoxLayout()
        layout.addWidget(self.text_label)
        self.widget_3.setLayout(layout)

    def link_show(self):
        self.label.setPixmap(QPixmap('app/images/svg/bilibili.svg'))
        # 创建一个QLabel用于显示超链接文本
        self.label_2.setText('<a href="https://www.bilibili.com/video/BV1Hb421a7Ym">点此观看视频教程</a>')

        # self.label_2.setFont(18)

        # 设置文本允许打开超链接
        self.label_2.setOpenExternalLinks(True)

        # 为超链接文本连接点击事件
        self.label_2.linkActivated.connect(self.open_link)

    @staticmethod
    # 定义超链接点击事件的槽函数
    def open_link(url):
        QDesktopServices.openUrl(QUrl(url))


class ScriptWindow(QWidget, Ui_Script):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initWindow()
        # 重构输入框
        self.replace_widget(CustomLineEdit, self.lineEdit_4)
        # self.lineEdit_4 = CustomLineEdit(self.lineEdit_4)
        self.replace_widget(CustomLineEdit, self.lineEdit_9)
        self.replace_widget(CustomLineEdit, self.lineEdit_5)
        self.replace_widget(CustomLineEdit, self.lineEdit_6)
        self.replace_widget(CustomLineEdit, self.lineEdit_7)
        self.replace_widget(CustomLineEdit, self.lineEdit_8)
        self.replace_widget(CustomLineEdit, self.lineEdit_10)
        self.replace_widget(CustomLineEdit, self.lineEdit_11)
        self.replace_widget(CustomLineEdit, self.lineEdit_12)
        self.replace_widget(CustomLineEdit, self.lineEdit_13)
        self.replace_widget(CustomLineEdit, self.lineEdit_14)
        self.replace_widget(CustomLineEdit, self.lineEdit_15)
        self.replace_widget(CustomLineEdit, self.lineEdit_16)
        self.replace_widget(CustomLineEdit, self.lineEdit_17)
        self.replace_widget(CustomLineEdit, self.lineEdit_18)
        self.replace_widget(CustomLineEdit, self.lineEdit_50)
        self.replace_widget(CustomLineEdit, self.lineEdit_51)
        self.replace_widget(CustomLineEdit, self.lineEdit_52)
        self.replace_widget(CustomLineEdit, self.lineEdit_53)
        self.replace_widget(CustomLineEdit, self.lineEdit_54)
        self.replace_widget(CustomLineEdit, self.lineEdit_55)
        self.replace_widget(CustomLineEdit, self.lineEdit_56)

        self.completer = QCompleter(self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)  # 不区分大小写
        self.replace_widget(TextEdit, self.textEdit)
        self.textEdit.setCompleter(self.completer)

    def replace_widget(self, new_widget, old_widget):
        # 创建一个新的TextEdit控件
        new_text_edit = new_widget()

        # 获取原始textEdit的父对象
        parent_widget = old_widget.parent()

        # 获取原始textEdit的索引
        index = parent_widget.layout().indexOf(old_widget)

        # 删除原始textEdit控件
        parent_widget.layout().removeWidget(old_widget)
        old_widget.deleteLater()

        # 将新的textEdit控件添加到相同的位置
        parent_widget.layout().insertWidget(index, new_text_edit)

        # 更新属性引用
        if hasattr(self, old_widget.objectName()):
            setattr(self, old_widget.objectName(), new_text_edit)

    def guide(self, _):
        text = self.listWidget_2.selectedItems()[0].text()
        try:
            self.stackedWidget.setCurrentIndex(TASK_SHOW[text][0])
            self.scrollArea.verticalScrollBar().setValue(TASK_SHOW[text][1])
        except KeyError:
            pass

    def guide_2(self, _):
        text = self.listWidget.selectedItems()[0].text()
        try:
            self.stackedWidget.setCurrentIndex(TASK_SHOW[text][0])
            self.scrollArea.verticalScrollBar().setValue(TASK_SHOW[text][1])
        except KeyError:
            pass

    def add(self, _):
        selected_items = self.listWidget_2.selectedItems()
        for item in selected_items:
            # 判断执行列表是否已有
            if item.text() not in [self.listWidget.item(i).text() for i in range(self.listWidget.count())]:
                if item.text() not in EXCLUDE_ADD_TASK:
                    self.listWidget.addItem(item.text())
            else:
                QMessageBox.warning(self, "警告", "该项已存在！")

    def remove(self, _):
        # 获取执行列表选中项
        selected_items = self.listWidget.selectedItems()
        for item in selected_items:
            row = self.listWidget.row(item)
            self.listWidget.takeItem(row)

    def clear(self, _):
        self.listWidget.clear()

    def initWindow(self):
        [self.listWidget_2.addItem(text) for text in TASK_MAPPING.keys() if text not in EXCLUDE_TASK_MAPPING]

        self.listWidget_2.clicked.connect(self.guide)
        self.listWidget.clicked.connect(self.guide_2)
        self.listWidget.doubleClicked.connect(self.remove)
        self.listWidget_2.doubleClicked.connect(self.add)
        self.pushButton_2.clicked.connect(self.clear)

        self.spinBox.setMaximum(2000)
        self.spinBox_2.setMaximum(2000)
        self.spinBox_3.setMaximum(2000)

        self.spinBox.setValue(100)
        self.spinBox_2.setValue(50)
        self.spinBox_3.setValue(100)

        self.spinBox_8.setMaximum(20)
        self.spinBox_8.setMinimum(1)
        self.spinBox_8.setValue(1)

        self.spinBox_9.setMinimum(1)
        self.spinBox_9.setMaximum(5)

        self.comboBox_9.setCurrentIndex(0)
        self.comboBox_10.setCurrentIndex(1)
        self.comboBox_11.setCurrentIndex(2)
        self.comboBox_12.setCurrentIndex(3)
        self.comboBox_13.setCurrentIndex(4)
        self.comboBox_14.setCurrentIndex(5)
        self.comboBox_15.setCurrentIndex(6)
        self.comboBox_16.setCurrentIndex(7)
        # self.checkBox.stateChanged.connect(lambda: self.task_append('课业任务'))
        # self.checkBox_2.stateChanged.connect(lambda: self.task_append('帮派任务'))
        # self.checkBox_3.stateChanged.connect(lambda: self.task_append('世界喊话'))
        # self.checkBox_4.stateChanged.connect(lambda: self.task_append('江湖英雄榜'))
        # self.checkBox_9.stateChanged.connect(lambda: self.task_append('日常副本'))
        # self.checkBox_10.stateChanged.connect(lambda: self.task_append('悬赏任务'))
        # self.checkBox_5.stateChanged.connect(lambda: self.task_append('侠缘喊话'))

        # self.pushButton.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        # self.pushButton_2.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        # self.pushButton_3.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        # self.pushButton_7.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        # self.pushButton_7.clicked.connect(lambda: self.show_page(1, 0))
        # self.pushButton_8.clicked.connect(lambda: self.show_page(1, 0))
        # self.pushButton_10.clicked.connect(lambda: self.show_page(1, 105))
        # self.pushButton_9.clicked.connect(lambda: self.show_page(1, 365))

        # 启用拖拽
        self.listWidget.setDragEnabled(True)
        # 设置拖拽模式为内部移动
        self.listWidget.setDragDropMode(QListWidget.DragDropMode.InternalMove)

    # def show_page(self, index, height):
    #     self.stackedWidget.setCurrentIndex(index)
    #     self.scrollArea_3.verticalScrollBar().setValue(height)


class RunWindow(QWidget, Ui_Run):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.struct_task_dict = {}
        self.mask_window = {}
        self.initWindow()
        self.start = StartTask()

        # connect button
        self.StartButton.clicked.connect(self.start_task)
        self.StopButton.clicked.connect(self.stop)
        self.ResumeButton.clicked.connect(self.resume)
        self.UnbindButton.clicked.connect(self.unbind)
        self.StopAllButton.clicked.connect(self.stop_all)
        self.ResumeAllButton.clicked.connect(self.resume_all)
        self.UnbindAllButton.clicked.connect(self.unbind_all)
        self.PersonaTableWidget.doubleClicked.connect(self.win_up)
        publicSingle.state.connect(self.set_state)
        publicSingle.journal.connect(self.journal)
        publicSingle.set_character.connect(self.set_character)
        publicSingle.start.connect(self.start_task)

        try:
            self.journal([-1, f'当前版本: {open("../version.txt").read()}'])
        except FileNotFoundError:
            pass

    # initialization
    def initWindow(self):
        # 设置运行角色信息表格的行和列
        self.PersonaTableWidget.setRowCount(10)
        self.PersonaTableWidget.setColumnCount(2)

        # 设置状态栏
        for i in range(10):
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.PersonaTableWidget.setItem(i, 1, item)

        # 禁止多选
        self.PersonaTableWidget.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        # 禁止编辑整个表格
        self.PersonaTableWidget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        # 禁用垂直滚动条
        self.PersonaTableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 禁用水平滚动条
        self.PersonaTableWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 设置列标题
        self.PersonaTableWidget.setHorizontalHeaderLabels(['角色信息', '任务状态'])

        # 设置表格部件所有列自动拉伸
        self.PersonaTableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

        # 设置表格部件所有行自动拉伸
        self.PersonaTableWidget.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)

    # 窗口调度
    def win_up(self, _):
        time.sleep(1)
        row = self.PersonaTableWidget.currentIndex().row()
        col = self.PersonaTableWidget.currentIndex().column()
        if row in self.struct_task_dict and col == 0:
            user = self.struct_task_dict[row]
            # 将窗口显示在前台
            win32gui.PostMessage(user.handle, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
            # # 如果需要，可以使用SetForegroundWindow来将窗口置于前台
            win32gui.SetForegroundWindow(user.handle)
            time.sleep(0.1)
            user.mask_window.activateWindow()
            # # # win32gui.PostMessage(user.mask_window.winId(), win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)
            # # # 如果需要，可以使用SetForegroundWindow来将窗口置于前台
            # win32gui.SetForegroundWindow(user.mask_window.winId())

    def journal(self, message):
        # 获取当前时间
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")

        if message[0] != -1:
            # 构造带有时间信息的文本
            formatted_message = f"[{current_time}] [窗口: {message[0] + 1} ] >>> {message[1]}\n"
        else:
            # 构造带有时间信息的文本
            formatted_message = f"[{current_time}] >>> {message[1]}\n"

        # 在文本框中追加信息
        self.textEdit.insertPlainText(formatted_message)

        # 滚动到最底部
        scrollbar = self.textEdit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def stop(self, _):
        try:
            row = self.PersonaTableWidget.currentIndex().row()
            if not self.struct_task_dict[row].flag:
                publicSingle.stop.emit(row)
                self.struct_task_dict[row].close()
                self.struct_task_dict[row].flag = True
        except KeyError:
            pass
        except Exception as e:
            logging.error(e)

    def stop_all(self, _):
        for index in range(10):
            self.PersonaTableWidget.setCurrentCell(index, 0)
            self.stop(None)

    def resume(self, _):
        try:
            row = self.PersonaTableWidget.currentIndex().row()
            if self.struct_task_dict[row].flag:
                publicSingle.resume.emit(row)
                self.struct_task_dict[row].resume()
                self.struct_task_dict[row].flag = False
        except KeyError:
            pass
        except Exception as e:
            logging.error(e)

    def resume_all(self, _):
        for index in range(10):
            self.PersonaTableWidget.setCurrentCell(index, 0)
            self.resume(None)

    def unbind(self, _):
        try:
            row = self.PersonaTableWidget.currentIndex().row()
            if row in self.struct_task_dict.keys():
                if self.struct_task_dict[row].flag:
                    self.resume(None)
                publicSingle.unbind.emit(row)
                self.struct_task_dict[row].close()
                self.remove_character()
                del self.struct_task_dict[row]
        except KeyError:
            pass
        except Exception as e:
            logging.error(e)

    def unbind_all(self, _):
        for index in range(10):
            self.PersonaTableWidget.setCurrentCell(index, 0)
            self.unbind(None)

    def set_state(self, message):
        item = QTableWidgetItem(message[1])
        # 设置字号为11号
        font = item.font()
        font.setPointSize(11)
        item.setFont(font)

        # 设置文字居中
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.PersonaTableWidget.setItem(message[0], 1, item)

    def set_character(self, row):
        try:
            temp_dir = tempfile.gettempdir()
            temp_img_path = os.path.join(temp_dir, f'template_image{row}')
            image = basic_functional.screen_shot(self.struct_task_dict[row].handle)
            cv2.imwrite(f'{temp_img_path}\\1.bmp', image)
            # 定义要保留的区域的坐标和尺寸
            x, y, width, height = 117, 730, 115, 20

            # 从原始截图中复制指定区域
            img = image[y:y + height, x:x + width]
            cv2.imwrite(f'{temp_img_path}\\person_{row}.bmp', img)

            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            # # 创建 QPixmap 对象，加载图片
            # pixmap = QPixmap(f'{temp_img_path}\\person_{row}.bmp')
            # 将OpenCV图像转换为Qt可用的格式
            height, width, channel = img.shape
            bytesPerLine = 3 * width
            qImg = QPixmap.fromImage(QImage(img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888))

            # 将图片添加到表格中
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DecorationRole, qImg)  # 使用 Qt::DecorationRole 用于图像数据
            self.PersonaTableWidget.setItem(row, 0, item)
        except KeyError:
            pass

    def remove_character(self):
        try:
            row = self.PersonaTableWidget.currentIndex().row()
            # 获取 QTableWidgetItem
            item = self.PersonaTableWidget.item(row, 0)
            # 移除图像，将图像数据设置为 None 或空 QPixmap
            if item is not None:
                item.setData(Qt.ItemDataRole.DecorationRole, None)

            item = self.PersonaTableWidget.item(row, 1)

            item.setText(None)
        except AttributeError:
            pass

    def window_inspection(self):
        row = self.PersonaTableWidget.currentIndex().row()
        handle = basic_functional.get_handle()
        if handle is not None and handle not in [object.handle for object in self.struct_task_dict.values()]:
            if row == -1 or row in self.struct_task_dict.keys():
                if empty_window := [item for item in TABLE_WINDOW if item not in self.struct_task_dict.keys()]:
                    row = empty_window[0]
                else:
                    TimingQMessageBox.information(self, '提示', '游戏窗口已经被绑定')
                    return -1
        elif handle is None:
            TimingQMessageBox.information(self, '提示', '请勿绑定非游戏窗口')
            return -1
        else:
            TimingQMessageBox.information(self, '提示', '游戏窗口已经被绑定')
            return -1
        self.struct_task_dict[row] = StructureTask(row, handle)
        return row

    def start_task(self, _):
        try:
            if (row := self.window_inspection()) != -1:
                # self.set_character(self.struct_task_dict[row].row)
                Thread(target=self.start.start, args=(row, self.struct_task_dict[row].handle)).start()
        except Exception as e:
            logging.error(f'线程发生错误{e}')


class SettingWindow(QWidget, Ui_Setting):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.parent = parent
        # 重构输入框
        self.replace_widget(ShortCutLineEdit, self.lineEdit)
        # print(int(self.parent.winId()))
        # 信号连接
        self.lineEdit.textChanged.connect(lambda: self.short_cut_key(START_ID))
        # basic_functional.update_hotkey(self.parent.winId(), 2, 'Ctrl+P')

    def short_cut_key(self, ID):
        sender_widget = self.sender()
        if sender_widget is not None and isinstance(sender_widget, QtWidgets.QLineEdit):
            basic_functional.update_hotkey(self.parent.winId(), ID, sender_widget.text())

    def replace_widget(self, new_widget, old_widget):
        # 创建一个新的TextEdit控件
        new_text_edit = new_widget()

        # 获取原始textEdit的父对象
        parent_widget = old_widget.parent()

        # 获取原始textEdit的索引
        index = parent_widget.layout().indexOf(old_widget)

        # 将新的textEdit控件添加到相同的位置
        parent_widget.layout().insertWidget(index, new_text_edit)

        # 删除原始textEdit控件
        parent_widget.layout().removeWidget(old_widget)
        old_widget.deleteLater()

        # 更新属性引用
        if hasattr(self, old_widget.objectName()):
            setattr(self, old_widget.objectName(), new_text_edit)


class EditorWindow(QWidget, Ui_Editor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.textDict = {
            '界面判断': '界面判断()',
            '目标创建': '目标(目标:N)'
        }
        self.parameters = {
            '界面判断': ['图片名称'],
            '目标创建': []
        }
        self.treeWidget.setDragEnabled(True)

        # 连接拖放事件处理函数
        self.treeWidget.itemPressed.connect(self.startDrag)

    def startDrag(self, event, _):
        cursor = self.textEdit.textCursor()
        cursor.select(cursor.SelectionType.BlockUnderCursor)
        old_text = cursor.selectedText()
        self.clearCursor()
        item = self.treeWidget.currentItem()

        mimeData = QMimeData()
        mimeData.setText(self.textProcess(item.text(0), old_text))
        drag = QDrag(self.treeWidget)
        drag.setMimeData(mimeData)
        drag.exec(Qt.DropAction.CopyAction)

        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine,  QTextCursor.MoveMode.MoveAnchor)
        self.textEdit.setTextCursor(cursor)
        self.textEdit.setFocus()

    def textProcess(self, text, old_text):

        if text in self.textDict:
            return self.textDict[text]

        if text.split(':')[0] in self.parameters.get(old_text.split('(')[0].strip()) if\
                self.parameters.get(old_text.split('(')[0].strip()) is not None else []:
            if (text.split(':')[0] not in
                    [s.strip() for s in old_text[old_text.find('(') + 1:old_text.rfind(')')].split(':')]):
                new_text = old_text[:old_text.rfind(')')] + ' ' + text + old_text[old_text.rfind(')'):]
                # new_text = old_text.replace(')', ' ' + text + ')', 1)
            else:
                return old_text
        else:
            return old_text
        return new_text

    def clearCursor(self):
        cursor = self.textEdit.textCursor()
        # 移动光标到当前行的开头并保持选中状态
        cursor.movePosition(QTextCursor.MoveOperation.StartOfLine,  QTextCursor.MoveMode.KeepAnchor)

        # 将光标移动到当前行的末尾并保持选中状态
        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor)
        cursor.insertText('    ')
        # # 将光标设置回行首
        # cursor.movePosition(QTextCursor.MoveOperation.StartOfLine, QTextCursor.MoveMode.KeepAnchor)
        # # 插入一个空的字符串来保留当前行
        self.textEdit.setTextCursor(cursor)


# 任务信息构造类
class StructureTask:
    def __init__(self, row, handle):
        self.row = row
        self.handle = handle
        self.flag = False
        self.set_window()
        self.mask_window = MaskWindow(self.handle)
        self.create_temp_file()
        self.block_window()
        publicSingle.write_json.emit(self.row)

    def set_window(self):
        scale_factor = QApplication.primaryScreen().devicePixelRatio()
        basic_functional.set_window(self.handle, DPI_MAPP[scale_factor])

    def create_temp_file(self):
        temp_dir = tempfile.gettempdir()
        temp_img_path = os.path.join(temp_dir, f'template_image{self.row}')

        shutil.rmtree(temp_img_path, ignore_errors=True)  # 删除目标文件夹及其内容，如果存在的话
        shutil.copytree(f'app/images/Img', temp_img_path)

    def block_window(self):
        if Mask:
            basic_functional.DisableTheWindow(self.handle)
            basic_functional.DisableTheWindow(self.mask_window.winId())

    def unblock_window(self):
        if Mask:
            basic_functional.UnDisableTheWindow(self.handle)
            basic_functional.UnDisableTheWindow(self.mask_window.winId())

    def resume(self):
        self.mask_window.mask_show()
        self.block_window()

    def close(self):
        self.unblock_window()
        self.mask_window.close()


class MaskWindow(QWidget):

    def __init__(self, handle):
        super().__init__()
        self.handle = handle
        # self.setWindowTitle(f"Child Window {self.row}")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowOpacity(0.35)  # 设置窗口半透明
        self.mask_show()
        self.activateWindow()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.RenderHint.Antialiasing)
        # painter.fillRect(self.rect(), QColor(0, 0, 0, 255))  # 半透明黑色背景

        font = QFont()
        font.setPointSize(50)
        painter.setFont(font)

        text = "脚本正在运行中 ✅\n窗口绑定成功😄\n鼠标可以移动啦🖱️\n使用过程中有任何问题 Q群:744646753✅"
        text_rect = painter.boundingRect(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
        painter.setPen(QColor(255, 0, 0))  # 设置画笔颜色为红色
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)

    def mask_show(self):
        rect = GetWindowRect(self.handle)
        scale_factor = QApplication.primaryScreen().devicePixelRatio()
        x, y, width, height = int(rect[0] / scale_factor), int(rect[1] / scale_factor), int(
            rect[2] / scale_factor), int(rect[3] / scale_factor)
        self.setFixedSize(width - x + 20, height - y + 20)
        self.move(x - 15, y - 15)
        if Mask:
            self.show()

    def mask_close(self):
        self.close()
