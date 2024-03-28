import json
import os
import shutil
import tempfile
import time
import configparser
import cv2
import logging
from PyQt6 import QtWidgets
from PyQt6.QtCore import QSize, QObject, Qt, QRect, QTimer, pyqtSignal, QSettings, QDateTime
from PyQt6.QtGui import QIcon, QPainter, QColor, QFont, QPixmap, QStandardItem
from PyQt6.QtWidgets import QWidget, QScrollArea, QPushButton, QVBoxLayout, QApplication, QDialog, QTableWidget, \
    QTableWidgetItem, QMessageBox, QListWidgetItem, QListWidget, QLineEdit, QDialogButtonBox
from win32gui import GetWindowRect
from threading import Thread

from app.view.Ui.MainWindow import Ui_MainWindow
from app.view.Ui.HomeWindow import Ui_Home
from app.view.Ui.ScriptWindow import Ui_Script
from app.view.Ui.RunWindow import Ui_Run
from app.Script.BasicFunctional import basic_functional
from app.Script.Task import StartTask, TASK_MAPPING, TASK_SHOW
from app.view.Public import publicSingle, TABLE_WINDOW, DPI_MAPP


class MainWindow(QWidget, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initWindow()
        self.index = 0
        # self.menu_layout = QVBoxLayout(self.menu_widget)

        # create window
        self.home = HomeWindow()
        self.script = ScriptWindow()
        self.run = RunWindow()
        # self.main_layout = QVBoxLayout(self.main_widget)

        self.initNavigation()
        self.script.pushButton_11.clicked.connect(self.save_config)
        self.script.pushButton_12.clicked.connect(self.delete_config)

        self.script.comboBox.currentTextChanged.connect(self.load_config)
        self.load_config('')
        self.load_system_config()
        self.main_widget.setCurrentIndex(0)

    # add window
    def initNavigation(self):
        self.addSubInterface(self.home, '主页')
        self.addSubInterface(self.script, '脚本')
        self.addSubInterface(self.run, '运行')
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

    # SwitchPages
    def show_page(self, index):
        """
        show page for index
        :param index:
        :return:
        """
        self.main_widget.setCurrentIndex(index)

    def return_data(self):
        return {
            "执行列表": [self.script.listWidget.item(i).text() for i in range(self.script.listWidget.count())],
            "世界喊话内容": self.script.lineEdit_2.text(),
            "世界喊话次数": self.script.spinBox_4.value(),
            "江湖英雄榜次数": self.script.spinBox_7.value(),
            "副本人数": self.script.comboBox_2.currentIndex(),
            "副本自动匹配": self.script.checkBox.isChecked(),
            "侠缘昵称": self.script.lineEdit_3.text(),
            # "侠缘喊话内容": self.script.textEdit_2.toPlainText(),
            "山河器": self.script.checkBox_2.isChecked(),
            "帮派铜钱捐献": self.script.checkBox_8.isChecked(),
            "帮派银两捐献": self.script.checkBox_7.isChecked(),
            "银票礼盒": self.script.checkBox_3.isChecked(),
            "商会鸡蛋": self.script.checkBox_5.isChecked(),
            "榫头卯眼": self.script.checkBox_6.isChecked(),
            "锦芳绣残片": self.script.checkBox_4.isChecked(),
            "摇钱树": self.script.checkBox_9.isChecked(),
            "摇钱树目标": self.script.comboBox_3.currentIndex(),
            "扫摆摊延迟1": self.script.spinBox.value(),
            "扫摆摊延迟2": self.script.spinBox_2.value(),
            "扫摆摊延迟3": self.script.spinBox_3.value(),
            "华山论剑次数": self.script.spinBox_6.value(),
            "华山论剑秒退": self.script.checkBox_10.isChecked(),
            "背包": self.script.lineEdit_14.text(),
            "好友": self.script.lineEdit_15.text(),
            "队伍": self.script.lineEdit_16.text(),
            "地图": self.script.lineEdit_17.text(),
            "设置": self.script.lineEdit_18.text(),
            "采集线数": self.script.comboBox_7.currentIndex(),
            "指定地图": self.script.comboBox_8.currentText(),
            "采集加速延迟": self.script.spinBox_8.value(),
            "切角色1": self.script.checkBox_14.isChecked(),
            "切角色2": self.script.checkBox_15.isChecked(),
            "切角色3": self.script.checkBox_16.isChecked(),
            "切角色4": self.script.checkBox_17.isChecked(),
            "切角色5": self.script.checkBox_18.isChecked(),
            "混队模式": self.script.checkBox_13.isChecked(),
            "采集方法": next(button.text() for button in [self.script.radioButton, self.script.radioButton_2,
                                                          self.script.radioButton_6] if button.isChecked()),
            "采集种类": next(button.text() for button in [self.script.radioButton_3, self.script.radioButton_4,
                                                          self.script.radioButton_5] if button.isChecked()),
            "采集目标": next(combo.currentText() for button, combo in
                             zip([self.script.radioButton_3, self.script.radioButton_4, self.script.radioButton_5],
                                 [self.script.comboBox_5, self.script.comboBox_4, self.script.comboBox_6]) if button.isChecked()),
            "自定义采集坐标": [(self.script.lineEdit_19.text(), self.script.lineEdit_20.text()),
                         (self.script.lineEdit_21.text(), self.script.lineEdit_22.text()),
                         (self.script.lineEdit_23.text(), self.script.lineEdit_24.text()),
                         (self.script.lineEdit_25.text(), self.script.lineEdit_26.text()),
                         (self.script.lineEdit_27.text(), self.script.lineEdit_28.text()),
                         (self.script.lineEdit_29.text(), self.script.lineEdit_30.text())],
            "技能列表": [
                self.script.lineEdit_4.text(),
                self.script.lineEdit_9.text(),
                self.script.lineEdit_5.text(),
                self.script.lineEdit_6.text(),
                self.script.lineEdit_7.text(),
                self.script.lineEdit_8.text(),
                self.script.lineEdit_10.text(),
                self.script.lineEdit_11.text(),
                self.script.lineEdit_12.text(),
                self.script.lineEdit_13.text(),

            ]

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
    def initWindow(self):
        publicSingle.write_json.connect(self.write_task_json)
        self.resize(1000, 610)
        self.setMinimumWidth(1000)
        # self.setWindowIcon(QIcon(':/gallery/images/logo.png'))
        self.setWindowTitle('时雪')

        desktop = QApplication.screens()[0].availableGeometry()
        w, h = desktop.width(), desktop.height()
        self.move(w // 2 - self.width() // 2, h // 2 - self.height() // 2)
        self.show()
        QApplication.processEvents()

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
                # 创建配置对象
                config = configparser.ConfigParser()

                # 添加一些配置项
                config['日常任务'] = {
                    "执行列表": [self.script.listWidget.item(i).text() for i in range(self.script.listWidget.count())],
                    "世界喊话内容": self.script.lineEdit_2.text(),
                    "世界喊话次数": self.script.spinBox_4.value(),
                    "江湖英雄榜次数": self.script.spinBox_7.value(),
                    "副本人数": self.script.comboBox_2.currentIndex(),
                    "副本自动匹配": self.script.checkBox.isChecked(),
                    "侠缘昵称": self.script.lineEdit_3.text(),
                    # "侠缘喊话内容": self.script.textEdit_2.toPlainText(),
                    "山河器": self.script.checkBox_2.isChecked(),
                    "帮派铜钱捐献": self.script.checkBox_8.isChecked(),
                    "帮派银两捐献": self.script.checkBox_7.isChecked(),
                    "银票礼盒": self.script.checkBox_3.isChecked(),
                    "商会鸡蛋": self.script.checkBox_5.isChecked(),
                    "榫头卯眼": self.script.checkBox_6.isChecked(),
                    "锦芳绣残片": self.script.checkBox_4.isChecked(),
                    "摇钱树": self.script.checkBox_9.isChecked(),
                    "摇钱树目标": self.script.comboBox_3.currentIndex(),
                    "扫摆摊延迟1": self.script.spinBox.value(),
                    "扫摆摊延迟2": self.script.spinBox_2.value(),
                    "扫摆摊延迟3": self.script.spinBox_3.value(),
                    "华山论剑次数": self.script.spinBox_6.value(),
                    "华山论剑秒退": self.script.checkBox_10.isChecked(),
                    "背包": self.script.lineEdit_14.text(),
                    "好友": self.script.lineEdit_15.text(),
                    "队伍": self.script.lineEdit_16.text(),
                    "地图": self.script.lineEdit_17.text(),
                    "设置": self.script.lineEdit_18.text(),
                    "采集线数": self.script.comboBox_7.currentIndex(),
                    "指定地图": self.script.comboBox_8.currentText(),
                    "采集加速延迟": self.script.spinBox_8.value(),
                    "地图搜索": self.script.radioButton_2.isChecked(),
                    "定点采集": self.script.radioButton_6.isChecked(),
                    "自定义坐标采集": self.script.radioButton.isChecked(),
                    "采草": self.script.radioButton_4.isChecked(),
                    "采草目标": self.script.comboBox_4.currentIndex(),
                    "伐木": self.script.radioButton_3.isChecked(),
                    "伐木目标": self.script.comboBox_5.currentIndex(),
                    "挖矿": self.script.radioButton_5.isChecked(),
                    "挖矿目标": self.script.comboBox_6.currentIndex(),
                    "切角色1": self.script.checkBox_14.isChecked(),
                    "切角色2": self.script.checkBox_15.isChecked(),
                    "切角色3": self.script.checkBox_16.isChecked(),
                    "切角色4": self.script.checkBox_17.isChecked(),
                    "切角色5": self.script.checkBox_18.isChecked(),
                    "混队模式": self.script.checkBox_13.isChecked(),
                    "自定义采集坐标": [
                        (self.script.lineEdit_19.text(), self.script.lineEdit_20.text()),
                        (self.script.lineEdit_21.text(), self.script.lineEdit_22.text()),
                        (self.script.lineEdit_23.text(), self.script.lineEdit_24.text()),
                        (self.script.lineEdit_25.text(), self.script.lineEdit_26.text()),
                        (self.script.lineEdit_27.text(), self.script.lineEdit_28.text()),
                        (self.script.lineEdit_29.text(), self.script.lineEdit_30.text())
                    ],
                    "技能列表": [
                        self.script.lineEdit_4.text(),
                        self.script.lineEdit_9.text(),
                        self.script.lineEdit_5.text(),
                        self.script.lineEdit_6.text(),
                        self.script.lineEdit_7.text(),
                        self.script.lineEdit_8.text(),
                        self.script.lineEdit_10.text(),
                        self.script.lineEdit_11.text(),
                        self.script.lineEdit_12.text(),
                        self.script.lineEdit_13.text(),

                    ],

                }

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

        # 创建配置对象
        config = configparser.ConfigParser()
        # 读取配置文件
        config.read(config_path, encoding='utf-8')
        if file_name == '':
            self.script.listWidget.clear()
            self.script.lineEdit_2.setText('')
            self.script.spinBox_4.setValue(1)
            self.script.spinBox_7.setValue(1)
            self.script.comboBox_2.setCurrentIndex(0)
            self.script.checkBox.setChecked(False)
            self.script.lineEdit_3.setText('')
            self.script.checkBox_2.setChecked(False)
            self.script.checkBox_8.setChecked(False)
            self.script.checkBox_7.setChecked(False)
            self.script.checkBox_5.setChecked(False)
            self.script.checkBox_3.setChecked(False)
            self.script.checkBox_6.setChecked(False)
            self.script.checkBox_4.setChecked(False)
            self.script.checkBox_9.setChecked(False)
            self.script.comboBox_3.setCurrentIndex(0)
            self.script.spinBox.setValue(100)
            self.script.spinBox_2.setValue(50)
            self.script.spinBox_3.setValue(100)
            self.script.spinBox_6.setValue(1)
            self.script.checkBox_10.setChecked(False)

            self.script.lineEdit_14.setText('B')
            self.script.lineEdit_15.setText('H')
            self.script.lineEdit_16.setText('T')
            self.script.lineEdit_17.setText('M')
            self.script.lineEdit_18.setText('ESC')
            self.script.comboBox_7.setCurrentIndex(0)
            self.script.comboBox_8.setCurrentText('江南')
            self.script.spinBox_8.setValue(250),
            # text = config.get('日常任务', '采集方法')

            self.script.lineEdit_19.setText('')
            self.script.lineEdit_20.setText('')
            self.script.lineEdit_21.setText('')
            self.script.lineEdit_22.setText('')
            self.script.lineEdit_23.setText('')
            self.script.lineEdit_24.setText('')
            self.script.lineEdit_25.setText('')
            self.script.lineEdit_26.setText('')
            self.script.lineEdit_27.setText('')
            self.script.lineEdit_28.setText('')
            self.script.lineEdit_29.setText('')
            self.script.lineEdit_30.setText('')

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
            self.script.spinBox_8.setValue(config.getint('日常任务', '采集加速延迟')),
            self.script.checkBox_13.setChecked(config.getboolean('日常任务', '混队模式'))
            # text = config.get('日常任务', '采集方法')

            text = iter(eval(config.get('日常任务', '自定义采集坐标')))
            coord = next(text)
            self.script.lineEdit_19.setText(coord[0])
            self.script.lineEdit_20.setText(coord[1])
            coord = next(text)
            self.script.lineEdit_21.setText(coord[0])
            self.script.lineEdit_22.setText(coord[1])
            coord = next(text)
            self.script.lineEdit_23.setText(coord[0])
            self.script.lineEdit_24.setText(coord[1])
            coord = next(text)
            self.script.lineEdit_25.setText(coord[0])
            self.script.lineEdit_26.setText(coord[1])
            coord = next(text)
            self.script.lineEdit_27.setText(coord[0])
            self.script.lineEdit_28.setText(coord[1])
            coord = next(text)
            self.script.lineEdit_29.setText(coord[0])
            self.script.lineEdit_30.setText(coord[1])

            kill_list = eval(config.get('日常任务', '技能列表'))
            self.script.lineEdit_4.setText(kill_list[0]),
            self.script.lineEdit_9.setText(kill_list[1]),
            self.script.lineEdit_5.setText(kill_list[2]),
            self.script.lineEdit_6.setText(kill_list[3]),
            self.script.lineEdit_7.setText(kill_list[4]),
            self.script.lineEdit_8.setText(kill_list[5]),
            self.script.lineEdit_10.setText(kill_list[6]),
            self.script.lineEdit_11.setText(kill_list[7]),
            self.script.lineEdit_12.setText(kill_list[8]),
            self.script.lineEdit_13.setText(kill_list[9]),

        except configparser.NoOptionError:
            pass
        except configparser.NoSectionError:
            pass
        except ValueError:
            pass
        except TypeError:
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

        # 创建配置对象
        config = configparser.ConfigParser()

        # 添加一些配置项
        config['界面设置'] = {
            '当前配置': self.script.comboBox.currentText(),
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

        # 加载配置文件
        for text in ini_files:
            self.script.comboBox.addItem(text)

        # 创建配置对象
        config = configparser.ConfigParser()
        # 读取配置文件
        config.read(f'{config_path}\\System.ini', encoding='utf-8')
        try:
            # 加载当前配置信息
            current_text = config.get('界面设置', '当前配置')
            if current_text in ini_files:
                self.script.comboBox.setCurrentText(current_text)
        except configparser.NoSectionError as e:
            print(e)

    # 重写关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认退出',
                                     "你确定要退出吗?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.save_system_config()
            self.run.unbind_all(None)
            event.accept()
        else:
            event.ignore()


class ConfigDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("保存配置文件")

        layout = QVBoxLayout()

        self.file_name_input = QLineEdit(self)
        self.file_name_input.setPlaceholderText("配置文件名称")
        layout.addWidget(self.file_name_input)

        self.list_widget = QListWidget(self)

        # 获取当前用户的路径
        user_path = os.path.expanduser('~')

        # 拼接文件路径
        config_path = os.path.join(user_path, 'ChronoSnow', 'TaskConfig')

        # 递归创建目录和文件
        os.makedirs(config_path, exist_ok=True)
        for file_name in os.listdir(config_path):
            if file_name.endswith(".ini"):
                # 去除文件名的后缀部分
                file_name_without_extension = os.path.splitext(file_name)[0]
                # 添加不带后缀的文件名到 QListWidget 中
                self.list_widget.addItem(file_name_without_extension)

        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selected_items(self):
        try:
            selected_item_text = self.list_widget.selectedItems()[0].text()

            all_items = [self.list_widget.item(index).text() for index in range(self.list_widget.count())]
            return all_items, selected_item_text
        except IndexError:
            pass


class DelConfigDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("删除配置文件")

        layout = QVBoxLayout()

        self.list_widget = QListWidget(self)

        # 获取当前用户的路径
        user_path = os.path.expanduser('~')

        # 拼接文件路径
        config_path = os.path.join(user_path, 'ChronoSnow', 'TaskConfig')

        # 递归创建目录和文件
        os.makedirs(config_path, exist_ok=True)
        for file_name in os.listdir(config_path):
            if file_name.endswith(".ini"):
                # 去除文件名的后缀部分
                file_name_without_extension = os.path.splitext(file_name)[0]
                # 添加不带后缀的文件名到 QListWidget 中
                self.list_widget.addItem(file_name_without_extension)

        layout.addWidget(self.list_widget)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def get_selected_items(self):
        selected_item_text = self.list_widget.selectedItems()[0].text()

        all_items = [self.list_widget.item(index).text() for index in range(self.list_widget.count())]
        return all_items, selected_item_text


class HomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.home = Ui_Home()
        self.home.setupUi(self)


class ScriptWindow(QWidget, Ui_Script):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.initWindow()
        # 重构输入框
        self.lineEdit_4 = CustomLineEdit(self.lineEdit_4)
        self.lineEdit_9 = CustomLineEdit(self.lineEdit_9)
        self.lineEdit_5 = CustomLineEdit(self.lineEdit_5)
        self.lineEdit_6 = CustomLineEdit(self.lineEdit_6)
        self.lineEdit_7 = CustomLineEdit(self.lineEdit_7)
        self.lineEdit_8 = CustomLineEdit(self.lineEdit_8)
        self.lineEdit_10 = CustomLineEdit(self.lineEdit_10)
        self.lineEdit_11 = CustomLineEdit(self.lineEdit_11)
        self.lineEdit_12 = CustomLineEdit(self.lineEdit_12)
        self.lineEdit_13 = CustomLineEdit(self.lineEdit_13)
        self.lineEdit_14 = CustomLineEdit(self.lineEdit_14)
        self.lineEdit_15 = CustomLineEdit(self.lineEdit_15)
        self.lineEdit_16 = CustomLineEdit(self.lineEdit_16)
        self.lineEdit_17 = CustomLineEdit(self.lineEdit_17)
        self.lineEdit_18 = CustomLineEdit(self.lineEdit_18)

    def task_append(self, text):
        sender = self.sender()
        if sender.isChecked():
            item = QListWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.listWidget.addItem(item)
        else:
            items = self.listWidget.findItems(text, Qt.MatchFlag.MatchExactly)
            row = self.listWidget.row(items[0])
            self.listWidget.takeItem(row)

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
        for text in TASK_MAPPING.keys():
            self.listWidget_2.addItem(text)
        # self.listWidget.clicked.connect(self.guide)
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

        self.spinBox_8.setMaximum(500)
        self.spinBox_8.setValue(250)
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
        publicSingle.state.connect(self.set_state)
        publicSingle.journal.connect(self.journal)
        publicSingle.set_character.connect(self.set_character)

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

    def journal(self, message):
        # 获取当前时间
        current_time = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")

        if isinstance(message, list):
            # 构造带有时间信息的文本
            formatted_message = f"[{current_time}] [窗口: {message[0] + 1} ] >>> {message[1]}\n"
        else:
            # 构造带有时间信息的文本
            formatted_message = f"[{current_time}] >>> {message}\n"

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
            cv2.imwrite(f'{temp_img_path}\\1.png', image)
            # 定义要保留的区域的坐标和尺寸
            x, y, width, height = 117, 730, 115, 20

            # 从原始截图中复制指定区域
            img = image[y:y + height, x:x + width]
            cv2.imwrite(f'{temp_img_path}\\person_{row}.png', img)
            # 创建 QPixmap 对象，加载图片
            pixmap = QPixmap(f'{temp_img_path}\\person_{row}.png')

            # 将图片添加到表格中
            item = QTableWidgetItem()
            item.setData(Qt.ItemDataRole.DecorationRole, pixmap)  # 使用 Qt::DecorationRole 用于图像数据
            self.PersonaTableWidget.setItem(row, 0, item)
        except KeyError:
            pass

    def remove_character(self):
        try:
            row = self.PersonaTableWidget.currentIndex().row()
            # 获取 QTableWidgetItem
            item = self.PersonaTableWidget.item(row, 0)
            # 移除图像，将图像数据设置为 None 或空 QPixmap
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
        if (row := self.window_inspection()) != -1:
            # self.set_character(self.struct_task_dict[row].row)
            Thread(target=self.start.start, args=(row, self.struct_task_dict[row].handle)).start()


# 任务信息构造类
class StructureTask:
    def __init__(self, row, handle):
        self.row = row
        self.handle = handle
        self.flag = False
        self.set_window(row)
        self.mask_window = MaskWindow(self.handle)
        self.create_temp_file()
        self.block_window()
        publicSingle.write_json.emit(self.row)

    def set_window(self, row):
        scale_factor = QApplication.primaryScreen().devicePixelRatio()
        basic_functional.set_window(self.handle, DPI_MAPP[scale_factor])

    def create_temp_file(self):
        temp_dir = tempfile.gettempdir()
        temp_img_path = os.path.join(temp_dir, f'template_image{self.row}')

        shutil.rmtree(temp_img_path, ignore_errors=True)  # 删除目标文件夹及其内容，如果存在的话
        shutil.copytree('app/images/Img', temp_img_path)

    def block_window(self):
        basic_functional.DisableTheWindow(self.handle)
        basic_functional.DisableTheWindow(self.mask_window.winId())

    def unblock_window(self):
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
        self.show()

    def mask_close(self):
        self.close()


class TimingQMessageBox:
    def __init__(self):
        ...

    @staticmethod
    def information(self, title, message):
        message_box = QMessageBox()
        message_box.setWindowTitle(title)
        message_box.setText(message)
        message_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        message_box.setDefaultButton(QMessageBox.StandardButton.Ok)
        QTimer.singleShot(700, lambda: message_box.button(QMessageBox.StandardButton.Ok).animateClick())
        message_box.exec()


class CustomLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        self.clear()
        try:
            key = KEY_DICT[event.key()]
        except KeyError:
            key = ''
        modifiers = str(event.modifiers())
        if modifiers == 'KeyboardModifier.KeypadModifier':
            key = 'Num' + key
        # 在这里添加你自定义的按键处理逻辑
        print("CustomLineEdit - Key pressed:", event.key(), key, modifiers)

        self.setText(key)

KEY_DICT = {
    48: '0',
    49: '1',
    50: '2',
    51: '3',
    52: '4',
    53: '5',
    54: '6',
    55: '7',
    56: '8',
    57: '9',
    65: 'A',
    66: 'B',
    67: 'C',
    68: 'D',
    69: 'E',
    70: 'F',
    71: 'G',
    72: 'H',
    73: 'I',
    74: 'J',
    75: 'K',
    76: 'L',
    77: 'M',
    78: 'N',
    79: 'O',
    80: 'P',
    81: 'Q',
    82: 'R',
    83: 'S',
    84: 'T',
    85: 'U',
    86: 'V',
    87: 'W',
    88: 'X',
    89: 'Y',
    90: 'Z',
    # 16777233: 'Num1',
    # 16777237: 'Num2',
    # 16777239: 'Num3',
    # 16777234: 'Num4',
    # 16777227: 'Num5',
    # 16777236: 'Num6',
    # 16777232: 'Num7',
    # 16777235: 'Num8',
    # 16777238: 'Num9',
    # 16777222: 'Num0',
    16777249: 'ctrl',
    16777216: 'ESC'
}
