import json
import os
import tempfile

from PyQt6.QtCore import QObject, pyqtSignal, QTimer, Qt, QRegularExpression
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import QVBoxLayout, QDialog, QListWidget, QDialogButtonBox, QLineEdit, QMessageBox
TABLE_WINDOW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DPI_MAPP = {1.0: (1350, 789), 1.25: (1352, 797), 1.5: (1356, 806), 1.75: (1360, 814)}


class PublicSingle(QObject):
    stop = pyqtSignal(int)
    resume = pyqtSignal(int)
    unbind = pyqtSignal(int)
    write_json = pyqtSignal(int)
    journal = pyqtSignal(list)
    state = pyqtSignal(list)
    set_character = pyqtSignal(int)
    login = pyqtSignal(str)
    offline = pyqtSignal()


publicSingle = PublicSingle()


class TaskConfig:

    @staticmethod
    def load_task_config(row):
        temp_dir = tempfile.gettempdir()
        temp_img_path = os.path.join(temp_dir, f'config{row}.json')
        try:
            with open(temp_img_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading JSON data from': {e}")
            return {}


LoadTaskConfig = TaskConfig()


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