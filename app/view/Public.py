import json
import os
import tempfile
import threading
import time

import win32con
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt6.QtCore import Qt, QStringListModel
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import QTextEdit, QCompleter
from PyQt6.QtWidgets import QVBoxLayout, QDialog, QListWidget, QDialogButtonBox, QLineEdit, QMessageBox

TABLE_WINDOW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DPI_MAPP = {1.0: (1350, 789), 1.25: (1352, 797), 1.5: (1356, 806), 1.75: (1360, 814), 2.0: (1360, 822)}
Mask = True
VERSION = '1.3.1'

START_ID = 1

SHORTKEY_ID_MAP = {
    START_ID: '开始快捷键'
}
# 定义修饰键与Win32常量的映射关系
MODIFIERS_MAP = {
    ('Shift', ): win32con.MOD_SHIFT,
    ('Alt',): win32con.MOD_ALT,
    ('Ctrl',): win32con.MOD_CONTROL,
    ('Shift', 'Alt'): win32con.MOD_SHIFT | win32con.MOD_ALT,
    ('Ctrl', 'Alt'): win32con.MOD_CONTROL | win32con.MOD_ALT,
    ('Ctrl', 'Shift'): win32con.MOD_SHIFT | win32con.MOD_CONTROL,
    ('Shift', 'Alt', 'Ctrl'): win32con.MOD_CONTROL | win32con.MOD_SHIFT | win32con.MOD_ALT,
    # 可以根据需要继续添加其他情况的映射关系
}


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
    start = pyqtSignal(str)


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


# 快捷键输入框
class ShortCutLineEdit(QLineEdit):

    def __init__(self, parent=None):
        super().__init__(parent)

    def keyPressEvent(self, event):
        self.clear()
        print(event.modifiers())
        key = ''
        if str(event.modifiers()) == "KeyboardModifier.ShiftModifier":
            key = 'Shift' + '+'
        elif str(event.modifiers()) == "KeyboardModifier.ControlModifier":
            key = 'Ctrl' + '+'
        elif str(event.modifiers()) == "KeyboardModifier.AltModifier":
            key = 'Alt' + '+'
        elif str(event.modifiers()) == "KeyboardModifier.ShiftModifier|AltModifier":
            key = 'Shift' + '+' + 'Alt' + '+'
        elif str(event.modifiers()) == "KeyboardModifier.ShiftModifier|ControlModifier":
            key = 'Ctrl' + '+' + 'Shift' + '+'
        elif str(event.modifiers()) == "KeyboardModifier.ControlModifier|AltModifier":
            key = 'Ctrl' + '+' + 'Alt' + '+'
        elif str(event.modifiers()) == "KeyboardModifier.ShiftModifier|ControlModifier|AltModifier":
            key = 'Ctrl' + '+' + 'Shift' + '+' + 'Alt' + '+'

        if 48 <= event.key() <= 90:
            key += KEY_DICT[event.key()]
        else:
            key = ''

        # try:
        #     key = KEY_DICT[event.key()]
        # except KeyError:
        #     key = ''
        # modifiers = str(event.modifiers())
        # if modifiers == 'KeyboardModifier.KeypadModifier':
        #     key = 'Num' + key
        # # 在这里添加你自定义的按键处理逻辑
        # print("CustomLineEdit - Key pressed:", event.key(), key, modifiers)
        #
        self.setText(key)


# 技能输入框
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
    16777248: 'shift',
    16777220: 'Enter',
    16777249: 'ctrl',
    16777216: 'ESC'
}


class TextEdit(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.textChangedSign = True
        self._completer = None
        self.searchWords_dict = {
            "技能": "技能[$CURSON$] 延迟[2000]ms <>\n",
        }
        self.matchWords_list = [
            '普攻',
            '绝学',
            '闪避',
            '技能1',
            '技能2',
            '技能3',
            '技能4',
            '技能5',
            '技能6',
            '技能7',
            '技能8'

        ]
        self.initAutoCompleteWords()
        self.completion_prefix = ''

    def initAutoCompleteWords(self):
        self.autoCompleteWords_list = []
        self.specialCursorDict = {}
        for i in self.searchWords_dict:
            if "$CURSON$" in self.searchWords_dict[i]:
                cursorPosition = len(self.searchWords_dict[i]) - len("$CURSON$") - self.searchWords_dict[i].find(
                    "$CURSON$")
                self.searchWords_dict[i] = self.searchWords_dict[i].replace("$CURSON$", '')
                self.specialCursorDict[i] = cursorPosition
        for i in self.matchWords_list:
            if "$CURSON$" in i:
                cursorPosition = len(i) - len("$CURSON$") - i.find("$CURSON$")
                self.matchWords_list[self.matchWords_list.index(i)] = i.replace("$CURSON$", '')
                self.specialCursorDict[i.replace("$CURSON$", '')] = cursorPosition

        self.autoCompleteWords_list = list(self.searchWords_dict.keys()) + self.matchWords_list

    def setCompleter(self, c):
        self._completer = c
        c.setWidget(self)
        c.popup().setStyleSheet('''
        QListView {
            color: red;
            padding-top: 0px;
            padding-bottom: 0px;
        }
        QListView::item:selected {
            background-color: #9C9C9C;
        }
        ''')

        c.popup().setFixedWidth(100)  # 设置宽度为50像素

        c.setModelSorting(QCompleter.ModelSorting.CaseSensitivelySortedModel)
        c.setFilterMode(Qt.MatchFlag.MatchContains)
        c.setWrapAround(False)
        c.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

        c.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        c.setModel(QStringListModel(self.autoCompleteWords_list, self._completer))

        c.activated.connect(self.insertCompletion)

    def insertCompletion(self, completion):
        if self._completer.widget() is not self:
            return

        tc = self.textCursor()

        if completion in self.searchWords_dict:
            for i in self._completer.completionPrefix():
                tc.deletePreviousChar()
            self.setTextCursor(tc)

            insertText = self.searchWords_dict[completion]
            tc.insertText(insertText)

            self._completer.popup().hide()

        else:
            for i in self._completer.completionPrefix():
                tc.deletePreviousChar()
            tc.insertText(completion)
            self._completer.popup().hide()

        tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
        if completion in self.specialCursorDict.keys():
            for i in range(self.specialCursorDict[completion]):
                tc.movePosition(QTextCursor.MoveOperation.PreviousCharacter)
            self.setTextCursor(tc)
        else:
            tc.movePosition(QTextCursor.MoveOperation.EndOfWord)
            self.setTextCursor(tc)

    def getLastPhrase(self):
        connect = self.toPlainText()
        lastPhrase = connect.split('\n')[-1].split(' ')[-1]
        return lastPhrase

    def keyPressEvent(self, e):
        isShortcut = False

        if self._completer is not None and self._completer.popup().isVisible():
            if e.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Escape, Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
                e.ignore()
                return

        if (self._completer is None or not isShortcut) and e.key() != 0:
            super().keyPressEvent(e)

        ctrlOrShift = e.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        if self._completer is None or (ctrlOrShift and len(e.text()) == 0):
            return

        eow = ''
        hasModifier = (e.modifiers() != Qt.KeyboardModifier.NoModifier) and not ctrlOrShift

        lastPhrase = self.getLastPhrase()
        self.completion_prefix = lastPhrase

        if not isShortcut and (len(e.text()) == 0 or len(lastPhrase) < 0):
            self._completer.popup().hide()
            return

        if lastPhrase != self._completer.completionPrefix():
            self._completer.setCompletionPrefix(lastPhrase)
            self._completer.popup().setCurrentIndex(self._completer.completionModel().index(0, 0))

        cr = self.cursorRect()
        cr.setWidth(self._completer.popup().sizeHintForColumn(
            0) + self._completer.popup().verticalScrollBar().sizeHint().width())
        self._completer.complete(cr)







