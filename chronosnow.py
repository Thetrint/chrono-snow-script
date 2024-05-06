import logging
import sys
from logging.handlers import RotatingFileHandler

from PyQt6.QtWidgets import QApplication, QStyleFactory
from app.view.AppWindow import MainWindow

app = QApplication(sys.argv)
app.setStyle(QStyleFactory.create('windowsvista'))
# print(QStyleFactory.keys())

# 创建RotatingFileHandler实例
handler = RotatingFileHandler(filename='app.log', maxBytes=1024*1024, backupCount=1)

# 配置全局日志记录器
logging.basicConfig(
    level=logging.INFO,             # 日志级别为INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[handler]              # 添加RotatingFileHandler实例
)

# create main window
w = MainWindow()

app.exec()
