import logging
import sys
from logging.handlers import RotatingFileHandler

from PyQt6.QtWidgets import QApplication
from app.view.AppWindow import MainWindow

app = QApplication(sys.argv)

# 创建RotatingFileHandler实例
handler = RotatingFileHandler(filename='app.log', maxBytes=1024*1024)

# 配置全局日志记录器
logging.basicConfig(
    level=logging.INFO,             # 日志级别为DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[handler]              # 添加RotatingFileHandler实例
)

# create main window
w = MainWindow()
w.show()

app.exec()
