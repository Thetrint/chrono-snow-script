import logging
import sys
from PyQt6.QtWidgets import QApplication
from app.view.AppWindow import MainWindow

app = QApplication(sys.argv)

# 配置全局日志记录器，设置级别为DEBUG
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# create main window
w = MainWindow()
w.show()

app.exec()
