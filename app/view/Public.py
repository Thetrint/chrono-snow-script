import json
import os
import tempfile

from PyQt6.QtCore import QObject, pyqtSignal
TABLE_WINDOW = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DPI_MAPP = {1.0: (1350, 789), 1.25: (1352, 797), 1.5: (1356, 806), 1.75: (1360, 814)}


class PublicSingle(QObject):
    stop = pyqtSignal(int)
    resume = pyqtSignal(int)
    unbind = pyqtSignal(int)
    write_json = pyqtSignal(int)
    journal = pyqtSignal(list)
    state = pyqtSignal(list)


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
