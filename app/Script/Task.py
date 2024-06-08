import heapq
import logging
import os
import random
import re
import tempfile
import cv2
import numpy
import time
from threading import Lock, Event, Thread
from sklearn.cluster import DBSCAN
from abc import abstractmethod
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from datetime import time as datetime_time

from app.Script.BasicFunctional import basic_functional
from app.view.Public import publicSingle, LoadTaskConfig


class EventStruct:
    def __init__(self):
        self.unbind = {}
        self.stop = {}
        self.task_config = {}
        self.stop_flag = False
        self.persona = {}
        self.switch = {}

        self.disrupted_event = {}
        self.object_storage = {}

    # 时间获取
    @staticmethod
    def get_time(year=datetime.now().year, month=datetime.now().month,
                 day=datetime.now().day, hour=datetime.now().hour,
                 minute=datetime.now().minute, second=datetime.now().second):
        return datetime(year, month, day, hour, minute, second)

    # 当前时间获取
    @staticmethod
    def get_current_time():
        return datetime.now()


event = EventStruct()


class TimerTask:

    def __init__(self, mapp):
        self.mapp = mapp
        self.running_time = 0
        self.stop_event = True
        self.running = False
        Thread(target=self.Timer).start()

    def Timer(self):
        while not event.unbind[self.mapp].is_set() and self.stop_event:
            if self.running:
                self.running_time += 1
            time.sleep(1)


class TaskSchedulingStrategy:

    def __init__(self, task_list, mapp, index, switch):
        self._list = task_list
        self.mapp = mapp
        self.index = index
        self.switch = switch
        self.execute_task = []
        # 创建调度器实例
        self.scheduler = BackgroundScheduler()

        self.dispatch_task()
        # threading.Thread(target=self.dispatch_task).start()

    @staticmethod
    def process_task(task_list, mapp, index, switch):
        obj = TaskSchedulingStrategy([(1, task) for task in task_list], mapp, index, switch)
        switch.dl = obj
        return obj

    def task1(self):
        if '邸宅农场' not in [item[1] for item in self._list] and event.task_config[self.mapp].get('邸宅农场'):
            self.insert_task('邸宅农场', 2)
            event.disrupted_event[self.mapp].set()

    def task2(self):
        if event.task_config[self.mapp].get('帮派修炼') and '帮派修炼' not in self.execute_task:
            self.insert_task('帮派修炼', 2)
            self.execute_task.append('帮派修炼')
            event.disrupted_event[self.mapp].set()

    def task3(self):
        self.switch.roles = [True, True, True, True, True, True]
        self.execute_task = []

    def dispatch_task(self):
        self.scheduler.add_job(
            self.task1,
            'interval',
            minutes=30
        )

        self.scheduler.add_job(
            self.task2,
            'cron',
            start_date=datetime.combine(datetime.today(), datetime_time(18, 45)),
            end_date=datetime.combine(datetime.today(), datetime_time(19, 25))
        )

        self.scheduler.add_job(
            self.task3,
            'date',
            run_date=datetime.combine(datetime.today(), datetime_time(5, 0)),
        )

        self.scheduler.start()

    def get_task(self):
        if not self._list:
            if event.task_config[self.mapp].get('切角色1') and self.switch.roles[1]:
                return None
            elif event.task_config[self.mapp].get('切角色2') and self.switch.roles[2]:
                return None
            elif event.task_config[self.mapp].get('切角色3') and self.switch.roles[3]:
                return None
            elif event.task_config[self.mapp].get('切角色4') and self.switch.roles[4]:
                return None
            elif event.task_config[self.mapp].get('切角色5') and self.switch.roles[5]:
                return None
            elif all(self.switch.roles):
                return None
            return '占位任务'  # 当列表为空时返回 None
        return self._list.pop(0)[1]  # 取出第一个元素并从列表中移除

    def insert_task(self, task, priority):
        if priority == 2:
            # 查找第一个优先级为1的位置
            for i, (p, _) in enumerate(self._list):
                if p == 1:
                    self._list.insert(i, (priority, task))
                    return
            # 如果没有找到优先级为1的任务，则将任务插入到列表末尾
            self._list.append((priority, task))
        elif priority == 1:
            # 查找最后一个优先级为2的位置
            for i in range(len(self._list) - 1, -1, -1):
                if self._list[i][0] == 2:
                    self._list.insert(i + 1, (priority, task))
                    return
            # 如果没有找到优先级为2的任务，则将任务插入到列表开头
            self._list.insert(0, (priority, task))


class StartTask:
    def __init__(self):
        self.initTask()
        self.index = 0
        self.mapping = {}

    def start(self, row, handle):
        # 获取取映射
        mapp = self.create_mapping(row)
        # 任务必要变量
        event.unbind[mapp] = Event()
        event.stop[mapp] = Lock()
        event.task_config[mapp] = LoadTaskConfig.load_task_config(row)
        event.persona[mapp] = Persona(row, handle, mapp)
        event.switch[mapp] = SwitchRoles(row, handle, mapp)
        event.disrupted_event[mapp] = Event()
        event.object_storage[mapp] = {}

        # 游戏初始化
        init = Initialize(row, handle, mapp)
        # init.implement()
        Thread(target=init.global_detection).start()

        while event.switch[mapp].implement() and not event.unbind[mapp].is_set():
            dl = TaskSchedulingStrategy.process_task(event.task_config[mapp].get('执行列表'), mapp,
                                                     self.dispatch_stop(event.switch[mapp]), event.switch[mapp])
            init.implement()
            publicSingle.set_character.emit(row)
            # dl.insert_task('邸宅农场', 2)
            while (item := dl.get_task()) is not None and not event.unbind[mapp].is_set():

                if item not in ['占位任务']:
                    self.set_state(row, item)

                obj = event.object_storage[mapp].get(item) or TASK_MAPPING[item](row, handle, mapp)
                result = obj.implement()
                event.disrupted_event[mapp].clear()
                if result == -1:
                    publicSingle.journal.emit([row, f'{item} 任务中止 重新归入队列'])
                    dl.insert_task(item, 1)
                    event.object_storage[mapp][item] = obj
                # Task.initialization()
                # Task.implement()

        # self.dispatch_stop(mapp, switch)
        if not event.unbind[mapp].is_set():
            publicSingle.state.emit([row, '任务结束'])

    @staticmethod
    # 结束上一个实例线程
    def dispatch_stop(switch):
        if switch.dl is not None:
            switch.dl.scheduler.shutdown(wait=False)
            switch.dispatch_index += 1
            return switch.dispatch_index

    def create_mapping(self, row):
        self.mapping[row] = self.index
        self.index += 1
        return self.mapping[row]

    def initTask(self):
        publicSingle.stop.connect(self.stop)
        publicSingle.resume.connect(self.resume)
        publicSingle.unbind.connect(self.unbind)

    def stop(self, row):
        event.stop[self.mapping[row]].acquire()

    def resume(self, row):
        event.stop[self.mapping[row]].release()

    def unbind(self, row):
        event.unbind[self.mapping[row]].set()
        event.switch[self.mapping[row]].dl.scheduler.shutdown(wait=False)

    @staticmethod
    def set_state(row, tk):
        publicSingle.state.emit([row, tk])
        publicSingle.journal.emit([row, f'{tk}开始'])


class BasicTask(object):

    def __init__(self, row, handle, mapp):
        self.row = row
        self.handle = handle
        self.mapp = mapp
        self.timeout = event.task_config[mapp].get('全局延迟') / 1000

    @abstractmethod
    def implement(self):
        """
        执行任务
        :return:
        """
        pass

    @abstractmethod
    def initialization(self):
        """
        初始化
        :return:
        """
        pass

    # 队伍检测
    def leave_team(self):
        # self.log('队伍状态检查')
        self.key_down_up(event.persona[self.mapp].team)
        self.Visual('退出队伍', canny_process=True, threshold=0.7)
        self.Visual('确定', canny_process=True, threshold=0.7)
        self.Visual('关闭', '关闭1', threshold=0.7, histogram_process=True)

    # 位置检测
    def location_detection(self):
        # self.log('位置检测')
        self.key_down_up(event.persona[self.mapp].map)
        if self.coord('当前坐标金陵', canny_process=True, threshold=0.85):
            self.close_win(1)
            return 0
        self.Visual('世界', laplacian_process=True)
        self.Visual('金陵', histogram_process=True, threshold=0.7)
        self.map_input(571, 484)

    # 到达检测
    def arrive(self):
        index = 0
        for _ in range(300):
            if event.unbind[self.mapp].is_set():
                break
            # if self.coord('过图标志', canny_process=True, threshold=0.85, search_scope=(953, 510, 1237, 756), show=True):
            #     self.journal('过图中')
            #     time.sleep(8)
            #     continue
            if not self.coord('自动寻路中', histogram_process=True, threshold=0.6, search_scope=(531, 498, 870, 615)):
                index += 1
                time.sleep(1)
                if index == 10:
                    break
                continue
            else:
                index = 0

            time.sleep(2)

    # 脱离卡死
    def escape_stuck(self):
        self.key_down_up('ESC')
        self.Visual('脱离卡死', laplacian_process=True)
        self.Visual('确定', laplacian_process=True)
        self.key_down_up('ESC', key_down_timeout=4)

    def world_shouts(self, message):
        self.mouse_down_up(309, 595)
        self.Visual('世界频道', canny_process=True, threshold=0.7, wait_count=1)
        if self.Visual('输入文字', wait_count=1, canny_process=True, threshold=0.7):
            self.input(message)
        self.Visual('发送', canny_process=True, threshold=0.7)

        self.Visual('互联世界频道', canny_process=True, threshold=0.7, wait_count=1)
        if self.Visual('输入文字', wait_count=1, canny_process=True, threshold=0.7):
            self.input(message)
        self.Visual('发送', canny_process=True, threshold=0.7)
        self.Visual('聊天窗口关闭', canny_process=True, threshold=0.75, search_scope=(616, 203, 827, 512))

    def close_win(self, count, random_tap=False, left_tap=False, right_tap=False):
        for i in range(count):
            if event.unbind[self.mapp].is_set():
                return 0
            if left_tap:
                self.mouse_down_up(0, 0)
            if right_tap:
                self.mouse_down_up(1334, 750)
            if not self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7, random_tap=random_tap,
                               wait_count=1):
                break

    def open_entrance(self, tem_name, laplacian_process, binary_process, threshold):
        count = 0
        for i in range(2):
            self.journal('尝试打开任务入口')
            self.key_down_up(event.persona[self.mapp].knapsack)
            self.Visual('活动入口', histogram_process=True, threshold=0.7)
            if not self.Visual(tem_name, laplacian_process=laplacian_process, binary_process=binary_process,
                               threshold=threshold):
                count += 1
                self.close_win(2, left_tap=True)
            else:
                break
            if count >= 2:
                self.journal('入口打开识别')
                self.close_win(2, left_tap=True)
                return 0

    def map_input(self, x, y):

        self.Visual('停止寻路', canny_process=True, threshold=0.6, search_scope=(0, 600, 530, 750), wait_count=1)

        self.Visual('世界搜索坐标展开', histogram_process=True, threshold=0.7, wait_count=1,
                    search_scope=(0, 600, 349, 750))

        self.Visual('横', canny_process=True, threshold=0.6, wait_count=1, search_scope=(0, 600, 349, 750))
        # self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=96, search_scope=(0, 631, 414, 694))
        self.input(str(x))
        # self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=233,
        #             search_scope=(0, 631, 414, 694))
        self.Visual('纵', canny_process=True, threshold=0.6, wait_count=1, search_scope=(0, 600, 349, 750))
        self.input(str(y))
        # self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=300,
        #             search_scope=(0, 631, 414, 694))
        self.Visual('前往图标', canny_process=True, threshold=0.6, wait_count=1, search_scope=(0, 600, 400, 750))
        self.close_win(1)
        self.arrive()

    # 日志信息
    def journal(self, message):
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                publicSingle.journal.emit([self.row, message])

    def back_interface(self):
        while not event.unbind[self.mapp].is_set():
            if not self.coord('副本挂机', '挂机', canny_process=True, threshold=0.7):
                self.Visual('聊天窗口关闭', canny_process=True, threshold=0.75, wait_count=1,
                            search_scope=(616, 203, 827, 512))
                if not self.Visual('关闭', histogram_process=True, threshold=0.75, wait_count=1):
                    self.mouse_down_up(0, 0)
                if not self.Visual('关闭', histogram_process=True, threshold=0.75, wait_count=1):
                    self.mouse_down_up(1325, 745)
                continue
            return 1

    def keep_activate(self, count):
        for _ in range(count):
            if event.unbind[self.mapp].is_set():
                return 0
            time.sleep(1)
            self.key_down_up('Tab', key_down_timeout=0)
            self.mouse_down_up(1330, 740, tap_after_timeout=0)

    def SIFT(self, *args, search_scope=(0, 0, 1334, 750), threshold=0.1, histogram_process=False):
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:

                x1, y1, x2, y2 = search_scope
                # 调用方法获取图片 Numpy 数组
                image = basic_functional.screen_shot(self.handle)
                # 切割图片
                image = image[y1:y2, x1:x2]
                # 获取图片数据
                image_height, image_width, _ = image.shape
                # 转换为灰度图像
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                result_coord = []
                coord_list = []
                w, h = 0, 0

                # 创建ORB特征检测器和描述符提取器
                sift = cv2.SIFT_create()

                for template_image_name in args:
                    # 临时路径
                    template_image_path = os.path.join(tempfile.gettempdir(), f'template_image{self.row}')
                    template_image_path = f'{template_image_path}\\' + template_image_name + '.bmp'
                    # 检查路径是否存在
                    if not os.path.exists(template_image_path):
                        template_image_path = os.getcwd() + '\\app\\images\\Img\\' + template_image_name + '.bmp'
                    template = basic_functional.cv_imread(template_image_path)
                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                    # 应用直方图均衡化
                    if histogram_process:
                        # 创建自适应直方图均衡化器
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        image = clahe.apply(image)
                        template = clahe.apply(template)

                    # 在大图像和小图像中找到关键点和描述符
                    kp1, des1 = sift.detectAndCompute(image, None)
                    kp2, des2 = sift.detectAndCompute(template, None)

                    # 使用暴力匹配器进行特征匹配
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)

                    for m, n in matches:
                        if m.distance < threshold * n.distance:
                            img1_idx = m.queryIdx
                            (x1, y1) = kp1[img1_idx].pt
                            result_coord.append(([y1], [x1]))
                            # # 绘制矩形框
                            # cv2.rectangle(image, (int(x1), int(y1)),
                            #               (int(x1) + template.shape[1], int(y1) + template.shape[0]), (0, 255, 0), 2)

                for coord in result_coord:
                    coord_list.append(numpy.array(list(zip(coord[1], coord[0]))))
                # 上一个坐标点
                prev_loc = None

                # 参数定义
                tolerance = 20  # DBSCAN的eps参数

                # 使用DBSCAN进行聚类
                dbscan = DBSCAN(eps=tolerance, min_samples=1, metric='euclidean')

                coordinates = []

                for coord in coord_list:
                    if coord.size != 0:
                        clusters = dbscan.fit_predict(coord)

                        # 遍历每个聚类的代表坐标
                        for cluster_label in numpy.unique(clusters):
                            cluster_coord = coord[clusters == cluster_label]
                            cluster_center = numpy.round(numpy.mean(cluster_coord, axis=0)).astype(int)

                            # 检查与上一个坐标的距离
                            if prev_loc is not None and numpy.sqrt(numpy.sum(
                                    (cluster_center - prev_loc) ** 2)) <= tolerance:
                                continue  # 如果距离小于或等于 tolerance，则跳过此坐标

                            # 如果距离大于 tolerance，则将坐标添加到结果列表中
                            center_x = round(cluster_center[0] + w / 2)
                            center_y = round(cluster_center[1] + h / 2)

                            # 将中心坐标加入 filtered_locations
                            coordinates.append((center_x + search_scope[0], center_y + search_scope[1]))

                            # 更新上一个坐标
                            prev_loc = cluster_center
                print(args, coordinates, threshold)
                return coordinates or []

    def coord(self, *args, search_scope=(0, 0, 1334, 750), histogram_process=False, laplacian_process=False, show=False,
              binary_process=False, canny_process=False, process=False, threshold=0.23, ignore_scope=(0, 0, 0, 0)):
        """
        获取坐标
        :param ignore_scope:
        :param show: 展示图片处理结果
        :param canny_process:
        :param process:
        :param binary_process: 图片处理 二值化处理
        :param threshold: 图片置信度阈值
        :param laplacian_process: 图片处理 拉普拉斯算子处理
        :param histogram_process: 图片处理 直方图处理
        :param search_scope: 模板图片 匹配范围
        :param args:
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:

                x1, y1, x2, y2 = search_scope

                # 调用方法获取图片 Numpy 数组
                image = basic_functional.screen_shot(self.handle)
                # 切割图片
                image = image[y1:y2, x1:x2]
                # 获取图片数据
                image_height, image_width, _ = image.shape
                # 转换为灰度图像
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                result_coord = []
                coord_list = []
                w, h = 0, 0
                for template_image_name in args:
                    # 临时路径
                    template_image_path = os.path.join(tempfile.gettempdir(), f'template_image{self.row}')
                    template_image_path = f'{template_image_path}\\' + template_image_name + '.bmp'
                    # 检查路径是否存在
                    if not os.path.exists(template_image_path):
                        template_image_path = os.getcwd() + '\\app\\images\\Img\\' + template_image_name + '.bmp'
                    template = basic_functional.cv_imread(template_image_path)

                    # 转换为灰度图像
                    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

                    # 获取模板图像的宽度和高度
                    h, w = template_gray.shape

                    # 应用直方图均衡化
                    if histogram_process:
                        # 创建自适应直方图均衡化器
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        img = clahe.apply(image_gray)
                        tem = clahe.apply(template_gray)

                    # 拉普拉斯算子
                    if laplacian_process:
                        img = cv2.Laplacian(image_gray, cv2.CV_64F)
                        tem = cv2.Laplacian(template_gray, cv2.CV_64F)

                    # 二值化处理
                    if binary_process:
                        _, img = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                        _, tem = cv2.threshold(template_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    # Canny边缘检测
                    if canny_process:
                        img = cv2.Canny(image_gray, 50, 200)
                        tem = cv2.Canny(template_gray, 50, 200)

                    if binary_process or laplacian_process or histogram_process:
                        # 图片转换
                        image_gray = img.astype(numpy.uint8)
                        template_gray = tem.astype(numpy.uint8)

                    if show:
                        cv2.imshow('1', img)
                        cv2.imshow('2', tem)
                        cv2.waitKey(0)

                    if process:
                        template1 = template_gray - 6
                        template3 = template_gray + 6
                        # 使用 cv2.matchTemplate 函数找到匹配结果
                        result1 = cv2.matchTemplate(image_gray, template1, cv2.TM_CCOEFF_NORMED)
                        result2 = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                        result3 = cv2.matchTemplate(image_gray, template3, cv2.TM_CCOEFF_NORMED)

                        result = result3 + result2 + result1
                    else:

                        if not all(element == 0 for element in ignore_scope):
                            # 创建一个和模板一样大小的遮罩
                            mask = numpy.ones(template_gray.shape, dtype=numpy.uint8)

                            mask[ignore_scope[1]:ignore_scope[3], ignore_scope[0]:ignore_scope[2]] = 0

                            result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED, mask=mask)
                        else:
                            result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

                    # 使用 numpy.where 找到匹配结果大于阈值的坐标
                    matched_coords = numpy.where(result >= threshold)

                    # 将匹配结果和坐标一起存储在 result_coord 中
                    result_coord.append(matched_coords)
                    # logging.info((template_image_name, result[matched_coords], threshold))

                for coord in result_coord:
                    coord_list.append(numpy.array(list(zip(coord[1], coord[0]))))

                # 上一个坐标点
                prev_loc = None

                # 参数定义
                tolerance = 20  # DBSCAN的eps参数

                # 使用DBSCAN进行聚类
                dbscan = DBSCAN(eps=tolerance, min_samples=1, metric='euclidean')

                coordinates = []

                for coord in coord_list:
                    if coord.size != 0:
                        clusters = dbscan.fit_predict(coord)

                        # 遍历每个聚类的代表坐标
                        for cluster_label in numpy.unique(clusters):
                            cluster_coord = coord[clusters == cluster_label]
                            cluster_center = numpy.round(numpy.mean(cluster_coord, axis=0)).astype(int)

                            # 检查与上一个坐标的距离
                            if prev_loc is not None and numpy.sqrt(numpy.sum(
                                    (cluster_center - prev_loc) ** 2)) <= tolerance:
                                continue  # 如果距离小于或等于 tolerance，则跳过此坐标

                            # 如果距离大于 tolerance，则将坐标添加到结果列表中
                            center_x = round(cluster_center[0] + w / 2)
                            center_y = round(cluster_center[1] + h / 2)

                            # 将中心坐标加入 filtered_locations
                            coordinates.append((center_x + search_scope[0], center_y + search_scope[1]))

                            # 更新上一个坐标
                            prev_loc = cluster_center
                print(args, coordinates, threshold)
                logging.info((args, coordinates, threshold))
                return coordinates or []

    def Visual(self, *args, **kwargs):
        """
        找出指定模板图片
        :param random_tap_timeout: 全部点击 延迟间隔
        :param tap_after_timeout: 点击后 延迟
        :param tap_ago_timeout: 点击前 延迟
        :param random_tap: 判断条件 是否对所有找到的左边依次点击
        :param tap: 判断条件 是否点击找到的左边
        :param args: 图片名
        :return:
        """
        search_scope = kwargs.get('search_scope', (0, 0, 1334, 750))
        histogram_process = kwargs.get('histogram_process', False)
        laplacian_process = kwargs.get('laplacian_process', False)
        binary_process = kwargs.get('binary_process', False)
        canny_process = kwargs.get('canny_process', False)
        process = kwargs.get('process', False)
        SIFT = kwargs.get('SIFT', False)
        threshold = kwargs.get('threshold', 0.23)
        tap = kwargs.get('tap', True)
        show = kwargs.get('show', False)
        ignore_scope = kwargs.get('ignore_scope', (0, 0, 0, 0))
        tap_ago_timeout = kwargs.get('tap_ago_timeout', 1)
        tap_after_timeout = kwargs.get('tap_after_timeout', 1)
        random_tap = kwargs.get('random_tap', True)
        double = kwargs.get('double', False)
        random_tap_timeout = kwargs.get('random_tap_timeout', 1)
        continuous_search_timeout = kwargs.get('continuous_search_timeout', 1.5)
        wait_count = kwargs.get('wait_count', 4)
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)

        for _ in range(wait_count):
            if event.unbind[self.mapp].is_set():
                break
            if SIFT:
                coordinates = self.SIFT(*args, search_scope=search_scope, threshold=threshold,
                                        histogram_process=histogram_process)
            else:
                coordinates = self.coord(*args, search_scope=search_scope, histogram_process=histogram_process,
                                         laplacian_process=laplacian_process, binary_process=binary_process, show=show,
                                         canny_process=canny_process, process=process, threshold=threshold,
                                         ignore_scope=ignore_scope)
            if coordinates:
                if random_tap:
                    coord = random.choice(coordinates)
                    if tap:
                        time.sleep(tap_ago_timeout)
                        self.mouse_down_up(coord[0] + x, coord[1] + y, tap_after_timeout=tap_after_timeout,
                                           double=double)
                    return coord
                else:
                    for coord in coordinates:
                        if tap:
                            time.sleep(tap_ago_timeout)
                            self.mouse_down_up(coord[0] + x, coord[1] + y, tap_after_timeout=tap_after_timeout,
                                               double=double)
                            time.sleep(random_tap_timeout + self.timeout)
                        else:
                            break
                    return coordinates
            time.sleep(continuous_search_timeout + self.timeout)

    def mouse_down(self, x, y):
        """
        鼠标按下(x,y)坐标
        :param x: x坐标
        :param y: y坐标
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.mouse_down(self.handle, x, y)

    def mouse_up(self, x, y):
        """
        鼠标抬起(x,y)坐标
        :param x: x坐标
        :param y: y坐标
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.mouse_up(self.handle, x, y)

    def mouse_down_up(self, x, y, tap_after_timeout=1.0, double=False):
        """
        鼠标抬起(x,y)坐标
        :param double:
        :param tap_after_timeout: 点击后 延迟
        :param x: x坐标
        :param y: y坐标
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                if double:
                    basic_functional.mouse_down(self.handle, x, y)
                    time.sleep(0.13)
                    basic_functional.mouse_up(self.handle, x, y)
                    time.sleep(0.2)
                    basic_functional.mouse_down(self.handle, x, y)
                    time.sleep(0.13)
                    basic_functional.mouse_up(self.handle, x, y)
                else:
                    basic_functional.mouse_down(self.handle, x, y)
                    time.sleep(0.13)
                    basic_functional.mouse_up(self.handle, x, y)
                    time.sleep(tap_after_timeout + self.timeout)
                logging.info((x, y))

    # 持续点击
    def mouse_Keep_clicking(self, x, y, keep_time=1):
        """

        :param x: x坐标
        :param y: y坐标
        :param keep_time: 持续点击时间
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.mouse_down(self.handle, x, y)
                time.sleep(keep_time)
                basic_functional.mouse_up(self.handle, x, y)

    def mouse_move(self, start_x, start_y, end_x, end_y, step=1, move_timeout=1.5):
        """
        鼠标滑动
        :param move_timeout:
        :param step: 滑动次数
        :param start_x: 起始坐标 x
        :param start_y: 起始坐标 y
        :param end_x: 结束坐标 x
        :param end_y: 结束坐标 y
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                # 定义滑动的持续时间（秒）
                duration = 2

                # 计算滑动路径上的每个点的位置
                num_points = int(duration * 10)  # 假设每秒采样10个点
                points = [(start_x + (end_x - start_x) * i / num_points,
                           start_y + (end_y - start_y) * i / num_points)
                          for i in range(num_points + 1)]

                for _ in range(step):
                    basic_functional.mouse_down(self.handle, start_x, start_y)

                    for point in points:
                        basic_functional.mouse_move(self.handle, int(point[0]), int(point[1]))
                        # 添加延时模拟真实拖动延迟
                        time.sleep(0.01)
                    # 释放最后一次的移动坐标
                    basic_functional.mouse_up(self.handle, end_x, end_y)
                    time.sleep(move_timeout + self.timeout)

    def key_down(self, key):
        """
        键盘按下 key
        :param key: 键值
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.key_down(self.handle, key)

    def key_up(self, key):
        """
        键盘抬起 kye
        :param key: 键值
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.key_up(self.handle, key)

    def key_down_up(self, key, key_down_timeout=2.0):
        """
        键盘点击 key
        :param key_down_timeout: 点击后延迟
        :param key: 键值
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.key_down(self.handle, key)
                time.sleep(0.05)
                basic_functional.key_up(self.handle, key)
                time.sleep(key_down_timeout + self.timeout)

    def input(self, text):
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.input_text(self.handle, text)

    def img_ocr(self, search_scope=(0, 0, 1334, 750)):
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                return basic_functional.img_ocr(self.handle, search_scope)


# 角色类
class Persona(BasicTask):

    def __init__(self, row, handle, mapp):
        """

        :rtype: object
        """
        super().__init__(row, handle, mapp)
        self.fight_stop = False
        self.process_kill_list = []
        self.kill_dict = {
            '普攻': event.task_config[mapp].get('普攻'),
            '绝学': event.task_config[mapp].get('绝学'),
            '闪避': event.task_config[mapp].get('闪避'),
            '关山': event.task_config[mapp].get('关山'),
            '自创1': event.task_config[mapp].get('自创1'),
            '自创2': event.task_config[mapp].get('自创2'),
            '自创3': event.task_config[mapp].get('自创3'),
            '技能1': event.task_config[mapp].get('技能1'),
            '技能2': event.task_config[mapp].get('技能2'),
            '技能3': event.task_config[mapp].get('技能3'),
            '技能4': event.task_config[mapp].get('技能4'),
            '技能5': event.task_config[mapp].get('技能5'),
            '技能6': event.task_config[mapp].get('技能6'),
            '技能7': event.task_config[mapp].get('技能7'),
            '技能8': event.task_config[mapp].get('技能8'),
        }
        self.process_kill_data(event.task_config[mapp].get('技能逻辑'))
        self.knapsack = event.task_config[mapp].get('背包')
        self.buddy = event.task_config[mapp].get('好友')
        self.team = event.task_config[mapp].get('队伍')
        self.map = event.task_config[mapp].get('地图')
        self.set = event.task_config[mapp].get('设置')
        self.faction = event.task_config[mapp].get('帮派')

    def initialization(self):
        pass

    def implement(self):
        pass

    def process_kill_data(self, data):
        pattern = r'技能\[(.*?)\] 延迟\[(.*?)\]ms <>'

        matches = re.findall(pattern, data)

        if matches:
            for match in matches:
                self.process_kill_list.append(match)
        else:
            self.process_kill_list = [('普攻', 200), ('技能1', 200), ('技能2', 200), ('技能3', 200), ('技能4', 200),
                                      ('技能5', 200), ('技能6', 200), ('技能7', 200), ('技能8', 200)]

    def fight(self):
        while not self.fight_stop and not event.unbind[self.mapp].is_set():
            for kill, defer in self.process_kill_list:
                if self.fight_stop or event.unbind[self.mapp].is_set():
                    continue
                print(kill)
                self.key_down_up(self.kill_dict[kill], key_down_timeout=int(defer) / 1000)

    def stop_fight(self):
        self.fight_stop = True

    def start_fight(self):
        self.fight_stop = False
        Thread(target=self.fight).start()


# 初始化
class Initialize(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.back_interface()
        self.key_down_up('ESC')
        self.Visual('端游模式', canny_process=True, threshold=0.8)
        self.key_down_up('ESC')
        time.sleep(2)

    def global_detection(self):
        while not event.unbind[self.mapp].is_set():
            if self.coord('梦仔', histogram_process=True, threshold=0.8):
                self.journal('梦仔弹窗')
                self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.75)
            time.sleep(2)


# 切换角色
class SwitchRoles(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.roles = [True, True, True, True, True, True]
        self.dl = None

    def initialization(self):
        pass

    def implement(self):
        if event.task_config[self.mapp].get('切角色1') and self.roles[1]:
            self.switch_roles(1)
            self.roles[1] = False
            return True
        elif event.task_config[self.mapp].get('切角色2') and self.roles[2]:
            self.switch_roles(2)
            self.roles[2] = False
            return True
        elif event.task_config[self.mapp].get('切角色3') and self.roles[3]:
            self.switch_roles(3)
            self.roles[3] = False
            return True
        elif event.task_config[self.mapp].get('切角色4') and self.roles[4]:
            self.switch_roles(4)
            self.roles[4] = False
            return True
        elif event.task_config[self.mapp].get('切角色5') and self.roles[5]:
            self.switch_roles(5)
            self.roles[5] = False
            return True
        elif all(self.roles):
            self.roles[0] = False
            return True

        return False

    def switch_roles(self, index):
        self.key_down_up('ESC')
        self.Visual('切换角色', binary_process=True, threshold=0.4)
        self.Visual('确定', binary_process=True, threshold=0.4)
        if self.Visual('进入游戏', binary_process=True, threshold=0.4, wait_count=10, tap=False):
            self.mouse_down_up(1274, 66 + 108 * (index - 1))
            self.Visual('进入游戏', binary_process=True, threshold=0.4)
            time.sleep(20)


# 世界喊话任务
class WorldShoutsTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.count_number = [0, 0]
        self.count_time = [0.0, 0.0]

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                self.Visual('聊天窗口关闭', canny_process=True, threshold=0.75, search_scope=(616, 203, 827, 512))
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('打开世界窗口')
                if not event.task_config[self.mapp].get('世界喊话') and not event.task_config[self.mapp].get(
                        '互联世界喊话'):
                    self.journal('请至少选则一项喊话')
                    self.cause_index = 0
                    continue
                self.mouse_down_up(309, 595)
                self.cause_index = 2
            elif switch == 2:
                self.keep_activate(1)
                self.Visual('世界频道', canny_process=True, threshold=0.7, wait_count=1)
                if self.Visual('输入文字', wait_count=1, canny_process=True, threshold=0.7):
                    self.input(event.task_config[self.mapp].get('世界喊话内容'))
                if not self.Visual('发送', canny_process=True, threshold=0.7):
                    self.cause_index = 1
                    continue
                self.count_time[0] = time.time()
                self.count_number[0] += 1
                self.journal(f'世界喊话{self.count_number[0]}次')
                self.cause_index = 4
            elif switch == 3:
                self.keep_activate(1)
                self.Visual('互联世界频道', canny_process=True, threshold=0.7, wait_count=1)
                if self.Visual('输入文字', wait_count=1, canny_process=True, threshold=0.7):
                    self.input(event.task_config[self.mapp].get('世界喊话内容'))
                if not self.Visual('发送', canny_process=True, threshold=0.7):
                    self.cause_index = 1
                    continue
                self.count_time[1] = time.time()
                self.count_number[1] += 1
                self.journal(f'互联世界喊话{self.count_number[1]}次')
                self.cause_index = 2
            elif switch == 4:
                if (self.count_number[0] >= int(event.task_config[self.mapp].get('世界喊话次数'))
                        or self.count_number[1] >= int(event.task_config[self.mapp].get('世界喊话次数'))):
                    self.cause_index = 0
                else:
                    self.cause_index = 3

    def determine(self):
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                self.Visual('聊天窗口关闭', canny_process=True, threshold=0.75, search_scope=(616, 203, 827, 512))

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            return 4

        elif self.cause_index == 3:
            if time.time() - self.count_time[0] > 25 and event.task_config[self.mapp].get('世界喊话'):
                return 2
            else:
                self.keep_activate(1)
                self.cause_index = 4

        elif self.cause_index == 4:
            if time.time() - self.count_time[1] > 25 and event.task_config[self.mapp].get('互联世界喊话'):
                return 3
            else:
                self.keep_activate(1)
                self.cause_index = 2

    def detect(self):
        time.sleep(1)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1


# 江湖英雄榜任务
class HeroListTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 1
        self.disrupted_event = True
        self.finish_count = 0

    def initialization(self):
        pass

    def implement(self):
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set() and self.disrupted_event:
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.key_down_up(event.persona[self.mapp].team)
                else:
                    self.journal('队伍打开失败 请检查键位')
                    return 0

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.cause_index = 3
            elif switch == 2:
                self.journal('位置检测')
                self.location_detection()
                self.key_down_up(event.persona[self.mapp].map)
                if self.coord('当前坐标金陵', canny_process=True, threshold=0.85):
                    self.cause_index = 2
                self.key_down_up(event.persona[self.mapp].map)
            elif switch == 3:
                self.journal('开始任务')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面纷争', canny_process=True, threshold=0.7, wait_count=1)
                self.Visual('江湖英雄榜', y=45, canny_process=True, threshold=0.7)
                if self.coord('江湖英雄榜界面', canny_process=True, threshold=0.7):
                    self.cause_index = 4

            elif switch == 4:
                if self.coord('江湖英雄榜次数', threshold=0.95, histogram_process=True):
                    self.cause_index = 0
                    continue
                self.disrupted_event = False
                self.Visual('匹配', '晋级赛', canny_process=True, threshold=0.7)
                self.Visual('确定', canny_process=True, threshold=0.7)
                if self.Visual('准备', wait_count=30, canny_process=True, threshold=0.7, tap=False):
                    if not event.task_config[self.mapp].get('江湖英雄榜秒退'):
                        if self.Visual('准备', wait_count=15, canny_process=True, threshold=0.7):
                            self.key_down('W')
                            time.sleep(6)
                            self.key_up('W')
                            event.persona[self.mapp].start_fight()
                            self.Visual('离开', wait_count=300, canny_process=True, threshold=0.7)
                            event.persona[self.mapp].stop_fight()
                    else:
                        if self.Visual('准备', wait_count=15, canny_process=True, threshold=0.7, tap=False):
                            self.Visual('退出江湖英雄榜', canny_process=True, threshold=0.7)
                            self.Visual('退出副本', canny_process=True, threshold=0.7)
                            self.Visual('离开', wait_count=300, canny_process=True, threshold=0.7)
                    self.finish_count += 1
                    self.journal(f'江湖英雄榜完成{self.finish_count}次')
                    self.Visual('江湖英雄榜界面', canny_process=True, threshold=0.7, tap=False, wait_count=10)
                    if self.finish_count >= int(event.task_config[self.mapp].get('江湖英雄榜次数')):
                        self.cause_index = 5

                    if not self.coord('江湖英雄榜界面', canny_process=True, threshold=0.7):
                        self.cause_index = 3
                    self.disrupted_event = True

    def determine(self):
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        # 位置检测
        elif self.cause_index == 1:
            return 2

        # 队伍检测
        elif self.cause_index == 2:
            if switch in [1]:
                return 1
            else:
                return -3

        # 开始任务
        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [2]:
                return 4
            else:
                self.cause_index = 3
                return -3

        elif self.cause_index == 5:
            if switch in [1]:
                self.cause_index = 0
            else:
                return -3

    def detect(self):
        time.sleep(1)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('江湖英雄榜界面', canny_process=True, threshold=0.7):
            return 2


# 日常副本
class DailyCopiesTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 1
        self.disrupted_event = True
        self.record_time = [0.0, 0.0, 0.0]
        self.record_count = [0]
        self.record_event = [True]

    def initialization(self):
        pass

    def initialize(self, cause_index):
        self.cause_index = cause_index
        if cause_index == 3:
            # 重置喊话时间
            self.record_time[0] = time.time()
        elif cause_index == 4:
            # 重置开启副本次数
            self.record_count[0] = 0
        elif cause_index == 5:
            # 记录副本开启时间
            self.record_time[2] = time.time()
            # 重置副本内匹配标志
            self.record_event[0] = True

    def implement(self):
        self.cause_index = 1
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if (event.disrupted_event[self.mapp].is_set() and self.disrupted_event
                    and not event.task_config[self.mapp].get('队伍模式') == '固定队模式'):
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if event.task_config[self.mapp].get('队伍模式') == '带队模式':
                if switch == 1:
                    self.journal('位置检测')
                    self.location_detection()
                    self.initialize(2)
                elif switch == 2:
                    self.key_down_up(event.persona[self.mapp].team)
                    if not self.coord('队伍界面', canny_process=True, threshold=0.7):
                        self.journal('队伍打开失败 请检查键位是否正确')
                        continue

                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.journal('已有队伍 改变队伍目标')
                        self.Visual('下拉', binary_process=True, threshold=0.7)
                        self.mouse_move(277, 215, 277, 315, 2)
                        self.Visual('无目标', binary_process=True, threshold=0.6, wait_count=1)

                    else:
                        self.journal('创建队伍')
                        self.Visual('创建队伍', canny_process=True, threshold=0.7)
                        self.Visual('下拉', binary_process=True, threshold=0.7)

                    self.Visual('队伍界面江湖纪事', canny_process=True, threshold=0.7)
                    self.Visual('队伍界面自动匹配', canny_process=True, threshold=0.7,
                                search_scope=(115, 330, 436, 629))
                    self.Visual('确定', canny_process=True, threshold=0.7)
                    self.Visual('确定', canny_process=True, threshold=0.7)

                    if (event.task_config[self.mapp].get('副本人数') + 1) != 1:
                        self.journal('开启自动匹配')
                        self.Visual('自动匹配', histogram_process=True, threshold=0.7)

                    if self.coord('日常', canny_process=True, threshold=0.7):
                        self.initialize(3)
                elif switch == 3:
                    self.key_down_up(event.persona[self.mapp].team)
                    # 判断离线队员
                    while not event.unbind[self.mapp].is_set():
                        if not self.Visual('离线', histogram_process=True, threshold=0.7):
                            break
                        self.Visual('请离队伍', binary_process=True, threshold=0.4)

                    if not self.coord('队伍界面', canny_process=True, threshold=0.7):
                        self.journal('队伍打开失败 请检查键位是否正确')
                        continue
                    self.Visual('普通喊话', canny_process=True, threshold=0.7)
                    num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))

                    if 10 - num >= (event.task_config[self.mapp].get('副本人数') + 1):
                        self.initialize(4)
                    self.key_down_up(event.persona[self.mapp].team)
                elif switch == 4:
                    self.world_shouts(f"{event.task_config[self.mapp].get('副本喊话内容')}")
                    # 重置喊话时间
                    self.record_time[0] = time.time()
                elif switch == 5:
                    self.journal('开启副本')
                    # 中断标志设置
                    self.disrupted_event = False
                    # 记录开启副本次数
                    self.record_count[0] += 1
                    # 开启副本次数上限
                    if self.record_count[0] > 3:
                        self.initialize(2)
                        continue
                    self.key_down_up(event.persona[self.mapp].team)
                    if not self.coord('队伍界面', canny_process=True, threshold=0.7):
                        self.journal('队伍打开失败 请检查键位是否正确')
                        continue
                    if not self.Visual('进入副本', wait_count=2, binary_process=True, threshold=0.4):
                        self.initialize(2)
                        continue
                    self.Visual('确认', wait_count=2, binary_process=True, threshold=0.4)
                    if not self.Visual('副本退出', '跳过剧情', histogram_process=True, threshold=0.7,
                                       search_scope=(1149, 0, 1334, 329), wait_count=25, tap=False):
                        continue
                    self.initialize(5)
                elif switch == 6:
                    if self.record_time[1] == 0:
                        self.journal('激活副本任务')
                        if self.record_event[0] and event.task_config[self.mapp].get('副本自动匹配'):
                            self.key_down_up(event.persona[self.mapp].team)
                            self.Visual('自动匹配', laplacian_process=True)
                            self.key_down_up(event.persona[self.mapp].team)
                            self.Visual('关闭', '关闭1', threshold=0.7, histogram_process=True)
                        self.Visual('主界面任务', '主界面任务2', canny_process=True, threshold=0.7, wait_count=1)
                        self.Visual('主界面任务1', histogram_process=True, threshold=0.7, wait_count=1)
                        self.Visual('副本日常任务', canny_process=True, threshold=0.7, search_scope=(36, 209, 152, 418),
                                    x=68, y=44)
                        # 副本内匹配标志
                        self.record_event[0] = False
                        # 激活副本时间
                        self.record_time[1] = time.time()

                    if self.coord('副本完成', histogram_process=True, threshold=0.6, search_scope=(1192, 160, 1334, 250)):
                        self.Visual('副本退出', histogram_process=True, threshold=0.7, search_scope=(1149, 107, 1334, 329))
                        if self.Visual('确定', binary_process=True, threshold=0.5):
                            self.Visual('副本挂机', wait_count=30, tap=False, canny_process=True, threshold=0.7)
                            self.cause_index = 0

                elif switch == 7:
                    self.journal('副本超时执行脱离卡死')
                    self.escape_stuck()
                    # 重置激活副本时间
                    self.record_time[1] = 0
                elif switch == 8:
                    self.journal('副本超时任务状态重置')
                    # 中断标志设置
                    self.disrupted_event = True
                    self.leave_team()
                    self.initialize(2)
                elif switch == 9:
                    self.journal('跳过剧情')
                    self.Visual('跳过剧情', wait_count=1, canny_process=True, threshold=0.7, tap_ago_timeout=0)

            elif event.task_config[self.mapp].get('队伍模式') == '固定队模式':
                if switch == 1:
                    if self.coord('副本完成', histogram_process=True, threshold=0.6, search_scope=(1192, 160, 1334, 250)):
                        self.Visual('副本退出', histogram_process=True, threshold=0.7, search_scope=(1149, 107, 1334, 329))
                        if self.Visual('确定', binary_process=True, threshold=0.5):
                            self.Visual('副本挂机', wait_count=30, tap=False, canny_process=True, threshold=0.7)
                        self.cause_index = 0
                elif switch == 2:
                    self.journal('副本确认')
                    self.Visual('确认', canny_process=True, threshold=0.7)
                    time.sleep(18)

    def determine(self):
        switch = self.detect()

        if event.task_config[self.mapp].get('队伍模式') == '带队模式':
            if self.cause_index == 0:
                if switch in [1]:
                    return 0
                else:
                    return -3

            elif self.cause_index == 1:
                if switch in [1]:
                    return 1
                else:
                    return -3

            elif self.cause_index == 2:
                if switch in [1]:
                    return 2
                else:
                    return -3

            elif self.cause_index == 3:
                if switch in [1]:
                    if switch == 1:
                        if (event.task_config[self.mapp].get('副本人数') + 1) == 1:
                            self.initialize(4)
                        elif time.time() - self.record_time[0] > 30:
                            return 4
                        else:
                            return 3
                else:
                    return -3

            elif self.cause_index == 4:
                if switch in [1]:
                    return 5
                else:
                    return -3

            elif self.cause_index == 5:
                if time.time() - self.record_time[1] > 360 and self.record_time[1] != 0:
                    return 7
                if time.time() - self.record_time[2] > 720:
                    return 8
                if switch in [1, 2, 3]:
                    if switch == 1:
                        return 6
                    elif switch == 2:
                        return 9

        elif event.task_config[self.mapp].get('队伍模式') == '固定队模式':
            if self.cause_index == 0:
                if switch in [1]:
                    return 0
                else:
                    return -3

            elif self.cause_index == 1:
                if switch in [1, 3]:
                    if switch == 1:
                        return 1
                    elif switch == 3:
                        return 2

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('跳过剧情', canny_process=True, threshold=0.7):
            return 2
        elif self.coord('日常1', canny_process=True, threshold=0.6):
            return 3


# 悬赏任务
class BountyMissionsTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 1
        self.disrupted_event = True
        self.record_time = [0.0, 0.0, 0.0, 0.0]
        self.record_count = [0, 0]
        self.record_event = [True]

    def initialization(self):
        pass

    def initialize(self, cause_index):
        self.cause_index = cause_index
        if cause_index == 3:
            # 世界喊话时间
            self.record_time[3] = time.time()
        elif cause_index == 5:
            # 重置开启副本上限
            self.record_count[1] = 0
        elif cause_index == 6:
            # 记录副本开启时间
            self.record_time[2] = time.time()
            # 重置激活时间
            self.record_time[1] = 0
            # 重置副本内匹配标志
            self.record_event[0] = True

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if (event.disrupted_event[self.mapp].is_set() and self.disrupted_event
                    and not event.task_config[self.mapp].get('队伍模式') == '固定队模式'):
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if event.task_config[self.mapp].get('队伍模式') == '带队模式':
                if switch == 1:
                    self.journal('位置检测')
                    self.location_detection()
                    self.initialize(2)

                elif switch == 2:
                    self.key_down_up(event.persona[self.mapp].team)

                    if not self.coord('队伍界面', canny_process=True, threshold=0.7):
                        self.journal('队伍打开失败 请检查键位是否正确')
                        continue

                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.journal('已有队伍 改变队伍目标')
                        self.Visual('下拉', binary_process=True, threshold=0.7)
                        self.mouse_move(277, 215, 277, 315, 2)
                        self.Visual('无目标', binary_process=True, threshold=0.6, wait_count=1)

                    else:
                        self.journal('创建队伍')
                        self.Visual('创建队伍', canny_process=True, threshold=0.7)
                        self.Visual('下拉', binary_process=True, threshold=0.7)

                    self.Visual('队伍界面江湖纪事', canny_process=True, threshold=0.7)
                    self.Visual('队伍界面自动匹配', canny_process=True, threshold=0.7,
                                search_scope=(115, 330, 436, 629))
                    self.Visual('确定', canny_process=True, threshold=0.7)
                    self.Visual('确定', canny_process=True, threshold=0.7)

                    if (event.task_config[self.mapp].get('副本人数') + 1) != 1:
                        self.journal('开启自动匹配')
                        self.Visual('自动匹配', histogram_process=True, threshold=0.7)

                    if self.coord('日常', canny_process=True, threshold=0.7):
                        self.initialize(3)

                elif switch == 3:
                    self.journal('悬赏检测')
                    self.key_down_up(event.persona[self.mapp].knapsack)
                    self.Visual('活动入口', canny_process=True, threshold=0.7)
                    self.Visual('活动', canny_process=True, threshold=0.7)
                    self.Visual('活动界面悬赏', canny_process=True, threshold=0.7)
                    # 悬赏数量记录
                    self.record_count[0] = len(self.coord('前往', canny_process=True, threshold=0.7))
                    if self.record_count[0] != 0:
                        self.initialize(4)

                elif switch == 4:
                    self.journal('接取悬赏')
                    num = len(self.coord('前往', canny_process=True, threshold=0.7))
                    if num == 3 or self.coord('悬赏完成标志', histogram_process=True, threshold=0.9):
                        if num == 0 and self.coord('悬赏完成标志', histogram_process=True, threshold=0.9):
                            self.cause_index = 0
                            continue
                        # 悬赏数量记录
                        self.record_count[0] = len(self.coord('前往', canny_process=True, threshold=0.7))
                        self.initialize(4)
                        continue
                    if not self.Visual('刷新', histogram_process=True, threshold=0.7):
                        continue
                    self.Visual('悬赏界面每日悬赏', y=330, search_scope=(267 + 231 * num, 182, 1197, 558),
                                histogram_process=True, threshold=0.7)
                    self.Visual('铜钱购买', histogram_process=True, threshold=0.7)

                elif switch == 5:
                    self.key_down_up(event.persona[self.mapp].team)
                    if not self.coord('队伍界面', canny_process=True, threshold=0.7):
                        self.journal('队伍打开失败 请检查键位是否正确')
                        continue
                    self.Visual('普通喊话', canny_process=True, threshold=0.7)

                    # 判断离线队员
                    while not event.unbind[self.mapp].is_set():
                        if not self.Visual('离线', histogram_process=True, threshold=0.7):
                            break
                        self.Visual('请离队伍', binary_process=True, threshold=0.4)

                    num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))
                    if 10 - num >= (event.task_config[self.mapp].get('副本人数') + 1):
                        self.initialize(5)
                    self.key_down_up(event.persona[self.mapp].team)

                elif switch == 6:
                    self.world_shouts(f"{event.task_config[self.mapp].get('副本喊话内容')}")
                    # 记录世界喊话时间
                    self.record_time[3] = time.time()
                elif switch == 7:
                    self.journal('开启副本')
                    # 任务中断标志设置
                    self.disrupted_event = False
                    # 记录开启副本次数
                    self.record_count[1] += 1
                    # 开启副本次数上限
                    if self.record_count[1] > 3:
                        self.leave_team()
                        self.initialize(2)
                        continue
                    self.key_down_up(event.persona[self.mapp].team)
                    if not self.coord('队伍界面', canny_process=True, threshold=0.7):
                        self.journal('队伍打开失败 请检查键位是否正确')
                        continue
                    if not self.Visual('进入副本', wait_count=2, binary_process=True, threshold=0.4):
                        self.initialize(2)
                        continue
                    self.Visual('确认', wait_count=2, binary_process=True, threshold=0.4)
                    if not self.Visual('副本退出', '跳过剧情', histogram_process=True, threshold=0.7,
                                       search_scope=(1149, 0, 1334, 329), wait_count=25, tap=False):
                        continue
                    self.initialize(6)
                elif switch == 8:
                    if self.record_time[1] == 0:
                        self.journal('激活副本任务')
                        if self.record_event[0] and event.task_config[self.mapp].get('副本自动匹配'):
                            self.key_down_up(event.persona[self.mapp].team)
                            self.Visual('自动匹配', laplacian_process=True)
                            self.key_down_up(event.persona[self.mapp].team)
                            self.Visual('关闭', '关闭1', threshold=0.7, histogram_process=True)
                        self.Visual('主界面任务', '主界面任务2', canny_process=True, threshold=0.7, wait_count=1)
                        self.Visual('主界面任务1', histogram_process=True, threshold=0.7, wait_count=1)
                        self.Visual('副本任务', histogram_process=True, threshold=0.6, search_scope=(36, 209, 102, 418),
                                    x=68, y=44)
                        self.record_event[0] = False
                        # 副本激活时间
                        self.record_time[1] = time.time()

                    if self.coord('副本完成', histogram_process=True, threshold=0.6, search_scope=(1192, 160, 1334, 250)):
                        self.journal('副本完成')
                        self.Visual('副本退出', histogram_process=True, threshold=0.7, search_scope=(1149, 107, 1334, 329))
                        if self.Visual('确定', binary_process=True, threshold=0.5):
                            self.Visual('副本挂机', wait_count=30, tap=False, canny_process=True, threshold=0.7)
                            self.disrupted_event = True
                            # 减少悬赏数量
                            self.record_count[0] -= 1
                            if self.record_count[0] == 0:
                                self.initialize(3)
                                continue
                            self.initialize(4)
                elif switch == 9:
                    self.journal('副本超时执行脱离卡死')
                    self.escape_stuck()
                    # 重置副本激活时间
                    self.record_time[1] = 0
                elif switch == 10:
                    self.journal('副本超时任务状态重置')
                    self.leave_team()
                    self.disrupted_event = True
                    self.initialize(2)
                elif switch == 11:
                    self.journal('跳过剧情')
                    self.Visual('跳过剧情', wait_count=1, canny_process=True, threshold=0.7, tap_ago_timeout=0)

            elif event.task_config[self.mapp].get('队伍模式') == '固定队模式':
                if switch == 1:
                    if self.coord('副本完成', histogram_process=True, threshold=0.6, search_scope=(1192, 160, 1334, 250)):
                        self.journal('副本完成')
                        # self.Visual('副本退出', histogram_process=True, threshold=0.7, search_scope=(1149, 107, 1334, 329))
                        # if self.Visual('确定', binary_process=True, threshold=0.5):
                        #     self.Visual('副本挂机', wait_count=30, tap=False, canny_process=True, threshold=0.7)
                        self.record_count[0] -= 1
                elif switch == 2:
                    self.journal('副本确认')
                    self.Visual('确认', canny_process=True, threshold=0.7)
                    time.sleep(18)
                elif switch == 3:
                    self.journal('悬赏检测')
                    self.key_down_up(event.persona[self.mapp].knapsack)
                    if not self.Visual('活动入口', canny_process=True, threshold=0.7):
                        continue
                    if not self.Visual('活动', canny_process=True, threshold=0.7):
                        continue
                    if not self.Visual('活动界面悬赏', canny_process=True, threshold=0.7):
                        continue
                    # 悬赏数量记录
                    self.record_count[0] = len(self.coord('前往', canny_process=True, threshold=0.7))
                    if self.record_count[0] != 0:
                        self.close_win(3, random_tap=False)
                        self.cause_index = 1
                elif switch == 4:
                    self.journal('接取悬赏')
                    num = len(self.coord('前往', canny_process=True, threshold=0.7))
                    if num == 3 or self.coord('悬赏完成标志', histogram_process=True, threshold=0.9):
                        if num == 0 and self.coord('悬赏完成标志', histogram_process=True, threshold=0.9):
                            self.cause_index = 0
                            continue
                        # 悬赏数量记录
                        self.record_count[0] = len(self.coord('前往', canny_process=True, threshold=0.7))
                        self.close_win(3, random_tap=False)
                        self.cause_index = 1
                        continue
                    if not self.Visual('刷新', histogram_process=True, threshold=0.7):
                        continue
                    self.Visual('悬赏界面每日悬赏', y=330, search_scope=(267 + 231 * num, 182, 1197, 558),
                                histogram_process=True, threshold=0.7)
                    self.Visual('铜钱购买', histogram_process=True, threshold=0.7)

    def determine(self):
        switch = self.detect()

        if event.task_config[self.mapp].get('队伍模式') == '带队模式':
            if self.cause_index == 0:
                if switch in [1]:
                    return 0
                else:
                    return -3

            elif self.cause_index == 1:
                return 1

            elif self.cause_index == 2:
                if switch in [1]:
                    return 2
                else:
                    return -3

            elif self.cause_index == 3:
                if switch in [1, 4]:
                    if switch == 1:
                        return 3
                    elif switch == 4:
                        return 4
                else:
                    return -3

            elif self.cause_index == 4:
                if switch in [1]:
                    if switch == 1:
                        if (event.task_config[self.mapp].get('副本人数') + 1) == 1:
                            self.initialize(5)
                        elif time.time() - self.record_time[3] > 30:
                            return 5
                        else:
                            return 6
                else:
                    return -3

            elif self.cause_index == 5:
                if switch in [1]:
                    if switch == 1:
                        return 7
                else:
                    return -3

            elif self.cause_index == 6:
                if time.time() - self.record_time[1] > 360 and self.record_time[1] != 0:
                    return 9
                if time.time() - self.record_time[2] > 720:
                    return 10
                if switch in [1, 2]:
                    if switch == 1:
                        return 8
                    elif switch == 2:
                        return 11

        elif event.task_config[self.mapp].get('队伍模式') == '固定队模式':
            if self.cause_index == 0:
                if switch in [1]:
                    return 0
                else:
                    return -3

            elif self.cause_index == 1:
                if self.record_count[0] == 0:
                    self.cause_index = 2
                if switch in [1, 3]:
                    if switch == 1:
                        return 1
                    elif switch == 3:
                        return 2

            elif self.cause_index == 2:
                if switch in [3, 4]:
                    if switch == 3:
                        return 2
                    elif switch == 4:
                        return 4
                else:
                    return 3

    def detect(self):
        time.sleep(1)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('跳过剧情', canny_process=True, threshold=0.7):
            return 2
        elif self.coord('日常1', canny_process=True, threshold=0.6):
            return 3
        elif self.coord('悬赏界面', canny_process=True, threshold=0.7):
            return 4


class FiberHomeTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 1
        self.disrupted_event = True
        self.record_time = []
        self.record_count = [0]
        self.record_event = []

    def initialization(self):
        pass

    def initialize(self, cause_index):
        self.keep_activate(5)
        self.cause_index = cause_index

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            # if not event.get_current_time().day() in ['3', '']:
            #     self.journal('活动不再开放时间')
            #     self.initialize(0)
            #     continue
            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.initialize(2)
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.key_down_up(event.persona[self.mapp].team)
                else:
                    self.journal('队伍打开失败 请检查键位')
                    return 0

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.initialize(3)
            elif switch == 3:
                self.journal('开始任务')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面纷争', canny_process=True, threshold=0.7)
                if not self.Visual('烽火雁门关', canny_process=True, threshold=0.7, y=45):
                    self.initialize(0)
                    continue
                self.close_win(2)
                self.initialize(4)

            elif switch == 4:
                self.Visual('确认', canny_process=True, threshold=0.7, wait_count=1)
                if not self.Visual('烽火雁门关1', canny_process=True, threshold=0.7, wait_count=1, tap=False):
                    continue
                event.persona[self.mapp].start_fight()
                self.initialize(5)

            elif switch == 5:

                self.Visual('复活点复活', canny_process=True, threshold=0.7, search_scope=(976, 223, 1334, 750), y=-50)

                if self.Visual('离开', histogram_process=True, threshold=0.7, wait_count=10, tap_after_timeout=0):
                    event.persona[self.mapp].stop_fight()
                    self.record_count[0] += 1
                    self.keep_activate(10)
                    self.initialize(3)

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                if self.record_count[0] >= event.task_config[self.mapp].get("烽火雁门关次数"):
                    self.initialize(0)
                else:
                    return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [1]:
                return 4

        elif self.cause_index == 5:
                return 5

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1


# 侠缘喊话
class ChivalryShoutTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = [0]
        self.record_event = []

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.key_down_up(event.persona[self.mapp].buddy)
                self.Visual('联系人', canny_process=True, threshold=0.7)
                self.Visual('输入编号或者玩家昵称', canny_process=True, threshold=0.7)
                self.input(f'{event.task_config[self.mapp].get("侠缘昵称")}')
                self.Visual('搜索图标', histogram_process=True, threshold=0.7)
                self.mouse_down_up(307, 327)
                self.cause_index = 2
            elif switch == 2:
                self.Visual('输入文字', canny_process=True, threshold=0.7)
                self.input('日出日落都浪漫,有风无风都自由')
                self.Visual('发送', laplacian_process=True)
                self.record_count[0] += 1
                self.journal(f'侠缘喊话{self.record_count[0]}次')

                if self.record_count[0] == 100:
                    self.cause_index = 0
                time.sleep(1.5)

    def determine(self):
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        if self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [2]:
                return 2
            else:
                self.cause_index = 1
                return -3

    def detect(self):
        time.sleep(1)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('好友界面', canny_process=True, threshold=0.7):
            return 2


# 帮派任务
class FactionTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = [0.0, 0.0]
        self.record_count = [0]
        self.record_event = [True]

    def initialization(self):
        pass

    def initialize(self, cause_index):
        self.cause_index = cause_index
        if cause_index == 4:
            # 记录任务开始时间
            self.record_time[0] = time.time()

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.initialize(2)
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.key_down_up(event.persona[self.mapp].team)
                else:
                    self.journal('队伍打开失败 请检查键位')
                    return 0

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.initialize(3)
            elif switch == 3:
                self.journal('开始任务')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面帮派', canny_process=True, threshold=0.7)
                self.Visual('帮派任务', canny_process=True, threshold=0.7, y=45)
                self.arrive()
                if self.Visual('帮派任务1', canny_process=True, threshold=0.7):
                    self.Visual('确定1', canny_process=True, threshold=0.7)
                    self.initialize(4)
            elif switch == 4:
                self.Visual('主界面任务', '主界面任务2', histogram_process=True, threshold=0.7, wait_count=1)
                self.Visual('主界面江湖', histogram_process=True, threshold=0.7, wait_count=1)
                self.mouse_move(158, 239, 198, 639, 2)
                self.Visual('主界面帮派任务', '主界面帮派任务1', histogram_process=True, threshold=0.7, wait_count=1)
                self.record_time[1] = time.time()
            elif switch == 5:

                if self.coord('获取途径', canny_process=True, threshold=0.7):
                    self.record_time[1] = 0
                    if self.coord('帮派仓库', canny_process=True, threshold=0.7) and self.record_event[0]:
                        self.journal('帮派仓库')
                        self.Visual('帮派仓库', canny_process=True, threshold=0.7, y=-45)
                        self.record_event[0] = False
                        self.close_win(1)
                        continue
                    self.record_event[0] = True
                    if self.coord('摆摊购买', canny_process=True, threshold=0.7):
                        self.journal('摆摊购买')
                        self.Visual('摆摊购买', canny_process=True, threshold=0.7, y=-45)
                        if self.Visual('查看全服', canny_process=True, threshold=0.7):
                            self.close_win(1)
                            continue
                        self.Visual('购买', canny_process=True, threshold=0.7)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.close_win(1)
                        continue

                    if self.coord('商城购买', canny_process=True, threshold=0.7):
                        self.journal('商城购买')
                        self.Visual('商城购买', canny_process=True, threshold=0.7, y=-45)
                        for _ in range(14):
                            if event.unbind[self.mapp].is_set():
                                break
                            self.mouse_down_up(970, 680, tap_after_timeout=0.2)
                        self.close_win(1)
                        continue

                if self.coord('一键提交', canny_process=True, threshold=0.7):
                    self.journal('任务提交')
                    self.Visual('一键提交', canny_process=True, threshold=0.7)
                    self.initialize(0)
                    continue
            elif switch == 6:
                self.close_win(2)

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if time.time() - self.record_time[1] > 30:
                return 4
            elif time.time() - self.record_time[0] > 720:
                self.initialize(0)
            if switch in [1, 2]:
                if switch == 1:
                    return 5
                elif switch == 2:
                    return 6

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('交易界面', '购买', histogram_process=True, threshold=0.7):
            return 2  # 交易界面


# 帮派设宴
class GangBanquet(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.get_count = 0
        self.cause_index = 1
        self.articles_index = 1

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('打开活动')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面帮派', canny_process=True, threshold=0.7, wait_count=1)

                if not self.Visual('帮派设宴', canny_process=True, threshold=0.7, y=45):
                    self.close_win(2)
                    self.cause_index = 0

                self.Visual('前往邀约', canny_process=True, threshold=0.7, wait_count=1)

                self.arrive()
                if not self.coord('邀请赴宴', '设宴界面', canny_process=True, threshold=0.7):
                    self.cause_index = 0

                self.Visual('邀请赴宴', canny_process=True, threshold=0.7)
                self.Visual('确认邀约', canny_process=True, threshold=0.7)
                if self.coord('设宴界面', canny_process=True, threshold=0.7):
                    self.cause_index = 2
            elif switch == 2:
                if self.articles_index == 9:
                    self.cause_index = 6
                    continue
                self.journal(f'物品{self.articles_index}')
                self.get_count = 0
                x = 633 + 172 * (3 if self.articles_index % 4 == 0 else self.articles_index % 4 - 1)
                y = 282 + 182 * (self.articles_index // 5)
                self.mouse_down_up(x, y)
                self.cause_index = 3
            elif switch == 3:
                if self.coord('获取', canny_process=True, threshold=0.7):
                    self.cause_index = 4
                    continue

                self.Visual('一键提交', canny_process=True, threshold=0.7)
                self.articles_index += 1
                self.cause_index = 2
            elif switch == 4:
                self.get_count += 1
                if self.get_count == 4:
                    self.articles_index += 1
                    self.cause_index = 2
                    continue
                if not self.Visual('获取', canny_process=True, threshold=0.7):
                    self.cause_index = 3

                self.journal('帮派仓库')

                self.Visual('帮派仓库', y=-71, canny_process=True, threshold=0.7)
                self.Visual('提交', canny_process=True, threshold=0.7)
                if self.coord('帮派仓库界面', canny_process=True, threshold=0.7):
                    self.close_win(1)

                if self.coord('一键提交', canny_process=True, threshold=0.7):
                    self.cause_index = 3
                    continue
                self.cause_index = 5
            elif switch == 5:
                if not self.Visual('获取', canny_process=True, threshold=0.7):
                    self.cause_index = 3
                if self.Visual('商城购买', y=-71, canny_process=True, threshold=0.7, wait_count=1):
                    self.journal('商城购买')
                    if self.coord('珍宝阁界面', canny_process=True, threshold=0.7):
                        self.mouse_down_up(988, 697)
                        self.close_win(1)

                    if self.coord('一键提交', canny_process=True, threshold=0.7):
                        self.cause_index = 3
                        continue
                    self.cause_index = 4
                    continue

                if self.Visual('摆摊购买', y=-71, canny_process=True, threshold=0.7, wait_count=1):
                    self.journal('摆摊购买')
                    self.Visual('查看全服', canny_process=True, threshold=0.7, wait_count=2)

                    if self.Visual('购买', canny_process=True, threshold=0.7):
                        self.Visual('确定', canny_process=True, threshold=0.7)

                        if self.coord('确认', canny_process=True, threshold=0.7):
                            self.Visual('确认', canny_process=True, threshold=0.7)
                            self.Visual('确认', canny_process=True, threshold=0.7, wait_count=20)

                    if self.coord('交易界面', laplacian_process=True):
                        self.close_win(1)

                    if self.coord('一键提交', canny_process=True, threshold=0.7):
                        self.cause_index = 3
                        continue
                    self.cause_index = 4
                    continue
            elif switch == 6:
                self.journal('提交完成, 开始设宴')
                self.Visual('开始设宴', canny_process=True, threshold=0.7)
                self.Visual('确定', canny_process=True, threshold=0.7)
                self.close_win(3)
                self.cause_index = 0

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [2]:
                return 2
            else:
                self.cause_index = 1

        elif self.cause_index == 3:
            if switch in [2]:
                return 3
            else:
                self.cause_index = 1

        elif self.cause_index == 4:
            if switch in [2]:
                return 4
            else:
                self.cause_index = 1

        elif self.cause_index == 5:
            if switch in [2]:
                return 5
            else:
                self.cause_index = 1

        elif self.cause_index == 6:
            if switch in [2]:
                return 6

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('设宴界面', canny_process=True, threshold=0.7):
            return 2


# 破阵设宴
class BreakingBanquet(GangBanquet):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.location_detection()
                self.journal('打开活动')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面帮派', canny_process=True, threshold=0.7, wait_count=1)

                if not self.Visual('破阵设宴', canny_process=True, threshold=0.7, y=45):
                    self.close_win(2)
                    self.cause_index = 0

                self.Visual('前往邀约', canny_process=True, threshold=0.7, wait_count=1)

                self.arrive()
                if not self.coord('邀请赴宴', '设宴界面', canny_process=True, threshold=0.7):
                    self.cause_index = 0

                self.Visual('邀请赴宴', canny_process=True, threshold=0.7)
                self.Visual('确认邀约', canny_process=True, threshold=0.7)
                if self.coord('设宴界面', canny_process=True, threshold=0.7):
                    self.cause_index = 2
            elif switch == 2:
                if self.articles_index == 9:
                    self.cause_index = 6
                    continue
                self.journal(f'物品{self.articles_index}')
                self.get_count = 0
                x = 633 + 172 * (3 if self.articles_index % 4 == 0 else self.articles_index % 4 - 1)
                y = 282 + 182 * (self.articles_index // 5)
                self.mouse_down_up(x, y)
                self.cause_index = 3
            elif switch == 3:
                if self.coord('获取1', canny_process=True, threshold=0.7):
                    self.cause_index = 4
                    continue

                self.Visual('一键提交2', canny_process=True, threshold=0.7)
                self.articles_index += 1
                self.cause_index = 2
            elif switch == 4:
                self.get_count += 1
                if self.get_count == 4:
                    self.articles_index += 1
                    self.cause_index = 2
                    continue
                if not self.Visual('获取1', canny_process=True, threshold=0.7):
                    self.cause_index = 3
                self.journal('帮派仓库')
                self.Visual('帮派仓库', y=-71, canny_process=True, threshold=0.7)
                self.Visual('提交', canny_process=True, threshold=0.7)
                if self.coord('帮派仓库界面', canny_process=True, threshold=0.7):
                    self.close_win(1)

                if self.coord('一键提交', canny_process=True, threshold=0.7):
                    self.cause_index = 3
                    continue
                self.cause_index = 5
            elif switch == 5:
                if not self.Visual('获取1', canny_process=True, threshold=0.7):
                    self.cause_index = 3

                if self.Visual('商城购买', y=-71, canny_process=True, threshold=0.7, wait_count=1):
                    self.journal('商城购买')
                    if self.coord('珍宝阁界面', canny_process=True, threshold=0.7):
                        self.mouse_down_up(988, 697)
                        self.close_win(1)
                    if self.coord('一键提交2', canny_process=True, threshold=0.7):
                        self.cause_index = 3
                        continue
                    self.cause_index = 4
                    continue

                if self.Visual('摆摊购买', y=-71, canny_process=True, threshold=0.7, wait_count=1):
                    self.journal('摆摊购买')
                    self.Visual('查看全服', canny_process=True, threshold=0.7, wait_count=2)

                    if self.Visual('购买', canny_process=True, threshold=0.7):
                        self.Visual('确定', canny_process=True, threshold=0.7)

                        if self.coord('确认', canny_process=True, threshold=0.7):
                            self.Visual('确认', canny_process=True, threshold=0.7)
                            self.Visual('确认', canny_process=True, threshold=0.7, wait_count=20)

                    if self.coord('交易界面', laplacian_process=True):
                        self.close_win(1)

                    if self.coord('一键提交2', canny_process=True, threshold=0.7):
                        self.cause_index = 3
                        continue
                    self.cause_index = 4
                    continue
            elif switch == 6:
                self.journal('提交完成, 开始设宴')
                self.Visual('开始设宴', canny_process=True, threshold=0.7)
                self.Visual('确定', canny_process=True, threshold=0.7)
                self.close_win(3)
                self.cause_index = 0

    def determine(self):
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [2]:
                return 2
            else:
                self.cause_index = 1

        elif self.cause_index == 3:
            if switch in [2]:
                return 3
            else:
                self.cause_index = 1

        elif self.cause_index == 4:
            if switch in [2]:
                return 4
            else:
                self.cause_index = 1

        elif self.cause_index == 5:
            if switch in [2]:
                return 5
            else:
                self.cause_index = 1

        elif self.cause_index == 6:
            if switch in [2]:
                return 6

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('设宴界面', canny_process=True, threshold=0.7):
            return 2


# 课业任务
class LessonTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = [0.0, 0.0]
        self.record_count = [0]
        self.record_event = []

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.cause_index = 2
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if not self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    self.journal('队伍打开失败 请检查键位')
                    self.cause_index = 0
                    continue
                if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                    self.Visual('确定', canny_process=True, threshold=0.7)

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.cause_index = 3

                self.key_down_up(event.persona[self.mapp].team)

            elif switch == 3:
                self.journal('开始任务')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', histogram_process=True, threshold=0.7)
                self.Visual('活动', binary_process=True, threshold=0.5)
                self.Visual('活动界面江湖', binary_process=True, threshold=0.6, wait_count=1)
                if not self.Visual('濯剑', '观梦', '漱尘', '止杀', '锻心', '吟风', '含灵', '寻道', '悟禅', '归义',
                                   histogram_process=True, threshold=0.7, y=45):
                    self.cause_index = 0
                    continue

                self.Visual('课业', binary_process=True, threshold=0.5, y=210)
                self.arrive()
                self.Visual('课业3', '悟禅1', canny_process=True, threshold=0.7)
                self.Visual('确定1', binary_process=True, threshold=0.4)
                self.record_count[0] = 0
                self.cause_index = 4

            elif switch == 4:
                if self.record_count[0] == 5:
                    self.cause_index = 0
                    continue

                if not self.Visual('困难课业', canny_process=True, threshold=0.7):
                    self.Visual('刷新1', binary_process=True, threshold=0.5, x=-55)
                    self.Visual('确定', binary_process=True, threshold=0.4)
                    self.record_count[0] += 1
                    continue

                if self.coord('已接取', binary_process=True, threshold=0.4):
                    self.journal('已有课业任务')
                    self.close_win(2)
                # 激活时间
                self.record_time[0] = time.time()
                # 开始时间
                self.record_time[1] = time.time()
                self.cause_index = 5
            elif switch == 5:
                if self.coord('商城购买', canny_process=True, threshold=0.7):
                    self.journal('商城购买')
                    self.Visual('商城购买', canny_process=True, threshold=0.7, y=-71)
                    for _ in range(14):
                        if event.unbind[self.mapp].is_set():
                            break
                        self.mouse_down_up(970, 680, tap_after_timeout=0.2)
                    self.close_win(1)
                elif self.coord('一键提交', canny_process=True, threshold=0.7):
                    self.journal('提交物品')
                    self.Visual('一键提交', canny_process=True, threshold=0.6, wait_count=1)
                    self.mouse_down_up(0, 0)
                    if self.Visual('确定', canny_process=True, threshold=0.7):
                        self.journal('课业任务完成')
                        self.close_win(3)
                        self.cause_index = 0
                elif self.coord('一大桶水', canny_process=True, threshold=0.75, search_scope=(775, 188, 1260, 600)):
                    self.Visual('使用1', canny_process=True, threshold=0.7)
            elif switch == 6:
                self.journal('定时激活课业任务')
                self.Visual('主界面任务', '主界面任务2', histogram_process=True, threshold=0.7, wait_count=1)
                self.Visual('主界面江湖', histogram_process=True, threshold=0.7, wait_count=1)
                self.mouse_move(158, 239, 198, 639, 2)
                self.Visual(
                    '观梦任务', '锻心任务', '归义任务', '止杀任务', '吟风任务', '漱尘任务', '含灵任务', '寻道任务',
                    '课业任务', '濯剑任务',
                    '观梦任务1', '锻心任务1', '归义任务1', '止杀任务1', '吟风任务1', '漱尘任务1', '含灵任务1',
                    '寻道任务1', '课业任务1', '濯剑任务1',
                    histogram_process=True, threshold=0.5, search_scope=(41, 211, 268, 422), wait_count=1)
                self.record_time[0] = time.time()
            elif switch == 7:
                self.journal('课业排序')
                for _ in range(10):
                    try:
                        if target := random.sample(self.coord('排序', binary_process=True, threshold=0.6,
                                                              search_scope=(412, 364, 1233, 576)), 2):
                            self.mouse_move(target[0][0], target[0][1], target[1][0], target[1][1], move_timeout=0)
                    except ValueError:
                        pass
            elif switch == 8:
                self.journal('课业杂货商人购买')
                self.Visual('铜钱购买', histogram_process=True, threshold=0.7, search_scope=(820, 517, 1242, 673))
            elif switch == 9:
                self.journal('课业答题任务')
                self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
            elif switch == 10:
                self.journal('摆摊购买')
                self.Visual('购买', canny_process=True, threshold=0.7)
                self.Visual('确定', canny_process=True, threshold=0.7)
                self.close_win(1)
            elif switch == 11:
                self.Visual('对话回答', canny_process=True, threshold=0.7, search_scope=(800, 286, 1334, 750),
                            ignore_scope=(24, 8, 428, 72))
                self.record_time[0] = 0  # 重新激活防止打错卡住

    def determine(self):
        switch = self.detect()
        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [2]:
                return 4
            else:
                self.cause_index = 3

        elif self.cause_index == 5:
            if time.time() - self.record_time[0] > 30:
                return 6
            elif time.time() - self.record_time[1] > 720:
                self.cause_index = 0
            if switch in [1, 3, 4, 5, 6, 7]:
                if switch == 1:
                    return 5
                elif switch == 3:
                    return 7
                elif switch == 4:
                    return 8
                elif switch == 5:
                    return 9
                elif switch == 6:
                    return 10
                elif switch == 7:
                    return 11

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('刷新1', canny_process=True, threshold=0.7):
            return 2  # 任务接取界面
        elif self.coord('倒计时', canny_process=True, threshold=0.7):
            return 3  # 华山课业排序界面
        elif self.coord('杂货商人', histogram_process=True, threshold=0.7):
            return 4  # 杂货商人界面
        elif self.coord('课业任务答题', histogram_process=True, threshold=0.65):
            return 5  # 答题界面
        elif self.coord('交易界面', '购买', histogram_process=True, threshold=0.7):
            return 6  # 交易界面
        elif self.coord('对话回答', canny_process=True, threshold=0.7, search_scope=(800, 286, 1334, 750),
                        ignore_scope=(24, 8, 428, 72)):
            return 7  # 对话任务


# 发布悬赏
class PostBounty(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = []
        self.record_event = [False, False]

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('打开活动')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面悬赏', canny_process=True, threshold=0.7)

                if self.coord('悬赏界面', canny_process=True, threshold=0.7):
                    self.cause_index = 2
            elif switch == 2:
                if all(self.record_event):
                    self.cause_index = 0
                    continue
                self.Visual('发布', search_scope=(968, 558, 1269, 700), canny_process=True, threshold=0.7)
                self.Visual('下拉', histogram_process=True, threshold=0.8)

                if not self.record_event[0]:
                    self.record_event[0] = True
                    if not self.Visual('行商次数', histogram_process=True, threshold=0.9):
                        self.mouse_down_up(0, 0)
                        self.close_win(1, random_tap=False)
                        continue
                    self.Visual('发布悬赏', canny_process=True, threshold=0.7)
                    self.Visual('确定', canny_process=True, threshold=0.7)
                    continue

                if not self.record_event[1]:
                    self.record_event[1] = True
                    if not self.Visual('聚义次数', histogram_process=True, threshold=0.9):
                        self.mouse_down_up(0, 0)
                        self.close_win(1, random_tap=False)
                        continue
                    self.Visual('发布悬赏', canny_process=True, threshold=0.7)
                    self.Visual('确定', canny_process=True, threshold=0.7)
                    continue

    def determine(self):
        switch = self.detect()
        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [2]:
                return 2
            else:
                return -3

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('悬赏界面', canny_process=True, threshold=0.7):
            return 2

    # def implement(self):
    #     self.key_down_up(event.persona[self.mapp].knapsack)
    #     self.Visual('活动入口', histogram_process=True, threshold=0.7)
    #     self.Visual('活动', binary_process=True, threshold=0.5)
    #     self.Visual('活动界面悬赏', laplacian_process=True)
    #
    #     self.Visual('发布', search_scope=(968, 558, 1269, 700), laplacian_process=True)
    #     self.Visual('下拉', histogram_process=True, threshold=0.8)
    #
    #     if self.Visual('行商次数', histogram_process=True, threshold=0.9):
    #         self.Visual('发布悬赏', laplacian_process=True)
    #         self.Visual('确定', laplacian_process=True)
    #     else:
    #         self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
    #
    #     self.Visual('发布', search_scope=(968, 558, 1269, 700), laplacian_process=True)
    #     self.Visual('下拉', histogram_process=True, threshold=0.8)
    #
    #     if self.Visual('聚义次数', histogram_process=True, threshold=0.9):
    #         self.Visual('发布悬赏', laplacian_process=True)
    #         self.Visual('确定', laplacian_process=True)
    #     else:
    #         self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
    #
    #     time.sleep(2)
    #     self.mouse_down_up(0, 0)
    #     self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
    #     self.mouse_down_up(0, 0)
    #     self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
    #     self.mouse_down_up(0, 0)
    #     self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)


# 茶馆说书
class TeaStory(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = []
        self.record_event = []

    def initialization(self):
        pass

    def initialize(self, cause_index):
        self.keep_activate(3)
        self.cause_index = cause_index

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                self.close_win(3)
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.initialize(2)
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.key_down_up(event.persona[self.mapp].team)
                else:
                    self.journal('队伍打开失败 请检查键位')
                    return 0

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.initialize(3)

            elif switch == 3:
                self.journal('开始活动')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面江湖', canny_process=True, threshold=0.7)
                self.Visual('茶馆说书', canny_process=True, threshold=0.7, y=45)
                self.arrive()
                if self.Visual('进入茶馆', canny_process=True, threshold=0.7, tap_after_timeout=5):
                    self.initialize(4)

            elif switch == 4:
                if self.coord('茶馆界面', canny_process=True, threshold=0.7):
                    if self.Visual('退出茶馆', canny_process=True, threshold=0.7, wait_count=1, tap_after_timeout=6,
                                   double=True):
                        self.close_win(3)
                        self.initialize(0)
                        continue
                    if not self.coord('甲1', '乙1', '丙1', '丁1', histogram_process=True, threshold=0.7,
                                      search_scope=(1089, 238, 1334, 750)):
                        self.Visual('甲', '乙', '丙', '丁', histogram_process=True, wait_count=1, threshold=0.65,
                                    search_scope=(1089, 238, 1334, 750))

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [1]:
                if switch == 1:
                    return 4

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1

    #             if self.Visual('退出茶馆', histogram_process=True, threshold=0.7, wait_count=1, tap_after_timeout=0.1):
    #                 time.sleep(6)
    #                 break
    #             time.sleep(1)
    #
    #         for _ in range(4):
    #             if event.unbind[self.mapp].is_set():
    #                 break
    #             self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)


# 华山论剑
class TheSword(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = [0]
        self.record_event = []

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.key_down_up(event.persona[self.mapp].map)
                if self.coord('当前坐标金陵', canny_process=True, threshold=0.85):
                    self.cause_index = 2
                self.key_down_up(event.persona[self.mapp].map)
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.key_down_up(event.persona[self.mapp].team)
                else:
                    self.journal('队伍打开失败 请检查键位')
                    return 0

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.cause_index = 3
            elif switch == 3:
                self.journal('开始任务')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面纷争', canny_process=True, threshold=0.7, wait_count=1)
                self.Visual('华山论剑', canny_process=True, threshold=0.7, y=45, x=-50)
                self.cause_index = 4
            elif switch == 4:

                if not self.coord('取消匹配', canny_process=True, threshold=0.7):
                    self.Visual('匹配1', canny_process=True, wait_count=1, threshold=0.7)

                self.Visual('确认', canny_process=True, threshold=0.7, wait_count=1)

                if not self.Visual('准备1', canny_process=True, threshold=0.7, wait_count=15, tap=False):
                    continue
                self.record_count[0] += 1
                self.cause_index = 4
                if event.task_config[self.mapp].get('华山论剑秒退'):
                    self.Visual('退出论剑', canny_process=True, threshold=0.7)
                    self.Visual('确定', canny_process=True, threshold=0.7)
                    self.Visual('离开', canny_process=True, threshold=0.7, wait_count=5)
                    continue
                self.Visual('准备1', canny_process=True, threshold=0.7, wait_count=1)
                event.persona[self.mapp].start_fight()
                self.Visual('离开', histogram_process=True, threshold=0.7, wait_count=180, tap_after_timeout=0)
                event.persona[self.mapp].stop_fight()
                self.keep_activate(5)

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:

            if switch in [1, 2]:
                if self.record_count[0] >= int(event.task_config[self.mapp].get('华山论剑次数')):
                    self.cause_index = 0
                elif switch == 1:
                    self.cause_index = 3
                elif switch == 2:
                    self.cause_index = 5

        elif self.cause_index == 5:
            return 4

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('论剑界面', canny_process=True, threshold=0.7):
            return 2


# 每日一卦
class HexagramDay(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up(event.persona[self.mapp].knapsack)
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', laplacian_process=True)
        self.Visual('活动界面游历', laplacian_process=True)
        if self.Visual('每日一卦', histogram_process=True, threshold=0.7, y=45):
            if self.Visual('算命占卜', canny_process=True, wait_count=400, threshold=0.7):
                self.Visual('落笔', canny_process=True, threshold=0.7)
                self.Visual('接受', binary_process=True, threshold=0.4)
                self.Visual('确定', binary_process=True, threshold=0.4)
                self.mouse_down_up(0, 0)

        self.close_win(2)


# 精进行当神厨-江湖急送
class UrgentDeliveryTask(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        implement = True
        buy = False
        for _ in range(30):
            if event.unbind[self.mapp].is_set():
                break
            if implement:
                self.journal('打开帮派接取急送任务')
                self.key_down_up('O')
                self.Visual('势力', histogram_process=True, threshold=0.7)
                self.Visual('江湖急送', laplacian_process=True)
                if self.Visual('订单上限', histogram_process=True, threshold=0.7, wait_count=1):
                    self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7, random_tap=False)
                    self.journal('今日订单已达上限')
                    break
                self.Visual('抢单', histogram_process=True, threshold=0.7)
                self.journal('接取成功')
                self.close_win(2, random_tap=False)
                implement = False

            self.Visual('主界面任务', '主界面任务2', histogram_process=True, threshold=0.7, wait_count=1)
            self.Visual('主界面江湖', histogram_process=True, threshold=0.7, wait_count=1)
            self.mouse_move(158, 239, 198, 639)
            if self.Visual('外卖', binary_process=True, threshold=0.4, wait_count=3, search_scope=(41, 211, 268, 422)):
                self.journal('激活急送任务')

            if self.Visual('前往购买', histogram_process=True, threshold=0.7, x=90, tap=False):
                if buy:
                    self.journal('当前订单购买次数上限 >>> 放弃本次订单')
                    self.Visual('放弃订单', binary_process=True, threshold=0.4)
                    self.Visual('确定', binary_process=True, threshold=0.4)
                    self.journal('放弃完成 >>> 开始再次购买')
                    implement = True
                    buy = False
                    continue
                self.Visual('前往购买', histogram_process=True, threshold=0.7, x=90)
                self.journal('准备商会购买物品')
                self.Visual('神厨商会', histogram_process=True, threshold=0.7, y=-45)
                self.Visual('菜品标签', histogram_process=True, threshold=0.7)
                if count := self.coord('选中标签', histogram_process=True, threshold=0.8):
                    self.mouse_down_up(1334, 750)
                    for i in range(len(count) + 2):
                        time.sleep(2.5)
                        if self.Visual('符合', histogram_process=True, threshold=0.7):
                            self.Visual('购买1', histogram_process=True, threshold=0.7)
                            self.Visual('确定', histogram_process=True, threshold=0.7)
                            buy = True
                            break
                        else:
                            if i == 0:
                                self.Visual('菜品品质', histogram_process=True, threshold=0.7)
                                self.journal('去掉一个标签')
                                self.Visual('选中标签', histogram_process=True, threshold=0.8)
                                self.mouse_down_up(1334, 750)
                            else:
                                self.Visual('菜品标签', histogram_process=True, threshold=0.7)
                                self.journal('去掉一个标签')
                                self.Visual('选中标签', histogram_process=True, threshold=0.8)
                                self.mouse_down_up(1334, 750)
                    self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
                    if not buy:
                        buy = True
                        self.journal('购买失败 >>> 放弃任务')
            elif self.Visual('领取食盆', histogram_process=True, threshold=0.7):
                self.Visual('菜品打包', histogram_process=True, threshold=0.6, wait_count=180)
                self.Visual('选择菜品', histogram_process=True, threshold=0.7, x=-120)
                self.Visual('选择', histogram_process=True, threshold=0.7)
                self.Visual('选择菜品', histogram_process=True, threshold=0.7)
                self.Visual('确定', laplacian_process=True)
            elif (self.coord('自动寻路中', histogram_process=True, threshold=0.7) or
                  self.coord('菜品送达', histogram_process=True, threshold=0.7, search_scope=(731, 200, 1234, 604))):
                for _ in range(20):
                    if event.unbind[self.mapp].is_set():
                        break
                    if self.Visual('菜品送达', histogram_process=True, threshold=0.7,
                                   search_scope=(731, 200, 1234, 604),
                                   wait_count=10):
                        self.Visual('菜品送达', histogram_process=True, threshold=0.7,
                                    search_scope=(731, 200, 1234, 604),
                                    wait_count=1)
                        if self.Visual('确认1', histogram_process=True, threshold=0.7, wait_count=20):
                            implement = True
                            buy = False
                            break
                    self.Visual('外卖', binary_process=True, threshold=0.4, wait_count=3,
                                search_scope=(41, 211, 268, 422))


# 精进行当豪侠-狂饮豪拳
class DrinkPunch(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = []
        self.record_event = []

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set() and self.disrupted_event:
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.key_down_up(event.persona[self.mapp].map)
                if self.coord('当前坐标金陵', canny_process=True, threshold=0.85):
                    self.cause_index = 2
                self.key_down_up(event.persona[self.mapp].map)
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.key_down_up(event.persona[self.mapp].team)
                else:
                    self.journal('队伍打开失败 请检查键位')
                    return 0

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.cause_index = 3

            elif switch == 3:
                self.journal('开始活动')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.mouse_move(1231, 544, 1231, 444)
                self.Visual('精进行当', canny_process=True, threshold=0.7)
                self.Visual('狂饮豪拳', canny_process=True, threshold=0.7)
                self.Visual('前往', canny_process=True, threshold=0.7)
                self.arrive()
                time.sleep(10)
                self.disrupted_event = False
                self.Visual('简单喝', canny_process=True, threshold=0.7)
                if self.coord('狂饮豪拳界面', canny_process=True, threshold=0.7):
                    self.cause_index = 4
            elif switch == 4:
                if self.coord('左一拳', histogram_process=True, threshold=0.7):
                    index = random.randint(1, 3)
                    if index == 1:
                        self.mouse_move(107, 445, 695, 430)
                    elif index == 2:
                        self.mouse_move(172, 566, 695, 430)
                    elif index == 3:
                        self.mouse_move(290, 673, 695, 430)
                elif self.coord('右一拳', histogram_process=True, threshold=0.6):
                    index = random.randint(1, 3)
                    if index == 1:
                        self.mouse_move(1237, 448, 695, 430)
                    elif index == 2:
                        self.mouse_move(1152, 568, 695, 430)
                    elif index == 3:
                        self.mouse_move(1047, 673, 695, 430)
                elif self.coord('缩一拳', histogram_process=True, threshold=0.7):
                    index = random.randint(1, 2)
                    if index == 1:
                        self.mouse_move(352, 454, 0, 750)
                    elif index == 2:
                        self.mouse_move(1007, 464, 1334, 750)

            elif switch == 5:
                self.key_down_up(event.persona[self.mapp].faction)
                self.Visual('返回帮派', canny_process=True, threshold=0.7, y=-45)
                self.arrive()
                self.location_detection()
                self.key_down_up(event.persona[self.mapp].map)
                if self.coord('当前坐标金陵', canny_process=True, threshold=0.85):
                    self.cause_index = 0
                self.key_down_up(event.persona[self.mapp].map)
            elif switch == 6:
                self.Visual('离开1', canny_process=True, threshold=0.7, wait_count=1)
                if self.Visual('确定', canny_process=True, threshold=0.7):
                    self.cause_index = 5

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [2]:
                return 4
            else:
                return 6

        elif self.cause_index == 5:
            if switch in [1]:
                return 5

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('狂饮豪拳界面', canny_process=True, threshold=0.7):
            return 2
    #         if self.Visual('离开1', laplacian_process=True, wait_count=1):
    #             self.Visual('确定', laplacian_process=True)
    #             break
    #         if self.coord('左一拳', histogram_process=True, threshold=0.7):
    #             index = random.randint(1, 3)
    #             if index == 1:
    #                 self.mouse_move(107, 445, 695, 430)
    #             elif index == 2:
    #                 self.mouse_move(172, 566, 695, 430)
    #             elif index == 3:
    #                 self.mouse_move(290, 673, 695, 430)
    #         elif self.coord('右一拳', histogram_process=True, threshold=0.6):
    #             index = random.randint(1, 3)
    #             if index == 1:
    #                 self.mouse_move(1237, 448, 695, 430)
    #             elif index == 2:
    #                 self.mouse_move(1152, 568, 695, 430)
    #             elif index == 3:
    #                 self.mouse_move(1047, 673, 695, 430)
    #         elif self.coord('缩一拳', histogram_process=True, threshold=0.7):
    #             index = random.randint(1, 2)
    #             if index == 1:
    #                 self.mouse_move(352, 454, 0, 750)
    #             elif index == 2:
    #                 self.mouse_move(1007, 464, 1334, 750)


# 聚义平冤
class PacifyInjusticeTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.team_detect = True
        self.start = 0
        self.task_start = 0
        self.cause = {
            0: '检测队伍',
            1: '检测队伍人数',
            2: '前往npc',
            3: '到达npc接取任务',
            4: '等待任务完成'
        }

    def initialization(self):
        pass

    def implement(self):
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if switch == -3:
                self.back_interface()
            elif switch == 0:
                self.journal('打开队伍')
                self.key_down_up(event.persona[self.mapp].team)
            elif switch == 1:
                self.journal('检测队伍')
                if not self.coord('聚义平冤队伍', binary_process=True, threshold=0.7):
                    self.leave_team()
                    self.pacify_injustice_Task_1()
                else:
                    self.key_down_up(event.persona[self.mapp].team)
                self.team_detect = False
                self.cause_index += 1
            elif switch == 2:
                self.journal('检测队伍人数')
                self.pacify_injustice_Task_2()
                self.cause_index += 1
            elif switch == 3:
                self.journal('打开背包')
                self.key_down_up(event.persona[self.mapp].knapsack)
            elif switch == 4:
                self.journal('进入活动页面')
                self.Visual('活动入口', histogram_process=True, threshold=0.7)
                self.Visual('活动', binary_process=True, threshold=0.5)
            elif switch == 5:
                self.journal('查找聚义平冤任务')
                self.Visual('活动界面行当', binary_process=True, threshold=0.6)

                if not self.Visual('聚义平冤', binary_process=True, threshold=0.6, y=45):
                    self.journal('任务已完成')
                    return 0
                self.journal('前往npc')
                self.cause_index += 1
                self.start = time.time()
            elif switch == 6:
                self.journal('到达npc')
                self.Visual('聚义平冤1', binary_process=True, threshold=0.5, tap_after_timeout=2)
                self.Visual('确定4', binary_process=True, threshold=0.4, double=True)
                if self.Visual('自动寻路中', histogram_process=True, threshold=0.7, wait_count=2, tap=False):
                    self.cause_index += 1
                    self.task_start = time.time()
                else:
                    self.close_win(2)
                    self.team_detect = True
                    self.cause_index = 1
            elif switch == 7:
                time.sleep(37)
                self.journal('任务完成')
                return 0
                # # if 聚义平冤 目标 不离开队伍
                #
                # # 不是离开队伍并创建 对应目标
                # self.leave_team()
                #
                # self.cause_index += 1

    def determine(self):
        switch = self.detect()

        if switch not in [0, 1] and self.cause_index == 0:
            return -3  # 返回主界面
        elif switch in [0, 1] and self.cause_index == 0:
            if self.team_detect and switch == 0:
                return 0  # 打开队伍
            elif self.team_detect and switch == 1:
                return 1  # 检测队伍

        if switch not in [0] and self.cause_index == 1:
            return -3  # 返回主界面
        elif switch in [0] and self.cause_index == 1:
            return 2  # 检测队伍人数

        if switch not in [0, 2, 3] and self.cause_index == 2:
            return -3  # 返回主界面
        elif switch in [0, 2, 3] and self.cause_index == 2:
            if switch == 0:
                return 3  # 打开背包
            elif switch == 2:
                return 4  # 进入活动页面
            elif switch == 3:
                return 5  # 查找活动

        if switch not in [4] and self.cause_index == 3 and time.time() - self.start > 120:
            self.team_detect = True
            self.cause_index = 1
            return -3  # 返回主界面
        elif switch in [4] and self.cause_index == 3:
            return 6  # 接取任务

        if switch not in [5] and self.cause_index == 4 and time.time() - self.task_start > 600:
            return -3  # 返回主界面
        elif switch in [5] and self.cause_index == 4:
            return 7  # 破门中

    def detect(self):
        time.sleep(2)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            if self.coord('破门中', binary_process=True, threshold=0.6):
                return 5  # 破门中
            return 0  # 主界面
        elif self.coord('队伍界面', histogram_process=True, threshold=0.7):
            return 1  # 队伍界面
        elif self.coord('物品界面', histogram_process=True, threshold=0.7):
            return 2  # 物品界面
        elif self.coord('活动界面', binary_process=True, threshold=0.4):
            return 3  # 活动界面
        elif self.coord('聚义平冤1', '确定4', binary_process=True, threshold=0.5):
            return 4  # 聚义npc

    # 创建队伍目标
    def pacify_injustice_Task_1(self):
        self.key_down_up(event.persona[self.mapp].team)
        self.Visual('创建队伍', histogram_process=True, threshold=0.6)
        self.Visual('下拉', binary_process=True, threshold=0.7)
        self.Visual('队伍界面行当玩法', histogram_process=True, threshold=0.6)
        self.Visual('聚义平冤目标', histogram_process=True, threshold=0.6)
        # self.Visual()('队伍界面自动匹配')
        self.Visual('确定', binary_process=True, threshold=0.4)
        # self.get_tap('关闭', '关闭1', process=False, threshold=0.7)
        self.key_down_up(event.persona[self.mapp].team)

    # 检测队伍人数
    def pacify_injustice_Task_2(self):
        time_start = 0
        self.key_down_up(event.persona[self.mapp].team)
        while not event.unbind[self.mapp].is_set():
            num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))
            if 5 - num >= 3:
                time.sleep(1)
                while not event.unbind[self.mapp].is_set():
                    if self.Visual('离线', histogram_process=True, threshold=0.7):
                        self.Visual('请离队伍', binary_process=True, threshold=0.4)
                    else:
                        break
                num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))
                if 5 - num >= 3:
                    if self.Visual('一键召回', binary_process=True, threshold=0.45):
                        time.sleep(15)
                        while not event.unbind[self.mapp].is_set():
                            if self.Visual('暂离', histogram_process=True, threshold=0.7):
                                self.Visual('请离队伍', binary_process=True, threshold=0.4)
                            else:
                                break
                    self.key_down_up(event.persona[self.mapp].team)
                    break
            self.Visual('普通喊话', binary_process=True, threshold=0.4)
            if time.time() - time_start > 30:
                self.key_down_up(event.persona[self.mapp].team)
                self.world_shouts(event.task_config[self.mapp].get('江湖行商喊话内容'))
                self.key_down_up(event.persona[self.mapp].team)
                time_start = time.time()


# 江湖行商
class MerchantLake(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        # 任务标志
        self.create_team = True  # 创建队伍
        self.target_location = False  # 目标地点
        self.team_satisfied = False  # 队伍条件满足
        self.task_execution = False  # 执行任务
        self.current_location = 0  # 当前位置
        self.count = 0  # 完成计数

    def initialization(self):
        pass

    def implement(self):
        while not event.unbind[self.mapp].is_set():
            switch = self.detect()

            if switch == 0:  # 判断队伍是否被创建 没有则队伍检测, 创建行商目标队伍
                if self.create_team:
                    self.journal('创建队伍')
                    self.merchants_lakes_1()
                    self.create_team = False  # 创建完成队伍 设置创建队伍标志
                elif not self.create_team:
                    if not self.task_execution:
                        if not self.target_location:  # 判断是否在行商目标地点 不在则前往
                            if self.merchants_lakes_2():
                                self.target_location = True
                        elif self.target_location:  # 在行商目标地点 判断队伍人数
                            if not self.team_satisfied:
                                self.merchants_lakes_3()
                                self.merchants_lakes_2()
                    elif self.task_execution:
                        if self.Visual('一键上缴', binary_process=True, threshold=0.5, wait_count=1):
                            self.current_location = 0
                            self.close_win(3)
                            self.task_execution = False
                            self.target_location = False
                            # self.team_satisfied = False
                            self.count += 1
                            if self.count == int(event.task_config[self.mapp].get('江湖行商次数')):
                                self.leave_team()
                                break
            elif switch == 6:
                if self.team_satisfied:
                    self.Visual('参与行商', binary_process=True, threshold=0.4)
                    self.Visual('确认发起', binary_process=True, threshold=0.4, wait_count=1)
                    self.Visual('铜钱购买', threshold=0.75, wait_count=1)
                    if self.Visual('行商等待队员', binary_process=False, threshold=0.4,
                                   search_scope=(328, 0, 992, 107), tap=False):
                        if self.Visual('行商交易', y=84, histogram_process=True, threshold=0.7, wait_count=30):
                            self.task_execution = True
                    else:
                        self.team_satisfied = False
                        self.close_win(3)
                elif not self.team_satisfied:
                    self.close_win(2)
            elif switch == 1 and self.task_execution:
                self.merchants_lakes_4()
            elif switch == 5:
                self.Visual('行商交易', y=84, histogram_process=True, threshold=0.7, wait_count=1)
            elif switch == 3 and self.task_execution:
                self.Visual('江南', binary_process=True, threshold=0.6)
                if self.current_location == 0:
                    self.Visual('商人', histogram_process=True, search_scope=(1000, 137, 1187, 262), threshold=0.6,
                                wait_count=1)
                    self.Visual('确定', binary_process=True, threshold=0.5)
                    self.current_location += 1
                elif self.current_location == 1 or self.current_location == 3:
                    self.Visual('本体位置', '本体位置1', histogram_process=True, threshold=0.6)
                    self.Visual('确定', binary_process=True, threshold=0.5)
                    self.current_location += 1
                elif self.current_location == 2:
                    self.Visual('商人', histogram_process=True, search_scope=(914, 34, 1147, 132), threshold=0.6,
                                wait_count=1)
                    self.Visual('确定', binary_process=True, threshold=0.5)
                    self.current_location += 1
                elif self.current_location == 4:
                    self.current_location = 0

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 0  # 主界面
        elif self.coord('江湖行商交易界面', histogram_process=True, threshold=0.4):
            return 1  # 江湖行商交易界面
        elif self.coord('区域', '世界', binary_process=True, threshold=0.6):
            return 3  # 地图界面
        elif self.coord('行商交易', histogram_process=True, threshold=0.7):
            return 5  # 任务交易界面
        elif self.coord('参与行商', binary_process=True, threshold=0.4):
            return 6  # 任务接取界面
        else:
            self.mouse_down_up(0, 0)
        time.sleep(1)

    # 创建江湖行商目标队伍
    def merchants_lakes_1(self):
        self.key_down_up(event.persona[self.mapp].team)
        if not self.coord('创建队伍', binary_process=True, threshold=0.6):
            self.journal('当前已有队伍')
            self.Visual('下拉', binary_process=True, threshold=0.7)
            self.mouse_move(277, 215, 277, 315, 2)
            self.Visual('无目标', binary_process=True, threshold=0.6, wait_count=1)
            self.Visual('队伍界面行当玩法', histogram_process=True, threshold=0.6)
            self.Visual('江湖行商目标', histogram_process=True, threshold=0.6)
            self.Visual('确定', binary_process=True, threshold=0.4)
            self.key_down_up(event.persona[self.mapp].team)
            return 0
        self.key_down_up(event.persona[self.mapp].team)
        self.leave_team()
        self.key_down_up(event.persona[self.mapp].team)
        self.Visual('创建队伍', histogram_process=True, threshold=0.6)
        self.Visual('下拉', binary_process=True, threshold=0.7)
        self.Visual('队伍界面行当玩法', histogram_process=True, threshold=0.6)
        self.Visual('江湖行商目标', histogram_process=True, threshold=0.6)
        # self.Visual()('队伍界面自动匹配')
        self.Visual('确定', binary_process=True, threshold=0.4)
        # self.get_tap('关闭', '关闭1', process=False, threshold=0.7)
        self.key_down_up(event.persona[self.mapp].team)

    # 前往目标地点
    def merchants_lakes_2(self):
        self.key_down_up(event.persona[self.mapp].knapsack)
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面行当', binary_process=True, threshold=0.5, wait_count=1)
        self.Visual('江湖行商', '江湖行商1', histogram_process=True, threshold=0.65)
        self.Visual('前往', canny_process=True, threshold=0.7, search_scope=(716, 525, 1334, 750))
        if self.Visual('参与行商', binary_process=True, threshold=0.4, wait_count=180, tap=False):
            return True
        return False

    # 判断队伍人数
    def merchants_lakes_3(self):
        self.key_down_up(event.persona[self.mapp].team)
        if (not self.coord('创建队伍', binary_process=True, threshold=0.6)
                and not self.coord('江湖行商任务目标', binary_process=True, threshold=0.6)):
            self.journal('设置队伍目标')
            self.Visual('下拉', binary_process=True, threshold=0.7)
            self.mouse_move(277, 215, 277, 315, 2)
            self.Visual('无目标', binary_process=True, threshold=0.6, wait_count=1)
            self.Visual('队伍界面行当玩法', histogram_process=True, threshold=0.6)
            self.Visual('江湖行商目标', histogram_process=True, threshold=0.6)
            self.Visual('确定', binary_process=True, threshold=0.4)
            # self.key_down_up(event.persona[self.mapp].team)
        time_start = 0

        while not event.unbind[self.mapp].is_set():
            num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))
            if 5 - num >= 3:
                time.sleep(1)
                while not event.unbind[self.mapp].is_set():
                    if self.Visual('离线', histogram_process=True, threshold=0.7):
                        self.Visual('请离队伍', binary_process=True, threshold=0.4)
                    else:
                        break
                num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))
                if 5 - num >= 3:
                    if self.Visual('一键召回', binary_process=True, threshold=0.45):
                        time.sleep(15)
                        while not event.unbind[self.mapp].is_set():
                            if self.Visual('暂离', histogram_process=True, threshold=0.7):
                                self.Visual('请离队伍', binary_process=True, threshold=0.4)
                            else:
                                break
                    self.key_down_up(event.persona[self.mapp].team)
                    self.team_satisfied = True
                    break
            self.Visual('普通喊话', binary_process=True, threshold=0.4)
            if time.time() - time_start > 30:
                self.key_down_up(event.persona[self.mapp].team)
                self.world_shouts(event.task_config[self.mapp].get('江湖行商喊话内容'))
                self.key_down_up(event.persona[self.mapp].team)
                time_start = time.time()

    # 交易处理
    def merchants_lakes_4(self):
        if self.coord('购买', binary_process=True, threshold=0.4, search_scope=(871, 230, 1209, 631)):
            if coord := self.coord('值', binary_process=True, threshold=0.6):
                for c in coord:
                    if event.unbind[self.mapp].is_set():
                        break
                    self.mouse_down_up(c[0], c[1])
                    self.mouse_down(1029, 485)
                    time.sleep(3)
                    self.mouse_up(1029, 485)
                    self.Visual('购买', binary_process=True, threshold=0.4, search_scope=(871, 230, 1209, 631))
            if len(coord) <= 2:
                for _ in range(3):
                    self.mouse_down(1029, 485)
                    time.sleep(3)
                    self.mouse_up(1029, 485)
                    self.Visual('购买', binary_process=True, threshold=0.4, search_scope=(871, 230, 1209, 631))
            self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
        elif self.coord('出售', binary_process=True, threshold=0.4, search_scope=(871, 377, 1209, 631)):
            for _ in range(5):
                self.Visual('出售', binary_process=True, threshold=0.4, search_scope=(871, 377, 1209, 631))
            self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)


# 采集任务
class AcquisitionTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.coord_iter = None
        self.coords = None
        self.acquisition_1_flag = True
        self.acquisition_flag = False
        self.last_coord = None
        self.line_dict = {1: '1线', 2: '2线', 3: '3线', 4: '4线', 5: '5线', 6: '6线', 7: '7线', 8: '8线', 9: '9线',
                          10: '10线', 11: '11线', 12: '12线', 13: '13线', 14: '14线', 15: '15线', 16: '16线',
                          17: '17线',
                          18: '18线', 19: '19线', 20: '20线', 21: '21线'}
        self.coords = [
            event.task_config[self.mapp].get('坐标1'),
            event.task_config[self.mapp].get('坐标2'),
            event.task_config[self.mapp].get('坐标3'),
            event.task_config[self.mapp].get('坐标4'),
            event.task_config[self.mapp].get('坐标5'),
            event.task_config[self.mapp].get('坐标6'),
            event.task_config[self.mapp].get('坐标7'),
            event.task_config[self.mapp].get('坐标8'),
            event.task_config[self.mapp].get('坐标9'),
            event.task_config[self.mapp].get('坐标10'),
            event.task_config[self.mapp].get('坐标11'),
            event.task_config[self.mapp].get('坐标12'),
            event.task_config[self.mapp].get('坐标13'),
            event.task_config[self.mapp].get('坐标14'),
            event.task_config[self.mapp].get('坐标15'),
        ]

    def initialization(self):
        pass

    def implement(self):
        line_Limit = int(event.task_config[self.mapp].get('采集线数'))
        if event.task_config[self.mapp].get('地图搜索'):
            method = '地图搜索'
        elif event.task_config[self.mapp].get('定点采集'):
            method = '定点采集'
        else:
            method = '自定义坐标采集'
        self.coords = [sublist for sublist in self.coords if any(item != '' for item in sublist)]
        self.coord_iter = iter(self.coords)
        if method == '地图搜索':
            # 指定地图
            self.acquisition_2()
            # 设置采集物
            self.acquisition_7()
        # 启动采集辅助线程
        Thread(target=self.acquisition_1).start()
        Thread(target=self.acquisition_4).start()

        while not event.unbind[self.mapp].is_set():
            # 刷新切线数 默认当前为一线开始循环
            line = 2
            if method == '地图搜索':
                # 地图就近搜素
                self.acquisition_3()
            elif method == '自定义坐标采集':
                # 地图输入坐标
                self.acquisition_5()
            elif method == '定点采集':
                pass
                # 定点采集
            # 开始采集
            while not event.unbind[self.mapp].is_set():
                target = self.acquisition_6()
                if target == 3:
                    # 结束辅助线程
                    self.acquisition_1_flag = False
                    return 0
                # 判断是否有目标
                if not target:
                    self.acquisition_flag = False
                    # 执行范围搜素

                    # 没有目标 判断是否执行切线
                    # 切线
                    if line_Limit == 1:
                        time.sleep(2)
                        break
                    else:
                        if line <= line_Limit:
                            self.acquisition_8(line)
                            # self.mouse_down_up(1235, 20)
                            # self.Visual(line_dict[line], SIFT=True, threshold=0.3, search_scope=(910, 107, 1176, 643))
                            # self.mouse_down_up(0, 0)
                            line += 1
                            time.sleep(4)
                        else:
                            self.acquisition_8(1)
                            # self.mouse_down_up(1235, 20)
                            # self.Visual(line_dict[1], histogram_process=True, threshold=0.6,
                            #             search_scope=(910, 107, 1176, 643))
                            # self.mouse_down_up(0, 0)
                            time.sleep(4)
                            break
                else:
                    self.acquisition_flag = True
                    # 有目标开始采集
                    self.Visual('采草', '伐木', '挖矿', '拾取', '搜查', histogram_process=True, process=True,
                                threshold=1,
                                search_scope=(752, 291, 1153, 565), x=-20)
                    # 判断是否有工具
                    if self.coord('采集工具', binary_process=True, threshold=0.6):
                        # 没有工具 购买 或者 停止
                        print(1)
                    time.sleep(0.2)
                    self.Visual('采草', '伐木', '挖矿', '拾取', '搜查', histogram_process=True, process=True,
                                threshold=1, x=-155,
                                y=-74,
                                search_scope=(752, 291, 1153, 565))
                    time.sleep(10)

            self.acquisition_flag = False
        # 结束辅助线程
        self.acquisition_1_flag = False

    # 采集加速
    def acquisition_1(self):
        while self.acquisition_1_flag and not event.unbind[self.mapp].is_set():
            if self.acquisition_flag:
                if self.coord('采集指针', binary_process=True, threshold=0.4, search_scope=(534, 437, 802, 583)):
                    time.sleep(0.85)
                    self.mouse_down_up(666, 475)
            else:
                time.sleep(0.5)
            time.sleep(0.01)

    # 换线
    def acquisition_8(self, index):

        self.mouse_down_up(0, 0)
        self.mouse_down_up(1235, 20, tap_after_timeout=2)
        if 0 < index <= 2:
            self.Visual(self.line_dict[index], canny_process=True, threshold=0.7,
                        search_scope=(910, 40, 1176, 643))
        if 2 < index <= 4:
            self.Visual(self.line_dict[index], canny_process=True, threshold=0.8,
                        search_scope=(910, 40, 1176, 643))
        if 5 < index <= 6:
            self.Visual(self.line_dict[index], canny_process=True, threshold=0.9,
                        search_scope=(910, 40, 1176, 643))
        if 6 < index <= 10:
            self.mouse_move(1068, 561, 1068, 431)
            self.Visual(self.line_dict[index], canny_process=True, threshold=0.8,
                        search_scope=(910, 40, 1176, 643))
        elif 10 < index <= 14:
            self.mouse_move(1068, 561, 1068, 431, 2)
            self.Visual(self.line_dict[index], canny_process=True, threshold=0.8,
                        search_scope=(910, 40, 1176, 643))
        elif 14 < index <= 17:
            self.mouse_move(1068, 561, 1068, 431, 3)
            self.Visual(self.line_dict[index], histogram_process=True, threshold=0.85,
                        search_scope=(910, 40, 1176, 643))
        elif 17 < index <= 18:
            self.mouse_move(1068, 561, 1068, 431, 3)
            self.Visual(self.line_dict[index], histogram_process=True, threshold=0.95,
                        search_scope=(910, 40, 1176, 643))
        elif 18 < index <= 21:
            self.mouse_move(1068, 561, 1068, 431, 4)
            self.Visual(self.line_dict[index], histogram_process=True, threshold=0.8,
                        search_scope=(910, 40, 1176, 643))
        self.mouse_down_up(0, 0)
        time.sleep(4)

    # 指定地图采集目标
    def acquisition_2(self):
        map_name = event.task_config[self.mapp].get('指定地图')
        self.key_down_up(event.persona[self.mapp].map)
        self.Visual('世界', binary_process=True, threshold=0.6)
        self.Visual(map_name, binary_process=True, threshold=0.6)
        self.Visual('传送点', histogram_process=True, threshold=0.6)
        # self.Visual('传送点', histogram_process=True, threshold=0.6)
        # self.key_down_up(event.persona[self.mapp].map)
        self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
        self.arrive()

    # 地图就近查找目标
    def acquisition_3(self):
        search_scope = (0, 80, 1334, 750)
        count = 0
        while not event.unbind[self.mapp].is_set():
            if self.last_coord is not None:
                search_scope = (self.last_coord[0][0] - 80 * count, self.last_coord[0][1] - 80 * count,
                                self.last_coord[0][0] + 80 * count, self.last_coord[0][1] + 80 * count)

            self.key_down_up(event.persona[self.mapp].map)
            if coord := self.Visual('采集物3', histogram_process=True, search_scope=search_scope,
                                    threshold=0.55):
                self.last_coord = coord
                self.key_down_up(event.persona[self.mapp].map)
                self.arrive()
                self.journal('到达采集目标')
                break
            else:
                count += 1

    # 地图输入坐标
    def acquisition_5(self):
        try:
            coord = next(self.coord_iter)
        except StopIteration:
            self.coord_iter = iter(self.coords)
            coord = next(self.coord_iter)

        self.key_down_up(event.persona[self.mapp].map)
        # self.Visual('世界搜索坐标展开', histogram_process=True, threshold=0.7, wait_count=1,
        #             search_scope=(0, 647, 349, 750))
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=96, search_scope=(0, 631, 414, 694))
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=300,
                    search_scope=(0, 631, 414, 694))
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=96, search_scope=(0, 631, 414, 694))
        self.input(coord[0])
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=233,
                    search_scope=(0, 631, 414, 694))
        self.input(coord[1])
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=300,
                    search_scope=(0, 631, 414, 694))
        self.key_down_up(event.persona[self.mapp].map)
        self.arrive()

    # 判断是否有目标
    def acquisition_6(self):
        method = event.task_config[self.mapp].get('采集方法')
        if not self.coord('体力耗尽', binary_process=True, threshold=0.7, search_scope=(752, 291, 1153, 565)):
            if method == '定点采集' or method == '自定义坐标采集':
                coord = self.Visual('采草', '伐木', '挖矿', '拾取', '搜查', histogram_process=True, process=True,
                                    threshold=1,
                                    search_scope=(752, 291, 1153, 565), x=-20, tap=False)
            else:
                coord = self.Visual('采草', '伐木', '挖矿', '拾取', '搜查', histogram_process=True, process=True,
                                    threshold=1,
                                    search_scope=(752, 291, 1153, 565), x=-20, tap=False)
            if coord:
                return True
            return False
        else:
            if event.task_config[self.mapp].get('自动吃鸡蛋'):
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('搜索图标', binary_process=True, threshold=0.7)
                self.Visual('输入道具名称', binary_process=True, threshold=0.6)
                for _ in range(int(event.task_config[self.mapp].get('吃鸡蛋数量'))):
                    self.input('一筐鸡蛋')
                    self.Visual('搜索图标', binary_process=True, threshold=0.7)
                    self.Visual('一筐鸡蛋', binary_process=True, threshold=0.7)
                    self.Visual('使用', binary_process=True, threshold=0.7)
                self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7)
            else:
                return 3

    # 设置采集物
    def acquisition_7(self):
        self.key_down_up(event.persona[self.mapp].map)
        self.Visual('地图目标设置', binary_process=True, threshold=0.7)
        self.mouse_move(281, 418, 181, 400)
        if event.task_config[self.mapp].get('采草'):
            self.Visual('采草目标', laplacian_process=True, threshold=0.5)
            self.mouse_move(281, 418, 181, 290)
            target = event.task_config[self.mapp].get('采草目标')
        elif event.task_config[self.mapp].get('伐木'):
            self.Visual('伐木目标', binary_process=True, threshold=0.6)
            self.mouse_move(281, 418, 181, 270)
            target = event.task_config[self.mapp].get('伐木目标')
        else:
            self.Visual('挖矿目标', binary_process=True, threshold=0.6)
            self.mouse_move(281, 418, 181, 230)
            target = event.task_config[self.mapp].get('挖矿目标')
        self.Visual(target, binary_process=True, threshold=0.6)
        self.Visual('采集目标关闭', binary_process=True, threshold=0.6)
        self.key_down_up(event.persona[self.mapp].map)

    # 关闭掉落物品
    def acquisition_4(self):
        while self.acquisition_1_flag and not event.unbind[self.mapp].is_set():
            if self.coord('关闭', '关闭1', histogram_process=True, threshold=0.7, search_scope=(871, 230, 1209, 631)):
                for _ in range(4):
                    if event.unbind[self.mapp].is_set():
                        break
                    self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7,
                                search_scope=(871, 230, 1209, 631))
            time.sleep(3)


# 扫摆摊
class SweepStalls(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.sweep_stalls_1_flag = True

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up(event.persona[self.mapp].knapsack)
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('交易', histogram_process=True, threshold=0.7)
        self.Visual('摆摊', laplacian_process=True, threshold=0.3)
        self.Visual('关注', histogram_process=True, threshold=0.7, wait_count=1)
        timeout_1 = int(event.task_config[self.mapp].get('扫摆摊延迟1')) / 1000
        timeout_2 = int(event.task_config[self.mapp].get('扫摆摊延迟2')) / 1000
        timeout_3 = int(event.task_config[self.mapp].get('扫摆摊延迟3')) / 1000

        # 定义数据和对应的优先级
        data_priorities = {
            (547, 219): event.task_config[self.mapp].get('优先级1'),
            (961, 219): event.task_config[self.mapp].get('优先级2'),
            (547, 327): event.task_config[self.mapp].get('优先级3'),
            (961, 327): event.task_config[self.mapp].get('优先级4'),
            (547, 435): event.task_config[self.mapp].get('优先级5'),
            (961, 435): event.task_config[self.mapp].get('优先级6'),
            (547, 543): event.task_config[self.mapp].get('优先级7'),
            (961, 543): event.task_config[self.mapp].get('优先级8')
        }

        data = [(547, 219), (961, 219), (547, 327), (961, 327), (547, 435), (961, 435), (547, 543), (961, 543)]

        exclude = []
        if not event.task_config[self.mapp].get('关注1'):
            exclude.append((547, 219))
        if not event.task_config[self.mapp].get('关注2'):
            exclude.append((961, 219))
        if not event.task_config[self.mapp].get('关注3'):
            exclude.append((547, 327))
        if not event.task_config[self.mapp].get('关注4'):
            exclude.append((961, 327))
        if not event.task_config[self.mapp].get('关注5'):
            exclude.append((547, 435))
        if not event.task_config[self.mapp].get('关注6'):
            exclude.append((961, 435))
        if not event.task_config[self.mapp].get('关注7'):
            exclude.append((547, 543))
        if not event.task_config[self.mapp].get('关注8'):
            exclude.append((961, 543))

        data_set = set(data)

        Thread(target=self.sweep_stalls_1).start()
        while not event.unbind[self.mapp].is_set():
            # 初始化优先级队列 清空队列
            priority_queue = []
            coord_data = []
            coords = self.coord('商品数量', canny_process=True, threshold=0.8, search_scope=(365, 175, 1153, 592))
            for coord in coords:
                if 537 < coord[0] < 557 and 209 < coord[1] < 229:
                    coord_data.append((547, 219))
                elif 951 < coord[0] < 971 and 209 < coord[1] < 229:
                    coord_data.append((961, 219))

                elif 537 < coord[0] < 557 and 317 < coord[1] < 337:
                    coord_data.append((547, 327))
                elif 951 < coord[0] < 971 and 317 < coord[1] < 337:
                    coord_data.append((961, 327))

                elif 537 < coord[0] < 557 and 425 < coord[1] < 445:
                    coord_data.append((547, 435))
                elif 951 < coord[0] < 971 and 425 < coord[1] < 445:
                    coord_data.append((961, 435))

                elif 537 < coord[0] < 557 and 533 < coord[1] < 553:
                    coord_data.append((547, 543))
                elif 951 < coord[0] < 971 and 533 < coord[1] < 553:
                    coord_data.append((961, 543))

            if not (exclude_data := list(data_set - set(coord_data + exclude))):
                continue
            for coord in exclude_data:
                heapq.heappush(priority_queue, (-data_priorities[coord], coord))  # 使用负值实现最大堆
            if coord := heapq.heappop(priority_queue)[1]:
                self.sweep_stalls_1_flag = False
                self.mouse_down_up(coord[0], coord[1], tap_after_timeout=0.1)
                while not event.unbind[self.mapp].is_set():
                    if self.Visual('银两购买', histogram_process=True, threshold=0.6,
                                   search_scope=(468, 205, 563, 243), tap_ago_timeout=0, tap_after_timeout=timeout_1,
                                   continuous_search_timeout=0, wait_count=30):
                        if self.Visual('购买', histogram_process=True, threshold=0.7, tap_ago_timeout=0,
                                       tap_after_timeout=timeout_2, continuous_search_timeout=0, wait_count=20):
                            self.Visual('确定', histogram_process=True, threshold=0.7, tap_ago_timeout=0,
                                        tap_after_timeout=timeout_3, continuous_search_timeout=0, wait_count=20)
                            continue

                    else:
                        self.sweep_stalls_1_flag = True
                        self.mouse_down_up(189, 285, tap_after_timeout=1.5)
                        break

    def sweep_stalls_1(self):
        while not event.unbind[self.mapp].is_set():
            if self.sweep_stalls_1_flag:
                self.mouse_down_up(189, 285, tap_after_timeout=0)
            time.sleep(0.65)


# 扫集市
class SweepMarketTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.sweep_market_flag = True

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up(event.persona[self.mapp].knapsack)
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('交易', histogram_process=True, threshold=0.7, tap_after_timeout=3)
        self.Visual('集市', canny_process=True, threshold=0.7)
        self.mouse_down_up(1274, 244)
        self.Visual('关注', histogram_process=True, threshold=0.7, wait_count=1)

        # 开启刷新线程
        Thread(target=self.sweep_market_1).start()
        while not event.unbind[self.mapp].is_set():
            if self.coord('集市商品', canny_process=True, threshold=0.7, search_scope=(362, 165, 753, 270)):
                self.sweep_market_flag = False
                self.mouse_down_up(550, 220, tap_after_timeout=0)
                if self.Visual('购买', histogram_process=True, threshold=0.7, tap_ago_timeout=0,
                               tap_after_timeout=0, continuous_search_timeout=0, wait_count=10):
                    self.Visual('确定', histogram_process=True, threshold=0.7, tap_ago_timeout=0,
                                tap_after_timeout=0, continuous_search_timeout=0, wait_count=10)
                self.sweep_market_flag = True

    def sweep_market_1(self):
        while not event.unbind[self.mapp].is_set():
            if self.sweep_market_flag:
                self.mouse_down_up(189, 285, tap_after_timeout=0)
            time.sleep(0.65)


#
#     def __init__(self, row, handle):
#         super().__init__(row, handle)
#         self.sweep_stalls_1_flag = True
#
#     def implement(self):
#         self.key_down_up(self.package)
#         self.get_tap('活动入口', process=False, process_laplacian=True, threshold=0.25)
#         self.get_tap('交易')
#         self.get_tap('集市', process=False, process_laplacian=True, threshold=0.3)
#         self.mouse_x_y(196, 269, 196, 669, 2)
#         self.get_tap('关注')
#         timeout_1 = int(self.LoadTaskConfig.get('timeout1')) / 1000
#         timeout_2 = int(self.LoadTaskConfig.get('timeout2')) / 1000
#         timeout_3 = int(self.LoadTaskConfig.get('timeout3')) / 1000
#         Thread(target=self.sweep_stalls_1).start()
#         while not event.unbind[self.mapp].is_set():
#             if self.get_coordinates('银票购买', threshold=0.7, search_scope=(468, 205, 563, 243)):
#                 self.sweep_stalls_1_flag = False
#                 self.tap_x_y(533, 227, timeout=timeout_1)
#                 # if self.get_tap('银票购买', threshold=0.7, search_scope=(468, 205, 563, 243), timeout=0,
#                 #                 tap_timeout=timeout_1, tap_time_out=0, wait_count=10):
#                 self.get_tap('购买', threshold=0.7, timeout=0, tap_timeout=timeout_2, tap_time_out=0,
#                              wait_count=10)
#
#                 self.get_tap('确定', threshold=0.7, timeout=0, tap_timeout=timeout_3, tap_time_out=0,
#                              wait_count=10)
#             else:
#                 self.sweep_stalls_1_flag = True
#
#     def sweep_stalls_1(self):
#         while not event.unbind[self.mapp].is_set():
#             if self.sweep_stalls_1_flag:
#                 self.tap_x_y(189, 285, timeout=0)
#                 time.sleep(0.2)
#             else:
#                 time.sleep(1)
#
#
# 帮派积分
class GangPoints(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.timer = TimerTask(self.mapp)
        self.disrupted_event = True
        self.record_time = [0.0, 0.0]
        self.record_count = [0, 0]
        self.record_event = []

    def initialization(self):
        pass

    def initialize(self, cause_index):
        self.cause_index = cause_index
        if cause_index == 1:
            # 异常记录次数
            self.record_count[1] = 0

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                self.timer.running = False
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.cause_index = 2
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if not self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    self.journal('队伍打开失败 请检查键位')
                    self.initialize(2)
                    continue
                if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                    self.Visual('确定', canny_process=True, threshold=0.7)

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.initialize(3)

                self.key_down_up(event.persona[self.mapp].team)
            elif switch == 3:
                self.journal('返回帮派')
                self.key_down_up(event.persona[self.mapp].faction)
                self.Visual('返回帮派', canny_process=True, threshold=0.7, y=-50)
                self.close_win(1)
                self.arrive()
                self.key_down_up(event.persona[self.mapp].map)
                if self.coord('当前坐标帮派驻地', canny_process=True, threshold=0.85):
                    self.initialize(4)
                self.key_down_up(event.persona[self.mapp].map)
            elif switch == 4:
                self.key_down_up(event.persona[self.mapp].map)
                self.mouse_down_up(768, 530)
                self.key_down_up(event.persona[self.mapp].map)
                self.arrive()
                if self.Visual('清扫', canny_process=True, threshold=0.65, double=True, search_scope=(721, 288, 1233, 627)):
                    self.Visual('确认', canny_process=True, threshold=0.7)
                    self.timer.running = True
                    self.journal('开始清扫')
                    # self.journal(f'预计完成时间 {self.finish_time}')
                    self.initialize(5)
            elif switch == 5:
                self.key_down_up(event.persona[self.mapp].faction)
                self.Visual('帮派领地', canny_process=True, threshold=0.7)
                self.Visual('建造气力', canny_process=True, threshold=0.7, x=114)
                if force := self.img_ocr(search_scope=(1040, 224, 1140, 256)):
                    try:
                        force = force.split('/')
                        if self.record_count[0] is not None:
                            # 记录体力
                            if self.record_count[0] == int(''.join(re.findall(r'\d+', force[0]))):
                                self.record_count[1] += 1
                                if self.record_count[1] > 3:
                                    self.initialize(1)
                                self.journal('体力消耗异常 >>> 重置任务')
                        # 记录体力
                        self.record_count[0] = int(''.join(re.findall(r'\d+', force[0])))
                        if int(''.join(re.findall(r'\d+', force[0]))) == 0 or self.timer.running_time / 20 >= 300:
                            self.timer.running = False
                            self.initialize(6)
                        seconds = int(force[0]) * 20
                        self.journal(f'剩余体力: {force[0]}  预计剩余时间: {seconds}秒')
                    except ValueError:
                        pass
                self.mouse_down_up(0, 0)
                self.close_win(2)
                self.record_time[0] = time.time()
            elif switch == 6:
                self.key_down_up(event.persona[self.mapp].faction)
                self.Visual('帮派领地', canny_process=True, threshold=0.7)
                if not self.Visual('建造气力', canny_process=True, threshold=0.7, x=114):
                    continue
                coord_1 = self.coord('共舞恢复', canny_process=True, threshold=0.7)
                coord_2 = self.coord('鼓励恢复', canny_process=True, threshold=0.7)

                try:
                    force_1 = self.img_ocr(search_scope=(coord_1[0][0] + 41, coord_1[0][1] - 15, coord_1[0][0] + 110, coord_1[0][1] + 15))
                    force_2 = self.img_ocr(search_scope=(coord_2[0][0] + 41, coord_2[0][1] - 15, coord_2[0][0] + 110, coord_2[0][1] + 15))
                    force_3 = self.img_ocr(search_scope=(1040, 224, 1140, 256))

                    force = force_1.split('/')
                    if int(''.join(re.findall(r'\d+', force[0]))) < int(''.join(re.findall(r'\d+', force[1]))):
                        self.close_win(2)
                        self.escape_stuck()
                        self.key_down_up(event.persona[self.mapp].map)
                        self.mouse_down_up(760, 528)
                        self.key_down_up(event.persona[self.mapp].map)
                        self.arrive()
                        self.Visual('共舞', canny_process=True, threshold=0.65, double=True,
                                    search_scope=(721, 288, 1233, 627))
                        self.initialize(7)
                        continue
                    force = force_2.split('/')
                    if int(''.join(re.findall(r'\d+', force[0]))) < int(''.join(re.findall(r'\d+', force[1]))):
                        self.mouse_down_up(0, 0)
                        self.Visual('排名', canny_process=True, threshold=0.7)
                        self.Visual('全服', histogram_process=True, threshold=0.7)
                        self.Visual('暮雪帮派', canny_process=True, threshold=0.7)
                        self.Visual('参观', canny_process=True, threshold=0.7)
                        self.Visual('副本挂机', canny_process=True, threshold=0.7, tap=False, wait_count=30)
                        self.initialize(8)
                        continue
                    force = force_3.split('/')
                    if int(''.join(re.findall(r'\d+', force[0]))) == 0:
                        self.initialize(0)
                        self.mouse_down_up(0, 0)
                        self.close_win(2)
                        continue
                    if self.timer.running_time / 20 >= 300:
                        self.initialize(0)
                        self.mouse_down_up(0, 0)
                        self.close_win(2)
                        continue
                    self.initialize(4)
                    self.mouse_down_up(0, 0)
                    self.close_win(2)
                except ValueError:
                    pass
                except TypeError:
                    pass
            elif switch == 7:
                if self.coord('取消', canny_process=True, threshold=0.7):
                    self.initialize(6)
                    continue
                if self.coord('奖励结算', canny_process=True, threshold=0.7):
                    self.Visual('共舞1', canny_process=True, threshold=0.7, tap_ago_timeout=0)
                    continue
            elif switch == 8:
                self.keep_activate(10)
                self.Visual('副本退出', canny_process=True, threshold=0.7)
                self.Visual('离开2', canny_process=True, threshold=0.7)
                self.Visual('副本挂机', canny_process=True, threshold=0.7, tap=False, wait_count=30)
                self.mouse_down_up(309, 595)
                self.Visual('当前频道', canny_process=True, threshold=0.7)
                self.Visual('滚动', canny_process=True, threshold=0.7)
                self.Visual('鼓劲队', canny_process=True, threshold=0.7, x=-195)
                self.Visual('申请入队', histogram_process=True, threshold=0.85)
                if not self.coord('跟随确认', canny_process=True, threshold=0.7):
                    self.key_down_up(event.persona[self.mapp].faction)
                    self.Visual('帮派领地', canny_process=True, threshold=0.7)
                    self.Visual('排名', canny_process=True, threshold=0.7)
                    self.Visual('全服', histogram_process=True, threshold=0.7)
                    self.Visual('暮雪帮派', canny_process=True, threshold=0.7)
                    self.Visual('参观', canny_process=True, threshold=0.7)
                    self.Visual('副本挂机', canny_process=True, threshold=0.7, tap=False, wait_count=30)
                    continue
                self.keep_activate(10)
                self.Visual('聊天窗口关闭', canny_process=True, threshold=0.75, search_scope=(616, 203, 827, 512))
                self.leave_team()

                self.key_down_up(event.persona[self.mapp].faction)
                self.Visual('帮派领地', canny_process=True, threshold=0.7)
                self.Visual('建造气力', canny_process=True, threshold=0.7, x=114)
                coord_2 = self.coord('鼓励恢复', canny_process=True, threshold=0.7)
                try:
                    force_2 = self.img_ocr(search_scope=(coord_2[0][0] + 41, coord_2[0][1] - 15, coord_2[0][0] + 110, coord_2[0][1] + 15))
                    force = force_2.split('/')
                    if int(''.join(re.findall(r'\d+', force[0]))) == int(''.join(re.findall(r'\d+', force[1]))):
                        self.mouse_down_up(0, 0)
                        self.close_win(2)
                        self.key_down_up(event.persona[self.mapp].faction)
                        self.Visual('返回帮派', canny_process=True, threshold=0.7, y=-50)
                        self.close_win(1)
                        self.arrive()
                        self.initialize(6)
                except ValueError:
                    pass
                except TypeError:
                    pass
                self.record_time[1] = time.time()
                self.close_win(2)

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [1]:
                return 4
            else:
                return -3

        elif self.cause_index == 5:
            if time.time() - self.record_time[0] > 60:
                return 5
            # if switch in [1]:
            #     return 5

        elif self.cause_index == 6:
            if switch in [1]:
                return 6
            else:
                return -3

        elif self.cause_index == 7:
            if switch in [1]:
                return 7

        elif self.cause_index == 8:
            if time.time() - self.record_time[1] > 30:
                return 8

    def detect(self):
        if self.coord('副本挂机', histogram_process=True, threshold=0.7):
            return 1
            # if switch == 0:
            #     self.journal('打开帮派')
            #     self.key_down_up('O')
            #     self.cause_index = 1
            # elif switch == 1:
            #     self.journal('打开帮派排名')
            #     self.Visual('帮派领地', laplacian_process=True, threshold=0.25)
            #     self.Visual('建造气力', histogram_process=True, threshold=0.7, x=114)
            #     if force := self.img_ocr(search_scope=(1040, 224, 1140, 256)):
            #         try:
            #             force = force.split('/')
            #             seconds = int(force[0]) * 20
            #             self.finish_time = (
            #                     datetime.datetime.now() + datetime.timedelta(seconds=int(seconds))).strftime(
            #                 "%H:%M:%S")
            #         except ValueError:
            #             pass
            #     self.mouse_down_up(0, 0)
            #     self.Visual('排名', histogram_process=True, threshold=0.7)
            #     self.cause_index = 2
            # elif switch == 2:
            #     self.journal('进去全服帮派')
            #     self.Visual('全服', histogram_process=True, threshold=0.7)
            #     self.mouse_down_up(585, 189)
            #     self.Visual('参观', binary_process=True, threshold=0.4)
            #     time.sleep(3)
            #     self.cause_index = 3
            # elif switch == 3:
            #     self.key_down_up(event.persona[self.mapp].map)
            #     self.mouse_down_up(768, 530)
            #     self.key_down_up(event.persona[self.mapp].map)
            #     self.arrive()
            #     if self.Visual('清扫', canny_process=True, threshold=0.65, double=True):
            #         self.journal('开始清扫')
            #         self.journal(f'预计完成时间 {self.finish_time}')
            #         self.cause_index = 4
            # elif switch == 4:
            #     self.journal('过图中')
            #     time.sleep(5)
            # elif switch == 6:
            #     self.journal('打开帮派')
            #     self.key_down_up('O')
            # elif switch == 5:
            #     self.journal('检测剩余体力')
            #     self.Visual('帮派领地', laplacian_process=True, threshold=0.25)
            #     self.Visual('建造气力', histogram_process=True, threshold=0.7, x=114)
            #     if force := self.img_ocr(search_scope=(1040, 224, 1140, 256)):
            #         try:
            #             force = force.split('/')
            #             if self.last_stamina is not None:
            #                 if self.last_stamina == int(force[0]):
            #                     self.cause_index = 0
            #                     self.journal('体力消耗异常 >>> 重置任务')
            #             self.last_stamina = int(force[0])
            #             if int(force[0]) == 0:
            #                 self.journal('清扫任务完成')
            #                 return 0
            #             seconds = int(force[0]) * 20
            #             # self.finish_time = (datetime.datetime.now() + datetime.timedelta(seconds=int(seconds))).strftime("%H:%M:%S")
            #             self.journal(f'剩余体力: {force[0]} 预计剩余时间: {seconds}秒')
            #         except ValueError:
            #             pass
            #     self.mouse_down_up(0, 0)
            #     self.close_win(2)
            #     time.sleep(60)

    # def determine(self):
    #     switch = self.detect()
    #
    #     if switch not in [0] and self.cause_index == 0:
    #         return -3  # 返回主界面
    #     elif switch in [0] and self.cause_index == 0:
    #         return 0
    #
    #     if switch not in [1] and self.cause_index == 1:
    #         return -3  # 返回主界面
    #     elif switch in [1] and self.cause_index == 1:
    #         return 1
    #
    #     if switch not in [2] and self.cause_index == 2:
    #         return -3  # 返回主界面
    #     elif switch in [2] and self.cause_index == 2:
    #         return 2
    #
    #     if switch not in [3, 4] and self.cause_index == 3:
    #         return -3  # 返回主界面
    #     elif switch in [3, 4] and self.cause_index == 3:
    #         if switch == 3:
    #             return 3
    #         elif switch == 4:
    #             return 4
    #
    #     if switch not in [0, 1, 3] and self.cause_index == 4:
    #         return -3  # 返回主界面
    #     elif switch in [0, 1, 3] and self.cause_index == 4:
    #         if switch == 0:
    #             self.cause_index = 0
    #             return -3
    #         elif switch == 1:
    #             return 5
    #         elif switch == 3:
    #             return 6
    #
    # def detect(self):
    #     time.sleep(2)
    #     if self.coord('副本挂机', canny_process=True, threshold=0.7):
    #         if self.coord('跨服模式', '副本退出', canny_process=True, threshold=0.7,
    #                       search_scope=(1150, 80, 1334, 400)):
    #             return 3  # 帮派主界面
    #         return 0  # 大世界主界面
    #     elif self.coord('帮派界面', binary_process=True, threshold=0.4):
    #         return 1  # 帮派界面
    #     elif self.coord('领地拜访界面', binary_process=True, threshold=0.4):
    #         return 2  # 领地拜访界面
    #     elif self.coord('过图标志', canny_process=True, threshold=0.7, search_scope=(953, 510, 1237, 756)):
    #         return 4  # 过图

        # self.Visual('排名', histogram_process=True, threshold=0.7)
        # self.Visual('全服', laplacian_process=True, threshold=0.25)
        # self.mouse_down_up(585, 189)
        # self.Visual('参观', laplacian_process=True, threshold=0.25)
        # self.Visual('跨服模式', laplacian_process=True, threshold=0.25, wait_count=30, tap=False)
        # while not event.unbind[self.mapp].is_set():
        #     self.key_down_up('O')
        #     self.Visual('帮派领地', laplacian_process=True, threshold=0.25)
        #     self.Visual('建造气力', histogram_process=True, threshold=0.7, x=114)
        #     if force := self.img_ocr(search_scope=(1040, 224, 1122, 256)):
        #         try:
        #             force = force.split('/')
        #             if int(force[0]) <= 10:
        #                 break
        #         except ValueError:
        #             pass
        #     self.key_down_up('O')
        #     self.mouse_down_up(1330, 740)
        #     self.key_down_up(event.persona[self.mapp].map)
        #     self.mouse_down_up(768, 530)
        #     self.mouse_down_up(768, 530)
        #     self.key_down_up(event.persona[self.mapp].map)
        #     if self.Visual('挑水', histogram_process=True, threshold=0.7, tap_ago_timeout=1, wait_count=30,
        #                    search_scope=(800, 270, 1033, 650)):
        #         time.sleep(1)
        #         self.Visual('挑水', histogram_process=True, threshold=0.7, wait_count=1,
        #                     search_scope=(800, 270, 1033, 650), tap_time_out=0)
        #         self.Visual('确认', laplacian_process=True)
        #
        #         if self.Visual('打水', histogram_process=True, threshold=0.65, tap_ago_timeout=1, wait_count=18,
        #                        search_scope=(800, 270, 1033, 650)):
        #             while not event.unbind[self.mapp].is_set():
        #                 if coord := self.coord('水滴', search_scope=(312, 580, 1014, 637), laplacian_process=True,
        #                                        threshold=0.25):
        #                     if coord[0][0] <= 666:
        #                         self.key_down_up('D', key_down_timeout=0)
        #                     elif coord[0][0] >= 666:
        #                         self.key_down_up('A', key_down_timeout=0)
        #                 else:
        #                     time.sleep(5)
        #                     break


# 坐观万象
class SittingObserving(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = []
        self.record_event = []

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('开始活动')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('活动', canny_process=True, threshold=0.7)
                self.Visual('活动界面游历', canny_process=True, threshold=0.7)
                if not self.Visual('坐观万象', canny_process=True, threshold=0.7, y=45):
                    self.cause_index = 0
                self.arrive()

                if self.coord('修炼中', histogram_process=True, threshold=0.7, search_scope=(485, 509, 769, 563)):
                    self.cause_index = 2
            elif switch == 2:
                self.keep_activate(1)
                if not self.coord('修炼中', histogram_process=True, threshold=0.7, search_scope=(485, 509, 769, 563)):
                    self.cause_index = 0

    def determine(self):
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                if switch == 1:
                    return 2
            else:
                return -3

    def detect(self):
        time.sleep(1)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1

    # def implement(self):
    #     self.key_down_up(event.persona[self.mapp].knapsack)
    #     self.Visual('活动入口', histogram_process=True, threshold=0.7)
    #     self.Visual('活动', binary_process=True, threshold=0.5)
    #     self.Visual('活动界面游历', laplacian_process=True)
    #     if self.Visual('坐观万象', histogram_process=True, threshold=0.7, y=45):
    #         self.Visual('修炼中', histogram_process=True, threshold=0.7, tap=False,
    #                     search_scope=(485, 509, 769, 563), wait_count=360)
    #         self.journal('修炼中')
    #         while not event.unbind[self.mapp].is_set():
    #             if not self.Visual('修炼中', histogram_process=True, threshold=0.7, tap=False,
    #                                search_scope=(485, 509, 769, 563)):
    #                 self.journal('修炼结束')
    #                 break
    #             time.sleep(5)
    #             self.mouse_down_up(1330, 740)
    #
    #     for _ in range(5):
    #         if event.unbind[self.mapp].is_set():
    #             break
    #         if not self.Visual('关闭', '关闭1', histogram_process=True, threshold=0.7):
    #             break


# 主线任务
class MasterStrokeTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = [0.0]
        self.record_count = []
        self.record_event = []

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                pass

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.Visual('任务1', '任务2', canny_process=True, threshold=0.7, wait_count=1)
                self.key_down_up('Y')
            elif switch == 2:
                self.Visual('指引框', canny_process=True, threshold=0.7, ignore_scope=(45, 43, 95, 92))
            elif switch == 3:
                self.Visual('对话回答', canny_process=True, threshold=0.7, search_scope=(800, 286, 1334, 750),
                            ignore_scope=(24, 8, 428, 72))
            elif switch == 4:
                self.Visual('跳过剧情', canny_process=True, threshold=0.7)
                self.Visual('确定', canny_process=True, threshold=0.7)
            elif switch == 5:
                self.Visual('装备', canny_process=True, threshold=0.7)
            elif switch == 6:
                self.close_win(1)
            elif switch == 7:
                self.Visual('交互', '交互1', '交互2', '交互3', '交互4', canny_process=True, threshold=0.75,
                            search_scope=(715, 422, 1104, 650))
                time.sleep(7)
            elif switch == 8:
                self.key_down_up('Y')
                event.persona[self.mapp].start_fight()
                time.sleep(8)
                event.persona[self.mapp].stop_fight()
            elif switch == 9:
                self.Visual('剧情选择', canny_process=True, threshold=0.7, ignore_scope=(41, 7, 346, 95))
            elif switch == 10:
                self.Visual('点此关闭', canny_process=True, threshold=0.7)
            elif switch == 11:
                self.Visual('绝世妖姬', canny_process=True, threshold=0.7)
                self.Visual('设为目标', canny_process=True, threshold=0.7)
            elif switch == 12:
                self.Visual('指引框1', canny_process=True, threshold=0.7, x=125)
            elif switch == 13:
                self.Visual('江湖1', histogram_process=True, threshold=0.6)
            elif switch == 14:
                self.Visual('活动入口1', canny_process=True, threshold=0.7)
                self.mouse_down_up(1253, 460)
                self.close_win(2)
            elif switch == 15:
                self.mouse_down_up(0, 0)
            elif switch == 16:
                self.key_down_up('space')
                self.escape_stuck()
                self.record_time[0] = time.time()
            elif switch == 17:
                self.Visual('任务1', canny_process=True, threshold=0.7)
                self.Visual('申请加入帮派', canny_process=True, threshold=0.7)
                self.Visual('一键申请', canny_process=True, threshold=0.7)
                self.close_win(2)
            elif switch == 18:
                self.Visual('个性标签', canny_process=True, threshold=0.7, y=120)
                self.Visual('这就是我', canny_process=True, threshold=0.7)
                self.Visual('聊天窗口关闭', canny_process=True, threshold=0.7)
            elif switch == 19:
                self.close_win(2)
            elif switch == 20:
                self.Visual('防守', canny_process=True, threshold=0.7)

    def determine(self):
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                pass

        elif self.cause_index == 1:
            if time.time() - self.record_time[0] > 300:
                return 16
            if switch in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
                if switch == 1:
                    return 1
                elif switch == 2:
                    return 2
                elif switch == 3:
                    return 3
                elif switch == 4:
                    return 4
                elif switch == 5:
                    return 5
                elif switch == 6:
                    return 6
                elif switch == 7:
                    return 7
                elif switch == 8:
                    return 8
                elif switch == 9:
                    return 9
                elif switch == 10:
                    return 10
                elif switch == 11:
                    return 11
                elif switch == 12:
                    return 12
                elif switch == 13:
                    return 13
                elif switch == 14:
                    return 14
                elif switch == 15:
                    return 17
                elif switch == 16:
                    return 18
                elif switch == 17:
                    return 19
                elif switch == 18:
                    return 20
            else:
                return 15

    def detect(self):
        time.sleep(1)
        if self.coord('挂机', canny_process=True, threshold=0.7):
            if self.coord('装备', canny_process=True, threshold=0.7):
                return 5
            elif self.coord('个性标签', canny_process=True, threshold=0.7):
                return 16
            elif self.coord('等级开启', canny_process=True, threshold=0.7):
                return 13
            elif self.coord('目标达成', canny_process=True, threshold=0.7):
                return 14
            elif self.coord('帮派解锁', canny_process=True, threshold=0.7):
                return 15
            elif self.coord('交互', '交互1', '交互2', '交互3', '交互4', canny_process=True, threshold=0.75,
                            search_scope=(715, 422, 1104, 650)):
                return 7  # 交互
            elif self.coord('战', canny_process=True, threshold=0.7, search_scope=(49, 147, 316, 359)):
                return 8  # 战斗
            elif self.coord('领悟新技能', canny_process=True, threshold=0.7, search_scope=(893, 216, 1268, 582)):
                return 6
            elif self.coord('指引框', canny_process=True, threshold=0.7, ignore_scope=(45, 43, 83, 92)):
                return 2
            elif self.coord('指引框1', canny_process=True, threshold=0.7):
                return 12
            return 1
        if self.coord('对话回答', canny_process=True, threshold=0.7, search_scope=(800, 286, 1334, 750),
                      ignore_scope=(24, 8, 428, 72)):
            return 3
        elif self.coord('跳过剧情', canny_process=True, threshold=0.7):
            return 4
        elif self.coord('剧情选择', canny_process=True, threshold=0.7, ignore_scope=(41, 7, 346, 95)):
            return 9
        elif self.coord('点此关闭', canny_process=True, threshold=0.7):
            return 10
        elif self.coord('绝世妖姬', canny_process=True, threshold=0.7):
            return 11  # 奇遇百态人生
        elif self.coord('点香阁', canny_process=True, threshold=0.7):
            return 17  # 点香阁
        elif self.coord('防守', canny_process=True, threshold=0.7):
            return 18
    # def implement(self):
    #     while not event.unbind[self.mapp].is_set():
    #         switch = self.detect()
    #         if switch == 0:
    #             if time.time() - self.start > 300:
    #                 self.key_down_up('space')
    #                 self.escape_stuck()
    #                 self.start = time.time()
    #             self.key_down_up('Y')
    #
    #         elif switch == 1:
    #             self.Visual('对话回答', binary_process=True, threshold=0.65)
    #         elif switch == -1:
    #             self.mouse_down_up(0, 0)
    #         elif switch == 2:
    #             self.Visual('装备', binary_process=True, threshold=0.4)
    #         elif switch == 3:
    #             self.Visual('交互', '交互1', '交互2', '交互3', '交互4', histogram_process=True, threshold=0.7)
    #             time.sleep(7)
    #             self.key_down_up('Y')
    #         elif switch == 4:
    #             self.Visual('点击关闭', binary_process=True, threshold=0.5)
    #         elif switch == 7:
    #             self.Visual('绝世妖姬', binary_process=True, threshold=0.5)
    #             self.Visual('设为目标', binary_process=True, threshold=0.5)
    #         elif switch == 8:
    #             self.Visual('脸谱商店', binary_process=True, threshold=0.5)
    #             self.Visual('购买', binary_process=True, threshold=0.5)
    #             self.close_win(2)
    #         elif switch == 9:
    #             self.Visual('天灵', binary_process=True, threshold=0.5, y=45)
    #             self.Visual('装备', binary_process=True, threshold=0.4)
    #             self.close_win(2)
    #         elif switch == 10:
    #             self.Visual('江湖目标吧', binary_process=True, threshold=0.5, x=186, y=-26)
    #             self.mouse_down_up(1253, 460)
    #             self.close_win(2)
    #         elif switch == 11:
    #             self.close_win(4)
    #         elif switch == 12:
    #             self.Visual('跑', histogram_process=True, threshold=0.7)
    #             self.flag_1 = False
    #         elif switch == 13:
    #             self.close_win(2)
    #             self.Visual('马儿', histogram_process=True, threshold=0.7)
    #         elif switch == 14:
    #             self.Visual('马儿', histogram_process=True, threshold=0.7, x=80)
    #             self.Visual('马儿', histogram_process=True, threshold=0.7, x=80)
    #         elif switch == 5:
    #             self.key_down_up('1')
    #             self.key_down_up('2')
    #             self.key_down_up('3')
    #             self.key_down_up('4')
    #             self.key_down_up('5')
    #             self.key_down_up('R')
    #             self.key_down_up('Y')
    #         elif switch == 6:
    #             self.Visual('随机选择', binary_process=True, threshold=0.7)
    #
    # def detect(self):
    #     time.sleep(2)
    #     if self.coord('挂机', histogram_process=True, threshold=0.65):
    #         if self.coord('装备', binary_process=True, threshold=0.4):
    #             return 2  # 装备
    #         elif self.coord('交互', '交互1', '交互2', '交互3', '交互4', histogram_process=True, threshold=0.7):
    #             return 3  # 交互
    #         elif self.coord('战', histogram_process=True, threshold=0.65, search_scope=(49, 147, 316, 359)):
    #             return 5  # 战斗
    #         elif self.coord('江湖目标吧', binary_process=True, threshold=0.5):
    #             return 10  # 江湖目标
    #         elif self.coord('跑', histogram_process=True, threshold=0.7) and self.flag_1:
    #             return 12  # 跑
    #         elif self.coord('经验', binary_process=True, threshold=0.5):
    #             return 13  # 经验
    #         elif self.coord('牵引', histogram_process=True, threshold=0.7):
    #             return 14  # B牵引
    #         return 0  # 主界面
    #     elif self.coord('对话回答', binary_process=True, threshold=0.65):
    #         return 1  # 对话界面
    #     elif self.coord('脸谱商店', binary_process=True, threshold=0.5):
    #         return 8  # 脸谱商店
    #     elif self.coord('点击关闭', binary_process=True, threshold=0.5):
    #         return 4  # 点击关闭
    #     elif self.coord('绝世妖姬', binary_process=True, threshold=0.5):
    #         return 7  # 奇遇百态人生
    #     elif self.coord('天灵', binary_process=True, threshold=0.5):
    #         return 9  # 脸谱界面
    #     elif self.coord('随机选择', binary_process=True, threshold=0.7):
    #         return 6  # 随机选择
    #     elif self.coord('点香阁', binary_process=True, threshold=0.5):
    #         return 11  # 点香阁
    #     else:
    #         return -1


# 邮件领取
class MailPickUpTask(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up(event.persona[self.mapp].buddy)
        self.Visual('飞鹰', canny_process=True, threshold=0.7)
        self.Visual('一键领取', binary_process=True, threshold=0.65)
        self.Visual('一键领取', binary_process=True, threshold=0.65)
        self.Visual('一键领取', binary_process=True, threshold=0.65)
        self.close_win(7)


# 行当绝活
class BusinessSkillsTask(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up(event.persona[self.mapp].knapsack)
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.mouse_move(1231, 544, 1231, 444)
        self.Visual('精进行当', binary_process=True, threshold=0.6)
        self.Visual('行当通用1', canny_process=True, threshold=0.8)
        self.Visual('前去裁衣', canny_process=True, threshold=0.6)
        self.Visual('服冠制样', '磨具打造', '鞋裤制样', '兵刃图样', canny_process=True, threshold=0.6, wait_count=180)
        self.Visual('炼制全部', canny_process=True, threshold=0.6)
        for _ in range(50):
            time.sleep(15)
            if force := self.img_ocr(search_scope=(253, 599, 360, 639)):
                try:
                    force = force.split('/')
                    if int(force[1].split('\n')[0]) < int(force[0]):
                        self.journal('剩余体力不足')
                        self.close_win(2)
                        return 0
                    self.journal(f'当前剩余体力: {force[1]}')
                except ValueError:
                    pass


# 斗地主
class BloodyBattleTask(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        while not event.unbind[self.mapp].is_set():
            if self.coord('提示', canny_process=True, threshold=0.7):
                self.Visual('提示', canny_process=True, threshold=0.7, wait_count=1)
                self.Visual('出牌', canny_process=True, threshold=0.7, wait_count=1)
            elif self.coord('要不起', canny_process=True, threshold=0.7):
                self.Visual('要不起', canny_process=True, threshold=0.6, wait_count=1)
            elif self.coord('准备2', canny_process=True, threshold=0.7):
                self.Visual('准备2', canny_process=True, threshold=0.6, wait_count=1)
            elif self.coord('叫地主', canny_process=True, threshold=0.7):
                self.Visual('叫地主', canny_process=True, threshold=0.7, wait_count=1)
            elif self.coord('不加倍', canny_process=True, threshold=0.7):
                self.Visual('不加倍', canny_process=True, threshold=0.7, wait_count=1)
            time.sleep(2)


# 限时开放
# 登峰造极
class TopPeakTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.start_way_finding = 0
        self.forward = True
        self.cause = {
            0: '到达npc',
            1: '开始任务',
            2: '开始战斗'
        }

    def initialization(self):
        pass

    def implement(self):
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if switch == 0:
                self.journal('打开活动')
                self.mouse_down_up(1205, 81)
                # self.Visual('立夏集', canny_process=True, threshold=0.6)
            elif switch == -3:
                self.mouse_down_up(0, 0)
                self.close_win(2)
            elif switch == 1:
                self.journal('查找登峰造极活动')
                self.Visual('登峰造极', canny_process=True, threshold=0.6)
                self.Visual('点击参与', canny_process=True, threshold=0.6)
            elif switch == 2:
                self.journal('前往挑战')
                self.Visual('前往挑战', canny_process=True, threshold=0.6)
                self.start_way_finding = time.time()
                self.cause_index += 1
            elif switch == 3:
                self.journal('到达npc')
                self.Visual('登峰造极1', canny_process=True, threshold=0.6)
            elif switch == 4:
                self.journal('开始挑战')
                self.Visual('开始挑战', canny_process=True, threshold=0.6)
                self.Visual('确认', canny_process=True, threshold=0.6)
                self.cause_index += 1
            elif switch == 5:
                self.journal('前进')
                self.key_down('W')
                time.sleep(2.5)
                self.key_up('W')
            elif switch == 6:
                self.journal('开始挑战')
                event.persona[self.mapp].start_fight()
                self.Visual('退出', canny_process=True, threshold=0.7, wait_count=300)
                event.persona[self.mapp].stop_fight()
                self.journal('挑战完成')
                self.forward = True
                self.start_way_finding = 0
                self.cause_index = 0
                time.sleep(8)

    def determine(self):
        switch = self.detect()

        if switch not in [0, 1, 2] and self.cause_index == 0:
            return -3  # 返回主界面
        elif switch in [0, 1, 2] and self.cause_index == 0:
            if switch == 0:
                return 0
            elif switch == 1:
                return 1
            elif switch == 2:
                return 2

        if switch not in [2, 3] and self.cause_index == 1 and time.time() - self.start_way_finding > 120:
            self.cause_index = 0
            return -3  # 返回主界面
        elif switch in [2, 3] and self.cause_index == 1:
            if switch == 3:
                return 3
            elif switch == 2:
                return 4

        if switch not in [4] and self.cause_index == 2:
            return -3  # 返回主界面
        elif switch in [4] and self.cause_index == 2:
            if switch == 4 and self.forward:
                self.forward = False
                return 5
            elif switch == 4 and not self.forward:
                return 6

    def detect(self):
        time.sleep(2)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            if self.coord('副本退出', histogram_process=True, threshold=0.7):
                return 4  # 挑战界面
            return 0  # 主界面
        elif self.coord('庆典日历', '活动时间', histogram_process=True, threshold=0.7):
            return 1  # 日历活动界面
        elif self.coord('登峰造极界面', histogram_process=True, threshold=0.7):
            return 2  # 登峰造极界面
        elif self.coord('登峰造极1', histogram_process=True, threshold=0.7):
            return 3  # 到达npc


# 邸宅农场
class MansionFarmTask(BasicTask):
    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 1

    def initialization(self):
        pass

    def initialize(self, cause_index):
        self.cause_index = cause_index

    def implement(self):
        self.cause_index = 1
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if switch == 0:
                self.Visual('副本退出', canny_process=True, threshold=0.7, search_scope=(1149, 107, 1334, 329))
                self.Visual('前往金陵', canny_process=True, threshold=0.7)
                return 0
            elif switch == -3:
                self.back_interface()
            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.cause_index = 2
            if switch == 2:
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('邸宅', canny_process=True, threshold=0.7)
                self.Visual('关闭', '关闭1', threshold=0.7, histogram_process=True, search_scope=(0, 0, 1220, 190))
                self.Visual('农场', canny_process=True, threshold=0.7)
                self.cause_index = 3
            elif switch == 3:
                if not self.coord('农场提交物品', canny_process=True, threshold=0.7) or not event.task_config[self.mapp].get('物品出售'):
                    self.Visual('农场种植', canny_process=True, threshold=0.7, x=400)
                    self.Visual('前往邸宅', canny_process=True, threshold=0.7)
                    self.Visual('农场种植', canny_process=True, threshold=0.7, x=400, wait_count=10)
                    self.cause_index = 4
                    continue
                self.Visual('农场提交物品', canny_process=True, threshold=0.7, tap_ago_timeout=2)
                self.Visual('农场提交', canny_process=True, threshold=0.7)

                self.Visual('遣退', canny_process=True, threshold=0.7, wait_count=1)
                self.Visual('铜钱', canny_process=True, threshold=0.7, search_scope=(326, 161, 1018, 581), wait_count=1)
                self.mouse_down_up(0, 0)
            elif switch == 4:
                self.Visual('养殖', canny_process=True, threshold=0.7)
                self.Visual('种植', canny_process=True, threshold=0.7)
                self.Visual('养殖', canny_process=True, threshold=0.7)
                self.Visual('种植', canny_process=True, threshold=0.7)
                self.Visual('一键收获', canny_process=True, threshold=0.7)
                self.Visual('一键播种', canny_process=True, threshold=0.7)
                self.Visual('养殖', canny_process=True, threshold=0.7, tap_after_timeout=10)
                self.Visual('一键收获', canny_process=True, threshold=0.7)
                self.Visual('一键育雏', canny_process=True, threshold=0.7)
                self.Visual('作坊', canny_process=True, threshold=0.7)
                self.cause_index = 5
            elif switch == 5:
                self.mouse_move(782, 632, 782, 332)
                if not self.coord('空闲中', canny_process=True, threshold=0.7):
                    self.cause_index = 0
                    continue
                self.Visual('空闲中', canny_process=True, threshold=0.7, tap_after_timeout=5)
                self.mouse_down_up(362, 523)
                self.mouse_down_up(461, 523)
                self.mouse_down_up(560, 523)
                self.mouse_down_up(416, 523)
                self.mouse_down_up(508, 490)
                for _ in range(5):
                    if not self.Visual('空闲队列', canny_process=True, threshold=0.7):
                        self.close_win(1)
                        break
                    for _ in range(2):
                        coords = self.coord('可制作', canny_process=True, threshold=0.7)
                        if len(coords) == 4:
                            self.mouse_move(569, 593, 569, 393)
                    coords = self.coord('可制作', canny_process=True, threshold=0.7)
                    for _ in range(3):
                        if coords is None:
                            self.mouse_move(569, 393, 569, 593)
                            coords = self.coord('可制作', canny_process=True, threshold=0.7)

                        if self.coord('0次', canny_process=True, threshold=0.7,
                                      search_scope=(coords[-1][0] + 40, coords[-1][1] - 21, coords[-1][0] + 85, coords[-1][1] + 18)):
                            coords.pop(-1)
                            continue
                        coord = coords.pop(-1)
                        self.mouse_down_up(coord[0], coord[1])
                        break

                    self.Visual('选择1', canny_process=True, threshold=0.7)
                    self.mouse_down_up(0, 0)
                self.close_win(1)

    def determine(self):
        time.sleep(1)
        if switch := self.detect() is None:
            return None

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [2]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [3]:
                return 4
            else:
                return -3

        elif self.cause_index == 5:
            if switch in [4]:
                return 5
            else:
                return -3

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('农场界面', canny_process=True, threshold=0.7):
            return 2
        elif self.coord('农田编制界面', canny_process=True, threshold=0.7):
            return 3
        elif self.coord('作坊界面', canny_process=True, threshold=0.7):
            return 4
        else:
            return None


class PlaceDingTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)

    def initialization(self):
        pass

    def implement(self):
        self.keep_activate(60)


# 帮派修炼
class FactionUnitingTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 1

    def initialization(self):
        pass

    def implement(self):
        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('返回帮派')
                self.key_down_up(event.persona[self.mapp].faction)
                self.Visual('返回帮派', canny_process=True, threshold=0.7, y=-50)
                self.close_win(1)
                self.arrive()
                self.key_down_up(event.persona[self.mapp].map)
                if self.coord('当前坐标帮派驻地', canny_process=True, threshold=0.85):
                    self.cause_index = 2
                self.key_down_up(event.persona[self.mapp].map)
            elif switch == 2:
                if self.Visual('帮派修炼打坐', '帮派修炼打坐1', canny_process=True, threshold=0.7,
                               search_scope=(737, 220, 1156, 626)):
                    self.cause_index = 3
            elif switch == 3:
                if self.Visual('恭喜获得', canny_process=True, threshold=0.7):
                    self.mouse_down_up(1334, 750)
                    self.cause_index = 0

    def determine(self):
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if event.get_time(hour=18, minute=45) < event.get_current_time() < event.get_time(hour=19, minute=15):
                self.keep_activate(60)
            elif event.get_time(hour=19, minute=15) < event.get_current_time() < event.get_time(hour=19, minute=25):
                return 2
            else:
                self.cause_index = 3

        elif self.cause_index == 3:
            if event.get_time(hour=19, minute=25) < event.get_current_time() < event.get_time(hour=19, minute=30):
                return 3
            elif event.get_current_time() > event.get_time(hour=19, minute=30):
                self.cause_index = 0

    def detect(self):
        time.sleep(1)
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1


# 山河器
class RiverTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = []
        self.record_event = []

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.cause_index = 2
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                        self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                        self.Visual('确定', canny_process=True, threshold=0.7)
                        self.key_down_up(event.persona[self.mapp].team)
                else:
                    self.journal('队伍打开失败 请检查键位')
                    return 0

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.cause_index = 3
            elif switch == 3:
                self.journal('山河器开始')
                self.key_down_up(event.persona[self.mapp].knapsack)
                self.Visual('活动入口', canny_process=True, threshold=0.7)
                self.Visual('山河器', canny_process=True, threshold=0.7)
                self.keep_activate(2)
                self.cause_index = 4
            elif switch == 4:
                if self.Visual('前往探索', canny_process=True, threshold=0.7):
                    self.journal('前往探索山河器')
                    self.arrive()
                    self.Visual('拾取', canny_process=True, threshold=0.6)
                    self.keep_activate(3)
                    continue

                if self.Visual('免费搜索', canny_process=True, threshold=0.7):
                    self.Visual('寻器', canny_process=True, threshold=0.7)
                    continue
                self.cause_index = 0

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [2]:
                return 4
            else:
                self.cause_index = 3
                return -3

    def detect(self):
        if self.coord('副本挂机', canny_process=True, threshold=0.7):
            return 1
        elif self.coord('山河器界面', canny_process=True, threshold=0.7):
            return 2


# 每日兑换
class DailyRedemption(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.cause_index = 0
        self.disrupted_event = True
        self.record_time = []
        self.record_count = []
        self.record_event = [True for _ in range(9)]

    def initialization(self):
        pass

    def implement(self):
        self.cause_index = 1

        while not event.unbind[self.mapp].is_set():
            switch = self.determine()

            if event.disrupted_event[self.mapp].is_set():
                return -1

            if switch == 0:
                return 0
            elif switch == -3:
                self.back_interface()

            if switch == 1:
                self.journal('位置检测')
                self.location_detection()
                self.cause_index = 2
            elif switch == 2:
                self.journal('队伍检测')
                self.key_down_up(event.persona[self.mapp].team)
                if not self.coord('队伍界面', histogram_process=True, threshold=0.7):
                    self.journal('队伍打开失败 请检查键位')
                    self.cause_index = 0
                    continue
                if not self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.Visual('退出队伍', histogram_process=True, threshold=0.65)
                    self.Visual('确定', canny_process=True, threshold=0.7)

                if self.coord('创建队伍', canny_process=True, threshold=0.7):
                    self.cause_index = 3

                self.key_down_up(event.persona[self.mapp].team)
            elif switch == 3:
                self.cause_index = 4
                if (self.record_event[0] and
                        (event.task_config[self.mapp].get('银票礼盒') or
                         event.task_config[self.mapp].get('吴越剑坯') or
                         event.task_config[self.mapp].get('白公鼎坯'))):
                    self.record_event[0] = False
                    self.key_down_up(event.persona[self.mapp].knapsack)
                    self.Visual('活动入口', canny_process=True, threshold=0.7)
                    self.Visual('珍宝阁', canny_process=True, threshold=0.7)
                    self.Visual('商城', canny_process=True, threshold=0.7)
                    self.Visual('搜索图标', histogram_process=True, threshold=0.85)

                    # 银票礼盒
                    if event.task_config[self.mapp].get('银票礼盒'):
                        self.journal('银票礼盒兑换')
                        self.Visual('输入道具名称', laplacian_process=True)
                        self.input('银票礼盒')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)
                        self.Visual('银票礼盒', laplacian_process=True)
                        for _ in range(30):
                            if event.unbind[self.mapp].is_set():
                                break
                            self.mouse_down_up(987, 699, tap_after_timeout=0.15)

                        self.Visual('搜索返回图标', binary_process=True, threshold=0.7)

                    # 吴越剑坯
                    if event.task_config[self.mapp].get('吴越剑坯'):
                        self.journal('吴越剑坯兑换')
                        self.Visual('输入道具名称', laplacian_process=True)
                        self.input('吴越剑坯')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)
                        self.Visual('吴越剑坯', binary_process=True, threshold=0.7)
                        for _ in range(5):
                            if event.unbind[self.mapp].is_set():
                                break
                            self.mouse_down_up(987, 699, tap_after_timeout=0.15)

                        self.Visual('搜索返回图标', binary_process=True, threshold=0.7)

                    # 白公鼎坯
                    if event.task_config[self.mapp].get('白公鼎坯'):
                        self.journal('白公鼎坯兑换')
                        self.Visual('输入道具名称', laplacian_process=True)
                        self.input('白公鼎坯')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)
                        self.Visual('白公鼎坯', binary_process=True, threshold=0.7)
                        for _ in range(2):
                            if event.unbind[self.mapp].is_set():
                                break
                            self.mouse_down_up(987, 699, tap_after_timeout=0.15)
                        self.Visual('搜索返回图标', binary_process=True, threshold=0.7)

            elif switch == 4:
                self.cause_index = 5
                if (self.record_event[1] and
                        (event.task_config[self.mapp].get('榫头卯眼') or
                         event.task_config[self.mapp].get('商会鸡蛋') or
                         event.task_config[self.mapp].get('碧铜马坯'))):
                    self.record_event[1] = False

                    self.key_down_up(event.persona[self.mapp].knapsack)
                    self.Visual('活动入口', canny_process=True, threshold=0.7)
                    self.Visual('珍宝阁', canny_process=True, threshold=0.7)
                    self.Visual('商会', canny_process=True, threshold=0.7)

                    # 榫头卯眼 鸡蛋
                    if event.task_config[self.mapp].get('榫头卯眼') or event.task_config[self.mapp].get('商会鸡蛋'):
                        self.Visual('商会宝石', histogram_process=True, threshold=0.7)
                        self.Visual('上拉', histogram_process=True, threshold=0.7)
                        # self.mouse_x_y(200,680,220,200)
                        self.Visual('江湖杂货', histogram_process=True, threshold=0.7)
                        self.mouse_move(580, 680, 600, 225)
                        time.sleep(1.8)

                        if event.task_config[self.mapp].get('商会鸡蛋'):
                            self.journal('商会鸡蛋兑换')
                            if self.Visual('鸡蛋', histogram_process=True, threshold=0.7):
                                for _ in range(5):
                                    if event.unbind[self.mapp].is_set():
                                        break
                                    self.mouse_down_up(970, 680, tap_after_timeout=0.2)

                        if event.task_config[self.mapp].get('榫头卯眼'):
                            self.journal('榫头卯眼兑换')
                            if self.Visual('榫头卯眼', histogram_process=True, threshold=0.7):
                                for _ in range(12):
                                    if event.unbind[self.mapp].is_set():
                                        break
                                    self.mouse_down_up(970, 680, tap_after_timeout=0.2)

                    # 碧铜马坯
                    if event.task_config[self.mapp].get('碧铜马坯'):
                        self.journal('碧铜马坯兑换')
                        self.Visual('商会宝石', histogram_process=True, threshold=0.7)
                        self.Visual('上拉', histogram_process=True, threshold=0.7)
                        self.mouse_move(200, 675, 200, 375)
                        self.Visual('古董材料', histogram_process=True, threshold=0.7)
                        self.Visual('碧铜马坯', binary_process=True, threshold=0.6)
                        for _ in range(3):
                            if event.unbind[self.mapp].is_set():
                                break
                            self.mouse_down_up(970, 680, tap_after_timeout=0.2)
                    self.close_win(5)
            elif switch == 5:
                self.cause_index = 6
                if self.record_event[2] and event.task_config[self.mapp].get('锦芳绣残片'):
                    self.record_event[2] = False
                    self.journal('锦芳绣残片兑换')
                    self.key_down_up(event.persona[self.mapp].knapsack)
                    self.Visual('积分', binary_process=True, threshold=0.5)
                    self.Visual('积分社交', histogram_process=True, threshold=0.7)
                    self.Visual('积分社交桃李值', x=350, histogram_process=True, threshold=0.7)
                    self.Visual('积分兑换搜索', canny_process=True, threshold=0.7)
                    self.input('锦芳绣')
                    self.Visual('搜索图标', histogram_process=True, threshold=0.7)
                    self.mouse_down_up(1000, 600, tap_after_timeout=0.2)
                    self.close_win(5)
            elif switch == 6:
                self.cause_index = 7
                if (self.record_event[3] and
                        (event.task_config[self.mapp].get('帮派铜钱捐献') or
                         event.task_config[self.mapp].get('帮派银两捐献'))):
                    self.record_event[3] = False
                    # 帮派铜钱捐献和银两捐献
                    if event.task_config[self.mapp].get('帮派铜钱捐献') or event.task_config[self.mapp].get(
                            '帮派银两捐献'):

                        self.key_down_up(event.persona[self.mapp].faction)
                        self.Visual('帮派福利', canny_process=True, threshold=0.7)
                        self.Visual('帮派捐献', canny_process=True, threshold=0.8)
                        # 帮派铜钱捐献
                        if event.task_config[self.mapp].get('帮派铜钱捐献'):
                            self.journal('帮派铜钱捐献')
                            for _ in range(3):
                                if event.unbind[self.mapp].is_set():
                                    break
                                self.Visual('捐献', search_scope=(146, 224, 450, 640), canny_process=True,
                                            threshold=0.7)
                                if self.Visual('不再提示', canny_process=True, threshold=0.7):
                                    self.Visual('确定3', canny_process=True, threshold=0.7)

                        # 帮派银两捐献
                        if event.task_config[self.mapp].get('帮派银两捐献'):
                            self.journal('帮派银两捐献')
                            for _ in range(3):
                                if event.unbind[self.mapp].is_set():
                                    break
                                self.Visual('捐献', search_scope=(513, 224, 805, 640), canny_process=True,
                                            threshold=0.7)
                                if self.Visual('不再提示', canny_process=True, threshold=0.7):
                                    self.Visual('确定3', canny_process=True, threshold=0.7)
                    self.close_win(5)
            elif switch == 7:
                self.cause_index = 8
                if self.record_event[4] and event.task_config[self.mapp].get('天幕雅苑'):
                    self.record_event[4] = False
                    self.journal('天幕雅苑')
                    self.key_down_up(event.persona[self.mapp].knapsack)
                    self.Visual('活动入口', histogram_process=True, threshold=0.7)
                    self.mouse_move(1231, 544, 1231, 444)
                    self.Visual('天幕雅苑', canny_process=True, threshold=0.6)
                    self.Visual('展馆排名', canny_process=True, threshold=0.6, tap_after_timeout=4)
                    self.Visual('进入展馆', canny_process=True, threshold=0.7, tap_after_timeout=8)

                    if self.Visual('进行中', canny_process=True, threshold=0.6, wait_count=10, tap=False):

                        if self.coord('盖章打卡进行中', canny_process=True, threshold=0.8):
                            self.journal('盖章打卡')
                            self.Visual('盖章打卡进行中', canny_process=True, threshold=0.8)
                            self.arrive()
                            self.Visual('签到盖章', '签到盖章1', canny_process=True, threshold=0.8)
                            self.Visual('盖章', canny_process=True, threshold=0.6)
                            self.close_win(2)

                        if self.coord('时装试穿进行中', canny_process=True, threshold=0.8):
                            self.journal('时装试穿')
                            self.Visual('时装试穿进行中', canny_process=True, threshold=0.8)
                            self.arrive()
                            self.Visual('试穿', '试穿1', canny_process=True, threshold=0.7)
                            self.close_win(2)
                    self.Visual('副本退出', canny_process=True, threshold=0.7)
            elif switch == 8:
                self.cause_index = 9
                if (self.record_event[5] and
                        (event.task_config[self.mapp].get('生活技能莲子') or
                         event.task_config[self.mapp].get('生活技能艾草'))):
                    self.record_event[5] = False
                    if event.task_config[self.mapp].get('生活技能莲子') or event.task_config[self.mapp].get('生活技能艾草'):

                        if event.task_config[self.mapp].get('生活技能艾草'):
                            self.journal('生活技能艾草')
                            self.key_down_up(event.persona[self.mapp].map)
                            self.Visual('世界', binary_process=True, threshold=0.6)
                            self.Visual('江南', binary_process=True, threshold=0.6)
                            self.Visual('地图目标设置', binary_process=True, threshold=0.7)
                            self.Visual('商人1', canny_process=True, threshold=0.7)
                            self.mouse_move(312, 616, 312, 416)
                            self.mouse_move(312, 616, 312, 416)
                            self.Visual('王韭菜', canny_process=True, threshold=0.7)
                            self.Visual('采集目标关闭', binary_process=True, threshold=0.6)
                            self.close_win(3)
                            self.arrive()
                            self.Visual('对话', canny_process=True, threshold=0.7)
                            self.Visual('新鲜蔬菜', canny_process=True, threshold=0.7)
                            self.Visual('输入名称搜索', canny_process=True, threshold=0.7)
                            self.input('艾草')
                            self.Visual('搜索图标', binary_process=True, threshold=0.7)
                            self.mouse_Keep_clicking(1180, 541, 2)
                            self.mouse_down_up(1011, 612)
                            self.close_win(2)

                        if event.task_config[self.mapp].get('生活技能莲子'):
                            self.journal('生活技能莲子')
                            self.key_down_up(event.persona[self.mapp].map)
                            self.Visual('世界', binary_process=True, threshold=0.6)
                            self.Visual('江南', binary_process=True, threshold=0.6)
                            self.Visual('地图目标设置', binary_process=True, threshold=0.7)
                            self.Visual('商人1', canny_process=True, threshold=0.7)
                            self.mouse_move(312, 616, 312, 116)
                            self.mouse_move(312, 616, 312, 116)
                            self.Visual('兔崽崽', canny_process=True, threshold=0.7)
                            self.Visual('采集目标关闭', binary_process=True, threshold=0.6)
                            self.close_win(3)
                            self.arrive()
                            self.Visual('对话', canny_process=True, threshold=0.7)
                            self.Visual('小虾小蟹', canny_process=True, threshold=0.7)
                            self.mouse_Keep_clicking(1042, 539, 2)
                            self.mouse_down_up(1011, 612)
                            self.close_win(2)
            elif switch == 9:
                self.cause_index = 10
                if (self.record_event[6] and
                        (event.task_config[self.mapp].get('精制面粉') or
                         event.task_config[self.mapp].get('土鸡蛋') or
                         event.task_config[self.mapp].get('鲜笋') or
                         event.task_config[self.mapp].get('猪肉') or
                         event.task_config[self.mapp].get('糯米') or
                         event.task_config[self.mapp].get('生活技能艾草'))):
                    self.record_event[6] = False

                    self.key_down_up(event.persona[self.mapp].map)
                    self.Visual('世界', binary_process=True, threshold=0.6)
                    self.Visual('中原', binary_process=True, threshold=0.6)
                    self.map_input(1272, 1724)
                    self.Visual('对话', canny_process=True, threshold=0.7)

                    self.Visual('购买食材', canny_process=True, threshold=0.7)

                    if event.task_config[self.mapp].get('精制面粉'):
                        self.journal('精制面粉')
                        self.Visual('输入名称搜索', canny_process=True, threshold=0.6)
                        self.input('精制面粉')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)

                        self.mouse_down(1180, 552)
                        time.sleep(2)
                        self.mouse_up(1180, 552)
                        self.mouse_down_up(1017, 620, tap_after_timeout=0.35)
                        self.Visual('搜索返回图标', canny_process=True, threshold=0.6)

                    if event.task_config[self.mapp].get('土鸡蛋'):
                        self.journal('土鸡蛋')
                        self.Visual('输入名称搜索', canny_process=True, threshold=0.6)
                        self.input('土鸡蛋')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)

                        self.mouse_down(1180, 552)
                        time.sleep(2)
                        self.mouse_up(1180, 552)
                        self.mouse_down_up(1017, 620, tap_after_timeout=0.35)
                        self.Visual('搜索返回图标', canny_process=True, threshold=0.6)

                    if event.task_config[self.mapp].get('鲜笋'):
                        self.journal('鲜笋')
                        self.Visual('输入名称搜索', canny_process=True, threshold=0.6)
                        self.input('鲜笋')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)

                        self.mouse_down(1180, 552)
                        time.sleep(2)
                        self.mouse_up(1180, 552)
                        self.mouse_down_up(1017, 620, tap_after_timeout=0.35)
                        self.Visual('搜索返回图标', canny_process=True, threshold=0.6)

                    if event.task_config[self.mapp].get('猪肉'):
                        self.journal('猪肉')
                        self.Visual('输入名称搜索', canny_process=True, threshold=0.6)
                        self.input('猪肉')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)

                        self.mouse_down(1180, 552)
                        time.sleep(2)
                        self.mouse_up(1180, 552)
                        self.mouse_down_up(1017, 620, tap_after_timeout=0.35)
                        self.Visual('搜索返回图标', canny_process=True, threshold=0.6)

                    if event.task_config[self.mapp].get('糯米'):
                        self.journal('糯米')
                        self.Visual('输入名称搜索', canny_process=True, threshold=0.6)
                        self.input('糯米')
                        self.Visual('搜索图标', histogram_process=True, threshold=0.85)

                        self.mouse_down(1180, 552)
                        time.sleep(2)
                        self.mouse_up(1180, 552)
                        self.mouse_down_up(1017, 620, tap_after_timeout=0.35)
                        self.Visual('搜索返回图标', canny_process=True, threshold=0.6)

                    self.close_win(2)
            elif switch == 10:
                self.cause_index = 11
                if self.record_event[7] and event.task_config[self.mapp].get('摇钱树'):
                    self.record_event[7] = False
                    self.journal('摇钱树')
                    self.key_down_up(event.persona[self.mapp].knapsack)
                    self.Visual('活动入口', histogram_process=True, threshold=0.7)
                    self.Visual('活动', binary_process=True, threshold=0.5)
                    self.Visual('活动界面帮派', laplacian_process=True)
                    self.Visual('摇钱树', histogram_process=True, threshold=0.7, y=45)
                    if self.Visual('前往1', canny_process=True, threshold=0.7):
                        self.arrive()
                        target = event.task_config[self.mapp].get('摇钱树目标')
                        if target == 0:
                            self.Visual('轻轻摇', canny_process=True, threshold=0.7, search_scope=(750, 213, 1137, 649))
                        elif target == 1:
                            self.Visual('用力摇', canny_process=True, threshold=0.7, search_scope=(750, 213, 1137, 649))
                        elif target == 2:
                            self.Visual('全力摇', canny_process=True, threshold=0.7, search_scope=(750, 213, 1137, 649))
            elif switch == 11:
                self.cause_index = 0
                if self.record_event[8] and event.task_config[self.mapp].get('商票上缴'):
                    self.record_event[8] = False
                    self.journal('商票上缴')
                    self.key_down_up(event.persona[self.mapp].faction)
                    self.Visual('帮派福利', binary_process=True, threshold=0.5)
                    self.Visual('商票上缴', binary_process=True, threshold=0.5)
                    self.Visual('商票上缴1', binary_process=True, threshold=0.5, wait_count=120)
                    self.Visual('高级商票', binary_process=True, threshold=0.5, random_tap=False)
                    self.Visual('上缴', binary_process=True, threshold=0.5)
                    self.close_win(3)

    def determine(self):
        time.sleep(1)
        switch = self.detect()

        if self.cause_index == 0:
            if switch in [1]:
                return 0
            else:
                return -3

        elif self.cause_index == 1:
            if switch in [1]:
                return 1
            else:
                return -3

        elif self.cause_index == 2:
            if switch in [1]:
                return 2
            else:
                return -3

        elif self.cause_index == 3:
            if switch in [1]:
                return 3
            else:
                return -3

        elif self.cause_index == 4:
            if switch in [1]:
                return 4
            else:
                return -3

        elif self.cause_index == 5:
            if switch in [1]:
                return 5
            else:
                return -3

        elif self.cause_index == 6:
            if switch in [1]:
                return 6
            else:
                return -3

        elif self.cause_index == 7:
            if switch in [1]:
                return 7
            else:
                return -3

        elif self.cause_index == 8:
            if switch in [1]:
                return 8
            else:
                return -3

        elif self.cause_index == 9:
            if switch in [1]:
                return 9
            else:
                return -3

        elif self.cause_index == 10:
            if switch in [1]:
                return 10
            else:
                return -3

        elif self.cause_index == 11:
            if switch in [1]:
                return 11
            else:
                return -3

    def detect(self):
        if self.coord('副本挂机', histogram_process=True, threshold=0.7):
            return 1


TASK_MAPPING = {
    # EXCLUDE
    '帮派修炼': FactionUnitingTask, '邸宅农场': MansionFarmTask, '占位任务': PlaceDingTask,

    # 导航任务
    '限时活动': None,
    # 基本任务
    '======': None,
    '课业任务': LessonTask, '帮派任务': FactionTask, '世界喊话': WorldShoutsTask, '江湖英雄榜': HeroListTask,
    '日常副本': DailyCopiesTask, '悬赏任务': BountyMissionsTask, '侠缘喊话': ChivalryShoutTask, '烽火雁门关': FiberHomeTask,
    '帮派设宴': GangBanquet, '破阵设宴': BreakingBanquet, '发布悬赏': PostBounty,
    '每日兑换': DailyRedemption, '坐观万象': SittingObserving, '狂饮豪拳': DrinkPunch,
    '茶馆说书': TeaStory, '山河器': RiverTask,
    '华山论剑': TheSword, '帮派积分': GangPoints, '每日一卦': HexagramDay,
    '江湖急送': UrgentDeliveryTask, '采集任务': AcquisitionTask,
    '江湖行商': MerchantLake, '邮件领取': MailPickUpTask,
    '聚义平冤': PacifyInjusticeTask, '行当绝活': BusinessSkillsTask,

    # 限时开放
    # '登峰造极': TopPeakTask,

    # 独立任务
    '######': None,
    '主线30级': MasterStrokeTask, '扫摆摊': SweepStalls, '扫集市': SweepMarketTask, '血战到底': BloodyBattleTask,
}

EXCLUDE_TASK_MAPPING = ['帮派修炼', '邸宅农场', '占位任务']

EXCLUDE_ADD_TASK = ['限时活动', '======', '######']

TASK_SHOW = {'课业任务': (0, 1074), '日常副本': (0, 0), '悬赏任务': (0, 0), '每日兑换': (0, 537), '限时活动': (0, 3759),
             '扫摆摊': (0, 1074), '侠缘喊话': (0, 1611), '世界喊话': (0, 1611), '华山论剑': (0, 2148), '烽火雁门关': (0, 3222),
             '江湖英雄榜': (0, 2148), '采集任务': (0, 2685), '切换角色': (0, 2148), '江湖行商': (0, 2148)}

# if __name__ == '__main__':
