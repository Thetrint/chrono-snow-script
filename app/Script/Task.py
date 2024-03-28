import logging
import os
import random
import tempfile
import time
import cv2
import numpy

from threading import Lock, Event, Thread
from sklearn.cluster import DBSCAN
from abc import abstractmethod

from app.Script.BasicFunctional import basic_functional
from app.view.Public import publicSingle
from app.view.Public import LoadTaskConfig


class EventStruct:
    def __init__(self):
        self.unbind = {}
        self.stop = {}
        self.task_config = {}
        self.stop_flag = False


event = EventStruct()


class StartTask:
    def __init__(self):
        self.initTask()
        self.index = 0
        self.mapping = {}
        self.execute = True

    def start(self, row, handle):
        mapp = self.create_mapping(row)
        event.unbind[mapp] = Event()
        event.stop[mapp] = Lock()
        event.task_config[mapp] = LoadTaskConfig.load_task_config(row)
        # 初始化
        init = Initialize(row, handle, mapp)
        # init.implement()
        Thread(target=init.global_detection).start()
        switch = SwitchRoles(row, handle, mapp)
        # if '切换角色' in event.task_config[mapp].get('执行列表'):
        for i in range(6):
            if '切换角色' in event.task_config[mapp].get('执行列表') and event.task_config[mapp].get('切角色1') and i == 1:
                if switch.switch_roles(i):
                    self.execute = True
            elif '切换角色' in event.task_config[mapp].get('执行列表') and event.task_config[mapp].get('切角色2') and i == 2:
                if switch.switch_roles(i):
                    self.execute = True
            elif '切换角色' in event.task_config[mapp].get('执行列表') and event.task_config[mapp].get('切角色3') and i == 3:
                if switch.switch_roles(i):
                    self.execute = True
            elif '切换角色' in event.task_config[mapp].get('执行列表') and event.task_config[mapp].get('切角色4') and i == 4:
                if switch.switch_roles(i):
                    self.execute = True
            elif '切换角色' in event.task_config[mapp].get('执行列表') and event.task_config[mapp].get('切角色5') and i == 5:
                if switch.switch_roles(i):
                    self.execute = True
            if self.execute:
                init.implement()
                publicSingle.set_character.emit(row)
                for task in event.task_config[mapp].get('执行列表'):
                    if not event.unbind[mapp].is_set() and task != '切换角色':
                        self.set_state(row, task)
                        Task = TASK_MAPPING[task](row, handle, mapp)
                        Task.initialization()
                        Task.implement()
                self.execute = False
        # BasicTask(row, handle, self.mapping[row]).coord('活动入口', laplacian_process=True)

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

    @staticmethod
    def set_state(row, task):
        publicSingle.state.emit([row, task])
        publicSingle.journal.emit([row, f'{task}开始'])


class BasicTask(object):

    def __init__(self, row, handle, mapp):
        self.row = row
        self.handle = handle
        self.mapp = mapp

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
        self.key_down_up('T')
        self.Visual('退出队伍', histogram_process=True, threshold=0.7)
        self.Visual('确定', laplacian_process=True)
        self.key_down_up('T')
        self.Visual('关闭', threshold=0.65, histogram_process=True)

    # 位置检测
    def location_detection(self):
        # self.log('位置检测')
        self.key_down_up('M')
        self.Visual('世界', laplacian_process=True)
        self.Visual('金陵', histogram_process=True, threshold=0.7)
        self.Visual('传送点', search_scope=(614, 236, 725, 342), laplacian_process=True)
        time.sleep(2)
        self.Visual('传送点', search_scope=(614, 236, 725, 342), laplacian_process=True)
        self.Visual('关闭', threshold=0.65, histogram_process=True)
        self.arrive()

    # 到达检测
    def arrive(self):
        for _ in range(300):
            if event.unbind[self.mapp].is_set():
                break
            if (self.coord('副本挂机', histogram_process=True, threshold=0.7)
                    and not self.coord('自动寻路中', histogram_process=True, threshold=0.6,
                                       search_scope=(531, 498, 870, 615))):
                time.sleep(3)
                if (self.coord('副本挂机', histogram_process=True, threshold=0.7)
                        and not self.coord('自动寻路中', histogram_process=True, threshold=0.6,
                                           search_scope=(531, 498, 879, 615))):
                    break
            time.sleep(2)

    # 脱离卡死
    def escape_stuck(self):
        self.key_down_up('ESC')
        self.Visual('脱离卡死', laplacian_process=True)
        self.Visual('确定', laplacian_process=True)
        self.key_down_up('ESC')

    def world_shouts(self, message):
        self.mouse_down_up(309, 595)
        self.Visual('世界频道', search_area=(0, 0, 145, 750), histogram_process=True, threshold=0.8)

        if self.Visual('输入文字', wait_count=1, binary_process=True, threshold=0.2):
            self.input(message)
            self.Visual('发送', histogram_process=True, threshold=0.7)
            self.Visual('聊天窗口关闭', histogram_process=True, threshold=0.7)

    def close_win(self, count, random_tap=False, left_tap=False, right_tap=False):
        for i in range(count):
            if event.unbind[self.mapp].is_set():
                return 0
            if left_tap:
                self.mouse_down_up(0, 0)
            if right_tap:
                self.mouse_down_up(1334, 750)
            if not self.Visual('关闭', histogram_process=True, threshold=0.65, random_tap=random_tap):
                break

    def open_entrance(self, tem_name, laplacian_process, threshold):
        count = 0
        for i in range(2):
            self.journal('尝试打开任务入口')
            self.key_down_up('B')
            self.Visual('活动入口', histogram_process=True, threshold=0.7)
            if not self.Visual(tem_name, laplacian_process=laplacian_process, threshold=threshold):
                count += 1
                self.close_win(2, left_tap=True)
            else:
                break
            if count >= 2:
                self.journal('入口打开识别')
                self.close_win(2, left_tap=True)
                return 0

    # 日志信息
    def journal(self, message):
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                publicSingle.journal.emit([self.row, message])

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
                    template_image_path = f'{template_image_path}\\' + template_image_name + '.png'
                    # 检查路径是否存在
                    if not os.path.exists(template_image_path):
                        template_image_path = os.getcwd() + '\\app\\images\\Img\\' + template_image_name + '.png'
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

    def coord(self, *args, search_scope=(0, 0, 1334, 750), histogram_process=False, laplacian_process=False,
              binary_process=False, process=False, threshold=0.23):
        """
        获取坐标
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
                    template_image_path = f'{template_image_path}\\' + template_image_name + '.png'
                    # 检查路径是否存在
                    if not os.path.exists(template_image_path):
                        template_image_path = os.getcwd() + '\\app\\images\\Img\\' + template_image_name + '.png'
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

                    if binary_process or laplacian_process or histogram_process:
                        # 图片转换
                        image_gray = img.astype(numpy.uint8)
                        template_gray = tem.astype(numpy.uint8)

                    if process:
                        template1 = template_gray - 6
                        template3 = template_gray + 6
                        # 使用 cv2.matchTemplate 函数找到匹配结果
                        result1 = cv2.matchTemplate(image_gray, template1, cv2.TM_CCOEFF_NORMED)
                        result2 = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                        result3 = cv2.matchTemplate(image_gray, template3, cv2.TM_CCOEFF_NORMED)

                        result = result3 + result2 + result1
                    else:
                        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

                    # 使用 numpy.where 找到匹配结果大于阈值的坐标
                    matched_coords = numpy.where(result >= threshold)

                    # 将匹配结果和坐标一起存储在 result_coord 中
                    result_coord.append(matched_coords)
                    logging.info((template_image_name, result[matched_coords], threshold))

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
        process = kwargs.get('process', False)
        SIFT = kwargs.get('SIFT', False)
        threshold = kwargs.get('threshold', 0.23)
        tap = kwargs.get('tap', True)
        tap_ago_timeout = kwargs.get('tap_ago_timeout', 1)
        tap_after_timeout = kwargs.get('tap_after_timeout', 1)
        random_tap = kwargs.get('random_tap', True)
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
                                         laplacian_process=laplacian_process, binary_process=binary_process,
                                         process=process, threshold=threshold)
            if coordinates:
                if random_tap:
                    coord = random.choice(coordinates)
                    if tap:
                        time.sleep(tap_ago_timeout)
                        self.mouse_down_up(coord[0] + x, coord[1] + y, tap_after_timeout=tap_after_timeout)
                    return coord
                else:
                    for coord in coordinates:
                        if tap:
                            time.sleep(tap_ago_timeout)
                            self.mouse_down_up(coord[0] + x, coord[1] + y, tap_after_timeout=tap_after_timeout)
                            time.sleep(random_tap_timeout)
                        else:
                            break
                    return coordinates
            time.sleep(continuous_search_timeout)

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

    def mouse_down_up(self, x, y, tap_after_timeout=1.0):
        """
        鼠标抬起(x,y)坐标
        :param tap_after_timeout: 点击后 延迟
        :param x: x坐标
        :param y: y坐标
        :return:
        """
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.mouse_down(self.handle, x, y)
                time.sleep(0.13)
                basic_functional.mouse_up(self.handle, x, y)
                time.sleep(tap_after_timeout)

    def mouse_move(self, start_x, start_y, end_x, end_y, step=1):
        """
        鼠标滑动
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
                    time.sleep(1.5)

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

    def key_down_up(self, key, key_down_timeout=2):
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
                time.sleep(key_down_timeout)

    def input(self, text):
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                basic_functional.input_text(self.handle, text)

    def img_ocr(self, search_scope=(0, 0, 1334, 750)):
        if not event.unbind[self.mapp].is_set():
            with event.stop[self.mapp]:
                return basic_functional.img_ocr(self.handle, search_scope)


# 初始化
class Initialize(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.close_win(3)
        self.key_down_up('ESC')
        self.Visual('端游模式', histogram_process=True, threshold=0.8)
        self.key_down_up('ESC')

    def global_detection(self):
        while not event.unbind[self.mapp].is_set():
            if self.coord('梦仔', histogram_process=True, threshold=0.8):
                self.journal('梦仔弹窗')
                self.Visual('关闭', histogram_process=True, threshold=0.65)
            time.sleep(2)


# 切换角色
class SwitchRoles(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        pass

    def switch_roles(self, index):
        self.key_down_up('ESC')
        self.Visual('切换角色', binary_process=True, threshold=0.4)
        self.Visual('确定', binary_process=True, threshold=0.4)
        if self.Visual('进入游戏', binary_process=True, threshold=0.4, wait_count=10, tap=False):
            self.mouse_down_up(1274, 66 + 108 * (index - 1))
            self.Visual('进入游戏', binary_process=True, threshold=0.4)
            return True
        return False


# 世界喊话任务
class WorldShoutsTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)

    def initialization(self):
        pass

    def implement(self):
        self.mouse_down_up(309, 595)
        self.Visual('世界频道', search_scope=(0, 0, 145, 750), laplacian_process=True)
        for count in range(int(event.task_config[self.mapp].get('世界喊话次数'))):
            if self.Visual('输入文字', wait_count=1, laplacian_process=True):
                self.input(event.task_config[self.mapp].get('世界喊话内容'))
                self.Visual('发送', laplacian_process=True)
                # self.log(f'世界喊话{count + 1}次')
                time.sleep(28)
            else:
                self.mouse_down_up(1330, 745)
                if self.Visual('输入文字', laplacian_process=True):
                    self.input(event.task_config[self.mapp].get('世界喊话内容'))
                    self.Visual('发送', laplacian_process=True)
                    # self.log(f'世界喊话{count + 1}次')
                    time.sleep(28)
        self.Visual('聊天窗口关闭', laplacian_process=True)


class FightTask(BasicTask):

    def __init__(self, row, handle, mapp, kill_list):
        super().__init__(row, handle, mapp)
        self.row = row
        self.handle = handle
        self.mapp = mapp
        self.kill_list = kill_list
        self.stop_flag = False

    def implement(self):
        pass

    def initialization(self):
        pass

    def fight(self):
        self.stop_flag = False
        while not self.stop_flag:
            for i in self.kill_list:
                if self.stop_flag:
                    break
                for _ in range(2):
                    if self.stop_flag:
                        break
                    self.key_down(i)
                    time.sleep(0.1)
                    self.key_up(i)
                    time.sleep(0.1)
                time.sleep(0.5)

    def stop(self):
        self.stop_flag = True

    def start(self):
        Thread(target=self.fight).start()


# 江湖英雄榜任务
class HeroListTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.fight = FightTask(row, handle, mapp, event.task_config[mapp].get("技能列表"))

    def initialization(self):
        pass

    def implement(self):
        self.leave_team()
        self.location_detection()
        self.key_down_up("B")
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面纷争', laplacian_process=True)
        self.Visual('江湖英雄榜', y=45, histogram_process=True, threshold=0.7)
        num = 0
        for _ in range(10):
            if event.unbind[self.mapp].is_set():
                break
            if self.Visual('江湖英雄榜次数', threshold=0.95, histogram_process=True):
                break
            self.Visual('匹配', '晋级赛', laplacian_process=True, threshold=0.25)
            self.Visual('确定', laplacian_process=True)
            if self.Visual('准备', wait_count=15, laplacian_process=True):
                self.key_down('W')
                time.sleep(6)
                self.key_up('W')
                self.fight.start()
                self.Visual('离开', wait_count=300, histogram_process=True, threshold=0.7)
                self.fight.stop()
                num += 1
                self.journal(f'江湖英雄榜完成{num}次')
                time.sleep(8)
                if num >= int(event.task_config[self.mapp].get('江湖英雄榜次数')):
                    break

        for _ in range(4):
            if event.unbind[self.mapp].is_set():
                break
            self.Visual('关闭', histogram_process=True, threshold=0.65, random_tap=False)


# 日常副本
class DailyCopiesTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)

    def initialization(self):
        pass

    def implement(self):
        self.leave_team()
        self.location_detection()
        self.daily_copies_1()
        for _ in range(30):
            # 判断队伍人数
            self.daily_copies_2()
            # 判断是否进入副本
            count_flag = self.daily_copies_3()

            if count_flag:
                self.leave_team()
                # 创建队伍
                self.daily_copies_1()
                continue

            self.daily_copies_4()
            break

    # 判断副本完成
    def daily_copies_4(self):
        start_time = 0
        count_stuck = 0
        while not event.unbind[self.mapp].is_set():
            if time.time() - start_time > 720:
                if count_stuck == 0 and event.task_config[self.mapp].get('副本自动匹配'):
                    self.key_down_up('T')
                    self.Visual('自动匹配', laplacian_process=True)
                    self.Visual('关闭', laplacian_process=True)

                # 脱离卡死
                if count_stuck != 0:
                    self.journal('副本超时执行脱离卡死')
                    self.escape_stuck()

                self.Visual('主界面任务', histogram_process=True, threshold=0.7)
                self.Visual('主界面任务1', histogram_process=True, threshold=0.7)
                if self.Visual('副本任务', histogram_process=True, threshold=0.7, search_scope=(36, 209, 102, 418),
                               x=68, y=44):
                    self.journal('激活任务等待副本完成')
                else:
                    self.journal('激活任务失败 >>> 尝试任务键激活')
                    self.key_down_up('Y')

                count_stuck += 1
                start_time = time.time()
            if count_stuck == 3:
                self.journal('脱离卡死次数上限 >>> 退队')
                self.leave_team()
                # 等待返回主界面
                self.Visual('副本挂机', wait_count=30, tap=False, histogram_process=True, threshold=0.7)
                break
            if self.Visual('跳过剧情', wait_count=1, laplacian_process=True, threshold=0.25, tap_ago_timeout=0):
                self.journal('跳过剧情')
            if self.coord('副本完成', histogram_process=True, threshold=0.6):
                time.sleep(2)
                if self.coord('副本完成', histogram_process=True, threshold=0.6):
                    self.journal('副本完成 >>> 退出副本')
                    self.Visual('副本退出', histogram_process=True, threshold=0.7, search_scope=(1149, 107, 1334, 329))
                    if self.Visual('确定', binary_process=True, threshold=0.5):
                        self.Visual('副本挂机', wait_count=30, tap=False, histogram_process=True, threshold=0.7)
                        self.journal('返回主界面')
                        break

    # 判断是否进入副本
    def daily_copies_3(self):
        count = 0
        start_time = time.time()
        while not event.unbind[self.mapp].is_set():
            self.Visual('进入副本', wait_count=2, laplacian_process=True, threshold=0.25)
            self.Visual('确认', wait_count=2, binary_process=True, threshold=0.4)
            while not event.unbind[self.mapp].is_set():
                if (self.coord('副本退出', histogram_process=True, threshold=0.65, search_scope=(1149, 107, 1334, 329)) or
                        self.Visual('跳过剧情', wait_count=1, laplacian_process=True, tap_ago_timeout=0, tap=False)):
                    time.sleep(2)
                    if (self.coord('副本退出', histogram_process=True, threshold=0.65, search_scope=(1149, 107, 1334, 329)) or
                            self.Visual('跳过剧情', wait_count=1, laplacian_process=True, tap_ago_timeout=0)):
                        self.journal('副本中 >>> 开始任务')
                        return False
                elif not self.coord('副本退出', histogram_process=True, threshold=0.65, search_scope=(1149, 107, 1334, 329)) and self.coord(
                        '副本挂机', histogram_process=True, threshold=0.7) and time.time() - start_time >= 20:
                    count += 1
                    start_time = time.time()
                    self.key_down_up('T')
                    self.journal('进入副本失败 >>> 再次尝试进入')
                    break
                if count == 4:
                    self.journal('多次进入失败 >>> 退队')
                    return True
                time.sleep(10)

    # 等待队伍人数
    def daily_copies_2(self):
        time_1 = 0
        if (event.task_config[self.mapp].get('副本人数') + 1) != 1:
            self.Visual('自动匹配', histogram_process=True, threshold=0.7)
        text = event.task_config[self.mapp].get('世界喊话内容')
        while True:
            if event.unbind[self.mapp].is_set():
                break
            num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))
            if 10 - num >= (event.task_config[self.mapp].get('副本人数') + 1):
                time.sleep(1)
                num = len(self.coord('队伍空位', threshold=0.8, histogram_process=True))
                if 10 - num >= (event.task_config[self.mapp].get('副本人数') + 1):
                    break
            self.Visual('普通喊话', laplacian_process=True, threshold=0.25)
            if time.time() - time_1 > 30:
                self.Visual('关闭', histogram_process=True, threshold=0.65)
                # (剩余: {self.number_copies}次)
                self.world_shouts(f'{text}')
                self.key_down_up("T")
                time_1 = time.time()

    # 创建日常副本
    def daily_copies_1(self):
        self.key_down_up("T")
        if not self.coord('日常', laplacian_process=True, threshold=0.3):
            self.Visual('创建队伍', laplacian_process=True)
            self.Visual('下拉', laplacian_process=True, threshold=0.35)
            self.Visual('队伍界面江湖纪事', laplacian_process=True)
            self.Visual('队伍界面自动匹配', laplacian_process=True)
            self.Visual('确定', laplacian_process=True)
            self.Visual('确定', laplacian_process=True)


# 悬赏任务
class BountyMissionsTask(DailyCopiesTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)

    def implement(self):
        self.location_detection()
        finish_flag = False
        while True:
            if event.unbind[self.mapp].is_set():
                break
            self.key_down_up('B')
            self.Visual('活动入口', histogram_process=True, threshold=0.7)
            self.Visual('活动', binary_process=True, threshold=0.5)
            if not self.Visual('活动界面悬赏', laplacian_process=True):
                continue
            if not self.coord('前往', laplacian_process=True):
                while True:
                    if event.unbind[self.mapp].is_set():
                        break
                    num = len(self.coord('前往', laplacian_process=True))
                    if num == 3 or self.coord('悬赏完成标志', histogram_process=True, threshold=0.9):
                        if num == 0 and self.coord('悬赏完成标志', histogram_process=True, threshold=0.9):
                            finish_flag = True
                        break
                    self.Visual('刷新', histogram_process=True, threshold=0.7)
                    self.Visual('悬赏界面每日悬赏', y=330, search_scope=(267 + 231 * num, 182, 1197, 558),
                                histogram_process=True, threshold=0.7)
                    self.Visual('铜钱购买', histogram_process=True, threshold=0.7)

            self.mouse_down_up(0, 0)
            self.Visual('关闭', histogram_process=True, threshold=0.65)

            if finish_flag:
                self.leave_team()
                break

            self.daily_copies_1()
            self.daily_copies_2()

            count_flag = self.daily_copies_3()

            if count_flag:
                self.leave_team()
                continue

            self.daily_copies_4()


# 侠缘喊话
class ChivalryShoutTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('H')
        self.Visual('联系人', laplacian_process=True)
        self.Visual('输入编号或者玩家昵称', laplacian_process=True)
        name = event.task_config[self.mapp].get('侠缘昵称')
        if name:
            self.input(name)
            self.Visual('搜索图标', histogram_process=True, threshold=0.7)
            self.mouse_down_up(307, 327)
            for num in range(100):
                if event.unbind[self.mapp].is_set():
                    break
                self.Visual('输入文字', laplacian_process=True)
                self.input('日出日落都浪漫,有风无风都自由')
                self.Visual('发送', laplacian_process=True)
                self.journal(f'侠缘喊话{num + 1}次')
                time.sleep(1.5)
                self.mouse_down_up(0, 0)
        self.Visual('关闭', histogram_process=True, threshold=0.65)


# 帮派任务
class FactionTask(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.open_entrance('活动', laplacian_process=True, threshold=0.2)
        # self.key_down_up('B')
        # self.Visual('活动入口', histogram_process=True, threshold=0.7)
        # self.Visual('活动', laplacian_process=True)
        self.Visual('活动界面帮派', laplacian_process=True)
        for _ in range(2):
            if event.unbind[self.mapp].is_set():
                break
            if self.Visual('帮派任务', histogram_process=True, threshold=0.7, y=45):
                if self.Visual('帮派任务1', laplacian_process=True, wait_count=180):
                    self.Visual('确定1', laplacian_process=True)
                    start_time = time.time()
                    start_time_1 = 0
                    start_num = 0
                    while not event.unbind[self.mapp].is_set():
                        if (time.time() - start_time_1 > 30 and not
                        self.coord('自动寻路中', histogram_process=True, threshold=0.7) and
                                self.coord('副本挂机', histogram_process=True, threshold=0.7)):
                            self.faction_task_2()
                            # self.Visual('主界面任务', histogram_process=True, threshold=0.7, wait_count=1)
                            # self.Visual('主界面江湖', histogram_process=True, threshold=0.7, wait_count=1)
                            # self.mouse_move(158, 239, 198, 639)
                            # self.Visual('主界面帮派任务', '主界面帮派任务1', histogram_process=True, threshold=0.7,
                            #             wait_count=1)
                            start_time_1 = time.time()
                        if time.time() - start_time > 900 and not self.coord('自动寻路中',
                                                                             histogram_process=True, threshold=0.7):
                            self.escape_stuck()
                            start_num += 1
                            start_time = time.time()
                        if start_num > 2:
                            self.journal('帮派任务失败 >>> 跳过任务')
                            break
                        if self.coord('交易界面', '购买', '摆摊购买', histogram_process=True, threshold=0.7):
                            self.journal('帮派任务购买')
                            for _ in range(5):
                                if event.unbind[self.mapp].is_set():
                                    break
                                if not self.Visual('关闭', histogram_process=True, threshold=0.7):
                                    break
                            lg = self.faction_task_1()
                            if lg == 0:
                                self.journal('帮派任务购买失败 >>> 跳过任务')
                                break
                            elif lg == 2:
                                time.sleep(30)
                                self.mouse_down_up(0, 0)
                                # 判断逻辑
                                break
                        if self.Visual('一键提交', histogram_process=True, threshold=0.7, wait_count=1):
                            time.sleep(1.5)
                            self.mouse_down_up(0, 0)
                            break

            for _ in range(3):
                if event.unbind[self.mapp].is_set():
                    break
                if not self.Visual('关闭', histogram_process=True, threshold=0.65):
                    break

    # 激活任务
    def faction_task_2(self):
        self.journal('激活帮派任务')
        self.Visual('主界面任务', histogram_process=True, threshold=0.7, wait_count=1)
        self.Visual('主界面江湖', histogram_process=True, threshold=0.7, wait_count=1)
        self.mouse_move(158, 239, 198, 639)
        self.Visual('主界面帮派任务', '主界面帮派任务1', binary_process=True, threshold=0.4,
                    wait_count=1, search_scope=(41, 211, 268, 422))

    # 帮派任务处理逻辑
    def faction_task_1(self):
        """

        :return: 0 购买失败 1 购买成功 2 帮派仓库提交
        """
        # 购买次数
        count = 0
        for i in range(1, 6):
            if event.unbind[self.mapp].is_set():
                break
            self.faction_task_2()
            # self.Visual('主界面任务', histogram_process=True, threshold=0.7, wait_count=1)
            # self.Visual('主界面江湖', histogram_process=True, threshold=0.7, wait_count=1)
            # self.mouse_move(158, 239, 198, 639)
            # self.Visual('主界面帮派任务', '主界面帮派任务1', histogram_process=True, threshold=0.7, wait_count=1)
            self.Visual('帮派仓库', y=-45, histogram_process=True, threshold=0.7, process=False)
            self.journal('尝试帮派仓库提交')
            if self.Visual('提交', laplacian_process=True):
                self.journal('帮派仓库提交成功')
                return 2
            else:
                self.Visual('关闭', histogram_process=True, threshold=0.65)
                self.Visual('主界面帮派任务', '主界面帮派任务1', histogram_process=True, threshold=0.7, wait_count=1)
                self.Visual('摆摊购买', y=-45, histogram_process=True, threshold=0.7)
                self.journal(f'尝试摆摊购买{i}次')
                if self.Visual('购买', laplacian_process=True):
                    if self.Visual('确定', laplacian_process=True):
                        count += 1
                        self.journal(f'成功购买{count}次')
                        if self.Visual('自动寻路中', histogram_process=True, threshold=0.7, wait_count=2):
                            self.journal('购买完成')
                            return 1
                else:
                    self.journal('没有商品可以购买')
                    self.Visual('关闭', histogram_process=True, threshold=0.65)

        return 0


# 帮派设宴
class GangBanquet(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)

    def initialization(self):
        pass

    def implement(self):
        flag = False
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面帮派', laplacian_process=True)
        if self.Visual('帮派设宴', y=45, histogram_process=True, threshold=0.7):
            self.Visual('前往邀约', laplacian_process=True)
            if self.Visual('邀请赴宴', wait_count=180, laplacian_process=True):
                self.Visual('确认邀约', laplacian_process=True)
                # self.update_log.emit([self.row, '开始设宴'])
                flag = True
            else:
                flag = False
                self.key_down_up('B')
                self.Visual('活动入口', histogram_process=True, threshold=0.7)
                self.Visual('活动', binary_process=True, threshold=0.5)
                self.Visual('活动界面帮派', laplacian_process=True)
                if self.Visual('帮派设宴', y=45, histogram_process=True, threshold=0.7):
                    self.Visual('前往邀约', laplacian_process=True)
                    if self.Visual('邀请赴宴', wait_count=180, laplacian_process=True):
                        self.Visual('确认邀约', laplacian_process=True)
                        # self.update_log.emit([self.row, '开始设宴'])
                        flag = True
                    else:
                        flag = False

        tap_x = 633
        tap_y = 282
        if flag:
            for col in range(2):
                if event.unbind[self.mapp].is_set():
                    break
                for row in range(4):
                    if event.unbind[self.mapp].is_set():
                        break
                    x = tap_x + 172 * row
                    y = tap_y + 182 * col
                    self.mouse_down_up(x, y)
                    if not self.coord('获取', laplacian_process=True):
                        self.Visual('一键提交', laplacian_process=True, threshold=0.3)
                        continue
                    else:
                        for _ in range(3):
                            if event.unbind[self.mapp].is_set():
                                break
                            self.Visual('获取', wait_count=1, laplacian_process=True)
                            self.gangs_set_up_feasts_1()
                            if self.Visual('获取', wait_count=1, laplacian_process=True):
                                self.gangs_set_up_feasts_2()
                                if self.Visual('获取', wait_count=1, laplacian_process=True):
                                    self.gangs_set_up_feasts_3()
                                    if not self.coord('获取', laplacian_process=True):
                                        self.Visual('一键提交', laplacian_process=True, threshold=0.3)
                                        break
                                else:
                                    self.Visual('一键提交', laplacian_process=True, threshold=0.3)
                                    break
                            else:
                                break
            self.Visual('开始设宴', laplacian_process=True)
            self.Visual('确定', laplacian_process=True)

        for _ in range(4):
            if event.unbind[self.mapp].is_set():
                break
            self.Visual('关闭', histogram_process=True, threshold=0.6)

    def gangs_set_up_feasts_1(self):
        if self.Visual('帮派仓库', y=-71, histogram_process=True, threshold=0.7):
            self.Visual('提交', laplacian_process=True)
            if self.coord('帮派仓库界面', laplacian_process=True):
                self.Visual('关闭', histogram_process=True, threshold=0.65)

    def gangs_set_up_feasts_2(self):
        if self.Visual('摆摊购买', y=-71, histogram_process=True, threshold=0.7):
            if self.Visual('购买', laplacian_process=True):
                self.Visual('确定', binary_process=True, threshold=0.4)
                self.Visual('确认', wait_count=12, binary_process=True, threshold=0.4)
                if self.coord('交易界面', laplacian_process=True):
                    self.Visual('关闭', histogram_process=True, threshold=0.65)
            else:
                self.Visual('关闭', histogram_process=True, threshold=0.65)
        else:
            self.mouse_down_up(0, 0)

    def gangs_set_up_feasts_3(self):
        if self.Visual('商城购买', y=-71, histogram_process=True, threshold=0.7):
            if self.coord('珍宝阁界面', laplacian_process=True):
                self.mouse_down_up(988, 697)
                self.Visual('关闭', histogram_process=True, threshold=0.6)
        else:
            self.mouse_down_up(0, 0)


# 破阵设宴
class BreakingBanquet(GangBanquet):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)

    def initialization(self):
        pass

    def implement(self):
        flag = False
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面帮派', laplacian_process=True)
        if self.Visual('破阵设宴', y=45, histogram_process=True, threshold=0.7):
            self.Visual('前往邀约', laplacian_process=True)
            if self.Visual('邀请赴宴', wait_count=180, laplacian_process=True):
                self.Visual('确认邀约', laplacian_process=True)
                self.journal('开始设宴')
                flag = True
            else:
                self.key_down_up('B')
                self.Visual('活动入口', histogram_process=True, threshold=0.7)
                self.Visual('活动', laplacian_process=True)
                self.Visual('活动界面帮派', laplacian_process=True)
                if self.Visual('破阵设宴', y=45, histogram_process=True, threshold=0.7):
                    self.Visual('前往邀约', laplacian_process=True)
                    if self.Visual('邀请赴宴', wait_count=180, laplacian_process=True):
                        self.Visual('确认邀约', laplacian_process=True)
                        # self.update_log.emit([self.row, '开始设宴'])
                        flag = True
                    else:
                        flag = False
        tap_x = 633
        tap_y = 282
        if flag:
            for col in range(2):
                if event.unbind[self.mapp].is_set():
                    break
                for row in range(4):
                    if event.unbind[self.mapp].is_set():
                        break
                    x = tap_x + 172 * row
                    y = tap_y + 182 * col
                    self.mouse_down_up(x, y)
                    if not self.coord('获取1', laplacian_process=True):
                        self.Visual('一键提交2', binary_process=True, threshold=0.4)
                        continue
                    else:
                        for _ in range(3):
                            if event.unbind[self.mapp].is_set():
                                break
                            self.Visual('获取1', wait_count=1, laplacian_process=True)
                            self.gangs_set_up_feasts_1()
                            if self.Visual('获取1', wait_count=1, laplacian_process=True):
                                self.gangs_set_up_feasts_2()
                                if self.Visual('获取1', wait_count=1, laplacian_process=True):
                                    self.gangs_set_up_feasts_3()
                                    if not self.coord('获取1', laplacian_process=True):
                                        self.Visual('一键提交2', binary_process=True, threshold=0.4)
                                        break
                                else:
                                    self.Visual('一键提交2', binary_process=True, threshold=0.4)
                                    break
                            else:
                                break
            self.Visual('开始设宴', laplacian_process=True)
            self.Visual('确定', laplacian_process=True)

        for _ in range(4):
            if event.unbind[self.mapp].is_set():
                break
            self.Visual('关闭', histogram_process=True, threshold=0.65)

    # def formation_set_up_banquet_1(self):
    #     if self.get_tap('帮派仓库', y=-71):
    #         self.get_tap('提交')
    #         if self.get_tap('帮派仓库界面'):
    #             self.get_tap('关闭')
    #
    # def formation_set_up_banquet_2(self):
    #     if self.get_tap('摆摊购买', y=-71):
    #         if self.get_tap('购买'):
    #             self.get_tap('确定', '确认')
    #             self.get_tap('确认', wait_count=10)
    #             if self.get_coordinates('交易界面'):
    #                 self.get_tap('关闭')
    #         else:
    #             self.get_tap('关闭')
    #     else:
    #         self.tap_x_y(0, 0)
    #
    # def formation_set_up_banquet_3(self):
    #     if self.get_tap('商城购买', y=-71):
    #         if self.get_coordinates('珍宝阁界面'):
    #             self.tap_x_y(988, 697)
    #             self.get_tap('关闭')
    #     else:
    #         self.tap_x_y(0, 0)


# 课业任务
class LessonTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.lesson_2_flag = True

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面江湖', laplacian_process=True)
        if self.Visual('濯剑', '观梦', '漱尘', '止杀', '锻心', '吟风', '含灵', '寻道', '悟禅', histogram_process=True,
                       threshold=0.7, y=45):
            self.Visual('课业', laplacian_process=True, y=210)
            self.Visual('课业1', '悟禅1', binary_process=True, threshold=0.4, wait_count=360)
            self.Visual('确定1', binary_process=True, threshold=0.4)
            for _ in range(8):
                if event.unbind[self.mapp].is_set():
                    break
                if not self.Visual('困难课业', histogram_process=True, threshold=0.7):
                    self.Visual('刷新1', laplacian_process=True, x=-55)
                    self.Visual('确定', laplacian_process=True)

                else:
                    self.Visual('关闭', histogram_process=True, threshold=0.65, wait_count=1)
                    break

            # 启动任务处理线程
            Thread(target=self.lesson_2).start()
            start_time = 0
            for _ in range(600):
                if time.time() - start_time >= 60 and self.coord('自动寻路中', histogram_process=True, threshold=0.7):
                    self.Visual('主界面任务', histogram_process=True, threshold=0.7, wait_count=1)
                    self.Visual('主界面江湖', histogram_process=True, threshold=0.7, wait_count=1)
                    self.mouse_move(158, 239, 198, 639, 2)
                    self.Visual('止杀任务', '吟风任务', '漱尘任务', '濯剑任务', '含灵任务', '寻道任务', '观梦任务',
                                histogram_process=True, threshold=0.75)
                    self.journal('课业任务检测中')
                    start_time = time.time()
                if event.unbind[self.mapp].is_set():
                    break
                if self.coord('交易界面', '购买', histogram_process=True, threshold=0.7):
                    self.journal('摆摊购买')
                    lg = self.lesson_1()
                    if lg == 0:
                        self.journal('课业任务购买失败')
                        if self.Visual('一键提交', laplacian_process=True):
                            self.mouse_down_up(0, 0)
                            if self.Visual('确定', laplacian_process=True):
                                break
                        break
                if self.coord('商城购买', histogram_process=True, threshold=0.7):
                    self.journal('商城购买')
                    self.mouse_down_up(1334, 750)
                    self.lesson_3()
                if self.Visual('一键提交', laplacian_process=True, wait_count=1):
                    self.mouse_down_up(0, 0)
                    if self.Visual('确定', laplacian_process=True):
                        break
                time.sleep(1)

        # 退出任务处理线程
        self.lesson_2_flag = False
        # 任务完成关闭多余弹窗
        for _ in range(5):
            if event.unbind[self.mapp].is_set():
                break
            self.Visual('关闭', histogram_process=True, threshold=0.65)

    # 课业分支1
    def lesson_1(self):
        """
        课业任务购买逻辑
        :return: 0 购买失败 1 购买成功
        """
        for _ in range(10):
            if self.Visual('购买', laplacian_process=True):
                self.Visual('确定', laplacian_process=True)
            if self.coord('一键提交', '自动寻路中', histogram_process=True, threshold=0.7):
                return 1
            for _ in range(3):
                if event.unbind[self.mapp].is_set():
                    break
                self.Visual('关闭', histogram_process=True, threshold=0.65, wait_count=1)
            self.Visual('主界面任务', histogram_process=True, threshold=0.7)
            self.Visual('主界面江湖', histogram_process=True, threshold=0.7)
            self.mouse_move(158, 239, 198, 639, 2)
            self.Visual('止杀任务', '吟风任务', '漱尘任务', '濯剑任务', '含灵任务', '寻道任务', '观梦任务',
                        histogram_process=True, threshold=0.7)

        return 0

    # 课业分支3
    def lesson_3(self):
        for _ in range(10):
            self.Visual('主界面任务', histogram_process=True, threshold=0.7)
            self.Visual('主界面江湖', histogram_process=True, threshold=0.7)
            self.mouse_move(158, 239, 198, 639, 2)
            self.Visual('止杀任务', '吟风任务', '漱尘任务', '濯剑任务', '含灵任务', '寻道任务', '观梦任务',
                        histogram_process=True, threshold=0.7)
            self.Visual('商城购买', histogram_process=True, threshold=0.7, y=-45)
            for _ in range(6):
                if event.unbind[self.mapp].is_set():
                    break
                self.mouse_down_up(970, 680, tap_after_timeout=0.2)
            self.Visual('关闭', histogram_process=True, threshold=0.65)
            if self.coord('一键提交', '自动寻路中', histogram_process=True, threshold=0.7):
                break

    # 课业分支2
    def lesson_2(self):
        while self.lesson_2_flag:
            # 分支1 止杀任务 漱尘任务
            if self.coord('课业任务答题', histogram_process=True, threshold=0.7):
                self.journal('课业答题任务')
                self.Visual('关闭', histogram_process=True, threshold=0.65)
            # 分支2 含灵任务 观梦任务 漱尘任务
            if self.coord('杂货商人', laplacian_process=True):
                self.Visual('铜钱购买', histogram_process=True, threshold=0.7, search_scope=(820, 517, 1242, 673))
            time.sleep(3)


# 发布悬赏
class PostBounty(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面悬赏', laplacian_process=True)

        self.Visual('发布', search_scope=(968, 558, 1269, 700), laplacian_process=True)
        self.Visual('下拉', histogram_process=True, threshold=0.8)

        if self.Visual('行商次数', histogram_process=True, threshold=0.9):
            self.Visual('发布悬赏', laplacian_process=True)
            self.Visual('确定', laplacian_process=True)
        else:
            self.Visual('关闭', histogram_process=True, threshold=0.65)

        self.Visual('发布', search_scope=(968, 558, 1269, 700), laplacian_process=True)
        self.Visual('下拉', histogram_process=True, threshold=0.8)

        if self.Visual('聚义次数', histogram_process=True, threshold=0.9):
            self.Visual('发布悬赏', laplacian_process=True)
            self.Visual('确定', laplacian_process=True)
        else:
            self.Visual('关闭', histogram_process=True, threshold=0.65)

        time.sleep(2)
        self.mouse_down_up(0, 0)
        self.Visual('关闭', histogram_process=True, threshold=0.65)
        self.mouse_down_up(0, 0)
        self.Visual('关闭', histogram_process=True, threshold=0.65)
        self.mouse_down_up(0, 0)
        self.Visual('关闭', histogram_process=True, threshold=0.65)


# 茶馆说书
class TeaStory(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面江湖', laplacian_process=True)
        if self.Visual('茶馆说书', histogram_process=True, threshold=0.7, y=45):
            self.Visual('进入茶馆', binary_process=True, threshold=0.4, wait_count=360)
            while True:
                if event.unbind[self.mapp].is_set():
                    break
                if not self.coord('甲1', '乙1', '丙1', '丁1', histogram_process=True, threshold=0.65,
                                  search_scope=(1089, 238, 1334, 750)):
                    self.Visual('甲', '乙', '丙', '丁', histogram_process=True, wait_count=1, threshold=0.65,
                                search_scope=(1089, 238, 1334, 750))
                if self.Visual('退出茶馆', histogram_process=True, threshold=0.7, wait_count=1, tap_after_timeout=0.1):
                    time.sleep(6)
                    break
                time.sleep(1)

            for _ in range(4):
                if event.unbind[self.mapp].is_set():
                    break
                self.Visual('关闭', histogram_process=True, threshold=0.65)


# 华山论剑
class TheSword(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.fight = FightTask(row, handle, mapp, event.task_config[mapp].get("技能列表"))

    def initialization(self):
        pass

    def implement(self):
        self.leave_team()
        self.location_detection()
        flag = True
        for _ in range(int(event.task_config[self.mapp].get('华山论剑次数'))):
            if flag:
                self.key_down_up('B')
                self.Visual('活动入口', histogram_process=True, threshold=0.7)
                self.Visual('活动', laplacian_process=True)
                self.Visual('活动界面纷争', laplacian_process=True)
                self.Visual('华山论剑', histogram_process=True, process=True, threshold=1, y=45)
            flag = True

            for _ in range(300):
                if event.unbind[self.mapp].is_set():
                    break
                if event.task_config[self.mapp].get('华山论剑秒退'):
                    if self.coord('准备1', histogram_process=True, threshold=0.7):
                        self.Visual('退出论剑', laplacian_process=True)
                        self.Visual('确定', laplacian_process=True)
                        time.sleep(5)
                        break
                else:
                    if self.Visual('准备1', histogram_process=True, threshold=0.7, wait_count=1):
                        self.fight.start()
                        self.Visual('离开', histogram_process=True, threshold=0.7, wait_count=180)
                        self.fight.stop()
                        time.sleep(5)
                        break
                if not self.coord('取消匹配', laplacian_process=True):
                    self.Visual('匹配1', laplacian_process=True, wait_count=1)
                self.Visual('确认', laplacian_process=True, wait_count=1)

            if self.coord('论剑界面', laplacian_process=True):
                flag = False
        for _ in range(4):
            if event.unbind[self.mapp].is_set():
                break
            self.Visual('关闭', histogram_process=True, threshold=0.65, wait_count=1)


# 每日一卦
class HexagramDay(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', laplacian_process=True)
        self.Visual('活动界面游历', laplacian_process=True)
        if self.Visual('每日一卦', histogram_process=True, threshold=0.7, y=45):
            self.close_win(1)
            if self.Visual('算命占卜', laplacian_process=True, wait_count=400):
                self.Visual('落笔', laplacian_process=True)
                self.Visual('接受', laplacian_process=True)
                self.Visual('确定', laplacian_process=True)
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
                    self.Visual('关闭', histogram_process=True, threshold=0.7, random_tap=False)
                    self.journal('今日订单已达上限')
                    break
                self.Visual('抢单', histogram_process=True, threshold=0.7)
                self.journal('接取成功')
                self.close_win(2, random_tap=False)
                implement = False

            self.Visual('主界面任务', histogram_process=True, threshold=0.7, wait_count=1)
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
                    self.Visual('关闭', histogram_process=True, threshold=0.65)
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
                  self.coord('菜品送达', binary_process=True, threshold=0.4, search_scope=(731, 200, 1234, 604))):
                for _ in range(20):
                    if event.unbind[self.mapp].is_set():
                        break
                    if self.Visual('菜品送达', binary_process=True, threshold=0.4, search_scope=(731, 200, 1234, 604),
                                   wait_count=10):
                        self.Visual('菜品送达', binary_process=True, threshold=0.4, search_scope=(731, 200, 1234, 604),
                                    wait_count=1)
                        if self.Visual('确认1', histogram_process=True, threshold=0.7, wait_count=20):
                            implement = True
                            buy = False
                            break
                    self.Visual('外卖', binary_process=True, threshold=0.4, wait_count=3,
                                search_scope=(41, 211, 268, 422))


# 精进行当豪侠-狂饮豪拳
class DrinkPunch(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.leave_team()
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('精进行当', laplacian_process=True)
        self.Visual('狂饮豪拳', histogram_process=True, threshold=0.7)
        self.Visual('前往', laplacian_process=True)
        self.Visual('简单喝', histogram_process=True, threshold=0.7, wait_count=360)
        while True:
            if event.unbind[self.mapp].is_set():
                break
            if self.Visual('离开1', laplacian_process=True, wait_count=1):
                self.Visual('确定', laplacian_process=True)
                break
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


#
#
# # 聚义平冤
# class PacifyInjustice(BasicTask):
#
#     def implement(self):
#         self.juyi_pingyuan_1()
#         for _ in range(30):
#             self.key_down_up(self.package)
#             self.get_tap('活动入口')
#             self.get_tap('活动')
#             self.get_tap('活动界面行当')
#             self.get_tap('聚义平冤', y=45)
#             self.get_tap('聚义平冤1', wait_count=360)
#             if not self.get_tap('确定1'):
#                 self.get_tap('关闭')
#                 continue
#
#             if not self.get_coordinates('自动寻路中'):
#                 self.get_tap('关闭')
#                 self.merchants_lakes_2()
#                 continue
#
#         while not event.unbind[self.mapp].is_set():
#             if not self.get_coordinates('副本完成'):
#                 time.sleep(8)
#                 if not self.get_coordinates('副本完成'):
#                     break
#
#     # 创建聚义平冤队伍目标
#     def juyi_pingyuan_1(self):
#         self.key_down_up(self.ranks)
#         self.get_tap('创建队伍')
#         self.get_tap('下拉')
#         self.get_tap('队伍界面行当玩法')
#         self.get_tap('聚义平冤目标')
#         # self.get_tap('队伍界面自动匹配')
#         self.get_tap('确定')
#         # self.get_tap('关闭', process=False, threshold=0.7)
#         self.key_down_up(self.ranks)
#
#     # 等待队伍人数
#     def merchants_lakes_2(self):
#         time_1 = 0
#         self.key_down_up(self.ranks)
#         while True:
#             if event.unbind[self.mapp].is_set():
#                 break
#             num = len(self.get_coordinates('队伍空位'))
#             if 5 - num >= 3:
#                 time.sleep(1)
#                 num = len(self.get_coordinates('队伍空位'))
#                 if 5 - num >= 3:
#                     if self.get_tap('一键召回'):
#                         time.sleep(20)
#                         while not event.unbind[self.mapp].is_set():
#                             if self.get_tap('暂离'):
#                                 self.get_tap('请离队伍')
#                             else:
#                                 break
#                     self.key_down_up(self.ranks)
#                     break
#             self.get_tap('普通喊话')
#             if time.time() - time_1 > 30:
#                 self.key_down_up(self.ranks)
#                 self.world_shouts_1('51来不跳车')
#                 self.key_down_up(self.ranks)
#                 time_1 = time.time()
#
#
# # 江湖行商
# class MerchantLake(BasicTask):
#
#     def __init__(self, row, handle):
#         super().__init__(row, handle)
#         self.merchants_lakes_3_flag = None
#         self.merchants_lakes_3_1_flag = None
#         self.merchants_lakes_flag = None
#
#     def implement(self):
#         # self.leave_party()
#         self.merchants_lakes_1()
#         Thread(target=self.merchants_lakes_3).start()
#         for _ in range(300):
#             if event.unbind[self.mapp].is_set():
#                 break
#             if self.merchants_lakes_flag:
#                 self.leave_party()
#                 break
#             self.key_down_up(self.package)
#             self.get_tap('活动入口')
#             self.get_tap('活动')
#             self.get_tap('活动界面行当')
#             self.get_tap('江湖行商', '江湖行商1', threshold=0.85)
#             self.get_tap('前往', search_scope=(716, 525, 1334, 750))
#             self.merchants_lakes_3_flag = True
#             self.get_tap('参与行商', wait_count=180)
#             self.get_tap('确认发起', wait_count=1)
#             self.get_tap('铜钱购买', threshold=0.75)
#             if self.get_coordinates('放弃', process=False, threshold=0.7):
#                 self.merchants_lakes_3_flag = False
#                 if not self.get_tap('行商交易', y=84, wait_count=30):
#                     self.get_tap('关闭')
#                     self.get_tap('关闭')
#                     self.get_tap('关闭')
#                     continue
#                 time.sleep(0.9)
#                 if coord := self.get_coordinates('值'):
#                     for c in coord:
#                         if event.unbind[self.mapp].is_set():
#                             break
#                         self.tap_x_y(c[0], c[1])
#                         self.mouse_down(1029, 485)
#                         time.sleep(3)
#                         self.mouse_up(1029, 485)
#                         self.get_tap('购买')
#                 if (d := self.get_coordinates('银饼', threshold=0.5, search_scope=(0, 0, 666, 666))) and len(
#                         coord) < 3:
#                     for a in d:
#                         if event.unbind[self.mapp].is_set():
#                             break
#                         self.tap_x_y(a[0], a[1])
#                         self.mouse_down(1029, 485)
#                         time.sleep(3)
#                         self.mouse_up(1029, 485)
#                         self.get_tap('购买')
#
#                 for _ in range(4):
#                     if event.unbind[self.mapp].is_set():
#                         break
#                     if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                         break
#                 self.get_tap('关闭')
#
#                 for _ in range(10):
#                     if event.unbind[self.mapp].is_set():
#                         break
#                     self.get_tap('江南')
#                     self.get_tap('商人', search_scope=(1000, 137, 1187, 262), threshold=0.6)
#                     self.get_tap('确定')
#                     self.get_tap('行商交易', y=84, wait_count=600)
#                     time.sleep(0.9)
#                     if c := self.get_coordinates('出售', search_scope=(811, 534, 1334, 750)):
#                         for _ in range(5):
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             self.tap_x_y(c[0][0], c[0][1])
#
#                     for _ in range(4):
#                         if event.unbind[self.mapp].is_set():
#                             break
#                         if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                             break
#                     self.get_tap('关闭')
#
#                     if self.get_tap('一键上缴'):
#                         for _ in range(4):
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                                 break
#                         self.get_tap('关闭')
#                         break
#
#                     self.get_tap('江南')
#                     self.get_tap('本体位置', '本体位置1', threshold=0.6)
#                     self.get_tap('确定')
#                     self.get_tap('行商交易', y=84, wait_count=30)
#                     time.sleep(0.9)
#                     if coord := self.get_coordinates('值'):
#                         for c in coord:
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             self.tap_x_y(c[0], c[1])
#                             self.mouse_down(1029, 485)
#                             time.sleep(3)
#                             self.mouse_up(1029, 485)
#                             self.get_tap('购买')
#                     if (d := self.get_coordinates('银饼', threshold=0.5, search_scope=(0, 0, 666, 666))) and len(
#                             coord) < 3:
#                         for a in d:
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             self.tap_x_y(a[0], a[1])
#                             self.mouse_down(1029, 485)
#                             time.sleep(3)
#                             self.mouse_up(1029, 485)
#                             self.get_tap('购买')
#
#                     for _ in range(4):
#                         if event.unbind[self.mapp].is_set():
#                             break
#                         if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                             break
#                     self.get_tap('关闭')
#
#                     self.get_tap('江南')
#                     self.get_tap('商人', search_scope=(914, 34, 1147, 132), threshold=0.6)
#                     self.get_tap('确定')
#                     self.get_tap('行商交易', y=84, wait_count=600)
#                     time.sleep(0.9)
#                     if c := self.get_coordinates('出售', search_scope=(811, 534, 1334, 750)):
#                         for _ in range(5):
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             self.tap_x_y(c[0][0], c[0][1])
#
#                     for _ in range(4):
#                         if event.unbind[self.mapp].is_set():
#                             break
#                         if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                             break
#                     self.get_tap('关闭')
#
#                     if self.get_tap('一键上缴'):
#                         for _ in range(4):
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                                 break
#                         self.get_tap('关闭')
#                         break
#                     self.get_tap('江南')
#                     self.get_tap('本体位置', '本体位置1', threshold=0.6)
#                     for _ in range(4):
#                         if event.unbind[self.mapp].is_set():
#                             break
#                         if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                             break
#                     self.get_tap('关闭')
#                     time.sleep(0.9)
#                     if coord := self.get_coordinates('值'):
#                         for c in coord:
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             self.tap_x_y(c[0], c[1])
#                             self.mouse_down(1029, 485)
#                             time.sleep(3)
#                             self.mouse_up(1029, 485)
#                             self.get_tap('购买')
#                     if (d := self.get_coordinates('银饼', threshold=0.5, search_scope=(0, 0, 666, 666))) and len(
#                             coord) < 3:
#                         for a in d:
#                             if event.unbind[self.mapp].is_set():
#                                 break
#                             self.tap_x_y(a[0], a[1])
#                             self.mouse_down(1029, 485)
#                             time.sleep(3)
#                             self.mouse_up(1029, 485)
#                             self.get_tap('购买')
#                     for _ in range(4):
#                         if event.unbind[self.mapp].is_set():
#                             break
#                         if not self.get_tap('关闭', search_scope=(871, 230, 1209, 631)):
#                             break
#                     self.get_tap('关闭')
#
#             else:
#                 self.merchants_lakes_3_flag = False
#                 self.get_tap('关闭')
#                 self.get_tap('关闭')
#                 if not self.merchants_lakes_flag:
#                     self.merchants_lakes_2()
#
#         self.merchants_lakes_3_1_flag = True
#
#     # 创建江湖行商目标队伍
#     def merchants_lakes_1(self):
#         self.key_down_up(self.ranks)
#         self.get_tap('创建队伍', process=False, threshold=0.8)
#         self.get_tap('下拉', process=False, threshold=0.8)
#         self.get_tap('队伍界面行当玩法', process=False, threshold=0.8)
#         self.get_tap('江湖行商目标', process=False, threshold=0.8)
#         # self.get_tap('队伍界面自动匹配')
#         self.get_tap('确定', process=False, threshold=0.8)
#         # self.get_tap('关闭', process=False, threshold=0.7)
#         self.key_down_up(self.ranks)
#
#     # 等待队伍人数
#     def merchants_lakes_2(self):
#         time_1 = 0
#         self.key_down_up(self.ranks)
#         while True:
#             if event.unbind[self.mapp].is_set():
#                 break
#             num = len(self.get_coordinates('队伍空位'))
#             if 5 - num >= 3:
#                 time.sleep(1)
#                 num = len(self.get_coordinates('队伍空位'))
#                 if 5 - num >= 3:
#                     if self.get_tap('一键召回'):
#                         time.sleep(20)
#                         while not event.unbind[self.mapp].is_set():
#                             if self.get_tap('暂离'):
#                                 self.get_tap('请离队伍')
#                             else:
#                                 break
#                     self.key_down_up(self.ranks)
#                     break
#             self.get_tap('普通喊话')
#             if time.time() - time_1 > 30:
#                 self.key_down_up(self.ranks)
#                 self.world_shouts_1('51来不跳车')
#                 self.key_down_up(self.ranks)
#                 time_1 = time.time()
#
#     # 行商次数判断
#     def merchants_lakes_3(self):
#         while True:
#             if event.unbind[self.mapp].is_set():
#                 break
#             if self.merchants_lakes_3_1_flag:
#                 break
#             if self.merchants_lakes_3_flag:
#                 if self.get_coordinates('行商次数耗尽', search_scope=(494, 299, 853, 369), threshold=0.7):
#                     self.merchants_lakes_flag = True
#                     break
#             else:
#                 time.sleep(2)


# 采集任务
class AcquisitionTask(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.coord_iter = None
        self.coords = None
        self.acquisition_1_flag = True
        self.acquisition_flag = False
        self.last_coord = None
        self.time_out = int(event.task_config[self.mapp].get('采集加速延迟')) / 1000

    def initialization(self):
        pass

    def implement(self):
        line_dict = {1: '1线', 2: '2线', 3: '3线', 4: '4线', 5: '5线'}
        line_Limit = int(event.task_config[self.mapp].get('采集线数'))
        method = event.task_config[self.mapp].get('采集方法')
        self.coords = [sublist for sublist in event.task_config[self.mapp].get('自定义采集坐标') if
                       any(item != '' for item in sublist)]
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
                # 判断是否有目标
                if not self.acquisition_6():
                    self.acquisition_flag = False
                    # 执行范围搜素

                    # 没有目标 判断是否执行切线
                    # 切线
                    if line_Limit == 1:
                        time.sleep(2)
                        break
                    else:
                        if line <= line_Limit:
                            self.mouse_down_up(1235, 20)
                            self.Visual(line_dict[line], SIFT=True, threshold=0.3, search_scope=(910, 107, 1176, 643))
                            self.mouse_down_up(0, 0)
                            line += 1
                            time.sleep(4)
                        else:
                            self.mouse_down_up(1235, 20)
                            self.Visual(line_dict[1], histogram_process=True, threshold=0.75, search_scope=(910, 107, 1176, 643))
                            self.mouse_down_up(0, 0)
                            time.sleep(4)
                            break
                else:
                    self.acquisition_flag = True
                    # 有目标开始采集
                    self.Visual('采草', '伐木', '挖矿', histogram_process=True, process=True, threshold=1,
                                search_scope=(752, 291, 1153, 565), x=-20)
                    # 判断是否有工具
                    if self.coord('采集工具', binary_process=True, threshold=0.6):
                        # 没有工具 购买 或者 停止
                        print(1)
                    time.sleep(0.7)
                    self.Visual('采草', '伐木', '挖矿', histogram_process=True, process=True, threshold=1, x=-155, y=-74,
                                search_scope=(752, 291, 1153, 565))
                    time.sleep(10)

            self.acquisition_flag = False
        # 结束辅助线程
        self.acquisition_1_flag = False

    # 采集加速
    def acquisition_1(self):
        while self.acquisition_1_flag and not event.unbind[self.mapp].is_set():
            if self.acquisition_flag:
                if self.coord('采集指针', histogram_process=True, threshold=0.7, search_scope=(534, 437, 802, 583)):
                    time.sleep(0.85)
                    self.mouse_down_up(666, 475)
            else:
                time.sleep(2)
            time.sleep(0.02)

    # 指定地图采集目标
    def acquisition_2(self):
        map_name = event.task_config[self.mapp].get('指定地图')
        self.key_down_up('M')
        self.Visual('世界', binary_process=True, threshold=0.6)
        self.Visual(map_name, binary_process=True, threshold=0.6)
        self.Visual('传送点', histogram_process=True, threshold=0.6)
        # self.Visual('传送点', histogram_process=True, threshold=0.6)
        # self.key_down_up('M')
        self.Visual('关闭', histogram_process=True, threshold=0.65)
        self.arrive()

    # 地图就近查找目标
    def acquisition_3(self):
        search_scope = (0, 80, 1334, 750)
        count = 0
        while not event.unbind[self.mapp].is_set():
            if self.last_coord is not None:
                search_scope = (self.last_coord[0][0] - 80 * count, self.last_coord[0][1] - 80 * count,
                                self.last_coord[0][0] + 80 * count, self.last_coord[0][1] + 80 * count)

            self.key_down_up('M')
            if coord := self.Visual('采集物3', histogram_process=True, search_scope=search_scope,
                                    threshold=0.55):
                self.last_coord = coord
                self.key_down_up('M')
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

        self.key_down_up('M')
        self.Visual('世界搜索坐标展开', binary_process=True, threshold=0.6, wait_count=1)
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=310,
                    search_scope=(0, 631, 414, 694))
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=96, search_scope=(0, 631, 414, 694))
        self.input(coord[0])
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=233, search_scope=(0, 631, 414, 694))
        self.input(coord[1])
        self.Visual('前往坐标', binary_process=True, threshold=0.6, wait_count=1, x=310, search_scope=(0, 631, 414, 694))
        self.key_down_up('M')
        self.arrive()

    # 判断是否有目标
    def acquisition_6(self):
        method = event.task_config[self.mapp].get('采集方法')
        if method == '定点采集' or method == '自定义坐标采集':
            coord = self.Visual('采草', '伐木', '挖矿', histogram_process=True, process=True, threshold=1,
                                search_scope=(752, 291, 1153, 565), x=-20, tap=False)
        else:
            coord = self.Visual('采草', '伐木', '挖矿', histogram_process=True, process=True, threshold=1,
                                search_scope=(752, 291, 1153, 565), x=-20, tap=False)
        if coord:
            return True
        return False

    # 设置采集物
    def acquisition_7(self):
        self.key_down_up('M')
        self.Visual('地图目标设置', binary_process=True, threshold=0.7)
        self.mouse_move(281, 418, 181, 400)
        if event.task_config[self.mapp].get('采集种类') == '采草':
            self.Visual('采草目标', laplacian_process=True, threshold=0.5)
            self.mouse_move(281, 418, 181, 290)
        elif event.task_config[self.mapp].get('采集种类') == '伐木':
            self.Visual('伐木目标', binary_process=True, threshold=0.6)
            self.mouse_move(281, 418, 181, 270)
        elif event.task_config[self.mapp].get('采集种类') == '挖矿':
            self.Visual('挖矿目标', binary_process=True, threshold=0.6)
            self.mouse_move(281, 418, 181, 230)
        self.Visual(event.task_config[self.mapp].get('采集目标'), binary_process=True, threshold=0.6)
        self.Visual('采集目标关闭', binary_process=True, threshold=0.6)
        self.key_down_up('M')

    # 关闭掉落物品
    def acquisition_4(self):
        while self.acquisition_1_flag and not event.unbind[self.mapp].is_set():
            if self.coord('关闭', histogram_process=True, threshold=0.65, search_scope=(871, 230, 1209, 631)):
                for _ in range(4):
                    if event.unbind[self.mapp].is_set():
                        break
                    self.Visual('关闭', histogram_process=True, threshold=0.65, search_scope=(871, 230, 1209, 631))
            time.sleep(3)


# 扫摆摊
class SweepStalls(BasicTask):

    def __init__(self, row, handle, mapp):
        super().__init__(row, handle, mapp)
        self.sweep_stalls_1_flag = True

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('交易', histogram_process=True, threshold=0.7)
        self.Visual('摆摊', laplacian_process=True, threshold=0.3)
        self.Visual('关注', histogram_process=True, threshold=0.7)
        timeout_1 = int(event.task_config[self.mapp].get('扫摆摊延迟1')) / 1000
        timeout_2 = int(event.task_config[self.mapp].get('扫摆摊延迟2')) / 1000
        timeout_3 = int(event.task_config[self.mapp].get('扫摆摊延迟3')) / 1000
        Thread(target=self.sweep_stalls_1).start()
        while not event.unbind[self.mapp].is_set():
            time.sleep(0.01)
            if not self.coord('商品数量', histogram_process=True, threshold=0.7, search_scope=(468, 205, 563, 243)):
                self.sweep_stalls_1_flag = False
                self.mouse_down_up(533, 227, tap_after_timeout=0)

                while not event.unbind[self.mapp].is_set():
                    if self.Visual('银两购买', histogram_process=True, threshold=0.6,
                                   search_scope=(468, 205, 563, 243), tap_ago_timeout=0, tap_after_timeout=timeout_1,
                                   continuous_search_timeout=0, wait_count=50):
                        self.Visual('购买', histogram_process=True, threshold=0.7, tap_ago_timeout=0,
                                    tap_after_timeout=timeout_2, continuous_search_timeout=0, wait_count=10)
                        self.Visual('确定', histogram_process=True, threshold=0.7, tap_ago_timeout=0,
                                    tap_after_timeout=timeout_3, continuous_search_timeout=0, wait_count=10)
                    else:
                        self.sweep_stalls_1_flag = True
                        self.mouse_down_up(189, 285, tap_after_timeout=1)
                        break

    def sweep_stalls_1(self):
        while not event.unbind[self.mapp].is_set():
            if self.sweep_stalls_1_flag:
                self.mouse_down_up(189, 285, tap_after_timeout=0)
            time.sleep(0.22)


# # 扫集市
# class SweepMarket(BasicTask):
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

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('O')
        self.Visual('帮派领地', laplacian_process=True, threshold=0.25)
        self.Visual('排名', histogram_process=True, threshold=0.7)
        self.Visual('全服', laplacian_process=True, threshold=0.25)
        self.mouse_down_up(585, 189)
        self.Visual('参观', laplacian_process=True, threshold=0.25)
        self.Visual('跨服模式', laplacian_process=True, threshold=0.25, wait_count=30, tap=False)
        while not event.unbind[self.mapp].is_set():
            self.key_down_up('O')
            self.Visual('帮派领地', laplacian_process=True, threshold=0.25)
            self.Visual('建造气力', histogram_process=True, threshold=0.7, x=114)
            if force := self.img_ocr(search_scope=(1040, 224, 1122, 256)):
                try:
                    force = force.split('/')
                    if int(force[0]) <= 10:
                        break
                except ValueError:
                    pass
            self.key_down_up('O')
            self.mouse_down_up(1330, 740)
            self.key_down_up('M')
            self.mouse_down_up(768, 530)
            self.mouse_down_up(768, 530)
            self.key_down_up('M')
            if self.Visual('挑水', histogram_process=True, threshold=0.7, tap_ago_timeout=1, wait_count=30,
                           search_scope=(800, 270, 1033, 650)):
                time.sleep(1)
                self.Visual('挑水', histogram_process=True, threshold=0.7, wait_count=1,
                            search_scope=(800, 270, 1033, 650), tap_time_out=0)
                self.Visual('确认', laplacian_process=True)

                if self.Visual('打水', histogram_process=True, threshold=0.65, tap_ago_timeout=1, wait_count=18,
                               search_scope=(800, 270, 1033, 650)):
                    while not event.unbind[self.mapp].is_set():
                        if coord := self.coord('水滴', search_scope=(312, 580, 1014, 637), laplacian_process=True,
                                               threshold=0.25):
                            if coord[0][0] <= 666:
                                self.key_down_up('D', key_down_timeout=0)
                            elif coord[0][0] >= 666:
                                self.key_down_up('A', key_down_timeout=0)
                        else:
                            time.sleep(5)
                            break


# 坐观万象
class SittingObserving(BasicTask):

    def initialization(self):
        pass

    def implement(self):
        self.key_down_up('B')
        self.Visual('活动入口', histogram_process=True, threshold=0.7)
        self.Visual('活动', binary_process=True, threshold=0.5)
        self.Visual('活动界面游历', laplacian_process=True)
        if self.Visual('坐观万象', histogram_process=True, threshold=0.7, y=45):
            self.Visual('修炼中', histogram_process=True, threshold=0.7, tap=False,
                        search_scope=(485, 509, 769, 563), wait_count=360)
            self.journal('修炼中')
            while not event.unbind[self.mapp].is_set():
                if not self.Visual('修炼中', histogram_process=True, threshold=0.7, tap=False,
                                   search_scope=(485, 509, 769, 563)):
                    self.journal('修炼结束')
                    break
                time.sleep(5)
                self.mouse_down_up(1330, 740)

        for _ in range(5):
            if event.unbind[self.mapp].is_set():
                break
            if not self.Visual('关闭', histogram_process=True, threshold=0.65):
                break


# 每日兑换
class DailyRedemption(BasicTask):

    def initialization(self):
        pass

    def implement(self):

        # 山河器
        if event.task_config[self.mapp].get('山河器'):
            self.key_down_up('B')
            self.Visual('活动入口', histogram_process=True, threshold=0.7)
            self.Visual('山河器', laplacian_process=True)
            for _ in range(5):
                if event.unbind[self.mapp].is_set():
                    break
                if self.Visual('前往探索', binary_process=True, threshold=0.5):
                    self.journal('前往探索山河器')
                    self.journal('等待到达目标地点')
                    self.arrive()
                    self.journal('到达目标地点查找目标')
                    if self.Visual('拾取', binary_process=True, threshold=0.6):
                        self.journal('查找目标成功')
                    else:
                        self.journal('目标查找失败')
                        self.key_down_up('B')
                        self.Visual('活动入口', histogram_process=True, threshold=0.7)
                        self.Visual('山河器', laplacian_process=True)
                        continue
                    time.sleep(6)
                    self.mouse_down_up(0, 0)
                else:
                    if not self.Visual('免费搜索', histogram_process=True, threshold=0.7):
                        self.journal('没有免费山河器次数')
                        break
                    else:
                        time.sleep(3)
                        self.Visual('寻器', laplacian_process=True)
                        self.journal('免费搜索')
                        time.sleep(3)
                        continue

            self.Visual('关闭', histogram_process=True, threshold=0.65)

        # 银票礼盒
        if event.task_config[self.mapp].get('银票礼盒'):
            self.key_down_up('B')
            self.Visual('活动入口', histogram_process=True, threshold=0.7)
            self.Visual('珍宝阁', laplacian_process=True)
            self.Visual('商城', histogram_process=True, threshold=0.7)
            self.Visual('搜索图标', histogram_process=True, threshold=0.85)
            self.Visual('输入道具名称', laplacian_process=True)
            self.input('银票礼盒')
            self.Visual('搜索图标', histogram_process=True, threshold=0.85)
            self.Visual('银票礼盒', laplacian_process=True)
            for _ in range(30):
                if event.unbind[self.mapp].is_set():
                    break
                self.mouse_down_up(987, 699, tap_after_timeout=0.15)
            for _ in range(5):
                if event.unbind[self.mapp].is_set():
                    break
                if not self.Visual('关闭', histogram_process=True, threshold=0.65):
                    break

        # 榫头卯眼 鸡蛋
        if event.task_config[self.mapp].get('榫头卯眼') or event.task_config[self.mapp].get('商会鸡蛋'):
            self.key_down_up('B')
            self.Visual('活动入口', histogram_process=True, threshold=0.7)
            self.Visual('珍宝阁', laplacian_process=True)
            self.Visual('商会', laplacian_process=True)
            self.Visual('商会宝石', histogram_process=True, threshold=0.7)
            self.Visual('上拉', histogram_process=True, threshold=0.7)
            # self.mouse_x_y(200,680,220,200)
            self.Visual('江湖杂货', histogram_process=True, threshold=0.7)
            self.mouse_move(580, 680, 600, 225)
            time.sleep(1.8)

            if event.task_config[self.mapp].get('商会鸡蛋'):
                if self.Visual('鸡蛋', histogram_process=True, threshold=0.7):
                    for _ in range(5):
                        if event.unbind[self.mapp].is_set():
                            break
                        self.mouse_down_up(970, 680, tap_after_timeout=0.2)

            if event.task_config[self.mapp].get('榫头卯眼'):
                if self.Visual('榫头卯眼', histogram_process=True, threshold=0.7):
                    for _ in range(12):
                        if event.unbind[self.mapp].is_set():
                            break
                        self.mouse_down_up(970, 680, tap_after_timeout=0.2)

            for _ in range(6):
                if event.unbind[self.mapp].is_set():
                    break
                if not self.Visual('关闭', histogram_process=True, threshold=0.65):
                    break

        # 锦芳秀
        if event.task_config[self.mapp].get('锦芳绣残片'):
            self.key_down_up('B')
            self.Visual('积分', laplacian_process=True)
            self.Visual('积分社交', histogram_process=True, threshold=0.7)
            self.Visual('积分社交桃李值', x=350, histogram_process=True, threshold=0.7)
            if self.Visual('积分兑换搜索', laplacian_process=True):
                self.input('锦芳绣')
                self.Visual('搜索图标', histogram_process=True, threshold=0.7)
                self.mouse_down_up(1000, 600, tap_after_timeout=0.2)
                for _ in range(5):
                    if event.unbind[self.mapp].is_set():
                        break
                    if not self.Visual('关闭', histogram_process=True, threshold=0.65):
                        break

        # 帮派铜钱捐献和银两捐献
        if event.task_config[self.mapp].get('帮派铜钱捐献') or event.task_config[self.mapp].get('帮派银两捐献'):

            self.key_down_up('O')
            self.Visual('帮派福利', laplacian_process=True)
            self.Visual('帮派捐献', laplacian_process=True)
            # 帮派铜钱捐献
            if event.task_config[self.mapp].get('帮派铜钱捐献'):
                for _ in range(3):
                    if event.unbind[self.mapp].is_set():
                        break
                    self.Visual('捐献', search_scope=(146, 224, 450, 640), laplacian_process=True)
                    if self.Visual('不再提示', laplacian_process=True):
                        self.Visual('确定3', laplacian_process=True)

            # 帮派银两捐献
            if event.task_config[self.mapp].get('帮派银两捐献'):
                for _ in range(3):
                    if event.unbind[self.mapp].is_set():
                        break
                    self.Visual('捐献', search_scope=(513, 224, 805, 640), laplacian_process=True)
                    if self.Visual('不再提示', laplacian_process=True):
                        self.Visual('确定3', laplacian_process=True)

            for _ in range(3):
                if event.unbind[self.mapp].is_set():
                    break
                if not self.Visual('关闭', random_tap=False, histogram_process=True, threshold=0.65):
                    break

        # 帮派摇钱树
        if event.task_config[self.mapp].get('摇钱树'):
            self.key_down_up('B')
            self.Visual('活动入口', histogram_process=True, threshold=0.7)
            self.Visual('活动', binary_process=True, threshold=0.5)
            self.Visual('活动界面帮派', laplacian_process=True)
            self.Visual('摇钱树', histogram_process=True, threshold=0.7, y=45)
            if self.Visual('前往1', laplacian_process=True):
                target = event.task_config[self.mapp].get('摇钱树目标')
                if target == 0:
                    if self.Visual('轻轻摇', laplacian_process=True, search_scope=(750, 213, 1137, 649),
                                   wait_count=360):
                        time.sleep(1)
                        self.Visual('轻轻摇', laplacian_process=True, search_scope=(750, 213, 1137, 649))
                elif target == 1:
                    if self.Visual('用力摇', laplacian_process=True, search_scope=(750, 213, 1137, 649),
                                   wait_count=360):
                        time.sleep(1)
                        self.Visual('用力摇', laplacian_process=True, search_scope=(750, 213, 1137, 649))
                elif target == 2:
                    if self.Visual('全力摇', laplacian_process=True, search_scope=(750, 213, 1137, 649),
                                   wait_count=360):
                        time.sleep(1)
                        self.Visual('全力摇', laplacian_process=True, search_scope=(750, 213, 1137, 649))

            for _ in range(4):
                if event.unbind[self.mapp].is_set():
                    break
                if not self.Visual('关闭', random_tap=False, histogram_process=True, threshold=0.65):
                    break


TASK_MAPPING = {'课业任务': LessonTask, '世界喊话': WorldShoutsTask, '江湖英雄榜': HeroListTask,
                '日常副本': DailyCopiesTask, '悬赏任务': BountyMissionsTask, '侠缘喊话': ChivalryShoutTask,
                '帮派设宴': GangBanquet, '破阵设宴': BreakingBanquet, '发布悬赏': PostBounty,
                '每日兑换': DailyRedemption, '坐观万象': SittingObserving, '扫摆摊': SweepStalls,
                '狂饮豪拳': DrinkPunch, '帮派任务': FactionTask, '茶馆说书': TeaStory,
                '华山论剑': TheSword, '帮派积分': GangPoints, '每日一卦': HexagramDay,
                '江湖急送': UrgentDeliveryTask, '采集任务': AcquisitionTask, '切换角色': None}

TASK_SHOW = {'课业任务': (0, 1074), '日常副本': (0, 2148), '悬赏任务': (0, 0), '每日兑换': (0, 537),
             '扫摆摊': (0, 1074), '侠缘喊话': (0, 1611), '世界喊话': (0, 1611), '华山论剑': (0, 2148),
             '江湖英雄榜': (0, 2148), '采集任务': (0, 2685), '切换角色': (0, 2148)}
