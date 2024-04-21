import datetime
import json
import threading
import time

import requests

from app.view.Public import publicSingle


class ClientServices:
    def __init__(self):
        self.stop = threading.Event()

    @staticmethod
    def POST(url, data):
        headers = {
            'Content-Type': 'application/json'
        }
        try:
            response = requests.post(url, json=data, headers=headers)
            return response.json()
        except KeyError as e:
            print(e)
            return False
        except requests.exceptions.ProxyError:
            return False
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False
        except requests.exceptions.RequestException:
            return False

    @staticmethod
    def login(username, password):
        url = 'https://75561x0s00.vicp.fun/login'
        data = {
            'username': username,
            'password': password,
            'token': TOKEN
        }
        if response := ClientServices.POST(url, data):
            # 从字典中获取成功信息
            success = response.get('success')
            message = response.get('message')
            return success, message
        return True, ''

    @staticmethod
    def signup(username, password):
        url = 'https://75561x0s00.vicp.fun/signup'
        data = {
            'username': username,
            'password': password
        }
        if response := ClientServices.POST(url, data):
            success = response.get('success')
            message = response.get('message')
            return success, message
        return False, '服务器请求失败'

    def heartbeat(self, username):
        while not self.stop.is_set():
            print('更新状态')
            try:
                url = 'https://75561x0s00.vicp.fun/heartbeat'
                data = {
                    'username': username,
                    'token': TOKEN
                }
                if response := ClientServices.POST(url, data):
                    success = response.get('success')
                    message = response.get('message')
                    if message == '登录凭证验证失败':
                        publicSingle.offline.emit()
                    if success:
                        for _ in range(55):
                            if self.stop.is_set():
                                return 0
                            time.sleep(1)
                time.sleep(5)
            except requests.exceptions.ProxyError:
                pass
            except requests.exceptions.ConnectionError:
                pass

    @staticmethod
    def get_public_ip():
        try:
            # 定义头文件
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                              ' Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0'
            }

            # 发送GET请求并添加头文件
            response = requests.get('https://searchplugin.csdn.net/api/v1/ip/get?ip=', headers=headers)

            return response.json()['data']['ip']
        except Exception as e:
            return str(e)


services = ClientServices()

TOKEN = f'{services.get_public_ip()} {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'


if __name__ == '__main__':
    print(services.get_public_ip())

