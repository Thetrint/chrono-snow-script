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
            for _ in range(60):
                if self.stop.is_set():
                    return 0
                time.sleep(1)

    @staticmethod
    def get_public_ip():
        try:
            response = requests.get('https://api.ipify.org?format=json')
            return response.json()['ip']
        except Exception as e:
            return str(e)


services = ClientServices()

TOKEN = f'{services.get_public_ip()} {datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")}'


if __name__ == '__main__':
    print(services.get_public_ip())

