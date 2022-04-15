# -*- encoding: utf-8 -*-

'''
@File    :   Data_report.py
@Time    :   2022/04/15 23:12:34
@Author  :   StarryJia 
@Version :   1.0
@IDE     :   VS Code
'''

import time

import socket
import json

class send_to_server(object):
    def __init__(self, user = 'test', password = 'test'):
        self.user = user
        self.password = password
    
    def send_data(self,data_dic):
        sk = socket.socket()        # 创建socket
        addess = ('152.136.115.85',5566)
        sk.connect(addess)        # 连接服务端IP地址和端口号
        data_dic_json = json.dumps(data_dic)
        sk.send(bytes(data_dic_json,encoding = 'utf-8')) # bytes编码后发送信息
        data = sk.recv(1024) # 接收信息
        print(str(data,'utf8')) #str解码后打印出来

    def send_status(self,status):
        tic = time.strftime('%Y-%m-%d %H-%M-%S')
        status_dic = {}
        status_dic['status'] = status
        status_dic['user'] = self.user
        status_dic['timestamp'] = tic
        print('send \n',status_dic)
        self.send_data(status_dic)
    
send_to_server().send_status('normal')