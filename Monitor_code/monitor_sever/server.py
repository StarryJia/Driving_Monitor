import os

import socket
import json
import threading

class monitor_server(object):
    def __init__(self) -> None:
        self.sk = socket.socket()        # 创建socket
        host = socket.gethostname()
        host = '10.0.8.8'
        port = 5566
        adress = (host,port)
        self.sk.bind(adress)            # 为socket绑定IP地址与端口号
        self.sk.listen(1)    # 客户端连接人数
        print("sever started")
        #client_hadle()
    
    def status_file(self,user_name,str_data):
        if os.path.exists(r"./用户数据"): # 判断是否存在文件夹，没有则创建
            pass
        else:
            os.mkdir("用户数据")
        file_name = r'./用户数据/' + user_name + '.txt'
        file = open(file_name,"a+") # 向文件中添加新数据
        file.write(str_data + '\n')
        file.close()
        

    def client_hadle(self) :
        while True :
            conn, addr = self.sk.accept()    # 等待客户端连接
            data = conn.recv(1024)    # 接收的比特流
            str_data = str(data,encoding='utf-8')#将信息转换成字符串
            data = json.loads(data) #转换成python的数据结构
            print(data)
            self.status_file(data['user'],str_data)
            conn.sendto(b'sever_received',addr)# bytes编码后发信息
            conn.close()


if __name__ == '__main__' :
    sever = monitor_server()
    client_handler1 = threading.Thread(target=sever.client_hadle,args=())
    client_handler1.start()