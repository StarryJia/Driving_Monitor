def _init():#初始化
    global test_dic
    test_dic = {"cigarette":0.0, "cellphone": 0.0}
    global has_sent
    has_sent = 0    #是否已向yolo传送图片
 
 
def set_dic_value(key,value):
    """ 定义一个全局变量 """
    test_dic[key] = value

def set_sent_value(value):
    has_sent = value
 
 
def get_dic_value(key):
    """ 获得一个全局变量,不存在则返回默认值 """
    return test_dic[key]

def get_sent_value():
    return has_sent

