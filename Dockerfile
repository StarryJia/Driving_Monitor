FROM ubuntu:18.04
# 基于的基础镜像

ADD ./face_recognize/env_test_code /code
# 将当前目录下的测试代码文件夹复制到/code路径下

# 更换apt-get源，同时更新索引，安装python和pip工具并更换pip源
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bk \
 && sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
 && apt-get update \
 && apt-get install -y python3 \
 && apt-get install -y python3-pip \
 && mkdir ~/.pip \
 && touch ~/.pip/pip.conf
RUN echo '\
[global]\n\
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/\n\
[install]\n\
trusted-host = pypi.tuna.tsinghua.edu.cn' > ~/.pip/pip.conf
RUN python3 -m pip install pip --upgrade \
 && python3 -m pip install cmake \
 && python3 -m pip install opencv-python
