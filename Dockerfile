FROM ubuntu:18.04
# 基于的基础镜像

ADD ./code /code
# 将当前目录下的code文件夹复制到/code路径下

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bk
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get update
# 换apt-get为国内源并更新

RUN apt-get install python3.6.9
RUN apt-get install python3-pip
# 安装python和pip工具
