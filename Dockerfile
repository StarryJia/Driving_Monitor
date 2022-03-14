FROM ubuntu:18.04
# 基于的基础镜像

ADD /face_recognize/env_test_code /code
# 将当前目录下的测试代码文件夹复制到/code路径下

# 更换apt-get源，同时更新索引，安装python和pip工具
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bk \
 && sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
 && apt-get update \
 && apt-get install python3.6.9 \
 && apt-get install python3-pip
