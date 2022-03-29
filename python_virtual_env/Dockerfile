# FROM arm32v7/ubuntu:20.04
FROM arm32v7/ubuntu:18.04
# 基于的ARM架构的镜像

#FROM ubuntu:18.04
# 基于x86架构的镜像

ADD ./Monitor_code/ /code
# 将当前目录下的测试代码文件夹复制到/code路径下
ADD ./install_python_env /install_python_env
# 将构建python虚拟环境的脚本相关文件复制到/install_python_env路径下
ADD ./python_pac /python_pac
# 将手动安装的资源复制到镜像中

# 更换apt-get源
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bk \
 && sed -i s@/ports.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
# ARM架构换源

#&& sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
# x86架构换元

# 更新索引，安装python和pip工具并创建虚拟环境
RUN apt-get update \
 && apt-get install -y zip \
 && apt-get install -y vim \
 && apt-get install -y python3 \
 && apt-get install -y python3-pip \
 && apt-get install -y gfortran libopenblas-dev liblapack-dev \
 && apt-get install -y libjpeg-dev zlib1g-dev \
 && apt-get install -y gawk bison gcc-multilib\
 && apt-get install -y python3-venv

# 更换pip源
RUN mkdir ~/.pip \
 && touch ~/.pip/pip.conf \
 && echo '\
[global]\n\
index-url = https://pypi.tuna.tsinghua.edu.cn/simple/\n\
[install]\n\
trusted-host = pypi.tuna.tsinghua.edu.cn' > ~/.pip/pip.conf