#!/bin/bash
###
 # @Description: install_python_env
 # @Author: jiayuchen
 # @Date: 2022-03-15 11:24:08
 # @LastEditTime: 2022-03-15 15:45:53
### 

apt-get install -y build-essential
apt-get install -y libssl-dev

# pip3 install --upgrade pip
# #更新pip
# pip3 install numpy
# pip3 install --upgrade numpy

# apt-get install -y libopencv-dev python3-opencv
# # 将会自动安装numpy
# # 安装完测试import cv2,numpy

# apt-get install python3-scipy 

# 手动安装cmake
# 安装前需要依赖libssl-dev 和 build-essential
# cmake 3.9.2版本无法安装
# echo "installing cmake,may take a while."
# cd /python_pac
# mkdir build_cmake
# tar -zxvf cmake-3.9.2.tar.gz -C build_cmake
# cd build_cmake/cmake-3.9.2
# ./bootstrap
# make
# make install
# echo "camke installed"
# cmake -version

#可以安装 cmake 3.21.6 的版本
echo "installing cmake,may take a while."
cd /python_pac
mkdir build_cmake
tar -zxvf cmake-3.21.6.tar.gz -C build_cmake
cd build_cmake/cmake-3.21.6
./bootstrap
make
make install
echo "camke installed"
cmake -version



# # 可以直接安装
# pip install dlib
# pip install six
# pip install imutils
# echo "installing scipy,may take few hours."

# #在安装pillow之前需要先安装libjpeg-dev zlib1g-dev
# pip install pillow

# #需要使用特定源来安装
# pip install torchvision -f https://cf.torch.kmtea.eu/whl/stable-cn.html
# pip install torch -f https://cf.torch.kmtea.eu/whl/stable-cn.html
# # pip install torch -f https://torch.maku.ml/whl/stable.html
# # 针对arm架构的torch源

# # 如果在import torch时报错: version `GLIBC_2.28' not found,请安装对应版本的glibc
# # 由于安装过程可能会出错导致系统无法正常运行，请手动安装，这里只是提供我安装的步骤，并且将会被注释掉


# # pip install -r requirements.txt
# # 安装相关依赖

# # pip install ...
# # 自定义安装

# rm -rf /python_pac
# # 删除安装包