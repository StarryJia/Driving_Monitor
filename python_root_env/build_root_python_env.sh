#!/bin/bash
###
 # @Description: install_python_env
 # @Author: jiayuchen
 # @Date: 2022-03-15 11:24:08
 # @LastEditTime: 2022-03-15 15:45:53
### 

# apt-get install -y build-essential
# apt-get install -y libssl-dev

pip3 install --upgrade pip
# 更新pip

apt-get install -y libopencv-dev python3-opencv
# 将会自动安装numpy
# 安装完测试import cv2,numpy
apt-get install -y python3-scipy
pip3 install --upgrade numpy
cd /python_pac
pip3 install torch-1.8.1-cp38-cp38-linux_armv7l.whl
pip3 install torchvision-0.9.1-cp38-cp38-linux_armv7l.whl
apt-get install -y python3-scipy
pip install six
pip install tflite-runtime

# 安装yolo相关依赖
pip3 install tqdm
apt-get install -y python3-pandas
pip3 install requests
pip3 install pyyaml
pip3 install seaborn
# pip3 install --upgrade matplotlib


# 手动安装cmake
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

# 可以安装 cmake 3.21.6 的版本
# 安装 3.21.6 前需要依赖libssl-dev 和 build-essential
# echo "installing cmake,may take a while."
# cd /python_pac
# mkdir build_cmake
# tar -zxvf cmake-3.21.6.tar.gz -C build_cmake
# cd build_cmake/cmake-3.21.6
# ./bootstrap
# make
# make install
# echo "camke installed"
# cmake -version

# 需要使用特定源来安装
# pip install torchvision -f https://cf.torch.kmtea.eu/whl/stable-cn.html
# pip install torch -f https://cf.torch.kmtea.eu/whl/stable-cn.html
# pip install torch -f https://torch.maku.ml/whl/stable.html
# 最后一个是git源，可能需要登陆
# 针对arm架构的torch源
