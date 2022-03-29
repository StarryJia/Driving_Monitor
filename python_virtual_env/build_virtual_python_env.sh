#!/bin/bash
###
 # @Description: install_python_env
 # @Author: jiayuchen
 # @Date: 2022-03-15 11:24:08
 # @LastEditTime: 2022-03-15 15:45:53
### 

# apt-get install cmake
# 安装cmake

dir='/mypy-env'
# 自定义环境路径

mkdir ${dir}
python3 -m venv ${dir}
# 创建虚拟环境

source ${dir}/bin/activate
# source /mypy-env/bin/activate
# 激活虚拟环境

pip install --upgrade pip
#更新pip

# 注:在arm架构下不支持pip安装cmake和opencv

# 如果需要手动安装cmake
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

# 先安装numpy，numpy是opencv的前提
pip install numpy

#手动安装opencv
echo "installing opencv-python."
cd /python_pac
unzip opencv-3.4.10.zip
unzip opencv_contrib-3.4.10.zip
cd opencv-3.4.10
mkdir build
cd build
cmake -D BUILD_opencv_python3=YES \
-D CMAKE_BUILD_TYPE=Release \
-D CMAKE_INSTALL_PREFIX=${dir}/opencv3.4 \
-D OPENCV_EXTRA_MODULES=../../opencv_contrib-3.4.10/modules \
-D PYTHON3_LIBRARIES=/usr/lib/arm-linux-gnueabihf/libpython3.8m.so \
-D PYTHON3_EXECUTABLE=${dir}/bin/python3 \
-D PYTHON3_NUMPY_INCLUDE_DIRS=${dir}/lib/python3.8/site-packages/numpy/core/include/ \
-D PYTHON3_PACKAGES_PATH=${dir}/lib/python3.8/site-packages ..
echo "making opencv,may take few hours."
make -j8
make install

# 可以直接安装
pip install dlib
pip install six
pip install imutils
echo "installing scipy,may take few hours."
# 在安装scipy之前需要先安装gfortran libopenblas-dev liblapack-dev
pip install scipy

#在安装pillow之前需要先安装libjpeg-dev zlib1g-dev
pip install pillow

#需要使用特定源来安装
pip install torchvision -f https://cf.torch.kmtea.eu/whl/stable-cn.html
pip install torch -f https://cf.torch.kmtea.eu/whl/stable-cn.html
# pip install torch -f https://torch.maku.ml/whl/stable.html
# 针对arm架构的torch源

# 如果在import torch时报错: version `GLIBC_2.28' not found,请安装对应版本的glibc
# 由于安装过程可能会出错导致系统无法正常运行，请手动安装，这里只是提供我安装的步骤，并且将会被注释掉


# pip install -r requirements.txt
# 安装相关依赖

# pip install ...
# 自定义安装

rm -rf /python_pac
# 删除安装包