#!/bin/bash
###
 # @Description: install_python_env
 # @Author: jiayuchen
 # @Date: 2022-03-15 11:24:08
 # @LastEditTime: 2022-03-15 15:45:53
### 

apt-get install cmake
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

# 手动安装cmake
echo "installing cmake,may take a while."
cd /python_pac
mkdir build_cmake
tar -zxvf cmake-3.9.2.tar.gz -C build_cmake
cd build_cmake/cmake-3.9.2
./bootstrap
make
make install
echo "camke installed"
cmake -version

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
-D PYTHON3_LIBRARIES=/usr/lib/arm-linux-gnueabihf/libpython3.6m.so \
-D PYTHON3_EXECUTABLE=${dir}/bin/python3 \
-D PYTHON3_NUMPY_INCLUDE_DIRS=${dir}/lib/python3.6/site-packages/numpy/core/include/ \
-D PYTHON3_PACKAGES_PATH=${dir}/lib/python3.6/site-packages ..
echo "making opencv,may take few hours."
make -j8
make install

pip install six
pip install imutils

# pip install -r requirements.txt
# 安装相关依赖

# pip install ...
# 自定义安装