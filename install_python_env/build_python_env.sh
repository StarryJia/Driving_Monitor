#!/bin/bash
###
 # @Description: install_python_env
 # @Author: jiayuchen
 # @Date: 2022-03-15 11:24:08
 # @LastEditTime: 2022-03-15 15:45:53
### 

dir='/mypy-env'
# 自定义环境路径

mkdir ${dir}
python3 -m venv ${dir}
# 创建虚拟环境

source ${dir}/bin/activate
# 激活虚拟环境

pip install --upgrade pip
#更新pip

pip install -r requirements.txt
# 安装相关依赖

#pip install xxxxx
# 自定义安装