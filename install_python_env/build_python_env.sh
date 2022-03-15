#!/bin/bash
###
 # @Description: install_python_env
 # @Author: jiayuchen
 # @Date: 2022-03-15 11:24:08
 # @LastEditTime: 2022-03-15 12:46:19
### 

mkdir /mypy-env
python3 -m venv /mypy-env
# 创建虚拟环境

source /mypy-env/bin/activate
# 激活虚拟环境

pip install -r requirements.txt
# 安装相关依赖

pip install requests
# 自定义安装