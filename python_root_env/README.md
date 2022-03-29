# README

在构建完docker之后可以执行脚本

```bash
build_root_python_env.sh
```

可能会出现错误

```
bad interpreter: No such file or directory
```

这是由于脚本是在windows环境下写的需要修改一下:

```bash
sed -i "s/\r//" a.sh
```

然后就可以正常执行脚本了。

```bash
pip3 install --upgrade pip
# 更新pip

apt-get install -y libopencv-dev python3-opencv
# 将会自动安装numpy
# 安装完测试import cv2,numpy
apt-get install -y python3-scipy
pip3 install --upgrade numpy
cd /python_pac
pip3 install torch-1.8.1-cp38-cp38-linux_armv7l.whl 
apt-get install -y python3-scipy
pip install six
pip install tflite-runtime
```

脚本主要安装了一些必要的python库，通过apt-get和pip结合的方式，以最快速度完成环境构建。

#### 出现的问题

在`pip3 install --upgrade numpy`的时候，我在ubuntu虚拟机上运行docker非常快，但是在Raspberry虚拟机中运行就是非常慢，我不确定这是为什么，但是能够正常升级numpy。

这个命令主要是升级numpy，为了支持torch，如果不升级的话在`import torch`的时候会报错。
并且之后的包也需要高版本numpy，所以如果认为不需要的话，可以手动安装这些包。