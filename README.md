# Driving_Monitor
驾驶员状态监控程序，实现对驾驶员的状态收集，并设置危险驾驶提醒。

实现统一的环境：

python==3.8.10

1.统一python版本，统一相同的库及版本。
2.使用docker实现封装和部署，在任意的机器上都能够部署和使用。

##### 为什么要用统一环境:

在实际项目开发中，我们通常会根据自己的需求去下载各种相应的库，如opencv、dlib等，并且由于团队开发，每个人负责的部分不同，用到的库以及版本和环境依赖也不同，这样需要我们根据需求不断的更新或卸载相应的库。这样的情况会让我们的开发环境和项目造成很多不必要的麻烦，管理也相当混乱，因此统一进行代码管理，免去后期移植及部署的麻烦。

##### 为什么要用docker:

由于我们需要将程序部署在树莓派上，在安装python环境的时候编译运行一些相当大的库例如opencv时可能需要用一两个小时，并且树莓派的性能也并不一定能够完成安装，而docker镜像可以在电脑上制作好后导入树莓派直接运行程序，主要目的是利用电脑的性能制作镜像，然后树莓派只负责运行镜像程序。并且在制作镜像的时候可以随时导出镜像进行存档，在后续步骤出现问题时能够有用于恢复的快照。

##### Dockerfile

可以使用Dockerfile构建基础的镜像，并根据自己的需求进行微调。
目前Dockerfile在ARM架构下运行正常，在x86架构下运行有导致pyhton第三方库安装依赖出现问题，体现于调用cv2的时候会报错。
因此使用Dockerfile构建基础镜像 + shell脚本安装python环境。

*注：因为树莓派的架构为arm，所以为了适配树莓派目前镜像是基于arm架构搭建的，但是在构建镜像的时候提供了x86的相关选择，将注释去掉即可，同时需要注意的是在x86环境下apt-get换源也需要微调，并且由于python第三方库对环境要求较为严苛，所以目前准备手动安装python的相关库。*

制作镜像：
先制作基础镜像在Dockerfile所在目录下执行

```bash
docker build -t ubuntu:base .
```

运行容器

```bash
docker run -it --privileged ubuntu:base
```

进入容器，搭建python虚拟环境

```bash
cd /install_python_env
./build_python_env.sh
```

#### 我的Docker环境：

使用树莓派虚拟机运行docker，docker基于镜像arm32v7/Ubuntu:20.04，并安装python3.8.10版本和相关依赖。

### Monitor_code

实现人脸识别、状态识别、注意力集中提醒。

### QUICK START

```bash
cd codes
python3 Monitor.py
```

#### python_pac

为镜像提供了torch和torchvison的whl文件。

#### install_python_env

构建python的虚拟运行环境的sh脚本，由于包含了编译的命令，运行时间会很长，可以手动运行安装相关的python第三方库。