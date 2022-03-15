# Driving_Monitor
驾驶员状态监控程序，实现对驾驶员的状态收集，并设置危险驾驶提醒。

实现统一的环境：

python==3.6.9

1.统一python版本，统一相同的库及版本。
2.使用docker实现封装和部署，在任意的机器上都能够部署和使用。

##### 为什么要用统一环境:

在实际项目开发中，我们通常会根据自己的需求去下载各种相应的库，如opencv、dlib等，并且由于团队开发，每个人负责的部分不同，用到的库以及版本和环境依赖也不同，这样需要我们根据需求不断的更新或卸载相应的库。这样的情况会让我们的开发环境和项目造成很多不必要的麻烦，管理也相当混乱，因此统一进行代码管理，免去后期移植及部署的麻烦。

##### 为什么要用docker:

利用Docker来构建打包应用镜像，这样可以一次构建到处运行，也可以充分利用Dockerfile自带的分层能力，可以方便地调整依赖包，这样在开发部署过程中格外高效。

#### Dockerfile

可以使用Dockerfile构建基础的镜像，并根据自己的需求进行微调。
目前Dockerfile在ARM架构下运行正常，在x86架构下运行有导致pyhton第三方库安装依赖出现问题，体现于调用cv2的时候会报错。
因此使用Dockerfile构建基础镜像 + shell脚本安装python环境。

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

使用Ubuntu虚拟机运行docker，docker基于镜像Ubuntu:18.04，并安装python3.6.9版本和相关依赖。

### face_recognize

实现人脸识别、状态识别、注意力集中提醒。

opencv_recognize.py定义了使用opencv识别人脸的方法库，供外部进行调用。