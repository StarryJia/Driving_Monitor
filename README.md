# Driving_Monitor
驾驶员状态监控程序，实现对驾驶员的状态收集，并设置危险驾驶提醒。

实现虚拟环境：
1.将python的环境依赖集成省去在树莓派上安装环境等步骤。
2.使用docker实现封装和部署，在任意的机器上都能够部署和使用。

为什么要用虚拟环境：

在实际项目开发中，我们通常会根据自己的需求去下载各种相应的库，如opencv、dlib等，并且由于团队开发，每个人负责的部分不同，用到的库以及版本和环境依赖也不同，这样需要我们根据需求不断的更新或卸载相应的库。这样的情况会让我们的开发环境和项目造成很多不必要的麻烦，管理也相当混乱，因此统一进行代码管理，免去后期移植及部署的麻烦。

### face_recognize

主要负责人：贾宇辰、穆含青

实现人脸识别、状态识别、注意力集中提醒。

#### jyc_code

opencv_recognize.py定义了使用opencv识别人脸的方法库，供外部进行调用。