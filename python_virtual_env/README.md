# 安装python虚拟环境(不推荐)

基于镜像ubuntu:18.04

**无法import pytorch，由于glibc版本为2.27所以无法import !!!!!!**

如果需要pytorch请移步[root_env](../python_root_env)该镜像基于ubuntu:20.04能够实现：

cmake + dlib + opencv + pytorch

该方法主要用于安装在虚拟环境中，所以无法使用apt-get install 安装一些python的包，导致运行速度缓慢，整体安装下来可能需要20小时左右，

### 不要直接运行`build_virtual_python_env.sh`!!!!!!

如果有必要的话请将sh脚本中的命令复制下来手动执行。

该方法仅仅用于记录我的一次安装过程，主要特点就是python包都安装在了虚拟环境中，但是这其实是一个被弃用的方案，尽管它能够实现基础的功能，但是有很多更快捷的方案来安装环境。

并且docker中完全可以当作一个虚拟环境，专docker专用。

但是它也是学习环境安装中重要的一环，并且其过程很大程度上帮助了之后的环境安装。

### 所以如果不是为了尝试学习或者其他什么原因的话请不要使用该方法安装环境。

请使用[root_env](../python_root_env)安装。