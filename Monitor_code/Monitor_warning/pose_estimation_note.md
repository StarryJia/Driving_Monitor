# pose_estimation代码笔记

## 引入模块

### python标准库

- argparse

argsparse是python的命令行解析的标准模块，内置于python，不需要安装。这个库可以让我们直接在命令行中就可以向程序中传入参数并让程序运行。

- logging

logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等；相比print，具备如下优点：

1. 可以通过设置不同的日志等级，在release版本中只输出重要信息，而不必显示大量的调试信息；
2. print将所有信息都输出到标准输出中，严重影响开发者从标准输出中查看其它数据；logging则可以由开发者决定将信息输出到什么地方，以及怎么输出；

- sys

该模块提供对解释器使用或维护的一些变量的访问，以及与解释器强烈交互的函数。它始终可用。

- time

Python 提供了一个 time 和 calendar 模块可以用于格式化日期和时间。

### 第三方库

1. cv2
2. ml
3. utils
4. numpy
5. data

### 代码理解

#### data.py