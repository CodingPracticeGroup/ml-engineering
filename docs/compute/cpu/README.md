# CPU {#cpu}

本章很短，因为机器学习工作由 GPU 主导——但 CPU 所占的比重一直在增长，而且与 GPU 不同，CPU 很容易在不知不觉中配置不足。

最初，机器学习工作负载除了数据处理（文本/音频/视频）之外，并不怎么使用 CPU。

2025 年前后，CPU 开始更多地被用于 RAG（检索增强生成，Retrieval-Augmented Generation）工作负载，这类负载需要执行数据库查询。

2026 年，由于强化学习（RL）工作负载中的 AI 工具调用，CPU 的负载开始上升。这些工作负载可能执行各种各样的计算机程序：可能要对生成的数据或代码做验证，要对生成的代码做编译和执行，还有许许多多别的事情，而且出于安全原因，这些通常都运行在沙箱（sandbox）中。这种需求有时要求配备专用的 CPU 节点，以免 GPU 被拖住而空等，因为与 GPU 同机部署的 CPU 核心可能不够用。

## 为 DataLoader 的工作需要多少 CPU 核心 {#how-many-cpu-cores-do-you-need-for-dataloader-work}

在典型的训练工作负载中，最消耗 CPU 核心的是 DataLoader。

每台加速器你需要：

1. 与加速器绑定的每个进程 1 个 CPU 核心
2. 每个 `DataLoader` worker 进程 1 个 CPU 核心——通常需要 2-4 个 worker。

对语言模型（LM）来说，2 个 worker 通常绰绰有余，尤其是数据已经预处理过的情况。

如果你需要做动态变换——计算机视觉模型或视觉语言模型（VLM）常常如此——你可能需要 3-4 个 worker，有时甚至更多。

目标是能够立即从 `DataLoader` 取到数据，不阻塞加速器的计算，也就是说，要在当前一轮迭代运行的同时，为下一轮迭代预先处理好一批样本。换句话说，准备下一批次（batch）数据所花的时间，不能超过一个同样大小的批次在加速器上完成单次迭代计算的时间。

除了预处理之外，如果你不是从本地存储、而是从云端动态拉取数据，你还需要确保数据预取足够快，喂得饱那些为“加速器熔炉”供料的 worker。

把上面的数量乘以加速器的数量，再给操作系统留几个核心（就算 4 个吧）。

如果节点有 8 台加速器，你设置了 `num_workers`，那么你需要 `8*(num_workers+1)+4` 个核心。如果你做 NLP，通常每台加速器配 2 个 worker 左右，于是 `8*(2+1)+4` => 28 个 CPU 核心。如果你做 CV 训练，假设每台加速器需要 4 个 worker，那么就是 `8(4+1)+4` => 44 个 CPU 核心。

如果非常活跃的进程数超过了 CPU 核心的总数，会发生什么？一些进程会被抢占（排进队列，等 CPU 核心空出来再运行），而你绝对要避免任何上下文切换。

不过现代云产品通常都有 50-100 多个 CPU 核心，所以一般来说，核心是够分的。

另见[异步 DataLoader](https://github.com/CodingPracticeGroup/ml-engineering/blob/master/training/performance/README.md#asynchronous-dataloader)。



## 沙箱化代码执行对 CPU 的需求 {#cpu-needs-for-sandboxed-code-execution}

为了给模型生成的内容打分，RL 的工具使用（tool-use）工作负载必须把它跑起来——可能是运行一个测试套件、做一次编译、安装一个软件包，或者做一次符号数学检查。这些都运行在沙箱里：一个隔离的环境，因为生成的代码天生不可信，而且一个样本的状态泄漏进下一个样本，会污染奖励信号。视所需的隔离强度而定，沙箱可以是一个带资源限制的子进程、一个容器、一个像 [gVisor](https://gvisor.dev/) 这样的用户态内核，或者一个像 [Firecracker](https://firecracker-microvm.github.io/) 这样的 micro-VM。

这些工作完全不涉及加速器，但数量极大——一个批次包含数百条轨迹（trajectory），每条轨迹又发起几十次工具调用，意味着每个训练步骤要运行数以万计的短小程序。

有两点会让它的代价比看上去高得多：

1. 隔离的代价落在系统调用（system call）上，而不是算术运算上。gVisor 不会给 CPU 指令增加任何开销，但它把每个系统调用都经由自己的用户态内核转发，所以在某些情况下，文件和进程密集的工作会慢上几倍——而跑一次测试或一次 `pip install` 恰恰就是这类工作（[gVisor 性能指南](https://gvisor.dev/docs/architecture_guide/performance/)）。micro-VM 避开了这笔开销，但取而代之的是每个沙箱都要付出一次启动代价，Firecracker 约 125ms。
2. 它处在加速器的关键路径（critical path）上。一个同步的 RL 步骤，在整个批次全部完成之前无法推进，因此 CPU 供应不足、某个沙箱拖沓落后，或者校验器（verifier）因一个病态输出而挂起，都会让整个作业里的每一台 GPU 空转。

给沙箱分配它们自己的核心，与训练进程及其 `DataLoader` worker 所使用的核心互不重叠。子进程会继承父进程允许的 CPU 核心，所以这个设置必须直接施加在沙箱本身，可通过 [`os.sched_setaffinity`](https://github.com/CodingPracticeGroup/ml-engineering/blob/master/training/performance/README.md#ossched_setaffinity) 或容器运行时的 cpuset 来设置。同一节还说明了这种划分在训练进程那一侧是如何实现的。要对依赖安装、测试运行这类高开销阶段限制并发，并给一切操作设置超时——否则模型生成的一个无限循环会一直占住一个核心，直到有什么东西把它杀掉。在规模化之后，正是这一点把沙箱推向了专用的 CPU 节点。

脚注：一篇 [Nebius 的 agentic RL 实践记录](https://nebius.com/blog/posts/in-agentic-rl-faster-tokens-are-not-enough)（2026-09）在 Terminal-Bench 负载上对此做了实测，模型是 GLM-5.2，跑在 32 台 B300 上。在保持处理中（in-flight）轨迹数为 1,024 的前提下，给沙箱分配确定的、互不重叠的 CPU 集合，把批次收集时间从 332.1s 降到 303.0s，而模型和 GPU 配置没有任何改动。



## CPU 卸载（offload） {#cpu-offload}

一些框架，例如 [DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/) 和 [FSDP](https://docs.pytorch.org/docs/stable/fsdp.html)，可以把一部分计算工作卸载（offload）到 CPU 上而不产生瓶颈。这种情况下，你会希望在现有用量之外配备更多的 CPU 核心。



## NUMA 亲和性 {#numa-affinity}

另见 [NUMA 亲和性](https://github.com/CodingPracticeGroup/ml-engineering/blob/master/training/performance/README.md#numa-affinity)。



## 超线程（Hyperthreads） {#hyperthreads}

[超线程（Hyper-Threads）](https://en.wikipedia.org/wiki/Hyper-threading) 把每个物理核心虚拟化为 2 个虚拟核心，使 CPU 核心数量翻倍，从而允许 2 个线程同时使用同一个 CPU 核心。视负载类型而定，这个特性可能提升整体性能，也可能不会。这项技术的发明者 Intel 给出的说法是，在某些情况下可能有 30% 的性能提升。

另见[是否启用超线程](https://github.com/CodingPracticeGroup/ml-engineering/blob/master/orchestration/slurm/performance.md#to-enable-hyper-threads-or-not)。
