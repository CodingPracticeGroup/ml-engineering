# 训练大语言模型的经验教训（一门课程） {#lessons-learned-from-training-llms-a-course}

![机器学习工程——训练大语言模型的经验教训](lesson-learned-cover.png)

一份精心整理的真实“实战故事”及其解决办法的合集，基于 Stas Bekman 的两本开源书籍：

- [Machine Learning Engineering (MLE)](../../index.md)
- [The Art of Debugging (AoD)](https://github.com/stas00/the-art-of-debugging)

下面每一条都是在实际大规模训练和运行大语言模型/图像模型过程中换来的、来之不易的经验教训，以及随之而来的硬件、存储、网络和调试方面的种种意外：

- BLOOM-176B（2022）
- IDEFICS-80B（2023）
- 在 Contextual.AI 的生产级 RAG/RL 系统（2024）
- 在 Snowflake 开展的各种 Arctic Training 开源工作（2025-2026）

每条经验的加粗标题是入口，其下的支撑事实被拆分成便于快速扫读的要点列表，而不是连贯的段落，这样你可以迅速浏览“是什么/为什么/怎么做”，想深入了解时再点开链接看完整故事。

这份材料是作为一门课程来设计的：读标题来定位自己感兴趣的领域，略读子要点了解背后的推理，如果还想要更多细节，顺着“延伸阅读”链接深入查看细节、脚本和基准测试。

你可以把它当作自学课程，也可以自己拿去教别人！

## 章节 {#chapters}

<!-- no toc -->
1. [存储与文件系统](#1-storage-and-file-systems)
2. [网络与互联](#2-network-and-interconnects)
3. [计算、加速器与内存](#3-compute-accelerators-and-memory)
4. [训练性能与吞吐量](#4-training-performance-and-throughput)
5. [数据加载器](#5-dataloaders)
6. [故障容错与可靠性](#6-fault-tolerance-and-reliability)
7. [训练稳定性](#7-training-stability)
8. [云服务商与采购](#8-cloud-providers-and-procurement)
9. [调试方法论](#9-debugging-methodology)

## 1. 存储与文件系统 {#1-storage-and-file-systems}

- 1.1. **快速并行存储是快速保存模型检查点（checkpoint）的决定性因素。**
  - GPFS、WekaIO 这类分布式并行文件系统，允许数百到数千个客户端同时访问同一存储而不产生热点——这带来的差别是巨大的。
  - **实例：** 在 JeanZay HPC（2021 年），一次 2.3TB 的检查点由 384 个进程并行写入，仅耗时 40 秒，因为该集群在 NVMe 盘上运行 GPFS。
  - “NVMe 上的 GPFS”与“普通云上的 NFS 挂载”之间的差距，按实际耗时（wall-clock time）计可以达到数倍之多。
  - **要点：** 针对你自己的具体负载给你的具体文件系统跑基准测试，弄清其中的开销。

  📖 [该选哪种文件系统](../../storage/README.md#which-file-system-to-choose)

- 1.2. **慢速共享存储会在不知不觉中扼杀开发者的生产力，而不仅仅是训练吞吐量。**
  - 在 GCP 的 Zonal FileStore（基于 NFS）上，一句简单的 `python -c "import torch"` 在无缓存时要花 20 秒（有缓存时约 2 秒），安装一个只含少量预编译软件包的 conda 环境可能要花 20-30 分钟。
  - 这种拖慢根本不会出现在训练吞吐量基准测试里，但它日复一日地消磨着每一位开发者，再乘以整个团队。
  - **解决办法：** 在对任何存储方案做出承诺之前，先跑一个“感知基准测试”（给 `import torch`、`conda create`、`git status`、`pytest --collect-only` 计时），并与本地 NVMe 基线对比。
  - 如果候选的共享文件系统只比你现有的快 2 倍，多半不值得切换——要争取达到比现状快 5-10 倍、接近本地 NVMe 速度的水平，它才能不再成为日常的烦恼来源。
  - 开始使用 `uv`——比 `pip install` 快 5-50 倍。

  📖 [该选哪种文件系统](../../storage/README.md#which-file-system-to-choose)

- 1.3. **Python 产生的数以万计的小文件，会在真正的磁盘空间耗尽之前先把 inode（索引节点）耗尽。**
  - 如果一个文件系统分区使用很大的块大小（例如 16MiB，为大文件优化的 GPFS 分区常见如此），而你的文件平均只有 16KiB，那么每个文件最多会浪费 1,024 倍的空间——更重要的是，消耗的是 inode 配额而不是字节配额。
  - JeanZay HPC 上发生的正是这种情况：conda 每安装一个环境就会产生数万个约 4KiB 的小文件，于是团队在普通 GPFS 分区上不断耗尽 inode，最后不得不申请一个专门存放 conda 环境的分区。
  - **解决办法：** 至少保留两个调优方向不同的分区——一个用小的（4-8KiB）块大小，存放数百万个小文件（venv、包缓存）；另一个用大的（2-16MiB）块大小，存放少数巨型文件（检查点、数据集）。现代 GPFS 的子块（sub-block）机制通过把小文件打包在一起，也能起到帮助。
  - 查看 `df -h` 时总要顺带检查 `df -i`，因为分区可能报告仍有大量剩余空间，实际上却已完全耗尽了可用的 inode。

  📖 [文件块大小](../../storage/README.md#file-block-size)

- 1.4. **训练期间直接从云存储加载数据是痛苦的，应尽一切可能避免。**
  - IDEFICS-80B 训练期间，本地磁盘空间放不下数 TB 的多模态数据集，团队不得不从云存储流式读取——为此花了好几周想让这个方案变得可靠，却始终没能真正做到。
  - **核心问题：** 当时的流式数据加载器（DataLoader）在恢复训练时不会保留 `DataSampler` 的随机数生成器（RNG）状态，于是每一轮崩溃-恢复要么跳过数据、要么重复数据——整个训练没有得到过一个完全干净、数据真正唯一的 epoch。
  - **经验：** 优先准备足够的本地磁盘用于数据加载，只把检查点异步卸载到云端（通过 crontab 或 SLURM 作业），并始终在本地保留最近几个检查点以便快速恢复。
  - 如果必须从云端流式读取，要确保你的具体方案能够精确恢复——不丢失、不重复数据，也不需要在本机启动数量庞大的数据加载器工作进程。

  📖 [本地存储优于云存储](../../storage/README.md#local-storage-beats-cloud-storage)


## 2. 网络与互联 {#2-network-and-interconnects}

- 2.1. **在拿一个带宽数字与实测值比较之前，务必先确认它是单向还是双向（双工）的。**
  - 大多数基准测试工具报告的是单向带宽，而厂商宣传几乎总是倾向于给出双工数字，因为它大约大一倍，看起来更漂亮。
  - **经验法则：** 如果你的实测带宽约为标称规格的 40%，那个标称值很可能是双工的——把它折半，你的真实效率就是常见的 ~80%。
  - **例子：** 一台标称节点内 600GBps 的 A100 节点，实测 235GBps 看起来只有令人失望的 40% 效率，直到与正确的 300GBps 单向规格比较，才确认约为 ~80%。
  - 同样的情况今天在 H200 上再次重演：NVLink 标称 900GBps（双工），而一次 all-reduce 基准测试达到 ~376GBps，对照 450GBps 的单向规格正好吻合。

  📖 [单向与双向（双工）](../../network/README.md#unidirectional-vs-bidirectional-duplex)

- 2.2. **缓慢的节点间网络可能左右你整个训练框架的选择，而不仅仅是限制吞吐量上限。**
  - 在 JeanZay HPC，Omni-Path 互联只有 135Gbps。
  - 仅凭这一点，BLOOM-176B 团队就被迫选择 Megatron-DeepSpeed（张量并行 + 流水线并行，通过网络传输的数据少得多），而不是简单得多的 DeepSpeed ZeRO（后者要对优化器状态和梯度做分片，需要高得多的带宽）。
  - 一旦持续数月的训练已经开始，这类决定几乎不可能回头修改——必须在项目一开始就了解网络速度，并将其纳入并行策略的考量。
  - 如果你被迫使用缓慢的节点间网络，优先选择只交换激活的并行策略（流水线/张量并行），而不是交换完整梯度和优化器状态的策略（ZeRO-DP/FSDP）。

  📖 [Omni-Path](../../network/README.md#omni-path)

- 2.3. **有时 all-reduce 基准测试报告的数字似乎超出了物理链路的理论上限，而这并不是测量错误——是 SHARP 在起作用。**
  - 在一台 H100 节点上，节点内 NVLink 4.0 的 all-reduce 对 4GiB 负载报告了 480GBps——而单向规格“只有” 450GBps。
  - **解释：** NVIDIA 的 SHARP 协议让 NCCL 可以直接在网络交换机硬件上完成归约。一旦 NCCL 检测到 InfiniBand/NVSwitch 的 SHARP 支持，就会切换到 `NVLS` 算法，它只需要 `N+1` 次发送，而不是通常的 `2N` 次。
  - 这也意味着，标准的 `busbw` 计算（假设 `2N` 模型）在 `NVLS`/SHARP 启用后会低估真实效率——看起来“好得离谱”的数字是更聪明的算法在起作用，而不是测量出了错。
  - 要获得这种加速，参与集合通信（collective）的 GPU 需要超过 4 块，并且环境变量 `NCCL_NVLS_ENABLE` 未被禁用。
  - SHARP 只适用于 `all-reduce`。整节点的 `all-gather` 或 `reduce-scatter` 仍走环形路径，落在单向规格常规的 ~80% 水平。

  📖 [SHARP](../../network/README.md#sharp)

- 2.4. **共享的节点间网络会让性能优化几乎无法进行，因为你分不清变化从何而来——是你自己的代码，还是“吵闹的邻居”（同网络上的其他用户作业）。**
  - 在 JeanZay HPC，BLOOM-176B 开始之前，完全相同的配置每一次运行都会产生明显不同的吞吐量，原因只是其他用户的作业以不可预知的强度共享同一个网络结构（fabric）。
  - 对认真的调优工作来说这是死胡同：如果两次运行之间环境噪声水平本身就有两位数的波动，你就无法把吞吐量变化归因于自己的改动。
  - **解决办法：** 在 BLOOM-176B 启动前不久，团队获准在（当时全新的）A100 上使用一个独占、隔离的 SLURM 分区，优化工作从此才变得可行。
  - 如果你在做认真的预训练性能工作，务必在基准测试阶段要求一个专用、隔离的网络分区。

  📖 [共享的节点间网络](../../network/README.md#shared-internode-network) + [至关重要的可复现性要求](../../network/benchmarks/README.md#crucial-reproducibility-requirements)

## 3. 计算、加速器与内存 {#3-compute-accelerators-and-memory}

- 3.1. **一致性（统一）内存硬件的行为确实难以预测，而解决办法是驱动层面的模式切换，不是改代码。**
  - 在 NVIDIA 的一致性内存平台（DGX Station、DGX Spark、GH200、GB300）上，当 CPU 内存耗尽时，Linux 可以透明地向 GPU 的 HBM 借用内存——但不能反向借用——而且 `nvidia-smi` 甚至可能不会报告这种借用。
  - 在 DGX Station 上为 Qwen3-32B 的后训练调优最大序列长度时（优化器状态被卸载到 CPU 内存），团队遇到了令人抓狂的 CPU-OOM 与 GPU-OOM 看似随机的交替出现，具体取决于当时恰好在运行哪些后台守护进程。
  - **解决办法**根本不是改代码，而是一个系统设置：通过 `/etc/modprobe.d/nvidia-openrm.conf` 并重启，把默认的 NUMA 一致性内存模式切换到 CDMM（“驱动”）模式，此后 CPU 与 GPU 内存就不再悄悄互相借用了。
  - 如果你在一致性内存的 NVIDIA 硬件上看到无法解释、看似非确定性的 OOM，先去检查 `CoherentGPUMemoryMode` 设置，再去调试你的训练脚本。

  📖 [克服一致性内存的不确定行为](../../debug/pytorch.md#overcoming-the-coherent-memory-uncertain-behavior)

- 3.2. **晦涩难懂的异步 CUDA 错误有时需要把“调试”开关永久打开，而这个开关有可能是免费的。**
  - CUDA 默认是异步的，因此等到错误显现时，程序的上下文早已往前走了——产生的堆栈跟踪指不到任何有用的地方，或者 NCCL 的报错信息干脆拒绝给出堆栈跟踪。
  - `CUDA_LAUNCH_BLOCKING=1` 强制同步执行，通常要付出真实的性能代价，但它也意味着出错时你终于能得到一个准确的、贴合上下文的堆栈跟踪。
  - BLOOM-176B 的*整个*训练过程都带着这个开关运行（因为否则训练会因为一个诊断不出原因的原因而挂起），出人意料的是，吞吐量没有受到任何可测量的影响。
  - **经验：** 在没有针对你的具体情况做过测量之前，不要假定调试开关一定会带来性能税——它也可能就是问题的解药。

  📖 [应对异步 CUDA 错误](../../debug/pytorch.md#dealing-with-async-cuda-bugs)

- 3.3. **内存剖析器能抓到代码审查中原本几乎看不见的真实内存泄漏。**
  - 在实现一个棘手的自定义 `TiledMLP` `torch.autograd.Function` 时，泄漏就藏在这份实现里，只有借 PyTorch 的内存剖析器（`torch.cuda.memory._record_memory_history()` + `_dump_snapshot()`，在 pytorch.org/memory_viz 查看）把它可视化之后才一目了然。
  - 大多数内存分配在图中表现为开始后又释放的条块。泄漏的张量则表现为连续的条块：从某一层的正向计算开始就再也没有结束，并且每一个后续层都如此重复。
  - 你可以点击任意条块，得到定位到确切分配代码行的 Python 级堆栈跟踪，几分钟内就能把“某处有泄漏”变成“某文件的第 X 行”。
  - 除了找泄漏，同一个工具还非常适合直观比较“同一”算法两种实现随时间使用 GPU 显存方式的差异（例如普通的激活检查点，即激活重计算，与卸载到 CPU 的激活检查点）。

  📖 [PyTorch 内存剖析器](../../debug/pytorch.md#pytorch-memory-profiler)

- 3.4. **策略性、精准放置的内存分配追踪也能发现框架的 bug，而不仅是你自己代码中的 bug。**
  - 能够明确控制*何时*采样 GPU/CPU 内存（在关键位置调用一次“立即采样”，通过一个源自 DeepSpeed `see_memory_usage` 的辅助函数实现），可以定位出被动剖析器可能漏掉的泄漏。
  - 借助这一技术，策略性追踪暴露出 PyTorch 自身 `all_gather_object` 内部的一个真实泄漏（后来已修复，同期还用同样的方法发现了其他几个 PyTorch 内部的泄漏）。
  - **细节：** 释放一个 Python 变量并不一定立刻释放张量，因为 Python 垃圾回收按自己的节奏运行。采样前先调用 `gc.collect()`，否则你会看到“仍然占用”的假象数字。
  - 在生产环境中必须关闭这套工具，否则会带来糟糕的性能。

  📖 [策略性内存分配追踪](../../debug/pytorch.md#strategic-memory-allocation-tracing)

- 3.5. **在某些硬件上，超线程（Hyper-Threading）可能严重破坏网络吞吐量，而唯一的判断方法就是在你的具体配置上把两种设置都实测一遍。**
  - 在 AWS p4 节点上，开启超线程（通常被视为“免费”的额外 CPU 并行能力）会让网络吞吐量降到关闭超线程时的大约四分之一。
  - 正是由于这一具体的实测结果，那套环境从此刻意保持超线程关闭，而不是沿用（默认往往开启超线程的）SLURM 配置。
  - 这件事在两个方向上都不普遍——在某些*其他*环境上恰恰相反，关闭超线程（`--hint=nomultithread`）才是拖累吞吐量的原因。
  - **唯一可靠的办法：** 在你的具体硬件上，分别在开启和关闭超线程时对你的真实负载做网络吞吐量基准测试，凭实测结果择优。

  📖 [要不要开启超线程](../../orchestration/slurm/performance.md#to-enable-hyper-threads-or-not)

## 4. 训练性能与吞吐量 {#4-training-performance-and-throughput}

- 4.1. **规格表给出的是无法达到的数字——一定要在自己的硬件上对 matmul 跑基准测试。**
  - 理论峰值 TFLOPS 假设矩阵形状完美、没有内存搬运开销、加速时钟频率可以无限维持——这些在真实训练中一个都不会发生。
  - **例子：** NVIDIA B200 SXM 在 BF16 下可能只能从 2250TFLOPS 的理论规格达到 ~1745.0TFLOPS（约 77.6% 的效率），而这个数字本身还取决于软件栈版本和具体的矩阵形状。
  - 这正是“最大可达 matmul FLOPS”（Maximum Achievable Matmul FLOPS，MAMF）这一指标独立于理论规格而存在的原因：它用 `mamf-finder.py` 在你的真实硬件/软件上暴力搜索 matmul 形状得到，为你提供一个可以据此优化的现实上限。
  - 一旦你实测的训练 TFLOPS 接近你自己的 MAMF 数字（而不是理论峰值），就该停止优化、开始训练了。

  📖 [最大可达与可持续 matmul FLOPS 对比表](../../compute/accelerator/README.md#maximum-achievable-and-sustainable-matmul-flops-comparison-table)

- 4.2. **吞吐量优化是一段收益递减的旅程，而不是一个可以立刻抵达的固定终点。**
  - 在 BLOOM-176B 的调优阶段，团队起点低于 100TFLOPS，经过几周的努力，到训练启动时提高到 150TFLOPS——没有任何一招制敌的银弹，只是一点点小胜利的累积。
  - **知道何时收手：** 当相对于当时同类环境所能达到的水平已经没有有意义的提升空间时（2022 年 80GB A100 的上限约为 155TFLOPS），团队就刻意停止优化，开始训练。
  - 在正式启动之前，为这个调优阶段明确预留日历时间——把它当成事后补充，就意味着要么以低效率浪费几个月的算力，要么为了把配置“打磨到完美”而无休止地推迟启动。

  📖 [把 TFLOPS 作为性能指标](../../training/performance/README.md#tflops-as-a-performance-metric)

- 4.3. **全局批次大小（Global Batch Size，GBS）的爬坡远比直觉认为的重要，配合流水线并行时尤其如此。**
  - 对 BLOOM-176B 来说，起始值太小是灾难性的：GBS=16 实测只有 8TFLOPS，因为在流水线并行下，小的 GBS 会产生占比很大的“流水线气泡”，即空闲的 GPU 时间。
  - 团队改为从 GBS=192（73TFLOPS）起步，每处理约 977 万个样本就增加 16，最终达到 GBS=2048、150TFLOPS。
  - **为什么要爬坡：** 训练早期本质上是随机的，还享受不到大批次、高精炼批次的好处，所以在那个阶段，也没有必要为小批次付出流水线气泡的代价。
  - 如果你使用流水线并行，在确定爬坡计划之前，先在几个候选的起始 GBS 值上实测你能达到的 TFLOPS。

  📖 [全局批次大小爬坡](../../training/hparams.md#global-batch-size-ramp-up)

## 5. 数据加载器 {#5-dataloaders}

- 5.1. **同一进程中的多个数据加载器可能随机挂起，而解决办法也许干脆就是不用其中一个。**
  - 在 BLOOM-176B 期间，训练数据加载器在 `num_workers=2` 下运行正常，但同时在加一个评估数据加载器后开始出现随机挂起——这是一类已知的、长期未解决的 PyTorch 多 worker 数据加载器 bug。
  - **解决办法：** 专门把评估加载器的 `num_workers` 设为 0（由于评估运行的频率低得多，可以接受更慢的阻塞式 IO），并最终完全去掉训练中的评估，改为在已保存的检查点上做异步的 lm-harness 风格评估。
  - **额外收获：** 去掉循环内的评估还让主训练循环变快了，因为它再也不用和第二个数据加载器共享任何资源。

  📖 [异步数据加载器](../../training/performance/README.md#asynchronous-dataloader)

- 5.2. **如果每个 worker 的内存占用很高而 CPU 内存有限，流式数据加载器会悄悄成为你的硬性瓶颈。**
  - 在 IDEFICS-80B 训练期间，多模态数据集的流式数据加载器每个 worker 都消耗巨大且不断尖峰的内存，而每台节点只有 ~1TiB CPU 内存，团队无法启动足够多的 worker 来喂饱流水线。
  - 与 GPU 显存瓶颈（通常会立刻抛出明确错误）不同，供给不足的数据加载器只会悄悄地把吞吐量压在加速卡能力之下，不一定让任何东西崩溃。
  - 当时没有时间去工程化一个更好的方案，于是团队就接受这个受数据加载器限制的上限，跑完了整个训练。
  - 如果你使用多模态或其他内存需求大的流式流水线，要为数据加载器 worker 仔细规划 CPU 内存余量，就像你为模型仔细规划 GPU 显存一样。

  📖 [异步数据加载器](../../training/performance/README.md#asynchronous-dataloader)

## 6. 故障容错与可靠性 {#6-fault-tolerance-and-reliability}

- 6.1. **警惕刚出厂的新 GPU：新一批加速卡常常有 3-10% 的故障率，而对一个全新世代的 GPU 来说，最早出厂的那些更严重。**
  - 这是一个众所周知的规律，因此才有“老化测试（burn-in）”这类做法，在硬件到达客户手中之前把它拦下来：OEM 通常会做 3-4 周的老化测试，云服务商随后再做 2-3 天的验收测试——尽管并非每家服务商都做得严格（或者说明他们是否做过）。
  - **标准应对措施：** 规划备用/冗余节点（多出 5-10% 不算过分），并确保你的调度器无需人工干预就能自动隔离（drain）并替换故障节点。
  - 一旦你花了几周时间从坏加速卡中筛出好的，就要坚持要求**固定**分配，让那些好节点在重启后仍归你使用，而不是回到共享池中。
  - 值得明确询问任何云服务商：你具体使用的节点在你接手之前做了多长时间、多严格的老化测试——只说“我们做过老化测试”而不给时长，不是有用的回答。

  📖 [永远要规划比需求更多的节点](../../training/fault-tolerance/README.md#always-plan-to-have-more-nodes-than-needed) + [加速卡需要做老化测试](../../insights/how-to-choose-cloud-provider.md#accelerators-need-to-be-burned-in) + [确保重启后仍保有你的好加速卡](../../insights/how-to-choose-cloud-provider.md#ensure-you-keep-your-good-accelerators-on-reboot)

- 6.2. **检查点开销与你的 IO 速度成反比。**
  - BLOOM-176B 的保存检查点很快：2.3TB 只要 40 秒，在约 3 个月的训练中每 ~3 小时保存一次——大约 720 个检查点，总共只有 ~8 小时开销，占实际耗时的比例仅为 ~0.37%。
  - 同样的保存频率，若 IO 慢 5 倍（高级与标准云存储层级之间的常见差距），开销就会膨胀到约占训练总时长的 2%——在大规模训练中这是一笔确实可观的成本。
  - **算术：** 实测一次保存的耗时，乘以你打算保存的次数，再把总和与预期的训练总时长做比较。
  - 如果必须把检查点从节点卸载到云端，务必在本地保留最近两个检查点以便快速恢复，因为最新的那一个可能因保存中途崩溃而损坏。

  📖 [频繁保存检查点](../../training/fault-tolerance/README.md#frequent-checkpoint-saving)

- 6.3. **做一个终止开关（kill-switch），因为 SLURM 没有“作业组所有权”的概念，这在实践中是一个真实的运维问题。**
  - 在 HPC 环境中，如果同事启动了一个长作业（或一个 SLURM 作业数组）就去睡觉了，没有 `sudo` 权限你根本无法停止或修改它——SLURM 的权限模型按用户划分，而不是按团队。
  - **BLOOM-176B 的办法：** 一个轮询文件的终止开关。训练循环在每次新的迭代开始前检查一个特定文件是否存在，若存在就保存一个检查点并干净地退出，从而让*任何*团队成员都能介入。
  - 在 `main()` 的最开头再加一次轮询，意味着睡觉同事留下的排队长作业数组可以几乎瞬间被“烧穿”，而不是每个作业都先跑满各自的完整时长。
  - 代码量很小，带来的运行安全性却很大——值得从第一天起就内置到任何共享的多用户训练环境中。

  📖 [终止开关](../../training/fault-tolerance/README.md#kill-switch)

- 6.4. **要看着那些“看守”：真正能抓住“训练悄悄停了却没人发现”的，是一个定时运行的作业状态看门狗。**
  - 在 BLOOM-176B 期间，一个 `slurm-status.py` 脚本大约每小时运行一次（通过 SLURM 上的 crontab 模拟实现，因为那里通常没有真正的 `crontab`），检查指定作业是在运行、或至少还在队列中。
  - 如果两个条件都不满足，它就发送告警邮件，并把这次检查写入主训练日志以留下记录。
  - 运行它的成本几乎为零，却能拦住一整类代价高昂的故障（算力被悄悄闲置几小时或几天），否则这类问题只有人碰巧手动检查时才会暴露。
  - **原则：** 不要只构建训练循环——还要构建看守训练循环的那个东西。

  📖 [作业是否在运行：看门狗](../../training/fault-tolerance/README.md#is-job-running-watchdog)

- 6.5. **积极地监控磁盘用量（以及 inode 用量），因为大量写检查点的训练消耗存储的速度远比直觉快。**
  - BLOOM-176B 保存检查点*每天*消耗约 18.4TB 磁盘，在还有其他作业也在写数据的共享集群上，这种消耗速度可以很快烧光一份“应该绰绰有余”的配额。
  - `fs-watchdog.py` 工具同时监控每个相关分区的原始磁盘使用百分比*和* inode 用量，因为这是“空间”耗尽的两条彼此独立的途径。
  - 告警阈值需要按分区分别调优，而不是设成一个一刀切的百分比——重要的是把百分比换算成“按我目前每天的消耗速度还能撑几天”。
  - 这就是把“凌晨 3 点磁盘满了，训练废了”变成“三天前就收到了邮件，及时扩充了存储”的关键。

  📖 [磁盘空间不足告警](../../training/fault-tolerance/README.md#low-disk-space-alerts)

- 6.6. **在某些分布式文件系统上，80% 可能就是新的 100%，所以 `df` 显示还有剩余空间并不是可靠的安全保证。**
  - 分布式存储由许多物理磁盘组成（每块通常只有 0.3-2TB）；如果重平衡做得不够频繁，任何单块磁盘都可能被写满到 100% 并拒绝写入，尽管从汇总的 `df` 来看使用率还“舒适地”停在 80%。
  - IDEFICS-80B 训练期间遇到的正是这种故障模式，在那套文件系统上，80% 真的成了实际的上限。
  - 这因技术而异：Lustre 已知有这个问题（80% 才算可靠），而 GPFS 服务商声称他们没有。
  - **稳妥做法：** 询问你的存储服务商，他们认为名义容量中可靠可用的百分比是多少，并把这个真实上限纳入资源规划。

  📖 [磁盘空间不足告警](../../training/fault-tolerance/README.md#low-disk-space-alerts)

- 6.7. **有时正确的工程决策是绕过泄漏，而不是把它追到根因，尤其是在时间紧迫的时候。**
  - 在 IDEFICS-80B 期间，一个缓慢的 CPU 内存泄漏要花好几天才积累到 OOM 崩溃——这种时间尺度让定位根因的迭代慢得令人痛苦。
  - 团队没有停下训练去追查源头，而是构建了一个循环内的看门狗，监控常驻内存，一旦越过阈值就主动退出，让下一个排队的作业在内存已回收的状态下恢复。
  - **具体模式：** 每次迭代检查一次 `psutil.virtual_memory().percent`（或在 `gc.collect()` + `torch.cuda.empty_cache()` 之后用 `torch.cuda.mem_get_info()` 检查 GPU 侧的对应值），越过阈值就提前退出——这是花小钱买保险，不管你当前是否怀疑有泄漏，都值得加上。

  📖 [应对缓慢的内存泄漏](../../training/fault-tolerance/README.md#dealing-with-slow-memory-leaks)

- 6.8. **优雅地处理被强制抢占：确切知道你的程序“保存并退出”需要多少时间余量，并把这个数字提前告诉它。**
  - 共享 HPC 集群上的 SLURM 作业有硬性时限（常见的是约 20 小时，尽管 BLOOM-176B 的作业最长跑到 100 小时），如果任由作业被杀掉，就会浪费自上一个检查点以来跑的步数。
  - **BLOOM-176B 的方案 A**（通过 `--exit-duration-in-mins`）：在启动时明确告诉程序它还有多少分钟。它启动一个内部计时器，并在每次迭代开始前检查（通过一个在所有 rank 之间由 `all_reduce` 同步的标志）时间预算是否已超——若已超，就在 SLURM 硬杀之前保存并干净退出。
  - **预算定多大：** 知道一次迭代耗时多久、保存一次检查点耗时多久，把两者相加再翻倍作为安全余量——提前退出不损失什么，退出晚了则赔掉整个检查点。
  - **方案 B：** 基于信号的方法（`--signal=B:USR1@<seconds>`，直接捕获信号，或在使用 `torchrun`/`accelerate`/`deepspeed` 时通过启动器转发信号），适用于无法事先干净地算出固定时间预算的情况。

  📖 [方案 A：在启动时告诉程序何时开始退出](../../training/fault-tolerance/README.md#approach-a-tell-the-program-at-launch-time-when-it-should-start-the-exiting-process)

## 7. 训练稳定性 {#7-training-stability}

- 7.1. **初始化 std 对稳定性的影响大得惊人，取值不当可能让一个大模型干脆挺不过最初的几千个迭代。**
  - 在 BLOOM 之前的 104B 实验中，团队始终无法突破训练早期阶段，直到他们意识到当时 Megatron-LM 默认的 `--init-method-std` 0.02 对那个隐藏层尺寸（hidden size）的模型来说大得太多。
  - 团队考虑了两个已发表的公式：“Transformers without Tears”（`sqrt(2/(NHIDDEN*5))`）和 530B 论文的公式（`sqrt(1/(NHIDDEN*3))`）——团队刻意选择更小的那个（530B）取值，因为它更保守。
  - 对 BLOOM-176B 的 `NHIDDEN=14336`，该公式给出 `std=0.00482`——一个微小、容易写错的数字，也正是 176B 训练从未遭受完全发散的关键原因之一。
  - **经验：** 初始化 std 应当按已发表的公式随隐藏层尺寸缩放，而不是作为固定常数从小模型的配置里照抄过来。

  📖 [初始化 STD](../../training/instabilities/README.md#std-init)

- 7.2. **从失败的训练中学习，而不是直接放弃它：BLOOM 之前 104B 实验的反复发散，直接催生了让 BLOOM-176B 成功的那套配方。**
  - 那些 104B 的尝试总是在早期就发散，事后分析指向两个嫌疑对象：fp16（动态范围更窄，容易上溢）以及包含大量未经清洗的网页抓取垃圾的训练数据。
  - 对于 BLOOM-176B，团队一次改了三件事：从 fp16 切换到 bf16、使用干净得多的数据、加入一个嵌入层 LayerNorm（对嵌入输出做层归一化）。这套组合“造成了决定性的差别”，得到一条几乎完全平滑的损失曲线（一次尖峰，200 步内恢复）。
  - 重点不只是“用 bf16”——一次真正失败了数月之久的实验，只要认真分析而不是弃之不顾，就直接产出了下一次更大尝试的修复方案。

  📖 [一次极其失败的训练](../../training/instabilities/training-loss-patterns.md#a-very-failed-training)
## 8. 云服务商与采购 {#8-cloud-providers-and-procurement}

- 8.1. **一通电话上到场的支持人员数量多，并不等于真正的支持，这一区别值得明确写进合同里谈下来。**
  - "我们会尽力"这种模式，往往意味着十几个人参加问题升级会议，却都没有真正解决该问题的能力——可能要几周甚至几个月（有私人关系会快一些）才能触达真正关键的两个人：一位产品经理、一位负责该问题的工程师。
  - 由于客户在法律上有义务持续付费，而且通常无法单方面解除合同，"尽力而为"这种承诺是不对等的——除非合同中针对每个关键组件都规定了具体的、可量化的交付指标。
  - **最值得谈下来的补救条款：**在约定的问题修复之前暂停付款义务——这是唯一能真正给对方创造动力、优先去修复问题的杠杆。
  - 这一点要在*签约之前*谈——事后再谈基本没有任何筹码。

  📖 ["我们会尽力"条款](../../insights/how-to-choose-cloud-provider.md#we-will-do-our-best-clause)

- 8.2. **出故障的绝大多数是加速卡（accelerator）——集群上约 95% 的硬件故障是加速卡本身造成的，而不是 CPU、存储或网络设备。**
  - 这一数字应当直接影响故障容错（fault tolerance）方面的规划：在存储/网络的韧性上与在 GPU 故障容错上投入同等的工程精力并不划算。
  - **实际做法：**建立一条快速的、最好是完全自动化的更换通道（通过 API，而不是提交支持工单），并明确询问服务商保留多少备机、承诺多快更换故障设备。
  - 由于"即时更换"几乎从未真正可得，标准的缓解措施是花钱购买比实际需要多约 10% 的节点，多出来的节点可以用于日常开发工作，并在生产节点出故障时充当即时替补。
  - 这种超量配置（overprovisioning）确实要花钱，但与动辄几百块 GPU 的作业停下来干等一张支持工单相比，这笔开销是便宜的。

  📖 [如何应对加速卡故障](../../insights/how-to-choose-cloud-provider.md#dealing-with-accelerator-failures)

- 8.3. **网络软件可能悄悄占用卖给你、你正在付费的那部分加速卡显存，而且事先没有人会告诉你。**
  - 在某个云上，还没等任何用户作业开始，服务商的网络软件就已经占用了*每块* GPU 显存中的约 5GiB——这是签约之后才发现的，销售阶段并未披露。
  - 随着加速卡显存越来越大（H200 有 141GiB，B200 可用 180GiB），固定的 5GiB 开销所占比例越来越小，但在 80GiB 的 H100 上，这是一个可观的比例，足以决定某个批次大小是放得下还是会 OOM。
  - **可执行的措施：**在签约之前问清楚服务商的软件栈到底保留了多少 GPU 显存、实际可用的是多少，并让这个数字以书面形式确认。

  📖 [网络会占用加速卡显存吗？](../../insights/how-to-choose-cloud-provider.md#does-the-network-steal-from-the-accelerator-memory)

- 8.4. **服务商的软件栈可能把你困在陈旧、不再受支持的操作系统版本上，原因仅仅是他们自己的内部工具还没跟上。**
  - 不止一次，服务商要求的驱动/编排/监控软件栈根本不支持某个较旧版本之后的任何 Ubuntu 版本，导致在一套现代机器学习栈下面被迫运行一个过时的操作系统。
  - 评估阶段很容易忽略这一点（没有人会演示"我能运行哪个 Ubuntu 版本"），但之后会变成实实在在的阻力：安全补丁、内核特性、驱动程序以及某些机器学习框架的依赖，都可能默认需要更新的基础操作系统。
  - **缓解措施：**在评估阶段（而不是签约之后）就问清楚支持并要求哪些操作系统/软件版本，并让对方承诺多快支持更新的版本。

  📖 [最新的软件/操作系统版本](../../insights/how-to-choose-cloud-provider.md#up-to-date-softwareos-versions)


## 9. 调试方法论 {#9-debugging-methodology}

以下内容引自 [The Art of Debugging](https://github.com/stas00/the-art-of-debugging)：

- 9.1. **把出错的 import 或组件单独抽成一条一行命令，而不是在一个庞大、启动缓慢的程序内部调试它。**
  - 如果一个大程序在启动期间失败（例如 `import torch` 失败，或加载模型/分词器/配置因为缺少 HF 认证令牌而失败），就恰好把那一步操作抽出来，做成一条独立的一行命令，直接在其中复现问题。
  - 这一点很重要，因为快速的调试循环和足够小的复现载荷是成功调试最重要的两个前提——每次尝试要等十分钟、试三十次，你会完全记不清自己到底验证过哪些假设。
  - 一行命令，比如 `python -c "import torch"`，还具有原子性（可以在 shell 历史中搜索），可以直接复制粘贴到远程机器上运行，也容易整理成一个包含各种可复用变体的文件。
  - 同样的技巧也可以用来在其他开发工作进行的同时预下载/缓存大模型或数据集：`python -c 'import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])' t5-small`

  📖 [一行命令的威力](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#the-power-of-one-liner-programs)

- 9.2. **用二分法（bisect）找出恰好引入问题的那次提交，而不是用眼睛手动翻检几百次提交。**
  - 常见场景：原本锁定在某个已知可用的发布版本上，为了一个新特性升级到最前沿的 `main` 分支，然后中间几百次提交里的某处出了问题。
  - `git bisect` 做的是经典的二分查找：给它一个已知可用的提交和一个已知有问题的提交，测试中间点，标记 `good`/`bad`，然后重复——1,024 个候选提交只需大约 10 轮测试就能收敛。
  - 复杂情况也有标准处理方式：对无法测试的提交使用 `git bisect skip`；当要找的是修复某问题的提交、而不是引入破坏的提交时，使用自定义术语（`--term-new=fixed --term-old=broken`）。
  - 一旦收敛，第一个坏提交对应的 GitHub 页面几乎总是链接到引入它的那个 PR——马上就能到一个知情的地方去求助。

  📖 [通过二分版本查找引入破坏的提交](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#finding-a-breaking-commit-by-bisecting-revisions)

- 9.3. **从核心转储文件（core file）中获取栈回溯（backtrace），把一大段毫无用处的跟踪信息变成确切、可操作的代码行。**
  - 当编译型扩展（CUDA 内核、NCCL、任何底层是 C/C++ 的库）发生段错误（segfault）时，Python 根本给不出有意义的异常——你要么什么都拿不到，要么只有一段读不懂的原生转储。
  - **解决办法：**执行 `gdb -c /path/to/core /path/to/program`，然后输入 `bt`，它会遍历崩溃那一刻的调用栈，并给出确切的源代码行。`bt full` 还会额外显示局部变量的值（例如某个指针显示为 `0x0`，说明它在未初始化的情况下就被解引用了）。
  - 对于多线程程序（基本上所有正经的 PyTorch/CUDA 程序都是），`thread apply all bt` 会显示所有线程的栈，因为真正的根因有时并不在崩溃的那个线程里，而在另一个线程中。
  - PyTorch 真实崩溃报告的根因定位正是这样做的：把一大段原生转储信息变成一行可操作的信息。

  📖 [从核心转储文件获取栈回溯](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#get-the-backtrace-from-the-core-file)

- 9.4. **直接对活着但已挂起的进程取栈回溯，而不是等它自己崩溃或超时。**
  - 有时问题是死锁而不是段错误——进程还活着，但始终没有任何进展，因此不会发生能触发核心转储的崩溃事件。
  - `gdb --pid=<pid>` 会附加到一个已在运行的进程上，而不必重启它，`thread apply all bt` 会显示每个线程正在做什么——曾经有一起真实的 PyTorch 死锁就是这样诊断出来的。
  - 如果你想保留现场以便稍后分析（或者立即释放资源），可以用 `gcore <pid>` 或 `kill -ABRT <pid>` 强制生成核心转储，不过这两种做法也都会终止该进程。
  - 这种"实时附加"技巧用于空转或完全挂起的进程；对于已经崩溃的进程，则使用上面介绍的核心转储文件技巧。
  - 如果它是在 Python 内部挂起的，`py-spy` 就是救命稻草！

  📖 [从仍在运行的进程获取栈回溯](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#get-the-backtrace-from-the-still-running-process) + [py-spy](https://github.com/stas00/the-art-of-debugging/tree/master/pytorch#py-spy)

- 9.5. **直接在 gdb 下运行不稳定或会段错误的测试，而不是盲目地设法复现崩溃。**
  - 对一个间歇性发生段错误的 `pytest` 测试，从一开始就在 gdb 下启动整个调用（`gdb -ex r --args python -m pytest -sv tests/test_failing.py`），而不是事后另起炉灶设法抓住它。
  - 当它发生段错误时，gdb 恰好会在那个时刻把你带回它的提示符。`bt`（或 `bt full`）的用法与核心转储文件的情况完全相同，之后如果需要，可以用 `c` 继续执行。
  - 这完全绕开了"可我自己单独复现的时候它又没崩"这种令人抓狂的局面。
  - 与上面两种栈回溯技巧结合起来，据信足以诊断机器学习代码库中现实世界里约 95% 的原生层崩溃。

  📖 [在 gdb 下运行程序](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#run-the-program-under-gdb)

- 9.6. **符号缺失往往不是神秘的运行时 bug，而是链接期或库版本的问题——用 `nm` 就能实际核实里面到底有什么。**
  - 运行时出现 `undefined symbol: util_b`，尽管这个函数就在你的代码库里，这种情况往往是由加载了错误版本的共享库导致的（例如 `LD_LIBRARY_PATH` 指向了一个旧的构建产物）。
  - `nm ./your_program | grep symbol_name` 能显示某个符号在特定二进制文件或库中是被定义了（`T`）还是仅被引用而未定义（`U`），从而把猜测变成可核查的事实。
  - 再配合 `ldd`（它能显示一个二进制文件实际解析到了哪些库文件），你可以直接比较两个"同名"库构建的 `nm` 输出，看出一个含有该符号而另一个没有。
  - 这套组合——用 `nm` 回答"这个符号在这里存在吗"，用 `ldd` 回答"实际加载的是哪个文件"——能解决"符号缺失"类运行时错误中的绝大部分。

  📖 [nm](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#nm)

---

*以上所有经验教训均出自 Stas Bekman 所著的 [Machine Learning Engineering](../../index.md) 和 [The Art of Debugging](https://github.com/stas00/the-art-of-debugging)，基于 [CC BY-SA 4.0](https://github.com/CodingPracticeGroup/ml-engineering/blob/master/LICENSE-CC-BY-SA) 协议发布。*
