# 模型并行 {#model-parallelism}


## 并行技术概览 {#parallelism-overview}

在现代机器学习中，各种并行方法被用于：

1. 突破 GPU 显存限制。示例：
   - 装下非常大的模型——例如 t5-11b 光是模型参数就有 45GB
   - 装下非常长的序列——例如 Llama 3.1 模型 128K 的上下文长度
2. 显著加速训练——把本来要训练一年的任务在几小时内完成

我们先深入讨论各种 1D 并行技术及其优缺点，然后再看如何把它们组合成 2D 和 3D 并行，以实现更快的训练并支持更大的模型。之后还会介绍其他若干种同样强大的替代方案。

虽然这些主要概念多半同样适用于其他框架，但本文聚焦于基于 PyTorch 的实现。

要让模型超出加速卡的显存仍能被训练和推理，主要有两种途径：
1. 3D 并行——对网络非常高效，但会对建模代码侵入很深，需要额外做大量工作才能让它正确运行
2. ZeRO 并行——对网络并不高效，但对建模代码几乎没有改动要求，非常容易让它跑起来。

## 可扩展性概念 {#scalability-concepts}

以下是对主要概念的简要说明，本文后文会逐一详细展开。

1. [数据并行](#data-parallelism)（DP）——同一套配置被复制多份，每份喂入数据的一个切片。处理是并行进行的，所有配置在每个训练步结束时同步。
2. [张量并行](#tensor-parallelism)（TP）——每个张量被切成多个块，整个张量不再驻留在单块 GPU 上，而是张量的每个分片驻留在指定的那块 GPU 上。在前向和反向处理中，各分片并发执行，并在与层相关的边界处交换部分激活值或梯度，通常使用 all-reduce、all-gather 或 reduce-scatter，而不是只在一步结束时才同步。这可以称为水平并行，因为切分发生在水平方向上。参见 [PyTorch Tensor Parallel 教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)。
3. [流水线并行](#pipeline-parallelism)（PP）——模型沿纵向（层级）切分到多块 GPU 上，使单块 GPU 上只放置模型的一层或若干层。每块 GPU 并行地处理流水线的不同阶段，并且每次只处理批次的一个小块。
4. [零冗余优化器](#zero-data-parallelism)（ZeRO）——也会进行张量分片，与 TP 有些相似，不同之处在于：整个张量会在前向或反向计算需要它时及时重建出来，因此模型不需要被修改。它还支持各种卸载（offloading）技术，以弥补 GPU 显存的不足。分片 DDP（Sharded DDP）是 ZeRO 这一基础概念在 ZeRO 其他各种实现中所使用的另一个名称。
5. [序列并行](#sequence-parallelism)——在很长的输入序列上训练需要海量 GPU 显存。这种技术把单个序列的处理切分到多块 GPU 上。
6. [专家并行](#expert-parallelism)——混合专家（Mixture-Of-Experts，MoE）模型可以被划分，使每个专家拥有一块专属 GPU（或几块）。

[Breadth-First Pipeline Parallelism](https://arxiv.org/abs/2211.05953) 这篇论文的引言部分，大概是我找到的对大多数常见并行技术最好的讲解之一。

## 数据并行 {#data-parallelism}

### DDP {#ddp}

哪怕只有 2 块 GPU，大多数用户也已经借助 `DataParallel`（DP）和 `DistributedDataParallel`（DDP）享受到了训练速度的提升，因为它们用起来几乎毫不费力。这是 PyTorch 的内置特性。

详情参见 [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

### ZeRO 数据并行 {#zero-data-parallelism}

基于 ZeRO 的数据并行（ZeRO-DP）在这篇[博客文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)的下图中做了说明：
![DeepSpeed-Image-1](images/parallelism-zero.png)

它可能不容易理解，但实际上概念相当简单。这就是平常的 `DataParallel`（DP），只不过每块 GPU 不再复制完整的模型参数、梯度和优化器状态，而是只保存其中的一片。然后在运行时，当仅对于当前这一层需要完整的层参数时，所有 GPU 相互同步，把自己缺少的部分从别处补齐——就是这样。

考虑下面这个简单的模型，它有 3 层，每层有 3 个参数：
```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```
层 La 有权重 a0、a1 和 a2。

如果我们有 3 块 GPU，分片 DDP（即 ZeRO-DP）会把模型像这样切分到 3 块 GPU 上：

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

从某种意义上说，这与张量并行的水平切分相同——如果你想象一下典型的 DNN 图示的话。纵向切分则是把整组整组的层放到不同的 GPU 上。但这只是起点。

现在，每块 GPU 都会像它在 DP 中工作时那样拿到通常的小批次（mini-batch）：
```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入未经修改——它们以为自己将要被一个普通的模型处理。

首先，输入到达层 La。

我们只看 GPU0：x0 需要 a0、a1、a2 这些参数来完成它的前向路径，但 GPU0 只有 a0——于是它从 GPU1 收到 a1、从 GPU2 收到 a2，把模型的所有部件凑到了一起。

与此同时，GPU1 拿到小批次 x1，它只有 a1，但需要 a0 和 a2 参数，于是它从 GPU0 和 GPU2 那里得到这两个参数。

GPU2 拿到输入 x2，发生的也是同样的事。它从 GPU0 和 GPU1 得到 a0 和 a1，再结合自己的 a2 重建出完整的张量。

3 块 GPU 全部重建出完整的张量，然后执行前向计算。

计算一结束，不再需要的数据就被丢弃——它只在计算期间被使用。这种重建通过预取（pre-fetch）高效地完成。

然后对层 Lb 重复整个过程，再依次前向处理 Lc，接着反向 Lc -> Lb -> La。

在我看来，这就像一种高效的徒步团队负重分配策略：

1. 成员 A 背着帐篷
2. 成员 B 背着炉子
3. 成员 C 背着斧头

每天晚上，大家都把自己有的东西拿出来与他人分享，再从别人那里拿到自己没有的东西，早上再收拾好各自负责的那类装备继续赶路。这就是分片 DDP / ZeRO DP。

对比一下那种每个人必须自己背帐篷、炉子和斧头的简单策略，后者要低效得多。这就是 PyTorch 中的 DataParallel（DP 和 DDP）。

在阅读这个主题的文献时，你可能会遇到以下同义词：Sharded（分片）、Partitioned（分区）。

如果你仔细观察 ZeRO 切分模型权重的方式——它看起来与后文将要讨论的张量并行非常相似。这是因为它切分/分片的是每一层的权重，而与接下来要讨论的纵向模型并行不同。

ZeRO-DP 阶段 1+2+3 的实现：
- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch](https://docs.pytorch.org/docs/stable/fsdp.html)（它最初在 [FairScale](https://github.com/facebookresearch/fairscale/) 中实现，后来被并入 PyTorch 核心代码库）
- [torchtitan](https://github.com/pytorch/torchtitan)

DeepSpeed ZeRO 集成：
- [HF Trainer 集成](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- [Determined.AI](https://github.com/determined-ai/determined/tree/main/docs/model-dev-guide/api-guides/apis-howto/deepspeed)

FSDP 集成：
- [HF Trainer 集成](https://huggingface.co/docs/transformers/main/en/fsdp)
- [Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)
- [torchtitan](https://github.com/pytorch/torchtitan)

重要论文：

关于 DeepSpeed 与 ZeRO 的一般性论文：
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/abs/2104.07857)
- [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)
- [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)
- [AMSP: Reducing Communication Overhead of ZeRO for Efficient LLM Training](https://arxiv.org/abs/2311.00257)

PyTorch：
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)

DeepSpeed ZeRO 主要资源：
- [项目的 GitHub](https://github.com/deepspeedai/DeepSpeed)
- [使用文档](https://www.deepspeed.ai/getting-started/)
- [API 文档](https://deepspeed.readthedocs.io/en/latest/index.html)
- [博客文章](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

#### 克服全局批次尺寸过大的问题 {#overcoming-the-huge-global-batch-size-issue}

假如你使用比如 1024 块加速卡，那么每块加速卡上的分片都会非常小，并且有大量空闲显存可用于微批次大小（micro-batch size，MBS），假设你能放下 MBS=32——最终得到的 GBS（全局批次大小）就是 32k——这多半不是你想要的。

因此，你要么需要部署[张量并行](#tensor-parallelism)，而它的实现并不简单；要么——通常做法简单得多——部署[序列并行](#sequence-parallelism)。我还没有在实际场景中用过它，但据我目前了解，对应方案为：

- DeepSpeed ZeRO 使用 [DeepSpeed-Ulysses](#deepspeed-ulysses-sp)
- [Arctic Long Sequence Training](#arctic-long-sequence-training)
- FSDP 使用 [Paged Ring Attention](https://github.com/lucidrains/ring-attention-pytorch)（[论文](https://arxiv.org/abs/2402.08268)）

请注意，它的效率很可能比不上[张量并行](#tensor-parallelism)——但我还不知道实际会多出多少额外开销。

#### 带多个副本的 ZeRO {#zero-with-multiple-replicas}

默认情况下，ZeRO 使用所有 GPU 组成单个模型副本——也就是模型被分散到所有 GPU 上。这带来若干限制，例如：

1. 全局批次大小不灵活——它始终是 `total_gpus*micro_batch_size` 的函数——在大型集群上，这可能导致极大的全局批次大小，而这对高效收敛是有害的。诚然，可以用很小的微批次大小来控制全局批次大小，但这会导致每块 GPU 上的矩阵变小，从而造成计算效率下降
2. 无法利用快得多的节点内网络的优势，因为较慢的节点间网络决定了通信的整体速度。

[ZeRO++](https://arxiv.org/abs/2306.10209) 通过引入面向 ZeRO 的层级权重划分（Hierarchical Weight Partition，hpZ）解决了第 2 个限制。在这种方法中，模型的完整权重不再被分散到所有 GPU 上，而是把每个模型副本限制在单个节点内。这会使显存用量按节点总数成倍增加，但现在用于收集分片组件的 2 次 `all_gather` 调用是在快得多的节点内连接上执行的。只有用于聚合并重新分发梯度的 `reduce_scatter` 调用会在较慢的节点间网络上执行。

第一个限制并没有被真正解决，因为整体的全局批次大小保持不变，但由于每个副本的效率更高，而且额外的显存压力很可能会限制每块 GPU 上可行的微批次大小，因此总体上应该会提升系统的吞吐量。

PyTorch FSDP 在 [shardingStrategy.HYBRID_SHARD](https://docs.pytorch.org/docs/stable/fsdp.html) 中实现了这一特性

论文：

- [ZeRO++: Extremely Efficient Collective Communication for Giant Model Training](https://arxiv.org/abs/2306.10209)
- [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)


#### ZeRO 的变体 {#zero-variations}

提出对 ZeRO 协议进行修改的已发表论文：

- [MiCS: Near-linear Scaling for Training Gigantic Model on Public Cloud](https://arxiv.org/abs/2205.00119)（2022）
- [AMSP: Super-Scaling LLM Training via Advanced Model States Partitioning](https://arxiv.org/abs/2311.00257)（2023）



## 流水线并行方法 {#pipeline-parallelism-methods}

### 朴素模型并行（纵向） {#naive-model-parallelism-vertical}

朴素模型并行（Naive Model Parallelism，MP）就是把模型的若干层组分散到多块 GPU 上。其机制相对简单——用 `.to()` 把所选的层切换到目标设备上，此后每当数据进出这些层时，就把数据切换到与该层相同的设备上，其余部分保持不变。

我们称之为纵向 MP，因为如果你还记得大多数模型是如何绘制的，就会明白我们是在垂直方向上切分层。例如，如果下图展示的是一个 8 层模型：

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```
我们只是把它垂直切成了 2 份，把第 0-3 层放到 GPU0 上，把第 4-7 层放到 GPU1 上。

现在，数据从第 0 层传到第 1 层、第 1 层传到第 2 层、第 2 层传到第 3 层时，就跟普通模型一样。但当数据需要从第 3 层传到第 4 层时，它必须从 GPU0 传到 GPU1，这就引入了通信开销。如果参与的 GPU 位于同一个计算节点（即同一台物理机）上，这种拷贝相当快；但如果 GPU 位于不同的计算节点（即多台机器）上，通信开销可能会大得多。

接下来第 4、5、6、7 层与普通模型并无二致。当第 7 层完成后，我们通常要把数据送回第 0 层——标签在那里（或者也可以把标签发送到最后一层）。这样就能计算损失，优化器也能开始工作了。

问题：
- 主要缺陷——也是它被称为“朴素”MP 的原因——在于任何给定时刻，除一块 GPU 外其余全部空闲。因此，如果使用 4 块 GPU，效果几乎等同于把单块 GPU 的显存扩大到四倍，而无视其余的硬件。此外还有在设备之间拷贝数据的开销。所以，4 块 6GB 的卡为被切分的层状态提供的总容量大致相当于 1 块 24GB 的卡，但单卡会更快完成训练，因为它避免了设备间传输。朴素 MP 主要是一种解决容量的技术：当某个工作负载与层关联的训练状态超出一块 GPU 的显存时，只要每个分区及其激活值能放进所分配的 GPU，它就能让这个负载运行起来。
- 共享的嵌入可能需要在 GPU 之间来回拷贝。

### 流水线并行 {#pipeline-parallelism}

流水线并行（Pipeline Parallelism，PP）与朴素 MP 几乎相同，但它解决了 GPU 空闲的问题：把传入的批次切分成微批次（micro-batches），并人为构造出一条流水线，使不同的 GPU 能够并发地参与计算过程。

下面这张来自 [GPipe 论文](https://research.google/blog/introducing-gpipe-an-open-source-library-for-efficiently-training-large-scale-neural-network-models/)的图示，上半部分展示的是朴素 MP，下半部分展示的是 PP：

![mp-pp](images/parallelism-gpipe-bubble.png)

从下半部分的图很容易看出，PP 中 GPU 空闲的死区更少。空闲的部分被称为“气泡”（bubble）。

图的上、下两部分展示的都是并行度为 4 的并行。也就是有 4 块 GPU 参与流水线。因此存在由 4 个流水线阶段 F0、F1、F2 和 F3 组成的前向路径，以及按相反顺序返回的反向传播路径 B3、B2、B1 和 B0。

PP 引入了一个新的需要调节的超参数，即 `chunks`，它定义依次通过同一流水线阶段发送多少个数据块。例如，在下半部分的图中可以看到 `chunks=4`。GPU0 对数据块 0、1、2、3 执行相同的前向路径（F0,0、F0,1、F0,2、F0,3），然后等待其他 GPU 完成各自的工作；只有当它们的工作即将完成时，GPU0 才再次开始工作，对数据块 3、2、1、0 执行反向路径（B0,3、B0,2、B0,1、B0,0）。

注意，从概念上讲，这与梯度累积步数（gradient accumulation steps，GAS）是同一回事。PyTorch 使用 `chunks`，而 DeepSpeed 把同一个超参数称为 GAS。

由于有了数据块，PP 引入了微批次（micro-batches，MBS）的概念。DP 把全局数据批次大小切分成小批次，所以如果你的 DP 并行度是 4，1024 的全局批次大小会被切成 4 份、每份 256 的小批次（1024/4）。而如果 `chunks`（即 GAS）的数量是 32，最终得到的微批次大小就是 8（256/32）。每个流水线阶段一次只处理一个微批次。

对于 DP + PP 组合，我们这样计算全局批次大小：`mbs*chunks*dp_degree`（`8*32*4=1024`）。

让我们回到那张图。

`chunks=1` 时会得到非常低效的朴素 MP。而 `chunks` 取值非常大时，微批次尺寸会变得极小，效率同样可能不高。所以必须通过实验，找到能让 GPU 获得最高效利用的取值。

虽然图中显示存在一段无法并行的“死”时间气泡——因为最后一个 `forward` 阶段必须等待 `backward` 走完整个流水线——但寻找 `chunks` 最佳取值的目的，正是让所有参与的 GPU 都保持很高的并发利用率，也就是把气泡的尺寸降到最低。

调度方案的选择对高效性能至关重要，最常见的调度方案按提出顺序如下：

- 顺序式（sequential）[Gpipe: Efficient training of giant neural networks using pipeline parallelism](https://arxiv.org/abs/1811.06965)
- 交错式 1F1B（interleaved 1F1B）[Pipedream: Fast and efficient pipeline parallel dnn training](https://arxiv.org/abs/1806.03377)
- 循环式、深度优先（looped, depth-first）[Efficient large-scale language model training on gpu clusters using Megatron-LM](https://arxiv.org/abs/2104.04473)
- 广度优先（breadth-first）[Breadth-First Pipeline Parallelism](https://arxiv.org/abs/2211.05953)
- Llama 3 的训练为了获得最佳性能，把深度优先和广度优先结合了起来，并且还能随着训练推进逐步修改全局批次大小——这在 PP 中通常很难做到。参见 [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) 的 3.3.2 节“用于模型扩展的并行技术”（Parallelism for Model Scaling）。

例如下面是一个交错式流水线：

![interleaved-pipeline-execution](images/parallelism-sagemaker-interleaved-pipeline.png)

这里通过优先处理反向传播，把气泡（空闲时间）进一步降到最低。

DeepSpeed、SageMaker 等都使用了它（仅举几例）。

[DeepSeek v3](https://arxiv.org/abs/2412.19437) 通过 DualPipe 引入了效率更高的 PP，它减小了气泡尺寸，并实现了更好的计算/通信重叠。具体细节见论文的 3.2.1 节。

![dualpipe](images/parallelism-pp-dualpipe.png)
（[来源](https://arxiv.org/abs/2412.19437)）

### PP 实践中的注意事项 {#pp-practical-caveats}

无论你使用哪个 PP 技术栈，以下几个约束都会反复出现：

- 必须对模型做相当大的修改，因为常规的模块图必须被切分成流水线阶段——有些 API 仍然要求提供接近于由各个阶段组成的 `nn.Sequential` 的东西——这可能要求修改模型的设计。
- 阶段边界所能传递的内容是受限的。如果在单进程的前向传播中有一堆 Python 变量要流经第一个阶段，你就得想办法绕开它。典型的接口要求阶段的输入和输出是单个 Tensor 或 Tensor 元组，并且这些张量的第一个维度必须是批次大小，因为流水线会把小批次再切分成微批次。
- 在流水线阶段这一层级上不可能有条件控制流——例如，T5 这样的编码器-解码器模型需要特别的变通方案来处理带条件的编码器阶段。
- 必须安排好每一层，使一个阶段的输出能成为下一个阶段的输入。
- 第一个阶段包含一个很重的嵌入层，如果词表很大，它可能相当庞大——这可能要求自定义的切分方式，使第一个阶段包含的 Transformer 块比其他阶段少。

现代技术栈减轻了部分样板工作，但它们并没有消除这些结构性约束。

### 目前可用的 PP 方案 {#where-to-get-pp-today}

实际可选的是自带 PP 的生产级训练技术栈，以及 PyTorch 的可组合 API：

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)——久经考验的 PP（交错式 1F1B 及相关调度）；与特定模型绑定，没有单独公开的 Pipe 风格 API
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)——作为 DeepSpeed 技术栈一部分的 PP（常与 ZeRO 结合使用）
- [torch.distributed.pipelining](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)——PyTorch 当前的 PP 工具包（由 [PiPPy](https://github.com/pytorch/pippy) 迁移而来）；仍处于 **alpha** 阶段 / API 可能变化。它能切分通用模型，内置 GPipe、1F1B、交错式 1F1B、循环式 BFS 以及 DualPipeV（DeepSeek 风格）等调度，并可与 DP/FSDP/TP 组合——3D 并行的 Llama 示例见 [torchtitan](https://github.com/pytorch/torchtitan)
- [torchtitan](https://github.com/pytorch/torchtitan) / [nanotron](https://github.com/huggingface/nanotron)——使用现代 PP 的完整训练技术栈
- [DeepSeek DualPipe](https://github.com/deepseek-ai/DualPipe)——来自 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 的调度（在 `torch.distributed.pipelining` 中也以 `ScheduleDualPipeV` 的形式出现）
- [SageMaker](https://arxiv.org/abs/2111.05972)——专有方案，仅限 AWS 使用

### 延伸阅读 {#related-reading}

- [Pipeline-Parallelism: Distributed Training via Model Partitioning](https://siboehm.com/articles/22/pipeline-parallel-training)



## 张量并行 {#tensor-parallelism}

在张量并行中，每块 GPU 只处理张量的一个切片，只有对于需要完整张量的操作才聚合出完整张量。

本节采用 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 论文 [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473) 中的概念与图示。

任何 Transformer 的主要构件都是全连接层 `nn.Linear`，后面跟着非线性激活函数 `GeLU`。

按照 Megatron 论文的记法，我们可以把其中的点积部分写成 `Y = GeLU(XA)`，其中 `X` 和 `Y` 是输入和输出向量，`A` 是权重矩阵。

如果我们以矩阵形式来看这个计算，就不难看出矩阵乘法可以如何拆分给多块 GPU：
![并行 GEMM](images/parallelism-tp-parallel_gemm.png)

如果我们把权重矩阵 `A` 按列切分到 `N` 块 GPU 上，并行地执行矩阵乘法 `XA_1` 到 `XA_n`，就会得到 `N` 个输出向量 `Y_1, Y_2, ..., Y_n`，它们可以各自独立地送入 `GeLU`：
![独立的 GeLU](images/parallelism-tp-independent-gelu.png)

利用这一原理，我们可以更新任意深度的 MLP，而且直到最后需要从分片重建输出向量之前，都不需要在 GPU 之间进行任何同步。Megatron-LM 论文的作者为此提供了一张很有助益的图示：
![并行分片处理](images/parallelism-tp-parallel_shard_processing.png)

对多头注意力层做并行化甚至更简单，因为它们由于拥有多个独立的注意力头，本身就天然具备并行性！
![并行自注意力](images/parallelism-tp-parallel_self_attention.png)

重要：TP 需要非常快的网络，而通常节点内网络比节点间网络快得多，因此不建议跨节点做 TP。实际上，如果一台节点有 4 块 GPU，那么最高的 TP 并行度就是 4。如果你需要 8 的 TP 并行度，就必须使用至少有 8 块 GPU 的节点。

TP 可以与其他并行方法组合使用。

其他名称：
- DeepSpeed 称之为[tensor slicing（张量切分）](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/)

实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 有一个内部实现，因为它与模型高度绑定
- [PyTorch](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [SageMaker](https://arxiv.org/abs/2111.05972)——这是一个只能在 AWS 上使用的专有方案。
- [OSLO](https://github.com/eleutherAI/Oslo) 提供了基于 Transformers 的张量并行实现。
- [nanotron](https://github.com/huggingface/nanotron)
- [torchtitan](https://github.com/pytorch/torchtitan)


### 异步 TP {#async-tp}

TP 的缺陷之一，是它的通信很难与计算重叠。PyTorch 提出了 [Async-TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487) 来克服这一点，它把有依赖关系的 all-gather + matmul 序列分解为一系列 cudaMemcpyAsync 调用和更小的部分矩阵乘法——而且它借助 `torch.compile` 自动替你完成了这件事！

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 也通过 `--tp-comm-overlap` 实现了这一功能。


### 延伸阅读 {#related-reading}

- [Tensor Parallelism and Sequence Parallelism: Detailed Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)

## 序列并行 {#sequence-parallelism}

像 DNA 测序这样的机器学习任务，可能需要以很长的序列长度（例如 256K）进行训练，而即使是普通的 LLM，也可能在 10k 甚至更长的序列上训练。

自注意力（Self-Attention）是 Transformer 的关键组件，对于标准的稠密注意力，其计算复杂度关于序列长度是二次的。那些需要完整实体化出“序列×序列”注意力矩阵的内核（kernel）还需要二次方量级的临时显存。像 [FlashAttention](https://arxiv.org/abs/2205.14135) 这样节省显存的内核避免了存储该矩阵，使注意力的显存占用关于序列长度保持线性，但并没有消除二次方的计算。

当序列足够长时，即使批次大小为 1 也可能放不进单块 GPU，因此序列并行沿着序列维度划分工作。这提高了所能支持的序列长度，但并不使其没有上限：实际上限仍取决于加速卡的显存总量、计算时间、通信开销以及内核的限制。

由于这种并行类型与本文描述的其他并行类型是正交的，它可以与其中任意一种组合，形成 4D 并行、ZeRO-DP+SP 等各种组合。

### DeepSpeed-Ulysses SP {#deepspeed-ulysses-sp}

论文：[DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509)

在这种实现中，有 2 类东西会被分片：
1. 多头注意力的权重被切分到参与的各块 GPU 上，使每块 GPU 只持有少数几个子头（sub-head）。这发生在模型被创建/加载的时候。这在某种程度上与[张量并行](#tensor-parallelism)类似。
2. 在训练过程中，每个输入序列被划分成若干块，每一块被发送到其中一块 GPU 上，这让人想起 ZeRO-3 的分片，只不过被分片的不是权重而是输入。

在计算时，每个序列块先被投影到 QKV，然后在每块设备上聚合（gather）成完整序列的 QKV；每块设备只对自己拥有的子头做计算，随后再聚合为供 MLP 块使用的完整注意力输出。

![deepspeed-ulysses sp](images/deepspeed-ulysses.png)

[来源](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses)

图中：
1. 输入序列 N 被划分到 P 个可用设备上。
2. 输入序列在本地的每个 N/P 分区都被投影为查询（queries，Q）、键（keys，K）和值（values，V）嵌入。
3. 接下来，本地的 QKV 嵌入通过参与计算的设备之间高度优化的 all-to-all 集合通信，聚合为全局 QKV。
4. 然后对每个头执行注意力计算：

![math](images/deepspeed-ulysses-math.png)

5. 最后，另一个 all-to-all 集合通信把注意力计算的输出上下文张量变换为序列（N/P）并行的形式，供 Transformer 层块其余模块中的后续算子（MLP 的矩阵乘法、层归一化等）使用。

示例：考虑 seqlen=8K、num_heads=128、单个节点 num_gpus=8 的情形

1. 每块 GPU 得到原始序列中一段 1K 长的块（`8K/8`）
2. 每块 GPU 被分配 16 个子头（`128/8`）
3. a. 在 gpu0 上，`forward` 之前把原始序列重新聚合回 8K 个词元
   b. 在前 16 个子头上完成注意力计算

其余 7 块 GPU 执行同样的逻辑，各自在自己的 16 个子头上计算 8k 的注意力

你可以在[这里](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses#significant-communication-volume-reduction)阅读这种非常高效的通信的细节。

DeepSpeed-Ulysses 通过按消息大小或序列长度成比例地增加 GPU 数量，使通信量保持一致。

### Arctic Long Sequence Training {#arctic-long-sequence-training}

Arctic Long Sequence Training 把 [DeepSpeed-Ulysses](#deepspeed-ulysses-sp) 移植到了 HuggingFace Transformers，同时对其更新以适配现代的注意力头机制，并进一步做了扩展：通过分块（tiling）计算和卸载激活检查点（activation checkpoints），支持长得多的序列长度（或更大的批次大小）。集成指南见[这里](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)。

- 论文：https://arxiv.org/abs/2506.13996
- 实现与集成：[ArcticTraining](https://github.com/snowflakedb/ArcticTraining/tree/main/projects/sequence-parallelism) 和 [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)

### Colossal-AI 的 SP {#colossal-ais-sp}

论文：[Sequence parallelism: Long sequence training from system perspective](https://arxiv.org/abs/2105.13120)

Colossal-AI 的 SP 实现使用环状自注意力（ring self-attention）：这是一种环形通信集合操作，其中查询投影在本地计算，而键和值的投影以环形方式传递来计算全局注意力，其通信复杂度关于消息大小 M 是线性的。

### Megatron-LM 的 SP {#megatron-lms-sp}

论文：[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

Megatron-LM 的 SP 与其 TP 紧密集成。Megatron-LM 沿序列维度划分序列，并使用 allgather 和 reduce-scatter 集合通信来聚合用于注意力计算的 QKV 投影。无论计算设备的数量多少，它的通信量都随消息大小（M）线性增长。

### Ring Attention 与分块 Transformer {#ring-attention-with-blockwise-transformers}

论文：[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

它也被称为环状自注意力（Ring Self-Attention，RSA），这个名字出自 [Colossal-AI 的 SP](#colossal-ais-sp) 那篇论文，该论文提出了同样的做法：键和值以环形方式传递，而查询保持在本地。两个名称都在被广泛使用，所以 [DistFlashAttn](#distflashattn) 中与环状自注意力（Ring Self-Attention）的对比，指的就是这项技术。

1. 张量自始至终沿序列维度分片：形状为（`seq_len // N, d_model`）
2. 在注意力层中，每块 GPU 先利用自己手头可用的分片，计算它能算出的那部分注意力分数。
3. 与此同时，其他序列块的键和值在各设备之间传递。
4. 一旦拿到另一个块的键/值，每块 GPU 就使用这段新序列的键/值张量继续自己的注意力计算
5. 如此继续，直到注意力计算完成。

SP 实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Colossal-AI](https://colossalai.org/)
- [torchtitan](https://github.com/pytorch/torchtitan)

PyTorch 也在开发这项功能，并将其称为上下文并行（Context Parallel，CP）。
### DistFlashAttn {#distflashattn}

据报告，[DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training](https://arxiv.org/abs/2310.03294) 比 Ring Self-Attention 快很多倍，因为它在执行序列并行（Sequence Parallelism）的同时，在多个工作进程之间对每个词元的 KVQ 计算做了负载均衡。

![distflashattn](images/dist-flash-attn.png)

### 相关读物 {#related-reading}

- [Tensor Parallelism and Sequence Parallelism: Detailed Analysis](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)


## TP+SP {#tpsp}

正如 [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) 所解释的，TP 可以与 SP 组合在同一个进程组（process group）中，以把通信开销降到最低。例如在 LLM 中，TP 用于嵌入（embedding）、注意力和线性层，当遇到 dropout 和层归一化时则改用 SP。



## DP+PP {#dppp}

下面这张图来自 DeepSpeed 的[流水线教程](https://www.deepspeed.ai/tutorials/pipeline/)，展示了如何把 DP 与 PP 结合起来。

![dp-pp-2d](images/parallelism-zero-dp-pp.png)

这里重要的是要看到：DP 的 rank 0 看不到 GPU2，DP 的 rank 1 也看不到 GPU3。对 DP 来说，它眼里只有 GPU 0 和 GPU 1，它就像只有 2 块 GPU 那样喂入数据。GPU0 借助 PP 把自身的一部分负载“偷偷”卸载（offload）给了 GPU2。GPU1 同样如此，把 GPU3 招来帮忙。

由于每个维度都至少需要 2 块 GPU，这里你至少需要 4 块 GPU。

实现：
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/eleutherAI/Oslo)
- [nanotron](https://github.com/huggingface/nanotron)
- [torchtitan](https://github.com/pytorch/torchtitan)



## DP+PP+TP {#dppptp}

要获得更高效的训练，可以使用 3D 并行，即把 PP 与 TP 和 DP 组合起来，从下面这张图可以看到这一点。

![dp-pp-tp-3d](images/parallelism-deepspeed-3d.png)

这张图来自一篇博客文章 [3D parallelism: Scaling to trillion-parameter models](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)，也值得一读。

由于每个维度都至少需要 2 块 GPU，这里你至少需要 8 块 GPU。

实现：
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) - DeepSpeed 还包含一种更高效的 DP，他们称之为 ZeRO-DP。
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/eleutherAI/Oslo)
- [nanotron](https://github.com/huggingface/nanotron)
- [torchtitan](https://github.com/pytorch/torchtitan)


## ZeRO DP+PP+TP {#zero-dppptp}

DeepSpeed 的主要特性之一是 ZeRO，它是 DP 的一个极具扩展性的扩展。我们在 [ZeRO 数据并行](#zero-data-parallelism) 一节中已经讨论过它。它通常是一个独立的功能，不需要 PP 或 TP。但它也可以与 PP 和 TP 组合使用。

当 ZeRO-DP 与 PP（以及可选的 TP）组合时，通常只启用 ZeRO 阶段 1（优化器分片）。

尽管理论上可以把 ZeRO 阶段 2（梯度分片）与流水线并行一起使用，但这会带来糟糕的性能影响。每个微批次都需要额外做一次 reduce-scatter 集合通信，以便在分片之前先把梯度聚合起来，这可能带来相当可观的通信开销。而流水线并行的本性是使用较小的微批次，重点反而是尽量在算术强度（微批次大小）与最小化流水线气泡（微批次数量）之间取得平衡。因此这些通信开销只会雪上加霜。

此外，由于 PP 的存在，层的数量本来就比正常情况少，所以显存节省也有限。PP 本来就已经把梯度大小降低到了原来的 ``1/PP``，在此基础上再做梯度分片，收益就不如纯 DP 那么显著了。

出于同样的原因，ZeRO 阶段 3 也不是好选择——需要更多的节点间通信。

而且既然我们用上了 ZeRO，另一个好处就是 ZeRO-Offload。由于这是阶段 1，优化器状态可以被卸载到 CPU。

实现：
- [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed) 以及 [来自 BigScience 的 Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)，后者是前者的一个分支。
- [OSLO](https://github.com/eleutherAI/Oslo)
- [torchtitan](https://github.com/pytorch/torchtitan)

重要论文：

- [Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model](
https://arxiv.org/abs/2201.11990)



## 专家并行 {#expert-parallelism}

当使用混合专家（Mixture-Of-Experts，MoE）时（尤其是在推理期间），可以给每个专家分配自己的一块加速器（如果一块不够，就多分几块），这就是所谓专家并行（Expert Parallelism，EP）。这又增加了一个并行维度，对于很可能命中所有专家的大批次来说，可以显著提速。EP 通信的不是模型权重，而是词元。EP 还能带来更高效的计算，因为矩阵乘法处理的输入变大了。

想详细了解，请参见：
- [DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale](https://arxiv.org/abs/2201.05596)
- [Mixture of Experts Explained](https://huggingface.co/blog/moe#parallelism)

## 其他并行方案 {#other-parallelism-approaches}

### FlexFlow {#flexflow}

[FlexFlow](https://github.com/flexflow/flexflow-train) 用一种略有不同的思路解决并行化问题。

论文：["Beyond Data and Model Parallelism for Deep Neural Networks" by Zhihao Jia, Matei Zaharia, Alex Aiken](https://arxiv.org/abs/1807.05358)

它在 Sample-Operator-Attribute-Parameter（样本-算子-属性-参数）之上做一种 4D 并行。

1. Sample（样本）= 数据并行（按样本并行）
2. Operator（算子）= 把单个运算并行化为若干子运算
3. Attribute（属性）= 数据并行（按长度并行）
4. Parameter（参数）= 模型并行（不分维度——水平或垂直皆可）

示例：
* Sample

取 10 个批次，序列长度为 512。如果按 sample 维度把它们并行到 2 块设备上，10 x 512 就变成了 5 x 2 x 512。

* Operator

层归一化需要均值和方差，因此标准差和均值并不是相互独立的运算。以一个跨 2 块设备（cuda:0、cuda:1）的算子分解为例说明：先把输入数据复制到两块设备上，然后让 cuda:0 归约 `sum(x)`，cuda:1 归约 `sum(x*x)`。结果可以这样合并：`mean = E[x]`、`variance = E[x*x] - E[x]**2`。直接使用这种矩公式可能损失数值精度，因此生产环境的实现通常改用稳定的归约算法。参见 PyTorch 的 [`torch.std_mean` 文档](https://docs.pytorch.org/docs/stable/generated/torch.std_mean.html)。

* Attribute

我们有 10 个批次、长度 512。如果按 attribute 维度把它们并行到 2 块设备上，10 x 512 就会变成 10 x 2 x 256。

* Parameter

这与张量模型并行或朴素的逐层模型并行类似。

![flex-flow-soap](images/parallelism-flexflow.jpeg)

这个框架的意义在于：它把 (1) GPU/TPU/CPU、(2) RAM/DRAM、(3) 快速的内部互联/较慢的跨节点互联这些资源都纳入考虑，并用算法自动优化所有这些因素，决定在哪里使用哪种并行方式。

有一个非常重要的方面：FlexFlow 是为具有静态、固定工作负载的模型优化 DNN 并行化而设计的，因为行为动态变化的模型可能在不同迭代中偏好不同的并行策略。

所以这个前景非常诱人——它在所选集群上跑一个 30 分钟的模拟，就能给出利用这一特定环境的最佳策略。如果你增加/删除/替换任何部件，它会重新运行并为新配置重新优化方案。然后你就可以训练了。不同的环境会得到各自专属的优化。


## 并行中的网络集合通信 {#parallelism-network-collectives}

由于节点内和节点间的速度相差一个数量级，针对通信是留在节点内还是跨越节点来选择不同的并行技术，这一点至关重要。例如，TP 对同步的需求极大，应当留在最快、延迟最低的域内——传统上这意味着留在节点内；但在 NVL72 这类机架级系统上，这个域横跨许多节点（称为超级节点，supernode），因此真正的规则是把 TP 保持在 scale-up（纵向扩展）网络之内，只有当实测表明开销可以承受、或者容量上别无选择时，才跨入较慢的 scale-out（横向扩展）网络。此外，一些加速器，比如较新的 AMD MI3\*\* 系列，其 GPU 到 GPU 的连接非常慢，这同样会影响如何安排并行才能达到最佳性能。

这里有一个有用的小知识：all-reduce 集合通信可以分解为两个独立的阶段：reduce-scatter 和 all-gather。

![all-reduce = reduce-scatter 和 all-gather](images/all-reduce-reduce-scatter-all-gather.png)
([来源](https://engineering.fb.com/2021/07/15/open-source/fsdp/attachment/fsdp-graph-2a/))

以下是各种并行策略分别使用哪些集合通信的拆解：

- DDP：对梯度做 1 次 all-reduce——理想情况下与计算重叠——总通信量：2 倍模型参数量的通信
- ZeRO-DP ZeRO-1/ZeRO-2：对梯度做 1 次 reduce-scatter，外加 1 次 all-gather 来收集所有更新后的参数——总通信量：2 倍模型参数量的通信
- ZeRO-DP ZeRO-3：对权重做 2 次 all-gather（`forward` 之前 + `backward` 之前），外加 1 次对梯度的 reduce-scatter——总通信量：3 倍模型参数量的通信（比 DDP 和 ZeRO-1/ZeRO-2 多 1.5 倍）
- TP：2 次 all-gather 和 2 次 reduce-scatter
- PP：2 次 send + 2 次 recv——在稳定阶段与计算重叠
- SP：取决于实现：设隐藏层维度为 h，序列长度为 N，并行度为 P
  * Megatron-LM：2 次 all-gather 和 2 次 reduce-scatter，每个 transformer 层的通信量为 `4*N*h` [论文 3.2 节](https://arxiv.org/abs/2309.14509)
  * DeepSpeed Ulysses：2 次 all-to-all，每个 transformer 层的通信量为 `4*N*h/P` [论文 3.2 节](https://arxiv.org/abs/2309.14509)

你可能会发现不同的实现使用不同的通信模式。


## 使用 ZeRO 时对节点间速度的要求 {#inter-node-speed-requirements-to-use-zero}

ZeRO 扩展协议，无论是 DeepSpeed ZeRO 还是 PyTorch FSDP，都比 TP+PP+DP 方案需要多得多的节点间流量，而且有时无法利用更快的节点内互联；因此如果你的节点间网络很慢，昂贵的 GPU 可能会被通信严重卡住瓶颈。

ZeRO 协议会把通信与计算部分重叠，所以理想情况是尽量接近 `comms_time <= compute_time`。这种重叠并不完美，所以总会留下一些网络瓶颈，但我们要确保 `comms_time` 不要远大于 `compute_time`。

在 ZeRO-3 中，`forward` 里对权重做 `all_gather`，然后在 `backward` 里再对权重做一次 `all_gather`，最后是在 backward 中对梯度做 `reduce_scatter`。总共有 3 次全局集合通信调用，每次发送的数据量是模型大小乘以每个参数占用的字节数。例如，一个 10B 参数的模型以 bf16 运行 ZeRO-3，需要发送 `10*2*3` = 60GB 的数据。

相比之下，[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)（DDP）只做一次 `all_reduce` 调用，但这需要 2 倍的数据传输量，所以一个 10B 参数的模型以 bf16 运行 DDP 需要发送 `10*2*2` = 40GB 的数据。

ZeRO-1 只分片优化器状态，和 DDP 一样，也需要传输 40GB 的数据（一次 `all_gather` 和一次 `reduce_scatter`）。

下面是如何以秒为单位计算通信和计算的时间：

- `comms_time = comms_multiplier * n_bytes * model_size_in_B / inter-node-throughput_in_GBps`
- `compute_time = n_passes * n_bytes * model_size_in_B * seqlen * global_batch_size / (total_gpus * 1e3 * tflops_wo_comms)`

计算时间的公式只是一个粗略估计，适用于任何基于 Transformer 块的模型。它忽略了所有小的计算，只计入那些大规模的 `matmul`。

通信时间公式同样是一阶估计。`comms_multiplier` 是单步中执行的各项集合通信的[busbw 修正因子](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth)之和——`all_reduce` 为 `2(n-1)/n`，`all_gather` 和 `reduce_scatter` 各为 `(n-1)/n`，其中 `n` 是 rank 数——DDP 的 2 和 ZeRO-3 的 3 就是这么来的。它略去了 `(n-1)/n` 这一项，在 rank 数很大时这是不错的近似（在下面用到的 512 个 rank 时是 0.998），但 rank 数很小时就过于乐观了（8 个 rank 时是 0.875）。而且代入的吞吐量必须是实测的 `busbw`，而不是 `algbw`，因为随着 rank 增多 `algbw` 会缩水——不过即便是 `busbw`，在相同负载下不同集合通信之间也可能有些差异，因为 NCCL 可能为每种通信选择不同的算法或协议。

作为一次实验，让我们使用 [IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b/) 训练中的数据点。

我们曾用 340Gbps 的 EFA 训练 IDEFICS-80B，在 A100 上用 DeepSpeed ZeRO-3 只能得到 90TFLOPS，而用 Megatron 的 TP+PP+DP 可以得到 150+TFLOPS。而且模型有很大一块是冻结的，因为我们是在一个语言模型和一个视觉模型的基础上构建新模型。所以我们的乘数其实小于 3。另一方面，为了节省显存我们使用了激活重计算，这就多了一次所有模型权重的传输；更雪上加霜的是，当时 nccl 还不支持真正的半精度归约，所以我们用 fp32 做梯度归约，这样我们的乘数实际上不是 3 而是接近 4.5。

IDEFICS-80B 训练所用的取值：
- `model_size_in_B` = `80`
- `n_bytes` = `2`，因为 bf16 是 2 字节
- `comms_multiplier` = ZeRO-3/FSDP 情况下为 `3`（1 次 reduce_scatter + 2 次 all_gather（前向 + 反向）），ZeRO-1 情况下为 2（1 次 reduce_scatter + 1 次 all_gather），
- 另外，在 IDEFICS-80B 的情况下，我们决定[用 fp32 归约梯度](../dtype.md#when-to-use-fp32-accumulators)，以减小 NCCL 的累积损失，所以我们实际上有 `comms_multiplier*n_bytes=3*2+2=4*2`，多出 2 字节；但由于一半的模型是冻结的，实际只发送了大约一半的梯度，所以乘数仍然是 3。
- `n_passes` = 使用激活重计算时为 `4`，不使用时为 `3`。模型每次 `forward` 只需要计算 1 次，每次 `backward` 计算 2 次（因为梯度要算两遍——一遍相对于输入，一遍相对于权重）。而使用激活重计算时会多做一次 `forward`。
- `total_gpus` = `512`
- `global_batch_size` = `3584`
- `seqlen` = `1024`
- `inter-node-throughput_in_GBps` = 42.5（340Gbps）（AWS EFA v1）
-`tflops_wo_comms` 是不含通信开销的 tflops。不是理论峰值，因为那是无法达到的；对 A100@BF16 而言大约可以取 80%，所以是 `312*0.8=~250` TFLOPS（取整）。

我们用 `all_reduce_bench.py` 以 4GiB 负载测得了 340Gbps 的节点间网络吞吐（该脚本现已改名为 [torch-dist-bench.py](../../network/benchmarks/torch-dist-bench.py) ）。IDEFICS-80B 有 80 层，所以每层大约对应 1B 参数。这意味着 bf16 张量时每层发送 2GB 数据，fp32 张量时是 4GB 数据，与网络基准测试的结果吻合。如果你的层要小得多，我建议把基准测试的负载调整为对应的规模。例如，如果你的层只有 100M 参数，那么 bf16 张量时负载就只有 0.2GB。由于这小了整整一个数量级，网络很可能给出更低的带宽，你应该把这个值代入计算。

脚注：如果你模型的一部分是冻结的，那么在同步梯度时发送的数据会更少。在 IDEFICS 中我们有超过一半的模型是冻结的，所以归约梯度时只有大约一半的流量。

于是我们得到：

- comms = `3 * 2 * 80 / 42.5` = 11 秒
- compute = `4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 250)` = 18 秒

对照我们 IDEFICS-80B 的日志，可以看到每次迭代大约用了 49 秒。

所以好消息是，数学上是吻合的：通信 + 计算与实测时间处于同一量级。

我们还可以做另一个合理性检验：把日志中记录的 90TFLOPS 代入计算时间公式，这时：

- compute = `4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 90)` = 51 秒

这样 49 秒和 51 秒相当接近。但这个检验说明不了什么，因为记录的 TFLOPS 本来就是按这个公式算出来的，所以它当然会对得上。

我期望的最好情况是：在公式中代入接近理论峰值的 TFLOPS，得到的计算时间估计与在系统上实测到的实际计算时间大致相同。要记得，由于通信与计算是交错的，我们测量 `forward`+`backward` 挂钟时间时，其中已经包含了通信。

结论是什么？我认为还需要更多调查，因为这里显然还有其他隐藏的瓶颈。我已经没有机会接触那套环境来继续调查，所以等我下次训练另一个较大的模型时，我会重新做一遍这套推算，并把更新后的数字分享给你们。不过这次推算应该能让你感受到幕后到底发生了什么，以及这些数字是如何彼此关联的。

另外，上面的讨论没有把梯度累积步数（GAS）计入这些计算。IDEFICS-80B 的情况没有用到它。如果 GAS>1，理论计算时间不变，但通信时间会从 `3*2*M/GBps` 变成 `GAS*3*2*M/GBps`。为 `forward` 和 `backward` 通过 `all_gather` 收集权重的操作，会随着梯度累积步数进行同样多次。理论上梯度只需要做一次，但由于没有地方存放每块 GPU 上已收集权重的中间梯度，梯度归约也不得不做 GAS 次。这是针对 ZeRO-2 和 ZeRO-3 而言。对 ZeRO-1，GAS>1 不需要额外通信。

我们这里也没有讨论 `DataLoader` 是否可能是瓶颈，不过我们测试过它在 1 秒以内，即开销非常小。

回到通信的计算，我们也没有考虑各种硬件延迟，但在负载很大时，它们不应该带来明显的额外开销。

现在你知道在自己的系统的网络上传输这么多 GB 需要多长时间了。例如，如果网络比我们训练 IDEFICS-80B 时用的慢 5 倍，即 8.5GBps（68Gbps），那么：

- comms = `3 * 2 * 80 / 8.5` = 56 秒

与更快的计算相比，这肯定是一个巨大的瓶颈。

如果网络快 5 倍，即 212GBps（1700Gbps），那么：

- comms = `3 * 2 * 80 / 212` = 2 秒

与计算时间相比这就微不足道了，尤其是其中一部分还能与计算成功重叠的话。

此外，DeepSpeed 团队还在 384 块 V100 GPU（24 台 DGX-2 节点）上实测[对 176B 模型做了基准测试](https://github.com/deepspeedai/DeepSpeed/issues/2928#issuecomment-1463041491)，发现：

1. 用 100Gbps 的 IB，我们每块 GPU 只有 <20TFLOPS（糟糕）
2. 用 200-400Gbps 的 IB，我们达到每块 GPU 约 30-40TFLOPS（可以接受）
3. 用 800Gbps 的 IB，我们达到每块 GPU 40+TFLOPS（优秀）

提醒一下，NVIDIA V100 在 fp16 下的峰值 TFLOPS 是 [125TFLOPS](https://www.nvidia.com/en-gb/data-center/tesla-v100/)。

但这里要小心——这个基准测试是在 V100 上做的！在半精度下，V100 比 A100 慢大约 2-3 倍，比 H100 慢 4-8 倍。所以要让 H100 节点在半精度下达到上表的水平，通信至少得快 4-8 倍。我们需要在更新的硬件上做更多基准测试。

脚注：2-3 倍这个范围——因为官方规格声称 V100→A100 和 A100→H100 各自都有 3 倍的 TFLOPS 提升，但实际测量两者差距的用户报告最多只有 2.5 倍。

他们还注意到，在大规模训练时，每块 GPU 的微批次越小，通信开销就越明显。而我们可能无法增大微批次大小，因为全局批次大小通常是固定的，以保证良好的模型收敛速率。这由新近提出的 [ZeRO++](#zero-with-multiple-replicas) 解决了。

最后，做上面的计算时，你需要知道自己在实际环境上能得到的真实带宽——它随负载大小变化——负载越大，带宽越好。要获得这个信息，你需要查看 DeepSpeed 配置文件中分别负责归约和预取的 `reduce_bucket_size` 与 `prefetch_bucket_size` 设置。默认是 0.5B 参数，半精度下即 1GB（0.5B x 2 字节），如果用 fp32 精度则是 2GB（0.5B x 4 字节）。因此，为了测量真实吞吐，你需要用这个负载跑一次 `all_reduce` 基准测试，看看报告的是什么带宽，然后把它代入上面的计算。



## 何时该用哪种策略 {#which-strategy-to-use-when}

下面是一个非常粗略的提纲，说明什么时候该用哪种并行策略。每条列表中排第一的通常更快。

**⇨ 单张 GPU**

* 模型能放进单张 GPU：

    1. 常规使用即可

* 模型放不进单张 GPU：

    1. ZeRO + 卸载（offload）到 CPU，可选再卸载到 NVMe
    2. 在上述基础上再加上以内存为中心的切分（详见下文），适用于最大的单层放不进单张 GPU 的情况

* 最大的单层放不进单张 GPU：

    1. ZeRO - 启用[以内存为中心的切分（Memory Centric Tiling）](https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling)（MCT）。它能把任意大的层自动切分开来并依次执行，从而让你可以运行任意大的层。MCT 减少 GPU 上同时存活的参数量，但不会影响激活显存。由于截至 2026-08 这种需求非常罕见，需要由用户手动覆盖 `torch.nn.Linear`。

**⇨ 单节点 / 多 GPU**

* 模型能放进单张 GPU：

    1. DDP - 分布式 DP
    2. ZeRO - 视情况和所用配置而定，可能更快，也可能不更快

* 模型放不进单张 GPU：

    1. PP
    2. ZeRO
    3. TP

    在有 NVLINK 或 NVSwitch 这种极快的节点内互联时，三者应该大体不相上下；没有它们时，PP 会比 TP 或 ZeRO 更快。TP 的并行度也可能带来差别。最好在你的具体环境上多做实验，找出胜者。

    TP 几乎总是用在单个节点内，也就是 TP size <= 每节点 GPU 数。

* 最大的单层放不进单张 GPU：

    1. 如果不使用 ZeRO，就必须用 TP，因为光靠 PP 装不下。
    2. 如果搭配 ZeRO，参见上面“单张 GPU”的对应条目。


**⇨ 多节点 / 多 GPU**

* 如果模型能先放进单个节点，优先尝试 [ZeRO 多副本](#zero-with-multiple-replicas)，因为这样你是在更快的节点内互联上跑 ZeRO，只在较慢的节点间网络上跑 DDP

* 当你的节点间互联很快时：

    1. ZeRO - 几乎不需要对模型做任何改动
    2. PP+TP+DP - 通信更少，但需要对模型做大量改动

* 当你的节点间互联很慢，且显存仍然吃紧时：

    1. DP+PP+TP+ZeRO-1
