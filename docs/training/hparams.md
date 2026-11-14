# 选择训练超参数与模型初始化 {#selecting-training-hyper-parameters-and-model-initializations}

要找到一组合适的超参数和模型初始化的起始配置，最简单的办法是直接借鉴某次你已知取得成功的类似训练。这里有一个[公开的 LLM/VLM 训练日志合集](https://github.com/CodingPracticeGroup/ml-engineering/blob/master/resources/README.md#publicly-available-training-llmvlm-logbooks)可以帮你起步。另一个常见来源是论文，前提是论文中披露了这些信息。如果论文没有发表这些细节，你也可以尝试联系作者，向他们询问。

## 术语表 {#glossary}

训练领域的行话使用大量缩写和术语，因此这里列出本章中一些重要的词条。

- BS：批次大小（Batch Size）——这里指每块 GPU 的批次大小，通常也被称为 MBS（micro-batch-size，微批次大小）
- GAS：梯度累积步数（Gradient Accumulation Steps）——完成一次完整迭代之前需要执行多少次前向/反向传播循环
- GBS：全局批次大小（Global Batch Size）——每次迭代的总批次大小——可能包含梯度累积
- PP：流水线并行（Pipeline Parallelism）
- TFLOPS：每秒万亿次浮点运算（Trillion FLOPs per second）——[FLOPS](https://en.wikipedia.org/wiki/FLOPS)

## 逐步提升全局批次大小 {#global-batch-size-ramp-up}

如果你打算用非常大的 GBS 来训练，比如 1024、2048 个样本甚至更多，那么在刚开始训练时，把这么大的批次喂给模型是非常浪费的。此时模型完全是随机的，无法从过于精细的数据中获益。因此，为了节省数据和资源，人们通常会在一段时间内逐步提升全局批次大小。

同样重要的是，起始 GBS 也不能太小，否则训练进度不会高效。数据太少时，计算（TFLOPS）效率很低，会拖慢整个训练。在使用流水线并行（PP）时尤其如此，因为 PP 调优最关键的是把 GPU 空闲气泡（bubble）控制在很小的范围内，而 GBS 越小，气泡就越大。

例如，在我们确实使用了 PP 的 BLOOM-176B 项目中，做完吞吐量基准测试后我们发现，从 GBS=16 开始训练慢得难以置信（8TFLOPS），所以我们最终从 GBS=192（73TFLOPS）起步，然后逐步提升直到 GBS=2048（150TFLOPS）——每训练 9_765_625 个样本就把 GBS 增加 16。



### STD 初始化 {#std-init}

这个超参数极其重要，必须借助数学推导才能确定正确的值。详情请见 [STD 初始化](https://github.com/CodingPracticeGroup/ml-engineering/blob/master/training/instabilities/README.md#std-init)。
