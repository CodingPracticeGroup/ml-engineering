# 不稳定的避免、恢复与理解 {#avoiding-recovering-from-and-understanding-instabilities}

子章节：

* [理解训练损失模式](training-loss-patterns.md) - 尖峰、发散、顿悟（grokking）时刻、恢复训练等各种类型。

## 从训练日志中学习 {#learning-from-training-logbooks}

最好的学习方式是阅读[公开的 LLM/VLM 训练日志（logbook）](../../resources/README.md#publicly-available-training-llmvlm-logbooks)，因为在那里你可以确切地看到当时发生了什么、问题是如何被克服的。


## STD 初始化 {#std-init}

正确初始化张量的初始分布，会对训练的稳定性产生巨大影响。`std` 值并不是固定的，它取决于隐藏维度的大小。

这在我们 BLOOM 之前的 104B 实验中被证明是一个非常关键的设置。我们一直无法突破最初几千次迭代，直到后来才发现 Megatron-LM 中 0.02 的默认 `--init-method-std` 对我们的模型来说实在太大了。

我们参考了这两个来源：

1. "Transformers without Tears" 论文 https://arxiv.org/abs/1910.05895 给出的公式是：`sqrt(2/(NHIDDEN*5))`

2. 530B 训练论文 https://arxiv.org/abs/2201.11990 使用了更小的初始化公式：`sqrt(1/(NHIDDEN*3))`

我们决定采用 530B 论文中的那一个，因为它能得到更小的初始化值。

为了便于比较这两个公式，可以把它们改写为：

1. `sqrt(0.4000/NHIDDEN)`
2. `sqrt(0.3333/NHIDDEN)`

因此在 `NHIDDEN=14336` 时，计算结果是 `sqrt(1/(14336*3)) = 0.00482`，我们使用的就是这个值。这当然不是我们在 BLOOM-176B 训练期间没有出现稳定性问题的唯一原因，但我认为它是关键原因之一。


## 数值不稳定 {#numerical-instabilities}

另见调试章节中的[检测有问题的张量值](../../debug/pytorch.md#detecting-problematic-tensor-values)和[下溢与上溢检测](../../debug/pytorch.md#underflow-and-overflow-detection)，其中有用于定位 `inf`/`nan` 值的工具。

在处理低精度数值时，某些数学运算可能是不稳定的。

例如，请参见这篇非常有趣的 [PyTorch 数值稳定性指南](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

现在我们来看一个这一概念实际发挥作用的具体示例。

在使用 fp16 混合精度的 104B 训练实验中，[Corby Rosset](https://github.com/corbyrosset) 提出了以下改进，用于让[自注意力更稳定](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118)。

具体来说，这一[行代码](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/c839a8aa30731f71b3738d56009be9668508e366/megatron/model/transformer.py#L303)表明，`norm_factor` 是在 Query * Key 矩阵乘法完成之后才被乘上去的。如果 Q 和 K 的维度非常大，输出可能会爆炸式增大，而 `norm_factor` 无法把它救回来。

提议：把 `norm_factor` 移到内部，使 Q 和 K 在矩阵乘法之前就先被缩小：
```python
        matmul_result = torch.baddbmm(
            matmul_result,
            1.0/math.sqrt(self.norm_factor) * query_layer.transpose(0, 1),   # [b * np, sq, hn]
            1.0/math.sqrt(self.norm_factor) * key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0 if alibi is None else 1.0, alpha=1.0)

        # 将视图形状变为 [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
```

要使该操作在数学上保持等价，把范数因子移到内部时，如果 n 是标量、A 和 B 是矩阵，就需要对 n 再取一次平方根：
```
n * (A dot B) === (sqrt(n) * A) dot (sqrt(n) * B)
```

这样一来，A 和 B 的维度就可以大得多。

同样的先乘后缩放问题在更下一层也存在，就在 CUDA 的 matmul API 本身之中。

对于 CUDA 内核编写者来说，[CuBlas](https://docs.nvidia.com/cuda/cublas/index.html) 的 `GemmStridedBatchedEx` 在写作本文时也存在类似的问题。其定义为：

```
C+i*strideC=αop(A+i*strideA)op(B+i*strideB)+β(C+i*strideC), for i ∈[0,batchCount−1]
```

问题在于 `alpha` 是在矩阵-矩阵乘法完成之后才相乘的，因此可能导致不稳定。

## “糟糕”的数据批次与模型参数状态的组合 {#bad-combination-of-data-batch-and-model-parameter-state}

PaLM 团队在训练更大的模型时，在“高度不规则的间隔”处观察到了数十次损失尖峰。虽然他们没能追查到底层根因，但通过从较早的检查点重新开始并跳过可能有问题的数据批次，缓解了这一问题。[第 5.1 节：训练不稳定](https://arxiv.org/pdf/2204.02311)


## Adam 中的时域相关性发散 {#time-domain-correlation-divergence-in-adam}

[A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871) 对在规模最大达 546B 参数的 LLM 训练过程中出现的发散尖峰进行了严谨的研究，并提出时域相关性（time-domain correlation）会导致 Adam 的发散。这是由 epsilon 值不够小触发的，此时梯度估计的各分量变得与 epsilon 相近。

在第 7.1 节中，他们提出了实用建议，其中最有趣的一条是把 epsilon 设为 0，并且可能需要处理除以零的情况。
