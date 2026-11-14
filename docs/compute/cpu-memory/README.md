# CPU 内存 {#cpu-memory}

本章篇幅很短，因为通常关于 CPU 内存需要了解的门道非常少——这是件好事！

ML 负载的大部分计算都发生在 GPU 上，但每个节点上的 CPU 内存通常应当至少与 GPU 上的显存一样多。例如，如果你在一台配备 8 块 80GiB GPU 的 H100 节点上，你就拥有 640GiB 的显存。因此你至少希望拥有同样多的 CPU 内存。目前大多数高端云套餐通常配备 1-2TiB 的 CPU 内存。

## ML 负载需要 CPU 内存做什么 {#what-cpu-memory-is-needed-for-in-ml-workloads}

- 加载模型权重，除非权重被直接加载到 GPU 上——这通常是临时的内存占用，一旦模型被移动到 GPU 上，占用就会回落到零。
- 保存模型权重。在某些情况下，每块 GPU 会直接把自己的检查点写入磁盘；在另一些情况下，模型会先在 CPU 上重新组装，然后再写入磁盘——这同样是临时的内存占用。
- 使用 [DeepSpeed](https://www.deepspeed.ai/tutorials/zero-offload/) 这类框架时，可能会把参数和优化器状态卸载（offloading）到 CPU。这种情况下可能需要相当多的 CPU 内存。
- 在 `forward` 过程中计算的激活值需要在 `backward` 路径中可用，它们也可以被卸载到 CPU，而不是先丢弃、再在反向传播期间重新计算，从而省去不必要的开销
- `DataLoader` 通常是 CPU 内存的主要使用者之一，有时会消耗非常大的内存量。通常每个节点上至少运行着 2x 8 个数据加载器工作进程，所以你需要足够的内存来支撑至少 16 个各自持有部分数据的进程。例如，在从云端流式读取数据的情况下，如果数据分片很大，这些进程很容易就能吃掉数百 GB 的 CPU 内存。
- 软件本身及其依赖库会占用一点 CPU 内存，但这部分数量通常可以忽略不计。

## 需要了解的事项 {#things-to-know}

- 如果 `DataLoader` 以 `mmap` 模式使用 HF `datasets`，常驻内存（Resident memory）占用看起来可能像是在海量使用 CPU 内存，因为它会试图把整个数据集映射到内存中。但这其实有误导性，因为如果别处需要内存，操作系统会把任何不需要的 mmap 映射页换出（page out）回系统。你可以[在这里](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/)阅读更多相关内容。当然，这种认识适用于任何使用 `mmap` 的数据集，我用 HF `datasets` 只是举个例子，因为它的使用非常广泛。
