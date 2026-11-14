# 训练 {#training}

**子章节**：

- [模型并行](model-parallelism/README.md)

- [性能](performance/README.md)

- [故障容错](fault-tolerance/README.md)

- [可复现性](reproducibility/README.md)

- [不稳定性](instabilities/README.md)

- [检查点](checkpoints/README.md)

- [训练超参数与模型初始化](hparams.md)

- [张量精度 / 数据类型](dtype.md)

- [仅用单个节点模拟多节点环境](emulate-multi-node.md) - 介绍如何仅用单个节点模拟多节点环境 - 这里我们使用 `deepspeed` 启动器。

- [使用微调示例从零重新训练 HF hub 模型](re-train-hub-models.md)

- [数据集](datasets.md)

**工具**：

- [printflock.py](tools/printflock.py) - 一个小巧的库，可让你的 `print` 调用在多 GPU 环境中不会交错输出。

- [multi-gpu-non-interleaved-print.py](tools/multi-gpu-non-interleaved-print.py) - 一个基于 `flock` 的 `print` 包装器，可防止多个进程同时打印时消息相互交错 - 在使用 `torch.distributed` 并配合多个 GPU 时就会出现这种情况。
