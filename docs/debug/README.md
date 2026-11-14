# 调试与故障排查 {#debugging-and-troubleshooting}


## 指南 {#guides}

- [调试 PyTorch 程序](./pytorch.md)

- [诊断多节点多 GPU Python 程序中的挂起与死锁](./pytorch.md#diagnosing-crashes-hangs-and-tracing-execution)

- [网络调试](../network/debug/README.md)

- [排查 NVIDIA GPU 故障](../compute/accelerator/nvidia/debug.md)

- [下溢与上溢检测](./pytorch.md#underflow-and-overflow-detection)



## 工具 {#tools}

- [调试工具](./tools.md)

- [torch-distributed-gpu-test.py](./torch-distributed-gpu-test.py) - 这是一个 `torch.distributed`
  诊断脚本，用于检查集群（一个或多个节点）中的所有 GPU 能否相互通信并分配显存。

- [NicerTrace](./NicerTrace.py) - 这是一个改进版的 `trace` python 模块，为其构造函数添加了多个额外的标志（flag），并提供了更有用的输出。
