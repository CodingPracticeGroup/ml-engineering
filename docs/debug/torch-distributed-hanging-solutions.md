# 诊断多节点多 GPU Python 程序中的挂起与死锁 {#diagnosing-hangings-and-deadlocks-in-multi-node-multi-gpu-python-programs}

**已移至[调试 PyTorch 程序](./pytorch.md#diagnosing-crashes-hangs-and-tracing-execution)。**

## 辅助工具 {#helper-tools}

**已移至[调试 PyTorch 程序](./pytorch.md#helper-tools)。**

## 诊断多 GPU 挂起/死锁的方法 {#approaches-to-diagnosing-multi-gpu-hanging--deadlocks}

**已移至[调试 PyTorch 程序](./pytorch.md#approaches-to-diagnosing-multi-gpu-hanging--deadlocks)。**

### py-spy {#py-spy}

**已移至[调试 PyTorch 程序](./pytorch.md#py-spy)。**

#### 多进程 py-spy {#multi-process-py-spy}

**已移至[调试 PyTorch 程序](./pytorch.md#multi-process-py-spy)。**

#### 通过 srun 在多节点上使用 py-spy {#multi-node-py-spy-via-srun}

**已移至[调试 PyTorch 程序](./pytorch.md#multi-node-py-spy-via-srun)。**

#### 通过 pdsh 在多节点上使用 py-spy {#multi-node-py-spy-via-pdsh}

**已移至[调试 PyTorch 程序](./pytorch.md#multi-node-py-spy-via-pdsh)。**

#### 通过 ds_ssh 在多节点上使用 py-spy {#multi-node-py-spy-via-ds_ssh}

**已移至[调试 PyTorch 程序](./pytorch.md#multi-node-py-spy-via-ds_ssh)。**

### 网络层面的挂起 {#network-level-hanging}

**已移至[调试 PyTorch 程序](./pytorch.md#network-level-hanging)。**

### 隔离有问题的 GPU {#isolate-problematic-gpus}

**已移至[调试 PyTorch 程序](./pytorch.md#isolate-problematic-gpus)。**

### python 的 `trace` {#python-trace}

**已移至[调试 PyTorch 程序](./pytorch.md#python-trace)。**

#### NicerTrace {#nicertrace}

**已移至[调试 PyTorch 程序](./pytorch.md#nicertrace)。**

#### 处理生成的 trace 文件 {#working-with-generated-trace-files}

**已移至[调试 PyTorch 程序](./pytorch.md#working-with-generated-trace-files)。**

### 老办法 `print` {#good-old-print}

**已移至[调试 PyTorch 程序](./pytorch.md#good-old-print)。**

### core 文件 {#core-files}

**已移至[调试 PyTorch 程序](./pytorch.md#core-files)。**

### 代码循环 {#code-loops}

**已移至[调试 PyTorch 程序](./pytorch.md#code-loops)。**

## 硬件相关的问题 {#hardware-specific-issues}

**已移至[调试 PyTorch 程序](./pytorch.md#hardware-specific-issues)。**

### AMD/ROCm 在启用 IOMMU 时出现挂起或缓慢 {#amdrocm-hangs-or-slow-with-iommu}

**参见[排查 AMD GPU 问题](../compute/accelerator/amd/debug.md#hangs-or-slow-multi-gpu-runs-and-iommu)。**
