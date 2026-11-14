# 处理数据集 {#dealing-with-datasets}

## 在主进程上预处理并缓存数据集 {#preprocessing-and-caching-datasets-on-the-main-process}

HF Accelerate 有一个非常巧妙的容器 [`main_process_first`](https://huggingface.co/docs/accelerate/v0.4.0/accelerator.html#accelerate.Accelerator.main_process_first)，它让你可以写出这样的代码：

```python
with accelerator.main_process_first():
    # 加载并预处理数据集
    dataset = datasets.load_dataset(...)
    # 可选：将其缓存，并让其余进程加载该缓存
```
而不必使用这种不够直观、还需要重复代码的写法：
```python
if rank == 0:
    dataset = datasets.load_dataset(...)
dist.barrier()
if not rank == 0:
    dataset = datasets.load_dataset(...)
```

你应当只在主进程上下载和处理数据，而不是在所有进程上做，因为所有进程会并行地重复做同一件事，而且它们很可能会写入同一个位置，导致写入交错、结果损坏。从 IO 的角度看，把这类工作串行化也要快得多。

接下来是 `main_process_first` 和 `local_main_process_first`——前者用于数据位于共享文件系统上、所有计算节点都能看到它的情况。后者用于数据只存在于每个节点本地的情况。

如果你不使用 HF Accelerate，我复刻了类似的容器，只是把它们命名为：

- `global_main_process_first`——用于共享文件系统
- `local_main_process_first`——用于节点本地文件系统

你可以在[这里](tools/main_process_first.py)找到它们。

那么，如果你想写一段通用的代码，让它能在共享文件系统和本地文件系统上都自动工作，该怎么办？我添加了另一个辅助函数，它会自动发现我们面对的是哪种类型的文件系统，并据此调用正确的容器。我把它命名为 `main_process_by_path_first`，用法如下：

```python
path = "/path/to/data"
with main_process_by_path_first(path):
    # 加载并预处理数据集
    dataset = datasets.load_dataset(...)
    # 可选：将其缓存，并让其余进程加载该缓存
```

你可以在[这里](tools/main_process_first.py)找到它。

当然，除了容器之外，你还会需要用于检查主进程类型的工具函数，因此有 3 个与这些容器一一对应的工具函数：

- `is_main_process_by_path(path)`
- `is_local_main_process()`
- `is_global_main_process()`

它们全部可以在[这里](tools/main_process_first.py)找到。

你可以通过运行下面的命令来看它们实际工作：

```bash
python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d tools/main_process_first.py
```
