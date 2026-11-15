# 使用 SLURM 的单节点与多节点启动器 {#single-and-multi-node-launchers-with-slurm}

下面这些是完整的 SLURM 脚本，演示如何把各种启动器与使用 `torch.distributed` 的软件集成起来（不过它们应该也很容易改造后用于其他分布式环境）。

- [torchrun](torchrun-launcher.slurm) - 与 [PyTorch distributed](https://github.com/pytorch/pytorch) 配合使用。
- [accelerate](accelerate-launcher.slurm) - 与 [HF Accelerate](https://github.com/huggingface/accelerate) 配合使用。
- [lightning](lightning-launcher.slurm) - 与 [Lightning](https://lightning.ai/)（“PyTorch Lightning”和“Lightning Fabric”）配合使用。
- [srun](srun-launcher.slurm) - 与 SLURM 原生的启动器配合使用 - 这里我们必须手动预先设置 `torch.distributed` 所期望的环境变量。

以上所有脚本都使用 [torch-distributed-gpu-test.py](../../../debug/torch-distributed-gpu-test.py) 作为演示脚本，你只需要执行下面的命令就可以把它复制到当前目录：
```bash
cp ../../../debug/torch-distributed-gpu-test.py .
```
这里假设你已经克隆了这个仓库。但你也可以把它替换成任何你需要的其他脚本。
