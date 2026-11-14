# 在 SLURM 环境中工作 {#working-in-slurm-environment}

除非你很幸运，拥有一个完全由自己掌控的专用集群，否则你多半得使用 SLURM，与其他人分时共用 GPU。而且，即使你在 HPC 环境中训练、并且被分配了一个专用分区，很多时候你仍然必须使用 SLURM。

SLURM 这个缩写的全称是：**Simple Linux Utility for Resource Management**（简单的 Linux 资源管理工具）——不过它现在改名为 The SLURM Workload Manager（SLURM 工作负载管理器）。它是一个免费、开源的作业调度器，适用于 Linux 及类 Unix 内核，被世界上许多超级计算机和计算机集群所采用。

本书的这几章不会试图把 SLURM 讲得面面俱到，因为外面已经有很多手册了；我们只介绍一些对训练过程很有帮助的具体细节。

- [SLURM 用户指南](./users.md) —— 在 SLURM 环境中开展训练，你需要知道的一切都在这里。
- [SLURM 管理](./admin.md) —— 如果你不走运，除了使用 SLURM 集群之外还得负责管理它，这份文档里有一份不断扩充的“食谱”（快速解决问题的操作配方），能帮你更快地把事情办完。
- [性能](./performance.md) —— SLURM 在性能方面的注意事项。
- [启动脚本](./launchers/README.md) —— 如何在 SLURM 环境中用 `torchrun`、`accelerate`、pytorch-lightning 等工具启动训练
