# Kubernetes {#kubernetes}

就我个人看来，Kubernetes 并不是做训练工作的合适环境，[SLURM](../slurm/README.md) 才是远为易用的编排系统。SLURM 构建在 Unix 之上，因此它只是让多节点协调变得更容易。而 k8s 则把轮子彻底重新发明了一遍，带来大量复杂性，它的功能集合中还缺少许多常规的 Unix 特性，使人在其中工作变得极具挑战：你不得不维持一支庞大的支持团队，仅仅为了应付那么多的 k8s 问题以及基础 Unix 特性的缺失（例如，你甚至无法拥有各自的 Unix 用户名，所有用户都在同一个 Unix 用户名下运行——想想责任追溯和安全性！）。

即便被迫使用 k8s，人们也总能想办法把工作完成，但训练团队为此要付出巨大的人力和金钱代价。

本章会非常简短，因为我无法想象该如何在这里把 k8s 的种种复杂性都涵盖进来，所以我只会介绍一些重要的坑，希望能让你的工作轻松一点。

## 环境配置 {#setup}

### 应对 CPU OOM 事件导致的作业重置 {#overcoming-job-reset-on-cpu-oom-event}

这个自 k8s v1.28 起默认启用的特性，放到交互式训练作业的语境里完全说不通。只要你的任何一个进程发生 CPU OOM（内存耗尽），就会被踢出，作业随之被重置。这个特性用于在生产环境中提供推理服务是好事，但对交互式训练工作来说是个大问题，比如当你在尝试寻找性能最优的训练配置时：本来你应该能从崩溃的进程中恢复、调整配置再重试，现在却只能从头开始，甚至连日志都没有——除非你及时手动把这些日志同步出来，否则问题根本无法定位。

所以要避免这一点，你需要让 k8s 在节点池级别或集群级别的配置中设置 `memory.oom.group = 0`——默认值是 `memory.oom.group = 1`，因此如果你的作业因 cpu-oom 事件被杀掉，请让你的 k8s 管理员做这项修改。取值 `0` 只会杀掉引发 cpu-oom 的那个进程。

具体做法如下。Kubernetes 1.32 引入了 kubelet 参数 [`singleProcessOOMKill`](https://github.com/kubernetes/kubernetes/pull/126096)，它允许你设置 `memory.oom.group = 0`。

```
compute:
  additionalNodePools:
    - name: foo
      kubeletConfig:
        singleProcessOOMKill: true
```


要在运行中的节点内部查看实际生效的设置，请执行：
```bash
$ cat /sys/fs/cgroup/memory.oom.group
```
取值为 `0` 表示修改已生效，发生 CPU OOM 事件时只会杀掉惹祸的那个进程；取值为 `1` 则是有问题的默认值，会把整个作业的进程组全部杀掉。
