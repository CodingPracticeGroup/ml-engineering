# SLURM 管理 {#slurm-administration}


## 在多个节点上运行命令 {#run-a-command-on-multiple-nodes}

1. 为了避免在你尚未登录过的每个新节点上都被询问：
    ```
    Are you sure you want to continue connecting (yes/no/[fingerprint])?
    ```
    你可以通过以下做法关闭这个检查：
    ```bash
    echo "Host *" >> ~/.ssh/config
    echo "  StrictHostKeyChecking no" >> ~/.ssh/config
    ```

    当然，请自行确认这样是否满足你的安全需求。我这里假设你已经在 SLURM 集群内部，并不是通过 ssh 连接到集群之外的机器。你也可以选择不做这个设置，那样你就必须逐个手动批准每一个新节点。

2. 安装 `pdsh`

现在你就可以在多个节点上运行想要的命令了。

例如，我们来运行 `date`：

```bash
$ PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26] date
node-25: Sat Oct 14 02:10:01 UTC 2023
node-21: Sat Oct 14 02:10:02 UTC 2023
node-23: Sat Oct 14 02:10:02 UTC 2023
node-24: Sat Oct 14 02:10:02 UTC 2023
node-26: Sat Oct 14 02:10:02 UTC 2023
```

我们来做点更有用、更复杂的事情。我们来杀掉所有在 SLURM 作业被取消后仍未退出的、占用 GPU 的进程：

首先，下面的命令会给出所有占用 GPU 的进程的进程 id：

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq
```

因此我们现在可以一次性把这些进程全部杀掉：

```bash
 PDSH_RCMD_TYPE=ssh pdsh -w node-[21,23-26]  "nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort | uniq | xargs -n1 sudo kill -9"
```


## SLURM 设置 {#slurm-settings}

查看 slurm 的设置：

```bash
sudo scontrol show config
```

配置文件位于 slurm 控制节点上的 `/etc/slurm/slurm.conf`。

更新 `slurm.conf` 之后，若要重新加载配置，请运行：
```bash
sudo scontrol reconfigure
```
（此命令需在控制节点上执行。）


## 节点管理 {#node-management}

### 更改节点的状态 {#changing-the-state-of-the-node}

更改操作由 `scontrol update` 完成

示例：

要解除一个已经准备好可供使用的节点的 drain 状态（即恢复其可用）：
```bash
scontrol update nodename=node-5 state=idle
```

要把一个节点从 SLURM 的可用节点池中移除：
```bash
scontrol update nodename=node-5 state=drain reason="maintenance"
```


### 解除因进程退出缓慢而被 drain 的节点的状态 {#undrain-nodes-killed-due-to-slow-process-exit}

有时在作业被取消之后，进程退出得很慢。如果 SLURM 被配置为不无限等待下去，它就会自动把这样的节点置为 drain 状态。但这些节点没有理由不能继续提供给用户使用。

下面是把这个过程自动化的方法。

关键是拿到那些因为 `"Kill task failed"` 而被 drain 的节点列表，可以用下面的命令获取：

```bash
sinfo -R | grep "Kill task failed"
```

然后提取并展开这个节点列表，确认这些节点上确实没有用户进程（或者先尝试把它们杀掉），再解除它们的 drain 状态。

前面你已经学过如何[在多个节点上运行命令](#run-a-command-on-multiple-nodes)，这个脚本就会用到那种做法。

下面这个脚本会替你完成所有这些工作：[undrain-good-nodes.sh](./undrain-good-nodes.sh)

现在你只要运行这个脚本，那些其实已经可以服务、但当前处于 drain 状态的节点就会被切换为 `idle` 状态，重新可供用户使用。


### 自动重启 {#auto-reboot}

如果需要安全地重启节点（例如镜像已经更新），请按需要修改节点列表，然后运行：

```bash
scontrol reboot ASAP node-[1-64]
```

对于每个非 idle 的节点，这条命令会等到当前作业结束之后，再重启该节点，并把它恢复为 `idle` 状态。

注意，要让它正常工作，需要在 `/etc/slurm/slurm.conf` 中设置：
```
RebootProgram = "/sbin/reboot"
```
（该配置文件位于控制节点上；如果你刚刚才把这一条目加进配置文件，还需要重新加载 SLURM 守护进程的配置。）


## 作业管理 {#job-management}

### 修改作业的时限 {#modify-a-jobs-timelimit}

要为作业设置新的时限（timelimit），例如 2 天：
```bash
scontrol update JobID=$SLURM_JOB_ID TimeLimit=2-00:00:00
```

要在之前的设置基础上追加时间，例如再加 10 小时：
```bash
scontrol update JobID=$SLURM_JOB_ID TimeLimit=+10:00:00
```

## 当 SLURM 出现问题时 {#when-something-goes-wrong-with-slurm}

分析 SLURM 日志文件中的事件日志：
```bash
sudo cat /var/log/slurm/slurmctld.log
```

例如，这可以帮助你弄清楚为什么某个节点上的作业会被提前取消，或者某个节点为什么被彻底移除。
