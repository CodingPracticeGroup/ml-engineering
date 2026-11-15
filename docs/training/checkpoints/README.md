# 检查点 {#checkpoints}

- [torch-checkpoint-convert-to-bf16](./torch-checkpoint-convert-to-bf16) - 将现有的 fp32 torch 检查点转换为 bf16。如果发现了 [safetensors](https://github.com/safetensors/safetensors)，也会一并转换。应当可以很容易地改造后用于其他类似的使用场景。

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py) - 这个脚本用于修复这样一类检查点：由于某种原因，它们在保存时存入了底层存储（storage）比其视图（view）更大的张量。脚本会克隆当前的视图，然后仅以当前视图对应的存储重新保存这些张量。
