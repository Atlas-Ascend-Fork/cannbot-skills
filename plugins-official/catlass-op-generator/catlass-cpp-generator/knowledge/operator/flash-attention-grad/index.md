# flash-attention-grad

- [FAG dense 反向模式](flash-attention-grad-patterns.md)

本 family 仅收录 `flash_attention_grad`（dense FAG，`torch_npu.npu_fusion_attention_grad` 语义）的
Ascend 950 领域知识。查询时必须指定 `--arch ascend950`；不得把其 ABI、tiling、资源或同步结论
外推到 BSAG（块稀疏梯度）、BSA Arch22 或其它家族。
