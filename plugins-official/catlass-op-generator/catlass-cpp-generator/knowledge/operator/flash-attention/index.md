# flash-attention

- [FA dense 前向 A2/A3 实现模式](flash-attention-infer-patterns.md)——含设计路由、标杆参考、实现规则、手搓手册、配方集、复合特性册、stage 设计、坑清单、评测方法论的全量正文
- [FA 变体族 A2/A3 手搓配方精要](fa-variants-patterns.md)——含 FA-Paged、fa-sink、fa-varlen、MLA（配方+契约）、FFA 六份手搓配方的全量逐步骤正文
- [FA-SageAttention Ascend 950 量化注意力实现模式](fa-sageattention-patterns.md)——含量化注意力架构规格、开发配方、精度取证、平台坑全集的全量正文

本 family 仅收录 `flash_attention`（dense FA 前向，MHA/GQA/fused attention，`aclnnFlashAttentionScore`
语义）领域知识，全量收录各配方的逐步骤正文、代码块与参数推导公式（自包含，闭卷可据此从零重生成
kernel）。标准 FA 与变体（paged/sink/varlen/q8/FFA/MLA）按 `--arch atlas_a2_a3` 查询；
SageAttention 为 Ascend 950（dav-3510）量化路径，按 `--arch ascend950` 查询。不得把本族的 ABI、
PAGED、tiling、资源或同步结论外推到 block 稀疏（BSA/BSAG）、MLA 或反向（FAG）家族。
