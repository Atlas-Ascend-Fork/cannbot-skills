---
type: CATLASS DSL Programming Concept
title: GEMM 主导 kernel 的优化阶梯
description: 当时间集中在 MAC 与 MTE2 两条管道时，按"定位 → 波次量化 → L1 双缓冲 → L0 双侧双缓冲 → 判停"的顺序推进，并记录各级的结构上限。
tags: [catlass-dsl, optimization-ladder, wave-quantization, l0-double-buffer, tiling]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-26T08:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-26T08:00:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: cube fractal 对齐与管道计时口径
  - id: measure
    resource: project-evidence:single-kernel-row-exclusive-optimization
    title: 行独占单 kernel 的逐轮实测与判停记录
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

主 kernel 的时间集中在 MAC 与 MTE2 两条管道时，优化收益几乎完全由结构性改动决定；
参数微调（K 步长、常量乘法、循环展开）通常只落在噪声范围内。按下面的阶梯推进，每级都有
可证伪的前置证据，级间不要跳步。

判定：fresh profile 中 `mac` 或 `mte2` 单项不低于墙钟的 0.7，且 `fixpipe` 占比低于 5%[^measure]。

# 用法

0. **定位关键路径**（探针消融，不改正式代码）
   - 依次把 A 侧 GM 加载量、B 侧 GM 加载量各削到 5%，再单独去掉 L1→L0 拷贝，各测一次。
   - 只有一条会明显降低墙钟；若 GM 两刀都不降而 L0 拷贝那刀降，说明瓶颈是"拷贝链与 MMAD
     串行"[^measure]，直接进入第 3 级，跳过第 2 级的收益预期。

1. **波次量化**（收益最大的一刀，先做）
   - 块数 = `ceil(M / BM)`，目标是块数接近核数的整数倍，使最后一波的空核比例最小。
   - `BM` 必须是 cube fractal 行数的整数倍[^api]。非整数倍的取值即使波次利用率接近 100%，
     MMA 结果也会错误；利用率指标必须在正确性复核通过后才可采信[^measure]。
   - 典型收益：波次利用率从约 75% 提到约 93%，墙钟下降 20% 以上。

2. **L1 双缓冲**（收益小，但解锁后续）
   - 半区偏移切分，每半区一套 ready/free flag，见
     [L1 半区偏移与双缓冲 flag 配对](pattern-l1-half-region-double-buffer.md)。
   - 单独做的典型收益只有 3%~5%，它的价值在于让第 3 级成立。

3. **L0 双侧双缓冲 + K 子块拆分**（决定性的一刀）
   - 把 K tile 拆成两个等长子块，L0A 与 L0B 同时双缓冲；只给单侧加槽几乎没有收益。
   - flag 结构：每个槽拥有自己的 ready/free 对，槽复用距离 2，依赖关系无环[^api]；跨核交接可参考
     [单个 cross flag 的双向互斥用法](pattern-cross-flag-bidirectional.md)。
   - 反例一：多个 ready flag 共享同一个 MTE1→CUBE 对会死锁，表现为 aicore 超时。
   - 反例二：改成一个共享 ready flag 虽不死锁，但搬运链只能领先一个子块，反而更慢。
   - 典型收益：20%~30%，是本阶梯里唯一能把拷贝链完整压进 MAC 的改动。

4. **判停**
   - `mac` 与 `mte2` 同时不低于墙钟的 0.95 时停止调参：两条管道都已贴住，单变量改动最多
     只剩几个百分点[^measure]。
   - 用"去掉某段工作后墙钟不变"的消融确认剩余工作确实被隐藏，再决定是否值得改写。

5. **结构上限的判断**
   - 行独占结构下每个行块都要重载整份权重，`mte2` 与块数成正比；继续增大 `BM` 会先损失
     波次利用率，MAC 时间反而上升[^measure]。
   - 越过这个上限需要改变 kernel 划分（例如沿 N/V 方向切分任务并引入第二段规约），
     属于结构变更而非调参，应在判停证据齐全后单独评估，并说明预期收益上限。

# 代码模式

各级的具体写法分别见 `pattern-capacity-driven-tiling`（容量反推）、
`pattern-l1-half-region-double-buffer`（半区与 flag 配对）、
`pattern-cross-flag-bidirectional`（跨核交棒）。

# 约束

- 每一级只改一个变量，改动前后各取一次 fresh `kernel_details.csv`，并复跑官方 benchmark。
- 正确性未复核的利用率或耗时数字不得写入结论。
- 连续两级出现中性结果时按第 4 步判停，不要在同一方向继续堆改动。

# 来源

[^api]: CATLASS core_api.py（cube fractal 对齐与管道计时口径）：
    https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
[^measure]: 行独占单 kernel 的逐轮实测记录（fresh `kernel_details.csv` 管道占比与官方 benchmark
    前后对比）：project-evidence。
