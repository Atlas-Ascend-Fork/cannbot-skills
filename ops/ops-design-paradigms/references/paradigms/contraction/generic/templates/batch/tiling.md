# contraction 范式 batch Tiling 策略

选块流程见 [公共选块](../../tiling-selection.md)；本模板提供实际任务、数据复用及资源模型，使用与其搬运方式匹配的成本公式。

本页定义单 batch 输出块的分层切分、广播坐标和 batch 任务展开；贯穿实例及资源定义见 [DAG](dag-buffers.md)。

公共地址与资源判定引用 [公共契约 §2–3](../../common-contract.md#2-存储与地址)；以下公式给出本模板具体槽数与形状。

## 1 UB 切分

不涉及。

## 2 L1 切分

仍按 `ML1×KL1`、`KL1×NL1` 的单任务窗口分配：`L1Required=2*span(AL1)+2*span(BL1)`。广播只选择 GM 源地址，不把输出 batch 数乘到单槽大小中。KL1 尾部遵循 [公共地址契约](../../common-contract.md#2-存储与地址)，有效长度取剩余K，首次物化负责填零。每个任务以 KL1 等步长推进，本例三个窗口有效长度为 256、256、128；L1 采用 [成对分区双缓冲](../../execution-strategies.md#2-l1-成对分区双缓冲)，真实分区间距由实际分配提供。

## 3 L0A 切分

`L0ARequired=2*span(AL0)`，逻辑尺寸 `ML0×KL0`，本模板 `ML0=ML1`。

## 4 L0B 切分

`L0BRequired=2*span(BL0)`，逻辑尺寸 `KL0×NL0`，本模板 `NL0=NL1`。

## 5 L0C 切分

`L0CRequired=span(CL0)`，逻辑尺寸 `ML0×NL0`。一个核顺序处理多批次时复用同一累加槽，每个任务重新初始化，首次累加前取得整槽写权限，最后 MMAD 完成后交给输出，输出完成才归还。

## 6 Bias[N] 组合扩展的 BiasTable 切分

原无组合扩展实例不涉及。hasBias=true 时按 [Bias[N] 组合扩展首段融合策略](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略) 分配 BiasTable 槽，并将中转槽记入 L1；增加资源后重新筛选分块。

## 7 多核切分

### 7.1 切分原理

先把输出 batch shape 展平为 `batchId`，再把 batch 内的 M/N 块交给基础 SWAT 映射。输出 batch 与输入物理 batch 分开：两输入分别按广播轴求物理基址，不能共用 `batchId*M*K`。

### 7.2 切分公式

令 `mTileCount=ceil(M/ML1)`、`nTileCount=ceil(N/NL1)`、`matrixTaskCount=mTileCount*nTileCount`，全局父任务数 `T=batchCount*matrixTaskCount`。使用 [公共尾轮份数选择](../../tiling-selection.md#9-尾轮拆分份数选择) 和 [SWAT 尾轮分派](../../swat-scheduler.md#4-尾轮分派与拆分)，得到 `expandedCount` 和每个发射任务的 `parentTaskId,mPartId,nPartId`。核按 `taskId=coreId+taskRound*coreCount` 遍历，小于 expandedCount 才进入。

先从全局父任务解码 `batchId=parentTaskId//matrixTaskCount`、`matrixTaskId=parentTaskId%matrixTaskCount`；再将 matrixTaskId 交给 SWAT 矩阵映射得到父矩形，最后加尾轮子块偏移并裁剪有效长度。解展平 batchId 后分别计算 A/B 广播索引。不能从展开后的 taskId 直接除余解码 batch，也不能对每个 batch 单独拆分尾轮。

尾轮拆分不改变 Batch 的归约语义：每个子块仍计算完整 K，仅拆分输出 M/N 区域。是否拆分由全局余数、搬运粒度和地址约束共同决定；增加子块也可能增加输入重读，任务数均衡不等于计算耗时必然下降。

### 7.3 贯穿实例

`matrixTaskCount=6`、总数 36。36 能被 4 整除，公共选择返回两轴份数均为 1；每个矩阵自身的 6 块余数不参与全局尾轮判断。若 `coreCount=4`，核 0 处理任务 `0,4,...,32`；任务 32 属于 `batchId=5` 即 `(p0,p1)=(1,2)`，矩阵局部任务 2。A 来自 `[1,0]`，B 来自 `[0,2]`。每个核都可能切换 batch，不能把 batch 基址固定在 Init。

拆尾例：batchCount=2、matrixTaskCount=3、P=4，请求并归一化为 M 两份、N 一份时，T=6 展开为 8。前四任务不变；核 0/1 处理父任务 4 的两个子块，核 2/3 处理父任务 5 的两个子块，均属于 batch 1，矩阵局部父编号分别为 1、2。

## 8 切分策略微调

batch 内 SWAT 改善矩阵窗口重访；batch 间顺序影响广播输入的跨 batch 重访。两者需结合任务分配评估。批次小矩阵需要在 L1/L0 同时容纳多个 batch 时，转到 [IterBatch](../iterbatch/tiling.md)；将批次合并为扩大矩阵计算时，转到 [MergeBatch](../mergebatch/tiling.md)。不能只改一个 batchCount 字段替代模板切换。

## 9 整体切分伪码

```text
normalize_and_validate_broadcast_shapes()
select_basic_matrix_tiles_and_physical_layouts()
validate_each_memory_capacity()
compute_batch_and_matrix_task_counts_with_checked_arithmetic()
select_core_count_and_swat_parameters()
select_tail_parts_from_global_parent_count()
normalize_tail_parts_and_compute_expanded_count()
emit_broadcast_strides_and_task_mapping()
```

## 10 切分后处理

### 10.1 UB 大小计算

不涉及。

### 10.2 Workspace 分配

不涉及。

### 10.3 TilingKey 设置

模板标识、Tin/Tacc/Tout 与 A/B 独立 ND/DN 编译期维度按 [公共契约 §1–2](../../common-contract.md) 绑定；C 存储方向按输出契约绑定。广播维度和步长是运行时描述，不把每种 shape 编成独立 key。

### 10.4 TilingData 设置

记录矩阵分块、输出 batch shape、各输入真实 shape/stride、输出 stride、SWAT 参数、全局父任务数、归一化后的尾轮份数、展开任务数和核数。所有乘法用可检查溢出的整数运算；保留元素/字节单位。

### 10.5 ScheduleMode 设置

本例选择 Cube 计算和直接结果写回，对应任务类型由后端入口绑定。

### 10.6 资源约束的输入与实例分配

用 DAG 账本逐层比较可用容量。输入容量仅依赖实际 shape，不以广播后的输出 shape 推算输入分配。

## 11 切分策略产出校验

枚举任务并给每个 `(p,m,n)` 计数，要求恰为 1；输入广播坐标必须在各自真实 shape 内。覆盖单 batch、多轴交叉广播、batch 数与核数不整除、M/N/K 尾块、全局整轮不拆分及尾轮子任务的父编号解码；错误地用发射编号求 batch 应被覆盖检查识别。资源筛选失败时改 tile 或模板，不能通过少发射核掩盖每核容量超限。

## 12 Bias[N] 组合扩展的分派与切分

原贯穿实例保持 hasBias=false；以下为同一形状接入 Bias[N] 组合扩展。Bias 向量在所有保留轴坐标间共享，仅按输出 N 区间切片。

hasBias 参与编译期 TilingKey 分派；Bias[N] 地址、N 方向 stride、槽基址与跨度由现有 Params 承接。按当前输出列块的物理跨度预留。目标布局和容量仍须逐层验证。

尾轮拆分沿用父任务 N 起点，子任务仅裁剪 Bias[N] 的有效区间，不改变索引映射。
