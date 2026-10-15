# contraction 范式 IterBatch Tiling 策略

选块流程见 [公共选块](../../tiling-selection.md)；本模板提供实际任务、数据复用及资源模型，使用与其搬运方式匹配的成本公式。

贯穿实例见 [DAG](dag-buffers.md)。此模板以完整矩阵驻留 L1 为前提；资源不满足时选择 [普通 batch 分块](../batch/tiling.md)，而不是继续发射相同布局。

公共地址与资源判定引用 [公共契约 §2–3](../../common-contract.md#2-存储与地址)；以下公式给出本模板具体槽数与形状。

## 1 UB 切分

不涉及。

## 2 L1 切分

`ML1=M,NL1=N,KL1=K`。令 `aL1BatchPitchBytes`、`bL1BatchPitchBytes` 为完整物理矩阵的预留间距：

`L1Required=2*BatchL1*aL1BatchPitchBytes + 2*BatchL1*bL1BatchPitchBytes`。

本例不在 L1 内分 K；KL0 是 L0 消费窗口。不能把公式替换成 `ML0*KL0` 来使不合法完整矩阵候选看似可装入。

## 3 L0A 切分

本例选择 `ML0=M,KL0<=K`；泛化候选可取 `ML0<=M`，但需同步闭合输出pitch，每 batch 仅搬当前 tile，但物理间距取所选布局的 `aL0BatchPitchBytes`。本例保留完整矩阵 pitch，`L0ARequired=2*BatchL0*aL0BatchPitchBytes`。紧凑布局是另一个需同时修改搬运与 MMAD 地址的候选，不能仅修改容量公式。

## 4 L0B 切分

同理 `L0BRequired=2*BatchL0*bL0BatchPitchBytes`，本例 `NL0=N`，当前tile为 `KL0×NL0`，间距可能大于 tile span。

## 5 L0C 切分

`L0CRequired=2*BatchL0*cL0BatchPitchBytes`。每 batch 有独立累加子区，首段初始化各自子区，跨全部 K 保持不变。MMAD 写入范围、输出读取范围和预留 pitch 必须一致。

## 6 BiasTable 切分

原无组合扩展实例不涉及。启用 Elementwise/Broadcast Bias[N] 组合扩展时，按 [Bias[N] 组合扩展策略](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略) 分配 BiasTable 槽，并将中转槽记入 L1；按当前 N 子区的实际布局核算资源，增加资源后重新筛选分块。

## 7 多核切分

### 7.1 切分原理

本例以连续 BatchL1 个 batch 为一个任务，任务内枚举 M/N 块，避免不同核重复装载相同完整矩阵组。任务数 `ceil(batchCount/BatchL1)`；核处理 `taskId=coreId+taskRound*coreCount`，任务 batch 起点 `taskId*BatchL1`。最后有效数量 `min(BatchL1,batchCount-batchTaskOffset)`。

### 7.2 贯穿实例

5个batch切为3个任务；2核时核0处理 `[0,1]`、`[4]`，核1处理 `[2,3]`。核0第二任务重新绑定 GM 基址，L1/L0槽按其实际完成边界继续轮转。任务内 N/M 可采用固定嵌套遍历，本例N外M内；不把基础模板 SWAT 的输出块任务编号直接当作 batch 组编号。

## 8 切分策略微调

BatchL1 决定完整矩阵驻留数量与任务并行度，BatchL0 决定一轮输入准备和计算拥有多少 batch 子区。增加 BatchL1 可能减少可分配任务；增加 BatchL0 会消耗 L0 资源，不使多个独立 MMAD 自动成为一次大矩阵乘。广播扩展需要为A/B分别定义驻留数量与索引映射，不能以输出batch数推算实际输入驻留需求。

## 9 整体切分流程

1. 归一化 A/B 矩阵存储方向与 batch 步长，确定完整矩阵布局及各层 batch 间距。
2. 枚举正整数 BatchL1、BatchL0 和合法 ML0/NL0/KL0；按 §2–5 计算两槽及批次子区资源，淘汰超容量或地址不闭合的候选。
3. 按 §7 展开实际任务、有效 batch 子组及 K 段，核算各核完成时间。每个输入 batch 的完整矩阵只装入 L1 一次；L0 搬运与 MMAD 次数随 M/N/K 分块及有效 batch 数变化。
4. 按 [公共选块成本与搜索](../../tiling-selection.md) 比较候选；批次数增加带来的任务数减少、组内串行 MMAD 和尾组空闲均计入成本。
5. 检查覆盖和固定槽间距，输出所选批次组织、布局、槽数和任务参数。

## 10 切分后处理

### 10.1 UB 大小计算

不涉及。

### 10.2 Workspace 分配

不涉及。

### 10.3 TilingKey 设置

独立模板维度选择 IterBatch，A/B 存储方向独立，类型与C输出契约引用 [公共契约 §1–2](../../common-contract.md)。不得给普通分块 kernel 传 BatchL1 字段后假定已选择 IterBatch。

### 10.4 TilingData 设置

BatchL1/BatchL0、M/N/K、ML0/NL0/KL0、每层batch pitch、物理槽跨度、任务和核数。由布局可推导字段可在编译期/入口推导，不重复保存另一个事实源。

### 10.5 ScheduleMode 设置

由运行入口绑定纯 Cube 执行模式；输出通路在本模板中为 L0C→GM。

### 10.6 资源约束的输入与实例分配

各层需求量见 DAG 账本，分别比较平台能力；同时检查 pitch 不小于物理访问跨度，槽基址满足对齐。本例选择下表的相对字节区间，再由分配结果绑定各存储空间的基址：

| 存储空间/角色 | 槽0 | 槽1 | 每槽 batch 子区间距 |
|---|---|---|---:|
| L1 A | [0,32 KiB) | [32,64 KiB) | 16 KiB |
| L1 B | [64,96 KiB) | [96,128 KiB) | 16 KiB |
| L0A | [0,32 KiB) | [32,64 KiB) | 16 KiB |
| L0B | [0,32 KiB) | [32,64 KiB) | 16 KiB |
| L0C | [0,32 KiB) | [32,64 KiB) | 16 KiB |

L0A、L0B、L0C 的零点属于不同空间。最后任务只有一个有效 batch，仍使用相同 32 KiB 槽跨度；L0 输入每个 batch 本轮实际写入 8 KiB，剩余预留不作为有效输入读取。

## 11 切分策略产出校验

枚举 `(task,batchInL1,batchInL0,mTile,nTile,kTile)`，每 `(b,m,n,k)` 乘积恰有一次贡献；每 `(b,m,n)` 恰有一次最终写回。检查 BatchL1 不整除 BatchL0、batchCount 不整除 BatchL1、多任务同核、M/N/K尾块。全部片上最大访问地址必须小于对应槽跨度。

## 12 可选 bias 的分派与切分

原贯穿实例保持 hasBias=false；组合扩展启用时输入仍是共享的一维 `[N]` 向量，在所有保留轴坐标间复用，并沿其他输出轴广播。

`hasBias` 参与编译期 TilingKey 分派；Params 只承接 Bias[N] 地址、N 方向真实 stride、槽基址与跨度，不承接逐坐标 bias 映射。BiasTable 按当前 N 子区的物理跨度预留；单槽可被顺序处理的保留轴坐标复用，不将坐标组大小机械乘入槽需求。目标布局和容量仍须逐层验证。

每个保留轴坐标对应的输出子区都在自身首个 K 段应用同一 `bias[n]`；单 BiasTable 槽按实际消费者顺序复用，不能因 K 分窗重复应用。
