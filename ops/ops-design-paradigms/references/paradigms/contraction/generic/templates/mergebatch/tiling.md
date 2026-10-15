# contraction 范式 MergeBatch Tiling 策略

选块流程见 [公共选块](../../tiling-selection.md)；本模板提供实际任务、数据复用及资源模型，使用与其搬运方式匹配的成本公式。

贯穿实例和D矩阵定义见 [DAG](dag-buffers.md)。本模板用于同形矩阵的小batch组，决定合并数时同时核算扩大计算量与完整L0C占用。

公共地址与资源判定引用 [公共契约 §2–3](../../common-contract.md#2-存储与地址)；以下公式给出本模板具体槽数与形状。

## 1 UB 切分

不涉及。

## 2 L1 切分

选择BatchMerge、KL1和每batch物理M/N间距。定义第b个矩阵的合并轴起点为 `mergedMStart(b)`、`mergedNStart(b)`，布局输出扩大物理尺寸 `mergedMExtent,mergedNExtent`。最大合并尺寸记为 ML1、NL1；紧凑情形为 BatchMerge*M 和 BatchMerge*N，有对齐空隙时使用实际 extent。ML0=ML1、NL0=NL1。尾组重新派生当前有效布局，但槽预留由最大组确定。

`L1Required=2*span(mergedAL1Layout)+2*span(mergedBL1Layout)`，逻辑K窗口为KL1。转置输入可能改变拼接所需padding与布局，不能只翻转GM步长而保留原资源估计。

## 3 L0A 切分

`L0ARequired=2*span(mergedAL0Layout)`，对应扩大M×KL0。各batch A必须沿同一K轴对齐，不能把batch拼入K导致数学语义改变。

## 4 L0B 切分

`L0BRequired=2*span(mergedBL0Layout)`，对应KL0×扩大N；与A使用同一有效K长度。

## 5 L0C 切分

`L0CRequired=2*span(mergedCL0Layout)`，完整扩大M×扩大N。无padding时为 `2*BatchMerge²*M*N*accElemBytes`，不是 `2*BatchMerge*M*N*accElemBytes`。所有K段保持同一合并形状和对角坐标。

## 6 BiasTable 切分

原无组合扩展实例不涉及。启用 Elementwise/Broadcast Bias[N] 组合扩展时，按 [Bias[N] 组合扩展策略](../../execution-strategies.md#8-biasn-组合扩展的首段融合策略) 分配 BiasTable 槽，并将中转槽记入 L1；增加资源后重新筛选分块。

## 7 多核切分

任务数 `ceil(batchCount/BatchMerge)`；任务batch起点 `taskId*BatchMerge`，有效数取剩余最小值。核按 `coreId+taskRound*coreCount` 领取任务。每任务完整计算对应batch组的K，核间不归约。

本例5个batch、合并2个，3任务；2核时核0处理batch `[0,1]`与`[4]`，核1处理`[2,3]`。末任务扩大有效形状为16×16，物理槽仍按32×32上限预留。

## 8 切分策略微调

无padding时，每完整组有用乘加数为`BatchMerge*M*N*K`，实际为`BatchMerge²*M*N*K`，有效比例`1/BatchMerge`。增加合并数减少MMAD发射但增加无用计算与L0C需求；应与普通batch、IterBatch候选比较发射开销、搬运量、并行任务数和扩大计算量，不单以发射次数选择。

缓存策略依据本模板实际重访：每个输入 batch 只由一个合并任务读取，其内部非对角复用发生在片上，不产生新的 GM 重访。结合 K 窗口的物理访问范围、缓存对齐与并发带宽选择 hint；不能直接套用基础模板的 M/N 块数判据。

## 9 整体切分流程

1. 固定合法合并数、KL1/KL0 候选顺序、能力和成本模型输入；按 [公共选块](../../tiling-selection.md#4-搜索与确定性) 确定比较与同分规则。
2. 对每个候选建立 batch 合并轴起点、最大及尾组布局，验证输入搬运与对角提取能力。
3. 核算 L1、L0A、L0B 双槽及完整扩大 L0C 双槽；容量、布局、输出能力不成立则淘汰该候选。
4. 按 §7 生成每核任务，展开共同 KL1 窗口及内部 KL0 子窗口，计算各层搬运量、实际扩大乘加量、发射数与依赖完成时间。
5. 按既定成本模型选择候选，输出分块、任务、布局、槽分配及对角提取参数。

贯穿例将 BatchMerge=2、KL1=128、KL0=64 作为已绑定候选，得 ML1=ML0=32、NL1=NL0=32，三个任务、每任务两次 MMAD。资源见 DAG 账本；合并数的优劣由 §8 的额外计算、搬运与核完成时间共同判定。

## 10 切分后处理

### 10.1 UB 大小计算

不涉及。

### 10.2 Workspace 分配

不涉及。

### 10.3 TilingKey 设置

独立MergeBatch模板维度和A/B独立布局方向；输入/累加/输出类型按 [公共契约](../../common-contract.md) 绑定，输出固定为 C[batch,M,N]。具体输出对角提取能力需通过后端映射筛选，不支持时选择其他合法模板。

### 10.4 TilingData 设置

BatchMerge、M/N/K、KL1/KL0、GM 步长、合并布局与对角起点描述、固定槽基址及跨度、任务数及核数。由布局推导的extent和字节span只维护一个事实源，kernel与容量检查复用同一推导。

### 10.5 ScheduleMode 设置

按 [所选编程接口映射](../../../patterns.md#编程接口选择) 选择 Cube 核执行模式；任务分配使用 §7 的轮转规则，输出使用直接 GM 对角提取。

### 10.6 资源约束的输入与实例分配

逐层验证 DAG 账本：所需 L1=32 KiB，L0A=8 KiB，L0B=8 KiB，L0C=8 KiB；每层分别与能力输入比较。扩大尺寸及平方资源量使用溢出检查。末组逻辑尺寸缩小不改变固定物理槽基址。

本例按最小连续跨度给出一个分配：L1 A 两槽起点为 0、8 KiB，B 两槽为 16、24 KiB；L0A/B 各自槽起点为 0、4 KiB，L0C 为 0、4 KiB。若分配器提供更大槽间距，容量检查与地址绑定使用实际间距；不能用有效字节量替代预留跨度。

## 11 切分策略产出校验

枚举全部任务和对角块，要求batch0…4各输出一次；计算实际扩大MMAD写入包围盒，要求不超L0C槽。测试batch尾组、K多窗口、M/N布局padding、不同存储方向；用BatchMerge=1退化场景核对与基础数学一致。

## 12 可选 bias 的分派与切分

原贯穿实例保持 hasBias=false；组合扩展启用时 GM 输入仍是共享的一维 `[N]` 向量。Tiling 按扩大 N 布局为 BiasTable 预留物理跨度，并将输入向量按局部 N 坐标映射到每个扩大 N 子区。

`hasBias` 参与编译期 TilingKey 分派；Params 承接唯一 Bias[N] 输入的地址、N 方向真实 stride、槽基址与跨度，不定义逐坐标 bias 输入。BiasTable 按重复映射后的扩大 N 物理 extent 预留，包括拼接 padding；目标布局和容量仍须逐层验证。

扩大列坐标 `nMerged` 映射到 `bias[nMerged % N]`；因此每个对角结果块得到 `bias[n]`，非对角结果仍不输出。验证所有扩大 N 子区都按此映射准备，且尾部拼接/padding 不越界读取 Bias[N]。
