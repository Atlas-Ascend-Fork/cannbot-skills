# SWAT 调度

SWAT 是一套确定的输出任务调度算法：沿 M 分组，组内 M 优先，相邻组反向遍历 N，末组吸收余数，核间按任务编号轮转，并按下述规则处理边缘尺寸和尾轮拆分。基础与 Batch 模板引用本页；数据搬运和同步由各自 Kernel 描述。其他窗口映射属于其他调度算法。

## 1 调度输入与结果

| 输入 | 含义 |
|---|---|
| M、N | 逻辑矩阵两轴长度 |
| ML1、NL1 | 常规输出块两轴长度 |
| mTileCount、nTileCount | 两轴输出块数 |
| mEdgeParts、nEdgeParts | 两轴末尾再均衡区的块数 |
| windowTileCount | 常规 M 调度窗口宽度，本算法取 4 |
| coreCount | 分配任务使用的核数 |
| mRegularCount、mTailMain、mTailLast | M 轴常规块数、边缘再均衡区的主块长度及末块长度；N 同理 |
| mTailParts、nTailParts | 调度尾轮中每个父块的 M/N 拆分份数 |

mRegularCount/nRegularCount 与 mTailLast/nTailLast 由 §3 推导；若调用方传入这些派生值，必须与原始尺寸一致，不作为独立可调参数。拆分份数由选块策略提供，本页不负责搜索份数；基础配置引用 [公共选择算法](tiling-selection.md#9-尾轮拆分份数选择)。

输出为每核任务的 `(mTaskOffset,nTaskOffset,mTaskValid,nTaskValid)`。输出矩形和任务编号是不同信息，不能用编号直接乘块尺寸求地址。

## 2 分组蛇形映射

沿 M 组织调度窗口，窗口内先推进 M 块，再推进 N 块；奇数窗口反向遍历 N。最后一组吸收余下的 M 块。

```text
mainWidth = min(windowTileCount, mTileCount)
mainGroupCount = mTileCount // mainWidth - 1
lastWidth = mTileCount - mainGroupCount * mainWidth

groupId = parentTaskId // (mainWidth * nTileCount)
if groupId < mainGroupCount:
    mTileId = groupId * mainWidth + parentTaskId % mainWidth
    nTileId = (parentTaskId // mainWidth) % nTileCount
else:
    groupId = mainGroupCount
    localId = parentTaskId - mainGroupCount * mainWidth * nTileCount
    mTileId = mainGroupCount * mainWidth + localId % lastWidth
    nTileId = localId // lastWidth
if groupId % 2 == 1:
    nTileId = nTileCount - 1 - nTileId
```

这里的整数除法均向下取整，输入块数为正。输入转置不改变该映射。沿 N 分组的对称遍历是另一种窗口调度，不属于本页 SWAT 的选择项。

例如 `(mTileCount,nTileCount)=(2,3)` 时，坐标依次为 `(0,0),(1,0),(0,1),(1,1),(0,2),(1,2)`。每个 N 块内重访 B，不同 N 块间重访 A。跨核开始时间由各核流水决定，编号相邻不意味着执行严格串行。

## 3 矩阵边缘再均衡

矩阵边缘再均衡与调度尾轮拆分分开描述。前者改变轴上最后若干块的尺寸与起点，后者把最后一轮父任务分给更多核。

以 M 轴为例，设最后 `mEdgeParts` 个块属于再均衡区：

```text
mRegularCount = mTileCount - mEdgeParts
mEdgeLength = M - mRegularCount * ML1
mTailLast = mEdgeLength - (mEdgeParts - 1) * mTailMain
if mTileId < mRegularCount:
    mParent = mTileId * ML1
    mParentValid = ML1
else:
    mParent = mRegularCount * ML1 + (mTileId - mRegularCount) * mTailMain
    mParentValid = mTailLast if mTileId == mTileCount - 1 else mTailMain
```

N 轴同理。没有边缘再均衡时，`mEdgeParts=1`，最后块长度为 `M-(mTileCount-1)*ML1`。再均衡参数必须使所有块非空、连续且覆盖完整轴；不能仍对再均衡区使用 `mTileId*ML1`。

## 4 尾轮分派与拆分

设单矩阵父任务数 `matrixTaskCount=mTileCount*nTileCount`、核数 P。单矩阵取 `T=matrixTaskCount`，Batch 取 `T=batchCount*matrixTaskCount`；两者共用全局尾轮分派，仅在 `T>P` 且 `r=T%P>0` 时尝试拆分尾轮，且 `r*mTailParts*nTailParts<=P`。不足一轮的整场景不在此步骤强行拆尾。

令 `S=mTailParts*nTailParts`、`tailBegin=(ceil_div(T,P)-1)*P`、`expandedCount=T+(S-1)*r`。每核从 `taskId=coreId` 开始按 P 递增。尾轮中父任务编号与子块坐标为：

```text
if taskId >= tailBegin and S > 1:
    parentTaskId = tailBegin + coreId // S
    mPartId = (coreId % S) % mTailParts
    nPartId = (coreId % S) // mTailParts
else:
    parentTaskId = taskId
    mPartId = 0
    nPartId = 0
```

仅执行 `taskId<expandedCount` 的任务；单矩阵的父任务编号经 §2、§3 求出父矩形；Batch 先以 `parentTaskId//matrixTaskCount` 求 batch 编号，以 `parentTaskId%matrixTaskCount` 经 §2、§3 求该矩阵的父矩形，子块保持同一 batch。尾轮各轴子步长分别为 `ceil_div(mParentValid,mTailParts)`、`ceil_div(nParentValid,nTailParts)`，起点加上 `partId*step`，有效长度取父区域剩余长度与 step 的较小值。起点超出父区域时跳过空任务。普通 ND 连续场景不额外对该步长向上对齐；其他布局需由其地址和搬运契约给出附加条件。

拆分份数先按末块实际长度归一化：`step=ceil_div(lastValid,requestedParts)`、`actualParts=ceil_div(lastValid,step)`，再计算 S 与 expandedCount。这里的 taskId 是发射编号，允许无有效输出的槽位；不得用“删掉空块后的紧凑列表”替换核号到父块的映射。

## 5 校验

验证多分组、奇数组反向、最后合并组、边缘再均衡及尾轮拆分。每个有效输出坐标只属于一个最终任务；拆分父任务不重复执行。尾轮例 `T=6,P=4,mTailParts=2,nTailParts=1` 产生八个任务：前四个不变，核 0/1 拆父任务 4，核 2/3 拆父任务 5。

任务枚举用于独立复算和核对实际 Device 控制流与对应生成产物，按 [公共证据选择](common-contract.md#52-设计一致性与证据选择) 记录检查；它不自动要求设备输出任务完成事件，也不能由 Host 计划伪造这些事件。多块、少于一轮、多于一轮和 M/N 尾块仍需设备 Golden，数值正确不单独证明任务映射的唯一性。
