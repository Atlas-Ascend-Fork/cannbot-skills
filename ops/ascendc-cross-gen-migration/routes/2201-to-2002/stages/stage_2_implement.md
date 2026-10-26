# 阶段 2：代码改造（含 m200 差异适配）

**前置条件**：`[GATE-1] LEVEL=L1/L2/L3` 已输出。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/m200-port-guide.md` | `[LOADED] m200-port-guide` |
| 2 | `references/m200-cube-guide.md`（cube 类算子 MUST） | `[LOADED] m200-cube-guide` |
| 3 | `references/multi-stage-guide.md`（2+ 算法阶段算子按需；数值为 3510 取值，310P 预算以 m200-port-guide §4 为准） | `[LOADED] multi-stage-guide` |

**★ 本文件 MUST 在代码改造前加载。** 阶段 1 扫描出的 m200 风险项，在此阶段逐项适配。
**禁止加载**（RegBase/MicroAPI 在 2002 不存在）：3510 路线的 l2-guide / api-mapping / l3-guide / api-diff-guide / cube-migration-guide、`ascendc-regbase-best-practice` sub-skill。加载即违规。

## 路由：根据层级加载实现指南

```
GATE-1 LEVEL 值
  ├─ L1 → Step 2.1（L1 改造清单）+ Step 2.4（m200 差异适配）
  ├─ L2 → Step 2.2（L2 改造清单）+ Step 2.4（m200 差异适配）
  └─ L3 → Step 2.3（L3 改造清单）+ Step 2.4（m200 差异适配）
```

---

## Step 2.1：L1 改造清单

### 命名纪律（310P，MUST）

| 项 | 310P 值 | 反模式 |
|----|--------|--------|
| `AddConfig()` | `AddConfig("ascend310p", aicore_config)`（复用 910b 配置——同为 MemBase，复用度高于 950） | 禁止写 ascend950 |
| config 目录 | `op_host/config/ascend310p/`（复制 binary.json + simplified_key.ini） | — |
| kernel 变体目录 | `op_kernel/arch20/`（ops-nn build.sh 先例） | ★ **禁止用 `arch35/` 目录名装 2002 代码**——目录名即架构承诺 |
| 平台宏 | `__NPU_ARCH__ == 2002` | 禁止 `__NPU_ARCH__ == 3510` / `__CCE_AICORE__ == 310` 残留误导 |

### 改造步骤

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `op_host/<op>_def.cpp` | 添加 `AddConfig("ascend310p", aicore_config)`（复用 910b 配置） |
| 2 | `op_host/config/ascend310p/` | 创建目录，复制 `binary.json` + `simplified_key.ini` |

**L1 适用条件（比 3510 路线宽）**：kernel 无 E1-E15 errata 命中、无特性缺失（bf16/fp64/fp8/int4）、UB ≤240KB。满足时纯 host 适配即可，kernel 零改动。

**标准 L1（需 kernel 改动时）**：

| 步骤 | 文件 | 操作 |
|------|------|------|
| 1 | `op_host/<op>_def.cpp` | 添加独立 `OpAICoreConfig` + `opFile.value = "<op>_apt"` |
| 2 | `op_kernel/<op>_apt.cpp` | 创建 310P 入口文件，include 指向 `arch20/` |
| 3 | `op_kernel/arch20/` | 复制 kernel 头文件，按 Step 2.4 处置 errata/特性缺失 |
| 4 | `op_host/config/ascend310p/` | 创建目录，复制配置文件 |

**cube 类算子**：追加 MUST READ `references/m200-cube-guide.md`，输出 `[LOADED] m200-cube-guide`。

---

## Step 2.2：L2 改造清单（kernel 适配，≠ RegBase）

**310P 的 L2 = dtype 降级 + errata 规避 + UB 收紧 + 核内流水同步盘点**（310P 无 RegBase/MicroAPI，3510 路线的 RegBase 改造内容全部不适用）。

### L2 改造步骤

1. **先完成 L1 全部步骤**（`_def.cpp` + `config/ascend310p/`，需要时 `_apt.cpp` + `arch20/`）
2. **errata 处置**：按阶段 1 扫描命中清单逐项处理，逐条按 stage_1 Step 1.3.2 的处置卡执行：
   - E1：DataCopyPad/`Copy<float,false>` → DataCopy + 手工 padding/展开，GM 侧保 32B 量子对齐
   - E10：fp32↔fp16 cast 全部 `RoundMode::CAST_NONE`（dav-m200 上 CAST_NONE=RNE，语义保持）
   - E11：Fixpipe → L0C→UB→GM（遵守 E12 −1 分形寻址 + guard buffer、E13 全量 sweep 与存储禁忌）
   - E2/E3/E5/E6/E15：Brcb 后恢复 FULL_MASK；核内流水显式 fence；tiling 标量 MTE2 物化；vector-only kernel 事件链改 PipeBarrier 按 stage 点
   - E14：GatherMask 源未 32B 对齐时改标量解交织
3. **dtype 降级**：按 `references/m200-port-guide.md` §2（bf16 移除或 host widen；fp32 matmul 按 §2.3 决策树）
4. **UB 重算**：按 §4（240KB 预算、8 核分片；`GetSubBlockNum()` 在 fuse 核实测恒 1——`GetBlockNum()*GetSubBlockNum()` 分片改为仅 `GetBlockNum()`）
5. **保持 outer shell 不变**（Init/Process/CopyIn/CopyOut 结构），只改命中点

### L2 改造检查点

- [ ] L1 步骤全部完成
- [ ] 阶段 1 扫描命中的 errata/特性缺失项已全部处置，未命中项已标注
- [ ] fp32↔fp16 cast 全部 CAST_NONE（E10）
- [ ] UB ≤240KB 与 8 核分片已重算
- [ ] **★ 性能形态三约束已落实**：每 kernel launch block_dim=8（或已按拆维/合并方案保证）；errata 替代路径无标量数据通路（向量化模式已实现）；矩阵乘结构已上 cube
- [ ] 命名纪律检查点通过（无 arch35/ 装 2002、无 3510 宏残留）
- [ ] 跨核同步点已按 Step 1.5 处置决策执行（fuse 核：AIC/AIV 粒度不适用已注明；核内流水按 E3/E6/E15 复核）

---

## Step 2.3：L3 改造清单（cube 深度重写）

**MUST READ**：`references/m200-cube-guide.md`
读取后输出：`[LOADED] m200-cube-guide`

### L3 改造步骤

1. 先完成 L1 全部步骤
2. 按 `references/m200-cube-guide.md` §2 决策树定 fp32/dtype 路线（写明门禁语义依据）
3. 按 §3 标准路线适配清单执行：高阶 Matmul API 边界、KTILE≠128（E4）、L0C→UB fp32 行主序 16×16 读出（2048B 尺寸陷阱）、ND2NZ 操作数 GM→L1 单跳（E9）
4. 性能重写算子：按 m200-port-guide §4 预算重排 tile/流水

**SIMT 在 2002 未实证（U2），默认不适用**——如确需，先回 stage 0 实证并回写 `references/m200-port-guide.md` §6。

---

### 同步协议处置原则（适用于所有层级）

- **跨核同步机制（CrossCoreSetFlag/WaitFlag、IBSet/IBWait、SyncAll、NotifyNextBlock/WaitPreBlock）按阶段 1 Step 1.5 盘点结果执行**：判定"沿用"的同步点原样保留；判定"禁用"的同步点**移除该路径，改用不依赖该同步的标准流程**。
- **无法验证跨核机制在目标平台模式下成立时，禁用该路径走标准流程，禁止原样移植。**
- 310P 为 fuse 核（无 AIC/AIV 分离）：AIC↔AIV 握手问题域不适用；核内流水同步按 `references/m200-port-guide.md` §5 与 E3/E6/E15 复核。
- 310P 无 RegBase：核内事件同步保留 `SetFlag/WaitFlag`，但 vector-only kernel 的手工事件链有间歇挂死风险（E15）——按 stage 点 PipeBarrier 处置；**跨核 CrossCoreSetFlag/WaitFlag 不受此影响**。

---

## Step 2.4：★ m200 差异适配（所有层级 MUST 执行）

**基于阶段 1 扫描结果和已加载的 `references/m200-port-guide.md`，逐项适配。**

| 维度 | 适配内容 | 参考 |
|------|---------|------|
| 特性缺失 | bf16 移除或 host widen；fp64/fp8 移除；int4+cube 先实证（U5）；fp32 matmul 按决策树 | `references/m200-port-guide.md` §2 |
| 硬件 errata | E1-E18 命中项逐条处置（读卡按卡的规避方案） | stage_1 Step 1.3.2 处置卡列 |
| 预算与核数 | UB ≤240KB、8 核分片、GetSubBlockNum=1 修正 | `references/m200-port-guide.md` §4 |
| subnormal | **本阶段不做预防性适配**——310P 行为未实证（U1），由阶段 4 实证驱动；eps/常量 dtype 可表示性检查仍 MUST 执行（方法论通用） | `references/m200-port-guide.md` §6 |

**适配检查点**：
- [ ] 特性缺失项已全部处置（dtype 矩阵与 def.cpp 同步裁剪）
- [ ] errata 命中项已全部按 §3 处置，未命中项已标注
- [ ] fp32 matmul 类算子已写明路线决策与门禁语义依据
- [ ] eps/常量 dtype 可表示性检查已完成（如适用）

---

## Gate 输出条件

- [ ] `m200-port-guide` 已 LOADED（cube 类加 `m200-cube-guide`）
- [ ] 所有改动文件已列出
- [ ] 代码改造检查点全部通过（含跨核同步点处置，无遗漏）
- [ ] **★ m200 差异三维度适配已完成并勾选检查点**
- [ ] 3510 路线资料（l2-guide/api-mapping/regbase sub-skill 等）未被加载（加载即违规）

**全部通过 → 输出 `[GATE-2] FILES_MODIFIED=[file1, file2, ...]` → 进入阶段 3**
