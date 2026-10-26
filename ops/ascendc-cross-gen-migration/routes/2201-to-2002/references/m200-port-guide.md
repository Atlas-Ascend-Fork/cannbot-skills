# 310P（dav-m200 / arch20）移植差异适配指南

> 本文件是 310P 路线的**风险扫描程序与知识索引**：定义"扫什么、命中后怎么办"，知识正文（信号/根因/规避的证据细节）已全部沉淀为 cannbot-knowledge 的 310P 卡（`platforms: ["310p"]`），本文件只留指针。
>
> **使用时机**：
> - 阶段 1 评估定级时，MUST 加载本文件做 m200 风险扫描（LOADED token：`[LOADED] m200-port-guide`）
> - 阶段 2 代码改造时，按 §3 索引逐命中项读卡处置
> - 阶段 4 精度验证时，按 §2 设计 dtype 覆盖矩阵、按 §6 未实证清单回填实证结论
>
> **卡片读取方式**：经 `knowledge-query` skill 按卡名检索（配置 `CANNBOT_KNOWLEDGE_ROOT`；卡路径相对知识仓 `knowledge/ops/ascendc/`）。卡与本文件冲突时以演练实证更新更晚者为准并上报。
>
> **性质提醒**：arch22→310P 是**下移植**（down-port）。310P 与 arch22 同为 MemBase 编程模型，**无 RegBase/MicroAPI**——凡是教你"改写 RegBase"的资料在 310P 路线**全部不适用**。

---

## 0. 总体结论

| 维度 | 影响面 | 适配策略 |
|------|--------|---------|
| **特性缺失** | 大 | dtype 降级或移除；fp32 matmul 按 §2.3 决策树选路线 |
| **硬件 errata** | 大（多处静默错误，编译无感） | 按 §3 索引逐项扫描，命中读卡处置；**禁止以编译通过推断正确** |
| **预算与核数** | 中 | UB 工作预算 ~240KB、8 个 AIV fuse 核重分片（事实真源见 §1 平台卡） |
| **同步语义** | 中 | fuse 核无 AIC/AIV 分离；风险转移到核内流水 fence |

---

## 1. 平台事实（真源已入卡）

平台事实全表（SoC 三套拼写 `--soc=ascend310p` / `Ascend310P3` / bisheng `dav-2002`、NpuArch=2002、UB/L1/L0C、8 核 fuse、cube fp16-only、dtype 矩阵、工具链）已沉淀为知识卡：

- **`runbooks/compilation/ascend310p_platform_facts_quick_reference.md`**（主卡，MUST 读）
- `common/platforms/concepts/target_ascend310p.md`（平台概念卡，互链）

stage 0 逐项实证后落盘 `.env_info.md` 的纪律不变（SoC 拼写三重实证、profiler 冒烟等见 stage_0 Step 0.8）。

---

## 2. 特性缺失扫描清单（阶段 1 Step 1.3 的扫描程序）

### 2.1 bf16 / fp64 / fp8 / int4

| 扫描特征 | 处置 |
|---------|------|
| `bfloat16_t` / `bf16` | 310P 无 bf16 硬件。首选从 dtype 支持集移除；业务必须时 host 侧 widen fp32（成本明示用户） |
| `double` / `fp64` | 硬件无。移除或降级 fp32 |
| `fp8` / `hifloat8` | 硬件无。移除 |
| int4 + Mmad | 310P cube 无整数路径先例，须先实证 int8 Mmad（U5），否则退 vector 点积 |

### 2.3 fp32 matmul 精度路线（cube 类算子 MUST 评估）

dav-m200 cube 只算 fp16（fp32 累加器），fp32 契约有精度地板。**路线决策树**（owner 裁决 2026-10-12：eager-parity 优先级最高，默认走 ①）：

1. **fp16-internal 直送 / 厂商对齐（默认）**——torch eager 的 fp32 matmul 在 310P 内部就降 fp16，语义相同；精度门 = ours vs fp64 ≤ max(2× eager vs fp64, 1e-3)
2. **hi/lo 拆分 3-Mmad**——仅当调用方显式要求 fp32-exact（3× cube 工作；固定项序保确定性）
3. **退 vector 路径**——K 小或 cube 收益不抵精度损失时

证据细节（fp16 量化地板量级、厂商 parity 锚点、eager-parity 门公式推导）见卡 **`runbooks/precision/fp16_cube_fp32_semantics_310p.md`**；参考相对门陷阱见卡 `npubench_reference_relative_gate_vendor_parity`（c 层 user_kb）。

### 2.4 950 新增特性回退

RegBase/MicroAPI、`PRECISION_1ULP_FTZ_FALSE`、FixpipeParamsArch3510、LoadDataWithTranspose 在 310P 全部不存在。来源算子含这些 = 来源不是纯 arch22，stop 并上报。

---

## 4. 预算与核数（迁移动作）

| 项 | 迁移动作（数值真源见 §1 平台卡） |
|----|-------------------------------|
| UB | tile 尺寸按 ~240KB 工作预算重算 |
| 核数 | block_dim 按 8 核重分片；来源硬编码核数必须参数化；`GetSubBlockNum()` 在 fuse 核恒 1——`GetBlockNum()*GetSubBlockNum()` 分片改为仅 `GetBlockNum()`（卡：compilation/fuse_core_getsubblocknum_always_one） |
| L1/L0C | cube 操作数驻留预算按平台卡数值重排 |
| double buffer | UB 紧张时优先降 depth 再缩 tile |
| cube K 维 | nEff≥64 规则在 m200 同样必要（卡：compilation/neff_ge_64_archive_rule_also_required_m200） |

---

## 5. 同步语义差异

- **fuse 核事实**：无 AIC/AIV 分离——3510 的握手问题域整体不存在；stage 1 Step 1.5 盘点注明"不适用"即可
- **风险转移**：核内流水同步是主战场（E3/E6/E15）；HardEvent 语义以 310P 平台文档为准
- **CrossCore\***：fuse 核下语义以 310P 文档为准；多核写出按 `GetCoreNumAiv()` 运行期分片

---

## 6. 未实证清单（stage 0/4 实证后回写本节，并入卡）

**仍开放**（遇到时实证，结果回写本节 + 沉淀新卡）：

| # | 问题 | 实证方式 |
|---|------|---------|
| U1 | 310P 是否裁剪 subnormal | stage 4 subnormal 用例实测（不预设 FTZ 结论） |
| U2 | SIMT 在 2002 的支持状态 | **不支持**（已确认，2026-10-14）——L3 SIMT 路线在本平台不适用 |
| U3 | `--enable_host_tiling` 在 310P MemBase tiling 下是否必需 | stage 3 先试不带，失败再加 |
| U5 | int8 Mmad cube 路径可用性 | 量化 matmul 需求出现时实证 |

**已闭环（知识入卡，本表只留指针）**：U4（profiler 自定义算子可产 kernel 级 CSV）→ performance/profiling_toolchain_310p_pipe_reading；U6（GetSubBlockNum 恒 1）→ 见 §4；U7（LoadData3DParamsV2 多 fractal 可用）；U8 → 见 §4；U9（大计数 vgather 有毒，只许 mfg 小计数 idiom）→ performance/dav_m200_data_movement_primitive_selection；U10（测量必须独占设备）→ performance/ol_15。

---

## 7. 检查清单（阶段 2 完成后逐项核对）

- [ ] bf16/fp64/fp8/int4 使用点已全部处置，dtype 矩阵与 def.cpp 同步
- [ ] fp32 matmul 类算子已按 §2.3 决策树选定路线并在分析摘要记录理由
- [ ] fp32↔fp16 cast 全部 CAST_NONE（E10）
- [ ] UB 预算与 8 核分片已按平台卡数值重算
- [ ] `AddConfig("ascend310p")`、`config/ascend310p/`、`arch20/`、`__NPU_ARCH__==2002` 命名纪律符合
- [ ] 未实证清单中本次命中的项已实证并回写（本文件 + 新卡）
