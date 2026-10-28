# A5 新特性使能清单（阶段 6 弹药库）

> 本文件是 `stages/stage_6_feature_enable.md` Step 6.1 的输入：逐项判定"适用/不适用"，适用项进入有界尝试。
> **定位**：Stage 0–4 只保证"正确迁移"，默认沿用 A2 写法；本清单列出 950 相对 2201 **新增或质变**的能力，用于把"能跑"提升到"吃满硬件"。
> **原则**：每条特性 = 适用信号 + 落地要点 + 验收指标 + 回退方式。没有验收指标的特性不进清单（无法判定是否该保留）。

## 一、特性总表

> **★ 知识真源纪律（stage_6 Step 6.1/6.2 强制）**：对判为"适用"且进入尝试的特性，下表「知识真源（尝试前 MUST READ）」列列出的文件**必须先读、并在尝试记录中引用（文件 + 小节）**。速览文档（l2-guide 速览表、本表本身）不替代真源——盲测实测：只看速览开写 RegBase 代码，卡在真源（CastTrait 家族条目 / 掩码按输出位宽重建）本可解答的怀疑点上，预算耗尽回退。

| # | 特性 | 适用信号（可静态判定） | 预期收益 | 知识真源（尝试前 MUST READ） |
|---|------|----------------------|---------|------|
| F1 | **L0C→UB 直通 + dualDstCtrl/subBlockId** | KFC MIX matmul 类；C 矩阵落 GM workspace 再读回 UB；per-tile 同步返回 | 消除每 tile 的 GM workspace 往返（写+读）与一次核间同步 | `cube-migration-guide.md` 改动 3；`$DEVKIT_PATH/examples/01_simd_cpp_api/00_introduction/03_fusion_operation/matmul_leakyrelu_advanced_api/`（`Iterate`+`GetTensorC` 标准形态）；`$DEVKIT_PATH/include/adv_api/matmul/matmul_client.h`（C 位置/VECIN 语义与 `End()` 时机）；`$DEVKIT_PATH/impl/adv_api/detail/matmul/matmul_impl.h`（`IterateAll` 对 Local/VECIN 的行为） |
| F2 | **L0C→GM 随路 NZ2DN** | 输出为列主序/DN 排布，原用 UB 转置或二次搬运实现 | 省掉转置与一次搬运 | `cube-migration-guide.md` 变更总表（`CFG_COLUMN_MAJOR`）；`ascendc-api-best-practices` 的 `references/api-cross-gen-fixpipe.md`（FixpipeConfig 全字段） |
| F3 | **UB→L1 通路** | 原先把 UB 数据经 GM 中转再装 L1（或反之） | 去掉一次 GM 往返 | `cube-migration-guide.md` 变更总表；`$DEVKIT_PATH/examples/02_simd_c_api/03_c_api/00_data_movement/data_copy_ub2l1/` |
| F4 | **双主模式 `enableMixDualMaster`** | 满足 cube-guide「保留高阶必查项」5 条静态约束；AIC/AIV 可独立推进而不依赖消息驱动 | 去掉 KFC 消息往返的部分串行化 | `cube-migration-guide.md`「保留高阶 Matmul API 的必查项」；`$DEVKIT_PATH/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/MatmulConfig.md` 的 enableMixDualMaster 参数说明 |
| F5 | **SSBuffer 跨核参数通道** | 每迭代向对端传小参数（tiling 派生值、循环控制信息），当前走 GM | 零拷贝、省 GM 延迟与同步成本 | `$DEVKIT_PATH/docs/zh/guide/cross_gen_migration_guide/instructions_for_new_features/3510_new_features.md` 特性 8；`$DEVKIT_PATH/impl/adv_api/detail/kfc/kfc_register_obj.h`（docs 缺失，以 impl 为准）；`$DEVKIT_PATH/examples/01_simd_cpp_api/05_best_practices/03_fusion_compute/ssbuf_aiv_aic_comm/` |
| F6 | **Mutex（950 新增）** | 核内异步流水之间同步点密集（SetFlag/WaitFlag 成对多于 3 组） | 简化同步、减少事件管理开销 | `cube-migration-guide.md`「Mutex」；`$DEVKIT_PATH/docs/zh/api/SIMD-API/` Mutex 相关条目 |
| F7 | **同步模式 4（AIC↔单个 AIV）** | 需单对粒度同步；当前用模式 2 全域等待导致空转 | 降低等待粒度，减少空等 | `$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md`（仅 950PR/DT；含 flagId 方向映射与双发双收实例化规则）；`$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/key_features.md`（参与集合定义） |
| F8 | **LoadData 扩展（MX / NDDMA / loop 模式）** | 装载来源非规则 2D 或有 ND-DMA 语义需求 | 提升装载效率/减少指令数 | `cube-migration-guide.md` 改动 2；`ascendc-api-best-practices` 的 `references/api-loaddata.md`（LoadData2DParamsV2 字段单位）；`$DEVKIT_PATH/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_load/LoadData_2D_V2.md` |
| F9 | **AIV 侧 RegBase 重写** | AIV Vector 路径为 MemBase 且 VEC 占比高（`aiv_vec_ratio` 显著非瓶颈但 scalar 高 / 指令数多） | 减少搬运与指令开销 | **下钻顺序固定**：① `l2-guide.md`（速览 + VF 封装规则）→ ② `api-mapping.md`（对照表）→ ③ **`ascendc-regbase-best-practice` 的 `references/api/regbase_api_reference.md`（CastTrait 家族 / Load·Store / 掩码语义，窄化 Cast 必查）与 `references/pitfalls/`（精度与布局陷阱）**——③ 是真源，①② 只是入口；SDK header（`$DEVKIT_PATH/include/basic_api/reg_compute/`）做最终确认 |
| F10 | **SIMT 优化（L3）** | Scatter/Gather、索引逻辑简单、无需 UB 中转 | 索引密集路径大幅加速 | `l3-guide.md`；`ascendc-simt-best-practices` |
| F11 | **低阶直跑替换高阶 Matmul（改动 2/3/4）** | 高阶 API 的 L1/L0 管理成为瓶颈（cube_util 低、per-tile 同步多）；或 flagId 冲突不可解 | 精细控制装载/回写时机 | `cube-migration-guide.md` 改动 2/3/4 + 「低阶直跑改造常见编译错误速查」；`$DEVKIT_PATH/examples/01_simd_cpp_api/06_compatibility_guide/data_copy_l1togm/`（硬事件链标准流水） |
| F12 | **溢出模式控制 `SetCtrlSpr`** | 数值范围可控（RMSNorm/Softmax 类）且溢出检测在热点路径 | 关闭溢出检测提吞吐 | `l2-guide.md` 改动 5（保存/恢复强制的原文）；`$DEVKIT_PATH/impl/` 中 GetCtrlSpr/SetCtrlSpr 定义确认 SPR 编号 |

## 二、重点特性落地要点

### F1：L0C→UB 直通（高阶 API 场景的"C 放 VECIN"）

**适用判定（逐条核对，全部为是才适用）**：

1. kernel 使用 Matmul 高阶 API（`REGIST_MATMUL_OBJ`）且为 MIX 模式；
2. 当前 C 目标位置为 `TPosition::GM`（典型：`IterateAll<true>(workspaceGm, 0)` 写 per-core workspace，再 `DataCopyPad` 读回 UB 做后处理）；
3. C 的后处理在 AIV 侧（Cast / Transpose / 激活 / 写回）。

**落地要点**：

- 高阶 API 形态：C 类型改为 `MatmulType<TPosition::VECIN, CubeFormat::ND, float>`，用 `Iterate` + `GetTensorC(mmOutLocal, ...)` 拿到 UB 结果（官方 `matmul_leakyrelu_advanced_api` 即此形态），替代 `IterateAll(workspaceGm)`；
- 低阶形态：`FixpipeConfig.isToUB = true`，配合 `dualDstCtrl` / `subBlockId`（`FixpipeParamsArch3510` 字段单位见 `api-cross-gen-fixpipe.md`）；
- **收益来源要算清**：省的是 `tile 元素数 × sizeof(C) × 2`（写+读 GM）的搬移量，以及每 tile 一次"回写完成→读回"的核间/流水同步；
- **★ 前置容量校验（先算，但不得据此否决特性本身）**：
  - **整块驻留变体**：C 整块常驻 UB 的容量 = `baseM × baseN × sizeof(C)`。典型 `256×256 fp32 = 256KB` > 单 AIV 可用 248KB（不是 512KB 共享池，见 `ub-budget-guide.md` 头部 ★ UB 空间归属）→ **该变体不可行**。
  - **流式变体（默认应评估的形态）**：高阶 API 的 `Iterate` + `GetTensorC` 本来就是**按 base 块**回写 UB，UB 只需容纳**单个 base 块**，不需要容纳整个 256×256 tile。把 `baseN`（或 baseM）调小到单块 ≤ 可用 UB 即可落地，例如 `baseN=128` → `128×256×4 = 128KB`；低阶形态则对 L0C 子区域多次 `Fixpipe` 分段回写。
  - 算子原有 buffer（src / trans / work 等）可能已占大量 UB（实测某 cube 算子合计约 176KB），须计入可用余量。
  - **判定纪律**：容量算式只能否决"整块变体"，**不能否决 F1**。缩小 base 块会降低 Cube 效率，因此 MUST 通过 A/B 测**净值**（省下的 GM 往返 + 同步 vs Cube 损失）；不得以"需要降 tile"为由跳过尝试（Stage 6 R2）。
  - 只有当"分块后单块仍放不下"（如 C 的 dtype/布局要求最小块就超 248KB）或"分块后 Cube 效率损失已被实测证明大于收益"时，才可判定 F1 不适用/收益不足——两种情况都必须附算式与实测数据。

**验收指标**：`AIC MTE2 ratio` 相对下降 ≥ 10% 或 `aic_fixpipe_ratio` 结构变化 + 中位耗时改善 ≥ 5%；精度全量 N/N。

**回退方式**：恢复 C 的 GM 位置与 `IterateAll(workspaceGm)` 路径；同步恢复 workspace 大小计算（host tiling 的 `GetWorkspaceSizes`）与 tilingKey 相关字段。

### F4：双主模式

**适用判定**：逐条核对 `cube-migration-guide.md`「保留高阶 Matmul API 的必查项」5 条（消息驱动不可依赖 / 核函数类型与 IBSHARE 组合 / 所有 Matmul 对象取值一致 / A、B、Bias 仅从 GM 搬入 / 结果只走 `IterateAll` 到 GM 或 LocalTensor）。**任一条不满足即不适用**，不要"试试看"。

**落地要点**：`MatmulType` 配置 `enableMixDualMaster`；核比例由框架自动推导（见 `cube-migration-guide.md`「MIX 任务类型由框架推导」）；确需显式声明时用入口内 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`，**禁止使用 `__mix__(1,N)` 属性形式**（ops-nn 入口按 tilingkey 改名后会编译失败，白耗构建循环）；注意双主模式下 AIC 不进入 KFC server 循环，原依赖消息驱动的自定义同步逻辑必须重新审查。

**验收指标**：中位耗时改善 ≥ 5%；`aic_mte2_ratio` / `cube_utilization` 改善；精度 N/N。

**回退方式**：移除 `enableMixDualMaster`（回到普通 MIX 消息驱动）。

### F5：SSBuffer 跨核参数

**适用判定**：AIC/AIV 之间有"每迭代都要传、量很小"的参数；当前经 GM workspace 传且伴随一次跨核 Set/Wait。

**落地要点**：写入方写 SSbuf → `CrossCoreSetFlag<4, PIPE_S>(flagId)`；读方 `CrossCoreWaitFlag<4, PIPE_S>(flagId)` 后直读标量。**flagId 计入占用区**（KFC 用 15/31，自定义建议 `[0,10]` 且避开 `[0,2N-1]`）。docs 标注"资料开发中"，以 `$DEVKIT_PATH/impl/` 源码为准。

**验收指标**：同步等待时间下降；中位耗时改善 ≥ 5%。

**回退方式**：恢复经 GM 传参。

## 三、判定纪律

1. **"不适用"必须写证据**：如"算子为纯 Vector 无 L0C"、"A/B 来自 UB 不满足双主约束第 4 条"。
2. **禁止把 F1/F11 当默认动作**：它们改变数据通路，只有在选为目标瓶颈时才尝试，且必须过全量精度。
3. **禁止跨特性搭便车**：不得在尝试 F9 时顺手改 F1，归因会失效。
4. **收益不足即回退**：改善 < 5% 且未命中该特性指定瓶颈指标 → 回退，记"收益不足"。
5. **文档置信度标注**：docs 缺失（如 SSBuffer、Mutex 细节）时以 `$DEVKIT_PATH/impl/` 源码为真源，记录所查文件路径作为证据。
6. **★ 先读真源再写代码**：进入尝试的特性 MUST 先读其「知识真源」列文件并在 `a5_feature_plan.md` 引用（文件 + 小节）。调试中每个"未验证怀疑点"先回真源查证，真源解答不了才允许带着怀疑点回退；速览文档（l2-guide 速览表、本表）不构成"已查证"。
