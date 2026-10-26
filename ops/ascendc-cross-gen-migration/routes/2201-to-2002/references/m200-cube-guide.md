# 310P（dav-m200 / arch20）Cube 类算子迁移指南

> 本文件是 310P 路线的 cube-migration-guide 对应物，覆盖 kernel 含 Mmad / 高阶 Matmul API（`lib/matmul_intf.h`、`REGIST_MATMUL_OBJ`、`IterateAll`）等 cube 特征时的 310P 适配。
> LOADED token：`[LOADED] m200-cube-guide`（cube 类算子在阶段 1 MUST 加载，与 m200-port-guide 同时）。
>
> **与 A5 版 cube-migration-guide 的关系**：A5 版覆盖 2201→3510 的 cube 数据通路变化（ZZ→NZ 分形、分离/双主模式、flagId 避让区、FixpipeParamsArch3510）——**这些内容在 310P 路线整体不适用**（310P 无分离核、无 3510 分形变化），不是"替换"而是"不加载"。310P 的 cube 差异集中在：**cube 仅 fp16**、**低阶 cube API 大面积缺席**、**m200 errata**（扫描与处置卡见 `../stages/stage_1_assess.md` Step 1.3.2）。
>
> **素材来源**：`/home/l00951729/xiaoqiaoling/sdpa_310p_20260924`（高阶 Matmul API + vector softmax 在 310P 标准 opp 路线的完整实证工程）、user_kb `m200_*.md` 卡（batch_mat_mul_v3 310P 战役）。

---

## 0. 触发条件

kernel 源码含以下任一信号即按 cube 类处理（与 A5 版信号表同构）：

- `Mmad` / `MatmulImpl` / `lib/matmul_intf.h` / `matmul::MatmulType` / `REGIST_MATMUL_OBJ`
- `LoadData` / `LoadData2D` / `LoadDataWithTranspose` / `Fixpipe` / `DataCopyCO12DstParams`
- `CrossCoreSetFlag` / `CrossCoreWaitFlag`（310P fuse 核下语义见 §4）

---

## 1. 首要决策：路线分流

| 路线 | 310P 可用性 | 说明 |
|------|------------|------|
| **标准 opp/aclnn 路线（本 skill 默认）** | ✅ 高阶 Matmul API 可用（sdpa_310p 实证：`matmul::MatmulType<TPosition::TSCM, CubeFormat::NZ, half,...>` + `SetTensorA/B` + `IterateAll` + `End` 全链路跑通） | tiling 用 `matmul_tiling::MatmulApiTiling` 生成 TCubeTiling 嵌入 tiling 结构 |
| raw-launch（torch-extension/ACLRT_LAUNCH_KERNEL）路线 | ✅ **高阶 Matmul 可用**（2026-10-15 mm_probe 实证：raw-launch 下 matmul::Matmul 正确运行——PB-117 的段错误记录系 GE 路径缺 tiling 工作空间所致，非 API 不可用） | conv2d 310P 演练即此形态；E18 冷启动 warmup 在本路径为 MUST（managed/opp 路径不需要）；裸 Mmad 骨架另见 user_kb `m200_bare_mmad_bmm_skeleton` 卡 |

---

### 1.1 已判决的不可行路线（310P 探针实证，勿重试）

| 路线 | 判决 | 证据 |
|------|------|------|
| 硬件 img2col（LoadData3D img2col 形态，fused 路径核心） | ❌ 2002 上全零/507014 挂死 | conv2d 310P 演练 img2col_probe（2026-10-15）；与高阶 Matmul 对象结构性互斥（img2col 产 L0A，mm 对象吃 L1） |
| direct-conv vector 直卷（dense conv） | ❌ 25× 慢（op 数 = Ci×kHkW/输出，V 管发射数锁死） | conv2d 310P 演练 direct 探针（2026-10-15）；depthwise（CiG=1）是唯一例外 |
| vendor 形态（v350 conv 指令） | ❌ AscendC 层不可达 | 指令不暴露 |

## 2. cube fp16-only 与 dtype 降级决策树

**核心约束**：dav-m200 cube 只接受 fp16 输入（fp32 累加器）。来源算子（arch22）若为 fp32/bf16/int 量化 matmul，必须走决策树：

```
来源 matmul 输入 dtype？
├─ fp16 → 直移（仅需 errata 处置 + 预算重算）
├─ fp32 → 三选一（详表见 m200-port-guide §2.3）：
│   ├─ ① 厂商对齐（默认）：RNE-fp16 输入 + fp16 Mmad fp32 累加 + 直接 fp32 输出
│   │     精度声明 = 与厂商 fp16-internal 对齐（parity 锚点 maxdiff~1.5e-2）
│   ├─ ② Sterbenz hi/lo 拆分 3-Mmad（理论残差 ~3.4e-5@K=256）
│   │     ★ 仅当门禁是绝对真值门（light skill 的 CPU 标杆即此类）时可选；
│   │       参考相对门下"更准"会挂（vendor-parity 卡实证），且该路线已归档为
│   │       负结果（m200_fp32_hilo_gemm_saga）——选它必须在分析摘要写明门禁语义依据
│   └─ ③ 退 vector 路径（K 小或精度优先于性能时）
├─ bf16 → 310P 无 bf16：移除该 dtype 或 host 侧 widen fp32（m200-port-guide §2.1）
└─ int4/int8 量化 → int8 Mmad 可用性未实证（U5）；未实证前退 vector 点积
```

---

## 3. 标准路线适配清单

### 3.1 高阶 Matmul API 可用性边界

- 可用：`matmul::MatmulType`（TSCM 等 TPosition）、`REGIST_MATMUL_OBJ`、`SetTensorA/B`、`SetOrgShape/SetTail`、`IterateAll`、`End`、`GetTensorC`
- 不可用/绕行：
  - `LoadData2D*` / `Fixpipe` → NOT_SUPPORT（E11）；操作数 ND→NZ 走 `DataCopy(L1, GM, Nd2NzParams)` 单跳（E9）
  - UB→L1 两种 DataCopy 形式是高阶 API 软件仿真（E9），自定义 staging 不要用
- **KTILE≠128**（E4：fixpipe NZ 输出缺陷），tiling 枚举避开
- L0C→UB fp32 读出图像 = **行主序 16×16**（元素 (m,n) 在下标 m*16+n），不是两个 16×8 子分形；正确存出 = 单条 strided DataCopy（blockCount=16，blockLen=2，srcStride=0，dstStride 按目标）；**尺寸陷阱**：BLOCK_MODE_MATRIX 下 float blockLen 单位=1024B，blockCount=2 实际搬 2048B，qCO2 类缓冲按 2048B 分配（m200_l0c_fp32_ub_rowmajor 卡）

### 3.2 tiling 侧

- `matmul_tiling::MatmulApiTiling` 在 310P 可用（sdpa_310p tiling.cpp :288-312 实证），为每个 bmm 生成 TCubeTiling 嵌入 tiling 结构
- 平台信息运行期直查 `GetPlatformInfo` / `GetCoreNumAiv()` / `GetCoreMemSize(UB/L1/L0_C)`（310P 不走 TilingParse CompileInfo 槽——sdpa_310p tiling.cpp :137-140 注释）
- **E5 适配**：kernel 侧读 tiling 标量必须 MTE2 物化到 UB 再读（禁止 GM 直解引用）
- L1 预算模式参考 sdpa_310p：concat carve + matmul share ≤ l1Size/2

### 3.3 vector 伴随计算

cube+vector 混合内核（如 SDPA 的在线 softmax）在 310P 实证可行，注意：

- 跨 pipe 同步用 `event_t` + `SetFlag/WaitFlag` + `PipeBarrier` 按 stage 点（E6：MTE2_V early-fire 处须 PIPE_ALL）
- TBuf→L1 方案在 sdpa_310p 实测 hang 已回退（kernel .h :143-150）——**不要复现这条死路**
- fp32↔fp16 cast 全部 CAST_NONE（E10）

---

## 4. 同步语义（310P 与 A5 的关键区别）

310P 是 fuse 核（无 AIC/AIV 分离）：A5 cube-guide 的"分离模式核能力/双主模式/flagId 15·31 避让区"全部**不适用**。310P 的同步风险在核内流水（E3/E6）与多核分片写出（按 `GetCoreNumAiv()` 运行期分片）。详细规则见 m200-port-guide §5。

---

## 5. 调试

死锁/卡死/结果错误的排查方法论（无同步路径对照法、mask 约定与消费粒度、取证打印、路径级验证）**通用**，继续用本包 `references/cube-debug-lessons.md`；命中 m200 特有条目（E1/E4/E5/E6/E10）时按 stage_1 Step 1.3.2 的处置卡执行。m200 战役沉淀的两条诊断法也可复用：

- **量级匹配诊断**：确定性近 miss 的实测 maxdiff 对照候选缺失项的解析量级（m200_pending_mte1m_event_envelope 卡；注意：收敛结果须交叉证据复核，该卡本身记录了被证伪史）
- **路线级退役信号**：同一失败签名 ×3 轮、每轮换独立根因假设都不动指标 = 别再迭代，换架构或升级用户裁决（m200_fp32_hilo_gemm_saga 卡）

---

## 6. 检查清单（cube 类算子 310P 迁移完成后逐项核对）

- [ ] 路线已声明（标准 opp/aclnn 或 torch-extension raw-launch 均可）；raw-launch + Matmul 对象时已加 discard warmup
- [ ] dtype 降级决策已按 §2 决策树选定并在分析摘要记录理由（fp32 matmul 必须写明门禁语义依据）
- [ ] KTILE 枚举无 128（E4）；L0C→UB 读出按行主序 16×16 + 2048B 尺寸陷阱（§3.1）
- [ ] ND2NZ 走 GM→L1 单跳（E9）；未使用 UB→L1 仿真路径
- [ ] tiling 标量 MTE2 物化（E5）；cast 全 CAST_NONE（E10）
- [ ] 未复现 TBuf→L1 hang 死路（§3.3）
- [ ] 精度报告中已附厂商 parity 锚点证据（fp32 matmul 选路线①时）
