# 阶段 5：性能验证（含 API 差异回归）

**前置条件**：`[GATE-4] PRECISION=N/N_PASS` 已输出。

本阶段使用 `torch_npu.profiler` 采集迁移后算子在 310P（Ascend310P3 / dav-m200）上的性能数据，生成性能评估报告。如果用户提供了 A2/A3 基线数据，则做跨平台对比。

**★ 本阶段 MUST 包含 m200 风险回归用例**（见阶段 1 的 m200 风险回归 shape 清单），验证 errata 替代路径开销（E1 标量替代、E13 读出 sweep、fp32-via-fp16 性能档）是否影响算子。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/m200-port-guide.md` | `[LOADED] m200-port-guide` |

**★ 若阶段 1 扫描发现算子命中 m200 风险路径，MUST 加载 `m200-port-guide` 并设计回归用例。**

## Step 5.0：全量仓性能优化参考（按需）

如需查阅性能优化技巧或参考同类算子的性能调优案例，按 `../../search-rules.md 路由到全量仓：

- **性能优化指南**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/simd_operator_optimization/` 查找访存优化、流水线优化、计算优化等
- **优秀实践案例**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/best_practices/` 查找 FlashAttention、GroupedMatmul 等标杆实现
- **性能分析方法**：`$DEVKIT_PATH/docs/zh/guide/operator_practice/performance_analysis/` 查找 profiler 使用方法

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

## Step 5.1：准备性能测试用例

**复用阶段 4 精度测试的用例设计**（shape + dtype + 边界值）。

### 用例筛选规则

从 `analysis_<op_name>.md` 和阶段 4 的测试用例中，按下表选取 **7-11 个代表性 shape**：

| 类别 | 数量 | 选取标准 |
|------|------|---------|
| 小规模 | 1-2 个 | launch 开销主导的典型 shape |
| 中规模 | 2-3 个 | 典型生产场景 shape |
| 大规模 | 1-2 个 | 带宽/算力主导的 shape |
| 边界 | 1 个 | 极端 shape 或 dtype |
| **★ m200 风险回归** | **2-3 个** | **按阶段 1 的 m200 风险回归 shape 清单设计** |

每种选取的 shape 遍历主要 dtype（310P 无 BF16：FP16/FP32），**用例总数 10-18 个**（shape 数与 dtype 数的乘积）。

### ★ m200 风险回归 shape 选取

根据阶段 1 的 m200 风险扫描结果，选取 errata 替代路径开销最大的 shape 做回归：

| 风险路径 | 高风险 shape | 说明 |
|---------|------------|------|
| E1 DataCopyPad 替代（标量合并/RMW） | 非对齐尾段占比大的 shape（小 W、奇数维、stride>1） | 标量合并开销随非对齐元素数线性增长 |
| E13 L0C→UB 全量 sweep 读出 | 多分形大 tile 的 GEMM shape | sweep 含无效分形约 2× 冗余读取 |
| E14 GatherMask→标量解交织 | stride-2 / 解交织类 shape | 标量路径吞吐远低于 vector |
| fp32-via-fp16 matmul | 深 K 累积 shape（K≥1024） | fp16-internal 精度-性能权衡档 |
| 核欠载 | 总 tile 数 < 8 的 shape | 310P 仅 8 核，欠载即性能地板 |

未命中任何风险路径时，报告明确标注"未命中 m200 风险路径，无需回归"。

### 生成性能测试脚本

基于阶段 4 的精度测试脚本改造，核心区别：

| 维度 | 精度测试（stage_4） | 性能测试（stage_5） |
|------|-------------------|-------------------|
| 用例数 | >= 30 | 10-18（7-11 代表 shape × 主要 dtype） |
| 运行次数 | 1 次 | warmup + active（多次） |
| 采集方式 | 无 profiler | `torch_npu.profiler` |
| 对比 | CPU 参考实现 | 仅 NPU 端，可选 A2/A3 基线 |

## Step 5.2：采集配置与执行

### ★ 唯一确定采集脚本（MUST，2026-10-14 owner 裁决）

**所有算子的性能对比 MUST 走 `ops-profiling` skill 的 `msprof_profile_run.sh`**（cannbot-skills `ops/ops-profiling/scripts/`），禁止各算子自造采集脚本：

```bash
bash <ops-profiling>/scripts/msprof_profile_run.sh --quick --output-dir=${OP_DIR} --warmup=3 --device=<N>
# 产物（与 910B 标杆同形同口径，可直接比）：performance.json / performance.log / perf_report.md
```

- **口径**：msprof kernel 级 Task_Duration（与既有标杆报告完全同方法同脚本——跨平台/跨算子可比性的来源）；产物三件套落 ${OP_DIR}
- **用例来源**：`<op>_perf_cases.jsonl`（标准命名优先）→ 任意 .jsonl → .json → model.py `get_input_groups()`
- **310P 适配注记**：① **310P 上 `--compare` 的深度采集组不可用**（其 7 组 aic-metrics 含 `L2Cache`，310P msprof 非法枚举——合法集：ArithmeticUtilization|PipeUtilization|Memory|MemoryL0|MemoryUB|ResourceConflictRatio；conv2d 实测全 case 采集失败）——**310P 一律用 `--quick`**；瓶颈画像用 torch_npu.profiler 的 kernel_details 补充路径 ② 多 kernel 流水线的 Task_Duration 包含 kernel 间 host 间隙——与标杆同方法即公平（910B 参照系同口径）
- **瓶颈定位补充**（非对比主口径）：kernel_details/pipe 画像仍可用 torch_npu.profiler 逐 case 采集（辅助分析，不入加速比表）



> **通用 profiler 机制**（`warmup/active` 固定约定、`*_ascend_pt` 命名、`op_statistic.csv` 定位、归一化口径）若有仓内技能 `ops/torch-ops-profiler` 可用，以其为参照；不可用时按本节配置执行——**本节已给出迁移场景所需的完整取值与目录约定，可独立执行**。（同插件的 `aog-perf-eval` 面向 op-gen 工程工作流，本阶段不适用。）

### 采集配置（瓶颈分析补充路径的固定取值）

> 以下是**瓶颈定位补充**路径（torch_npu.profiler 逐 case 采集 kernel_details/流水比），服务于瓶颈分析与形态取证；**加速比对比主口径已统一走上方「唯一确定采集脚本」（ops-profiling）**，本节不再承担对比职能。


```python
import torch
import torch_npu

# 最大数据量采集配置
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=True,
    msprof_tx=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None
)
```

**参数说明**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `profiler_level` | `Level1` | 采集 CANN 层 + NPU 硬件层数据（必须） |
| `aic_metrics` | `PipeUtilization` | 流水线利用率（VEC/MTE/Scalar 占比） |
| `l2_cache` | `True` | L2 Cache 命中率 |
| `export_type` | `Text` | 输出 CSV + JSON 格式 |

**Schedule（固定）**：

| 参数 | 值 | 说明 |
|------|-----|------|
| `wait` | 0 | 无等待 |
| `warmup` | 5 | 预热 5 步（消除 DVFS 影响） |
| `active` | 5 | 正式采集 5 步 |
| `repeat` | 1 | 不重复 |

**固定 warmup=5, active=5，不可修改。**

### 路径基准

**MUST 使用阶段 0 记录的 `${OP_DIR}`（算子工程根目录）作为绝对路径基准**，确保 Python 代码与 shell find 命令操作同一路径。

```python
import os
OP_DIR = "${OP_DIR}"  # 从阶段 0 环境记录获取，如 /home/user/code/ops-nn_reg
PROFILER_ROOT = os.path.join(OP_DIR, "perf", "op_name")  # profiler 输出根目录
```

Shell 中对应：
```bash
PROFILER_ROOT="${OP_DIR}/perf/op_name"
```

### 采集流程

```python
for case_idx, case in enumerate(perf_cases):
    inputs = build_inputs(case, device)

    # 每个 case 的输出路径（绝对路径）
    case_dir = os.path.join(PROFILER_ROOT, f"case_{case_idx:03d}")

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=5, active=5, repeat=1
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(case_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        experimental_config=experimental_config
    ) as prof:
        for step in range(10):  # warmup(5) + active(5)
            output = torch.ops.npu.op_name(*inputs)
            prof.step()

    torch.npu.synchronize()
    print(f"[INFO] case {case_idx} done -> {case_dir}", flush=True)
```

**关键**：每个 case 使用独立的 `with profile` 上下文，生成独立的 `*_ascend_pt` 目录。

### 输出目录结构

```
${OP_DIR}/perf/<op_name>/
├── case_000/
│   └── *_ascend_pt/                          # 前缀是时间戳，后缀 _ascend_pt 固定
│       └── ASCEND_PROFILER_OUTPUT/
│           ├── op_statistic.csv              # 算子耗时统计
│           ├── kernel_details.csv            # Kernel 详细信息
│           ├── operator_details.csv          # 算子详细信息
│           ├── api_statistic.csv             # API 调用统计（Level1）
│           ├── l2_cache.csv                 # L2 Cache 命中率（l2_cache=True）
│           └── trace_view.json              # Chrome Trace
├── case_001/
│   └── ...
└── ...
```

### 查找 CSV 文件

profiler 输出的 `*_ascend_pt` 目录前缀是时间戳（不可预测），后缀 `_ascend_pt` 是官方固定模式。利用这一点精确查找：

```bash
# 查找某个 case 下的 CSV 文件
CASE_DIR="${PROFILER_ROOT}/case_000"

# 核心指标文件
find ${CASE_DIR}/*_ascend_pt -name "op_statistic.csv"
find ${CASE_DIR}/*_ascend_pt -name "kernel_details.csv"

# 硬件指标文件（部分可能在子目录中，递归查找）
find ${CASE_DIR}/*_ascend_pt -name "*pipe_utilization*.csv"
find ${CASE_DIR}/*_ascend_pt -name "*l2_cache*.csv"
```

**注意**：
- `*_ascend_pt` 是 glob 通配符，匹配任意前缀的 `_ascend_pt` 目录
- `pipe_utilization` 和 `l2_cache` 文件名可能带时间戳后缀，用 `*pipe_utilization*` 匹配
- 搜索范围限定在 `case_NNN/*_ascend_pt/` 内，不会误读其他 case 的数据

## Step 5.3：解析性能数据

### CSV 解析优先级

| 优先级 | CSV 文件 | 关键指标 | 用途 |
|--------|---------|---------|------|
| 1 | `op_statistic.csv` | Total Time(us)、调用次数 | 目标算子耗时（**须先按名归因，禁止全表求和**） |
| 2 | `kernel_details.csv` | Kernel 执行时间、Block Dim | Kernel 级耗时 |
| 3 | `pipe_utilization.csv` | VEC/MTE2/MTE3/Scalar ratio | 瓶颈类型判定 |
| 4 | `l2_cache.csv` | total_hit_rate | Cache 命中率 |
| 5 | `operator_details.csv` | 各算子耗时明细 | 子算子拆解 |

### 核心指标提取

从 `op_statistic.csv` 提取：

| 指标 | 来源 | 说明 |
|------|------|------|
| **耗时(us)** | `op_statistic.csv` | **目标算子行**的 Total Time(us)：先按算子名精确筛选并校验调用数 = active 次，再除以 active(5)。**禁止全表求和**（原因见下方口径说明） |
| **Block Dim** | `kernel_details.csv` | 运行核数 |

从 `pipe_utilization.csv` 提取（最大 shape 的那个 case）：

| 指标 | 字段 | 含义 |
|------|------|------|
| VEC ratio | `aiv_vec_ratio` | Vector 计算占比（>50% = VEC Bound） |
| MTE2 ratio | `ai*_mte2_ratio` | GM 搬入占比（>50% = MTE2 Bound） |
| MTE3 ratio | `ai*_mte3_ratio` | GM 搬出占比 |
| Scalar ratio | `ai*_scalar_ratio` | 标量计算占比（>30% = Scalar Bound） |

从 `l2_cache.csv` 提取：

| 指标 | 字段 | 达标标准 |
|------|------|---------|
| 总命中率 | `total_hit_rate` | >80% 良好，<50% 需关注 |

### 耗时归一化与口径说明

```
per_step_us = 目标算子行 Total Time(us) / active_steps
其中 active_steps = 5（固定）
```

> **★ 主口径：ops-profiling 的 msprof kernel 级 Task_Duration**（见 Step 5.2「唯一确定采集脚本」）——与既有标杆报告同方法同脚本，跨平台可比。wall-clock 仅作"端到端体感"附列。（教训：310P 路线首演曾把 wall-clock 当主口径，与 910B 侧标杆无法对齐——同一 build 上 wall 0.19 / device-sum 0.043 / Task_Duration 0.029 三种口径三个数。）

> **多 kernel 算子的归因口径**：单 kernel 算子按算子名归因（禁止全表求和）；多 kernel 流水线算子（如本路线 conv 的 im2col→gemm→epilogue）一次调用含多个 kernel，**用整轮求和口径**（op_statistic 全部行求和，同 `ops/torch-ops-profiler`）——此时"算子"= 整条流水线，按名单独归因会漏掉流水线开销。

## Step 5.4：生成性能评估报告

### 报告内容（MANDATORY）

MUST 生成 `perf_report_<op_name>.md` 并在对话中展示：

```markdown
# 性能评估报告：<op_name> (310P ascend310p)

## 采集配置
- 平台: Ascend 310P3 (dav-m200)
- 迁移层级: L1/L2
- 采集工具: torch_npu.profiler (Level1, PipeUtilization, L2Cache)
- Schedule: warmup=5, active=5

## 性能数据

| Case | Shape | DType | 耗时(us) | Block Dim | 主导流水 |
|------|-------|-------|----------------|-----------|---------|
| 0 | [128, 4096] | fp16 | 9.75 | 28 | VEC (62%) |
| 1 | [2, 1024, 4096] | fp16 | 57.01 | 28 | VEC (58%) |
| ... | ... | ... | ... | ... | ... |

## 瓶颈分析（最大 shape 用例）

| 指标 | 值 | 说明 |
|------|-----|------|
| 算子耗时（per-step） | 75.60 us | 目标算子行的 Total Time / active |
| Block Dim | 28 核 | 用满可用核数 |
| VEC ratio | 62% | VEC Bound |
| MTE2 ratio | 18% | 正常 |
| MTE3 ratio | 12% | 正常 |
| Scalar ratio | 8% | 正常 |
| L2 Cache 命中率 | 85% | 良好 |

## ★ m200 风险回归分析

| 风险路径 | 高风险 Case | 耗时(us) | 对照 Case | 对照耗时(us) | 开销倍数 | 结论 |
|---------|------------|---------|----------|------------|---------|------|
| E1 标量替代 / E13 读出 sweep / E14 标量解交织 / fp32-via-fp16 / 核欠载（按实际命中列行） | ... | ... | ... | ... | ... | ... |

## ★ 性能形态合规（route.md 三约束）

| 约束 | 判定 | 证据 |
|------|------|------|
| 核数打满（block_dim=8） | 合规 / 不合规（列不合规 case） | 逐 kernel Block Num（kernel_details.csv） |
| 全程向量化（无标量数据通路） | 合规 / 不合规 | 逐 kernel scalar_ratio + 代码审查结论 |
| 能 cube 不纯 vector | 合规 / 不合规 | math 类 kernel 的 mac_ratio + 阶段 1 cube 化判定对照 |

## 跨平台对比（如用户提供 A2/A3 基线）

| Case | Shape | DType | 310P 耗时(us) | 910b 基线(us) | 比值(910b/310P) |
|------|-------|-------|-------------|--------------|---------------|
| ... | ... | ... | ... | ... | ... |

## 结论
- ...（3 条以上）
- ★ m200 风险回归结论（如命中）：errata 替代路径开销比例，是否在以正确性优先的可接受范围
```

### ★ m200 风险回归分析方法

对于命中 m200 风险路径的算子，MUST 做以下分析：

- **errata 替代路径开销归因**：用 kernel_details.csv 的流水比定位（如 E1 标量替代 → scalar_ratio 趋近 1；E13 sweep → gemm 内 mte2 主导、mac_ratio 极低），开销 = 替代路径耗时 vs 同形状理论直通耗时
- **fp32-via-fp16 性能档**：对比同形状 fp16 契约耗时（若有）或标注精度-性能权衡理由
- **核欠载**：Block Dim < 8 的 case 标注欠载比例（310P 仅 8 核）

### A2/A3 基线对比

**基线数据来源**：用户需要在 310P 测试完成后提供。

**基线可选，缺省不阻塞**：在报告中说明基线要求（下述三项），若用户已提供则追加「跨平台对比」表；未提供则跳过对比，仅展示 310P 数据，不因缺基线阻塞 GATE-5。

```
如需跨平台对比，请提供 A2/A3 上同一算子的性能数据：
1. 使用与 310P 相同的测试用例（shape + dtype 一致）
2. 提供格式：CSV 或 Markdown 表格（含 Shape、DType、Kernel 耗时(us)）
3. 采集方式不限（msprof op 或 torch_npu.profiler 均可）
```

## Step 5.5：在对话中展示结果

**MANDATORY**——MUST 在对话中展示以下内容，禁止只输出文件路径：

1. **性能数据表**：所有 case 的 Shape、DType、Kernel 耗时、Block Dim、主导流水（无法归因的 case 标注 `N/A（无法归因）` 及原因）
2. **瓶颈分析表**：最大 shape 用例的流水线占比、Cache 命中率等
3. **★ m200 风险回归分析表**：高风险 Case vs 对照 Case 的开销倍数
4. **跨平台对比表**（如用户提供基线）
5. **3 条以上结论**（含 ★ m200 风险回归结论）

## Step 5.6：产物存档与阶段交接（MANDATORY）

GATE-5 输出前 MUST 完成交接动作：

1. **产物存档**：把当前基线快照存到 `perf/<op>/snapshots/<YYYYMMDD_HHMM>/`（含 perf_report、perf_cases.json、profiler 关键 CSV、精度报告副本）——后续 stage 6 的每轮优化都与此基线对比，**禁止覆盖基线**
2. **向用户报告当前性能**（对话中展示，禁止只给路径）：
   - 性能数据表（各 case 耗时/加速比）
   - 标杆几何平均耗时 / 生成算子几何平均耗时 / 几何平均加速比
   - 形态合规表（三约束逐条状态）
   - 瓶颈 Top3 与优化项预估
3. **宣告下一步**（二选一，明确说出）：
   - 「迁移到此为止」（owner 无优化指令）——流程结束
   - 「**现在正在 stage 6 做循环优化**」（owner 有优化指令/目标值）——进入 `stage_6_optimize.md`

## Gate 输出条件

**★ Evidence-based Gate**——`[GATE-5] PERF=COLLECTED` 仅当以下 12 项证据全部存在时才允许输出。缺少任一项时 MUST 输出 `[GATE-5] PERF=NOT_COLLECTED BLOCKED`。

| 序号 | 必要证据 | 验证方法 |
|------|---------|---------|
| 1 | profiler 执行记录 | 对话中有 `torch_npu.profiler.profile(...)` 代码执行输出 |
| 2 | profiler 输出目录 | `find ${OP_DIR}/perf/<op_name>/case_*/*_ascend_pt/` 返回非空结果 |
| 3 | CSV 数据文件 | 至少存在 `op_statistic.csv` 和 `kernel_details.csv`（310P3+CANN9.0 对自定义算子已实证可产出；前提是 schedule+tensorboard_trace_handler 正确用法且用自定义算子冒烟，见 stage_0 Step 0.8） |
| 4 | `perf_report_<op_name>.md` | 文件存在于 `${OP_DIR}/perf/<op_name>/` 下 |
| 5 | 代表 shape ≥7 个（用例总数 10-18） | 报告中列出各 case 的 Shape + DType |
| 6 | m200 风险回归 case | 若阶段 1 扫描命中 m200 风险路径：报告含高风险 shape case；若未命中：报告明确标注"未命中 m200 风险路径，无需回归" |
| 7 | 性能数据表 | 报告含 Shape/DType/Kernel 耗时/Block Dim/主导流水 |
| 8 | 瓶颈分析表 | 报告含最大 shape 的流水比 + L2 Cache 命中率（310P 为 fuse 核无 aiv_/aic_ 分列；字段名以实测 CSV 为准，采集不到的指标标 N/A 并说明，禁止伪造） |
| 9 | m200 风险回归分析 | 报告含 errata 替代路径开销 / fp32-via-fp16 性能档分析（如命中）或标注"未命中" |
| 10 | 3 条以上结论 | 报告含结论部分，包括 ★ API 差异退化结论 |

### 形态阈值（310P 定值，经 SDPA310p 线与 mfg-ops-vision 线双重校准）

| 指标 | 性质 | 阈值与处理 |
|------|------|-----------|
| Block Num | **硬门** | units ≥ 8 的 case 每 kernel = 8。豁免（均需报告显式声明）：① 小形状 min(tasks, cores) 合法——实在分不了满核的可以不分（owner 裁决 2026-10-12），报告注明 tasks<cores；①b units 拆尽仍 < 8 的 floor 分支 ② 输出 < 一个 32B 量子 ③ CV-precheck 回落按 GetBlockNum() 实测均分。取证：tiling env 守卫 debug 打印每 case 的 path+核数 |
| 热路径标量 GM 写 | **硬门（静态检查）** | kernel 数据通路 `SetValue` on GlobalTensor 零容忍（E16：多核并发标量 GM 写会被硬件丢弃）；白名单豁免单核 launch |
| 热循环标量逐元素 | **硬门（静态检查）** | 主循环/tile 循环内 `GetValue`/`SetValue` 零容忍；per-launch ≤7 元素低频标量（张量尾）允许；装载侧用 E19 模式（对齐窗口+gather 表+哨兵） |
| 确定性 | **硬门** | 双跑 bit-exact（阶段 4 已有——此处复核性能形态改动后仍成立） |
| cube 覆盖 | **硬门（包络制）** | CV 包络（UB 容量/对齐）内的验收 case 必须 cvEligible=1 取证；包络外小形状（<1K 元素）允许纯 VEC 兜底 |
| kernel 级耗时 vs 标杆 | **分档参考门** | profiler/msprof kernel 级时间按 shape 档对比（aclnn 两段式启动地板 ~9us 会淹没小 shape；e2e 比值是 host 敏感指标（实测跨机差 3×），**禁止用 e2e 比值做硬门**）；不达标登记优化项+owner 裁决 |
| scalar_ratio / mac_ratio / pipe 画像 | **画像（不设门）** | scalar 高可能是 V 发射饱和正身；融合算子 mac 低是正确形态——仅作瓶颈分析素材 |

**禁止行为**：
- 禁止仅依据 Todo completed / Agent 自我声明 / "阶段已完成"输出 `PERF=COLLECTED`
- 禁止在没有 profiler 采集的情况下输出 `PERF=COLLECTED`
- 禁止用精度测试数据替代性能测试数据
- 禁止跳过 profiler 配置（Level1 + PipeUtilization + L2Cache + warmup=5 + active=5）

**310P 路线产物要求（route.md 约束 4）**：产物即算子工程本体，本阶段必须落盘：性能报告 `perf/<op>/perf_report_<op>.md`、profiler/计时原始数据（`perf/<op>/` 下 case 目录或等效布局）。加上阶段 4 的精度报告（`.md`+`.json`），产物即含完整精度+性能测试数据。

**注意**：GATE-5 的 token 是 `PERF=COLLECTED` 而非 `PERF=PASS/FAIL`。本阶段只做性能评估和数据采集，不做达标判定。`PERF=COLLECTED` 验证的是"性能数据是否真实采集并完成分析"，而不是强制要求性能提升。如需性能优化，后续可独立使用性能优化流程。
