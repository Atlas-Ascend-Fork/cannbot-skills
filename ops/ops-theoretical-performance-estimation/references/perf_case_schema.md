# A5 Pure Vector 理论性能 case 输入格式

性能评估脚本读取 JSONL，每行一个 case。每个 case 描述某个性能测试用例下，由算子设计推导出的计算流统计量。

## 最小示例

```json
{"case_id":"p0_fp32","hardware_variant":"ascend950pr_9589","variables":{"N":1048576},"active_vector_cores":64,"fixed_overhead_us":3.0,"memory":{"gm_read_bytes":"N*4","gm_write_bytes":"N*4"},"vector":{"instructions":[{"op":"Add","dtype":"fp32","elements":"N"}]}}
```

## 顶层字段

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `case_id` | 是 | case 唯一标识 |
| `hardware_variant` | 否 | 默认使用硬件 JSON 的 `default_variant` |
| `gm_bandwidth_model` | 否 | 默认 `chip_peak`；`vector_core_conservative`（INI 派生 VectorCore DDR 口径）；`l2_aware`（L2 驻留感知，标定曲线）；`l2_capacity`（确定性 L2 容量模型，按工作集 vs `C_eff` 阶跃切 L2/HBM，配 `l2_concurrent_streams`/`reuse_working_set`）。详见下方 GM 带宽口径 |
| `gm_rw_contention` | 否 | HBM 读写争用：默认关（读 mte2/写 mte3 视为完全 overlap）；`true`（硬件 α）或数值（显式 α）→ 合并 GM 时间 ≈ `max(mte2,mte3)+α·min(mte2,mte3)`，α≈0.3（`common.gm_bandwidth_measured.rw_contention_alpha`）。读+写流式算子用；`l2_aware` 已含读写争用故无需再开 |
| `variables` | 否 | 表达式变量，值必须可计算为数字 |
| `active_vector_cores` | 否 | 实际使用 VectorCore 数；默认取硬件 variant 的 `vector_core` |
| `tile_count` | 否 | 核内 UB tile 总数；驱动流水填充/排空，且在无 `core_split_units` 时回退驱动尾块均衡，见下方说明 |
| `tiling` | 否 | 由 UB 容量自动推 `tile_count` + 声明 `core_split_units`（核间分核单位），见下方说明 |
| `fixed_overhead_us` | 否 | 固定头开销，默认 0；依据独立实测显式给定，如 launch、调度、短 kernel 固定成本 |
| `fixed_overhead_model` | 否 | 兼容字段，仅支持 `none`；固定开销通过 `fixed_overhead_us` 显式给定 |
| `launch_floor_model` | 否 | 设备启动时延下限：`none`（默认，=0）或 `hardware`（`body = max(body, common.overheads.min_launch_floor_us)`，时延下限(floor)、非叠加，抬高小 shape 理论时延） |
| `gm_rampup_model` | 否 | 兼容字段，仅支持 `none`，不提供预置 GM/HBM 短传输爬坡修正 |
| `fill_drain` | 否 | `max_pipe` 下是否计入流水填充/排空项，默认 `true`；需 `tile_count` 已知才生效 |
| `target_efficiency` | 否 | CP4 建议阈值效率，默认 0.60 |
| `overlap_model` | 否 | 默认 `max_pipe`，也支持 `serial`；见下方 overlap 说明 |
| `memory` | 是 | GM/L2 读写字节数 |
| `vector` | 是 | RegBase 微指令计算流 |

## memory 字段

| 字段 | 说明 |
| --- | --- |
| `gm_read_bytes` | GM 读入字节数，进入 MTE2/GM roofline |
| `gm_write_bytes` | GM 写出字节数，进入 MTE3/GM roofline |
| `gm_reread_bytes` | 重复读取的字节数（同一张量被 copy_in 多次，如非全载重读 x），可选，默认 0。仅 `gm_bandwidth_model=l2_aware` 用：这部分按 L2 复用、与 HBM 流重叠，不计入 HBM 流量 |
| `l2_read_bytes` | L2 读字节数，可选 |
| `l2_write_bytes` | L2 写字节数，可选 |

字段值可以是数字，也可以是表达式：支持变量、数字、`+ - * / // % ( )`，以及 `ceil() / floor() / min() / max()` 四个函数（`ceil`/`floor` 单参，`min`/`max` 变参）。用于尾块对齐 padding，如 `"ceil(N/512)*512"`。结果须为有限非负数。

## vector.instructions 字段

指令性能来源于 asc-devkit 的[Reg 矢量计算接口理论性能汇总](https://asc.gitcode.com/api/appendix/reg_vector_compute_interface_theoretical_perf_summary.html)与[指令单双发汇总](https://asc.gitcode.com/api/appendix/reg_vector_compute_interface_instruction_single_dual_issue_summary.html)。固定修订号、仓库路径和链接见 `vector_instruction_types.json` 的 `source_documents`，每条指令的 `performance_source` 与 `documented_performance_dtypes` / `documented_performance_pairs` 指明已核对范围。

性能表给出单指令并行度 P（elements/cycle），本模型按 `throughput_cycles = 256 / dtype_width_bytes / P` 换算，再由单双发表确定 `issue_slots`；转换指令使用输入/输出最大位宽。例如 Log(fp16) 为 `256/2/16=8` 拍，浮点 MulDstAdd/MulAddDst 为 1 拍，文档列出的整数 MulAddDst 为 2 拍。这里的周期描述吞吐，不是依赖链延迟。

API 支持的数据类型与已核对性能范围分别记录。未列出的类型、特殊模式、复合指令和兼容别名的成本须按 JSON 中的模型假设或实测来源解读；执行单元分组和混合流 overlap 也属于模型层，其中 overlap 由随包微基准校准。

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `op` | 是 | RegBase 微指令名，如 `Add`、`Cast`、`ReduceSum` |
| `elements` | 是 | 该指令处理的元素数量 |
| `count` | 否 | 同类指令重复次数，默认 1 |
| `dtype` | 普通指令必需 | 元素 dtype |
| `src_dtype` / `dst_dtype` | Cast/Truncate 等转换指令使用 | 按大位宽计算并行度 |
| `dependency` | 否 | `independent` 默认，或 `dependent` |
| `efficiency` | 否 | 指令级效率修正，默认 1.0 |

`op` 与 dtype 的组合会按 `vector_instruction_types.json` 中该指令的 `supported_dtypes`（Cast/Pack/UnPack/Truncate 用 `supported_dtype_pairs` / `supported_conversions` 校验 `src_dtype->dst_dtype`）做合法性校验，不支持的组合直接报错而非静默出数。校验对别名做归一（`fp16`/`float16`/`half` 等价）。`elements` 缺失同样报错。

脚本优先使用 `vector_instruction_types.json` 中的硬件执行单元模型：`issue_per_cycle = issue_slots / throughput_cycles`。`issue_slots` 表示可用 issue 资源数：大多数 VALU 指令默认双发，所以 fp32 `Add` 是 2 个 issue slot、1 cycle throughput，约 2 issue/cycle；int32/uint32 `Add` 是 2 个 issue slot、2 cycle throughput，约 1 issue/cycle。规约、Pack/UnPack、Interleave/DeInterleave、tensor-mode `Duplicate` 使用单发 `vector` 资源，即 `issue_slots=1`。

混合 VF 计算流按 execution unit 聚合，`vector_time_us = max` over units。**同一 unit 内还分子管线（`pipe`）**：unit 时间 = `max(发射口时间, 各子管线各自的 Σ时间)`，不是该 unit 所有指令时间相加——发射口对该 unit 所有指令累加 `time/throughput_cycles`（每指令占一个发射槽），但不同子管线的指令并行、不相加。当前唯一非默认子管线是 `vector_transcendental`（Exp/Log/Div/Sqrt/Reciprocal/Rsqrt），与 1 拍 ALU 数据通路重叠。这是 Ascend950 微基准标定（`tests/calibration/instruments/regbase-microbench/docs/overlap_calibration_findings.md`），替代了旧的"unit 内所有指令时间相加"——后者对超越函数+廉价 ALU 混合流（softmax/gelu/normalize）高估 20–60%。例：实测 fp32 `Exp` 后接 `Sub` 的耗时基本等于单独 `Exp`，`Sub` 被超越函数子管线空隙隐藏。

`dependency=dependent` 只作为同资源强依赖链的保守修正输入；如果没有明确的 `dependent_issue_per_cycle` 或实测延迟模型，不应把不同资源的指令盲目顺序累加。

## GM 带宽口径

| `gm_bandwidth_model` | 说明 |
| --- | --- |
| `chip_peak` | 使用硬件 JSON 中的芯片级 GM 峰值带宽，适合作为理论上限 roofline |
| `vector_core_conservative` | 使用平台 INI 派生的 VectorCore DDR 聚合带宽，适合纯 Vector MTE 保守估算 |
| `l2_aware` | L2 驻留感知（标定，combined copy 曲线）：有效读写带宽随 HBM-streamed 工作集变化（小→L2 高带宽，大→HBM），曲线即实测 copy 速率已含读写争用（不再单独 α）；`gm_reread_bytes` 走 L2、与流重叠不计入。读+写流式算子(1R:1W，如 norm)适用，多输入流(add 2R:1W)需另标。参数见硬件 `common.gm_bandwidth_l2_aware`，标定源 `bench_results/membw_copy_sweep.csv` |
| `l2_capacity` | **确定性 L2 容量模型**（非曲线拟合）：逐方向按「复用工作集 W vs 有效容量 `C_eff`」阶跃切 L2/HBM 带宽。`C_eff = C / pass内并存流数`（见 `l2_concurrent_streams`）——一个 pass 里同时占 L2 的大流(reread 读 + 写出)互相挤,单流独占容量 = C/份数。物理：reread 在 LRU 下 W≤C_eff 全命中 L2、W>C_eff 全 miss HBM;写同理。读侧=冷读(HBM)+第二遍 reread(命中L2/missHBM)串行;写侧按 y 量切。常数全 membw 实测(`common.gm_bandwidth_measured`: `l2_residency_knee_bytes` C=128MiB、`l2_read/write_bandwidth`~4.5/4.95、`hbm_sustained_read/write`~1.5)。门禁 `check_calibration.py` 锁。上板验证(rms_norm 行级双 pass, streams=2 → C_eff=64MB): x=112/128MB 模型 156/179us ≈ 实测 158/180 ✓;过渡带(刚过拐点,如 x=96)阶跃判全 miss 偏保守 ~6%(安全)。⚠ 单一 L2 带宽常数,小 reread(≪C_eff)偏乐观(L2 区短传输效应尚未单独建模) |

L2 带宽：`l2_aware`/`chip_peak` 路径下显式 `l2_read/write_bytes` 仍用 `peak.l2_bandwidth_bytes_per_s`（厂商谱 6.4，实测够不到，算子一般不显式给）；`l2_capacity` 用实测 `l2_read/write_bandwidth`(~4.5/4.95)。

`reuse_working_set`（`memory` 内，可选）：`l2_capacity` 下「被复用张量两次访问之间需驻留的字节」= 复用跨度内数据量。缺省 = `gm_reread_bytes`（global 跨度，遍历完整张量再来第二遍，如 norm 行级双 pass）；tile 级局部复用应显式给 tile 字节（小 → 全命中）。

`l2_concurrent_streams`（模版级，可选，默认 1）：`l2_capacity` 下「一个 pass 内同时占 L2 的大流数」(reread 读 + 写出并存挤 L2)→ 引擎 `C_eff = C / 份数`。行级双 pass(pass2 = reread x + write y)= **2**(上板实测满命中拐点 ~64MB=C/2 印证);独占复用 = 1。`flow_runtime` 从模版透传(`row_split_2pass` 已标 2)。

### 读写争用（gm_rw_contention）
`max_pipe` 默认把读(mte2)/写(mte3)当成完全 overlap，但它们共享 HBM。`gm_rw_contention=true`（硬件 α）或给数值时，合并 GM 时间按 `max(mte2,mte3)+α·min(mte2,mte3)`（α≈0.3，`common.gm_bandwidth_measured.rw_contention_alpha`）计，更贴近读+写流式算子（如 softmax fp32 小 shape 误差从 −18%→+6.5%）。`l2_aware` 的 copy 曲线已折进读写争用，二者不叠加。芯片 GM 峰值 `peak.gm_bandwidth_bytes_per_s`(1.4–1.6 TB/s) 是 HBM 持续率；工作集 ≲112MB（L2）的中等算子可达 ~3.3 TB/s（`gm_bandwidth_measured.l2_assisted_peak_*`），尚未自动建模。

## 核间尾块负载均衡（core_split_units）

**两层 tiling 必须分清**：核间「怎么把工作分给核」(决定 core_balance) 与核内「一个 tile 在 UB 里多大」(决定 tile_count / fill_drain) 是**两件独立的事**。

核间均衡由 **`core_split_units`**（分核单位总数：elementwise=元素 `n`，行归约类=行数 `rows`）决定：`eta = (U/cores) / ceil(U/cores)`，对 `vector_time` 做 `÷eta` 修正。物理含义：分核单位不能被 `active_vector_cores` 整除时，最重核多干一份，整个 kernel 等它。**细粒度（元素）分核 eta≈1**（静态均分的 elementwise 几乎完美均衡）；**只有粗粒度（少量行）才显著 <1**。

⚠ **不要用 UB tile_count 当分核粒度**：tile_count 是「填满 UB 的大 tile 数」（核内访存量化），不是核间调度单位。早期引擎拿 tile_count 喂 core_balance → 对均分整除的算子误罚（如 vchain：265 个 UB-tile/64 核 → 假 21% 罚，而其 N 是均分整除的）。现在 core_balance 吃 `core_split_units`；flow 未声明时**回退 tile_count**（保持旧 case 行为）。

只修正 `vector_time` 而不修正 mte：mte 用芯片级聚合带宽（不随核数量化），矢量时间才由 `active_cores × freq` 派生。配合分核修正时把 `active_vector_cores` 设为硬件最大核数，由 eta 还原利用率，不要再手动调小。

不给 `core_split_units` / `tile_count` / `tiling` 时 `core_balance=1.0`。

## 由 UB 容量推导 tile_count（tiling）

`tiling` 块既声明**核间分核单位**（`core_split_units`，进 core_balance），又让脚本按 UB 容量自动推**核内 tile 数**（`tile_count`，进 fill_drain）：

```json
"tiling": {"core_split_units": "n", "total_tile_units": "n", "ub_bytes_per_unit": 16, "double_buffer": 2}
```

| 字段 | 必需 | 说明 |
| --- | --- | --- |
| `core_split_units` | 否 | **核间**分核单位总数（elementwise=`n`，行类=`rows`），决定 core_balance；缺省回退 `total_tile_units` 推出的 tile_count |
| `total_tile_units` | 是 | **核内** UB 切分的工作单元总数（如元素数、`rows*每行tile数`），表达式 |
| `ub_bytes_per_unit` | 是 | 每个工作单元在 UB 中的工作集字节（输入+输出+临时） |
| `double_buffer` | 否 | double buffer 倍数，默认 2 |

推导：`max_units_per_tile = floor(ub_size / (ub_bytes_per_unit × double_buffer))`，`tile_count = ceil(total_tile_units / max_units_per_tile)`。`ub_size` 取硬件 `common.memory_capacity_bytes_per_core.ub`。若一个单元（含 double buffer）都放不下 UB，判为 **UB 不可行**并报错。显式 `tile_count` 优先于 `tiling`。

输出 `tiling` 段含 `max_units_per_tile`、`ub_bytes_used`、`ub_utilization`、`core_split_units`，可用于判断 UB 利用率与分核粒度。

## 流水填充/排空（fill_drain）

`max_pipe` 是多 tile 稳态假设，tile 数有限时首尾 tile 的非瓶颈流水阶段无法被掩盖。脚本在 `tile_count` 已知且 `overlap_model="max_pipe"` 时一阶补一个填充/排空项：

```
fill_drain = (mte2 + vector + mte3)/tile_count − max(mte2, vector, mte3)/tile_count
body       = max(mte2, vector, mte3) + fill_drain
```

`tile_count=1` 时退化为 `serial`，`tile_count→∞` 时退化为纯 `max_pipe`，正好覆盖 SKILL 中"tile 太少别用稳态上限"的告警区间。设 `"fill_drain": false` 可关闭。这是一阶近似（per-tile 矢量时间用的是已含 `core_balance` 的值），tile 极少时与 `core_balance` 惩罚会叠加，符合"tile 少则核利用率和流水填充同时变差"的实际。

## 短传输与固定开销

GM/HBM 短传输不提供预置爬坡参数；`gm_rampup_model` 省略或设为 `none`，其他值报错。
固定开销默认 0。需要建模 launch、调度或收尾成本时，依据独立实测显式给出 `fixed_overhead_us`（可使用表达式），并记录测量方法与来源。`fixed_overhead_model` 仅接受 `none`，其他值报错。

`launch_floor_model="hardware"` 仍使用随技能提供的自测启动下限，通过 `max(body, floor)` 施加。它与加性 `fixed_overhead_us` 的口径不同，针对同一份启动成本应避免重复计入。

## overlap 模型

`max_pipe` 计算为 `max(mte2_time_us, vector_time_us, mte3_time_us)`，表示 MTE2、Vector、MTE3 在多 tile 稳态下充分 overlap。它适用于 tile 数充足、流水填充/排空开销相对主体很小、且 CopyIn/Compute/CopyOut 阶段可以稳定并行的设计。

`serial` 计算为 `mte2_time_us + vector_time_us + mte3_time_us`，适用于 tile 很少、无有效 double buffer overlap、常量搬运或收尾阶段占比明显、或者设计文档无法证明稳定 overlap 的场景。

如果 case 使用 `max_pipe` 但 tile 数很少，应把流水填充/排空、调度、常量特殊搬运等成本计入 `fixed_overhead_us` 或用指令/内存 `efficiency` 做保守修正。

## 输出口径

每个 case 输出：

- `mte2_time_us`: GM/L2 读侧理论耗时
- `mte3_time_us`: GM/L2 写侧理论耗时
- `vector_time_us`: 按 active cores、主频、指令并行度计算出的理论耗时（已含 `core_balance` 尾块修正）
- `vector_time_pre_balance_us`: 尾块均衡修正前的矢量耗时
- `core_balance`: 核间尾块负载均衡率 eta，∈(0,1]，由 `core_split_units`（缺省回退 `tile_count`）算；都无时为 1.0
- `tile_count`: 实际使用的 tile 数（显式或由 UB 推导），无则为 null
- `tiling`: tile 来源与 UB 推导细节（`source` / `max_units_per_tile` / `ub_bytes_used` / `ub_utilization`）
- `fill_drain_us`: 流水填充/排空补偿耗时
- `gm_rampup_model`: 兼容字段，恒为 `none`
- `fixed_overhead_source`: 固定头开销来源（`explicit` / `none`）
- `launch_floor_us`: 实际施加的设备启动下限（`max(body, ·)`，none 时为 0）
- `vector_resource_times_us`: 按 vector execution unit 聚合后的资源耗时
- `kernel_body_time_us`: 不含固定头开销的主体耗时
- `theoretical_time_us`: `fixed_overhead_us + kernel_body_time_us`
- `bound`: `mte2_bound` / `mte3_bound` / `vector_bound`
- `cp4_target_time_us`: `theoretical_time_us / target_efficiency`
