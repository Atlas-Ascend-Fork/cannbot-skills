# 算子计算流（flow）与模版编写规则

架构：**计算流定义（多模版）+ 通用解释器 + 通用 CLI + 共享标定引擎**。本文件是「给一个新算子写 flow」的契约与步骤。**新算子只写 JSON，不写 .py**。

```
计算流 <op>_flow.json (多模版: copy_in/vf/copy_out)   ← 新算子唯一要写的「脑力」文件
   │ 路由(shape/attr/UB可行性) + 解释
   ▼
flow_runtime.build  ──►  标准 case dict  ──►  共享引擎  ──►  理论时延/bound/CP4
   ▲                        可审计中间产物      evaluate_a5_vector_roofline
   │ test_design_conformance.py 独立按选中模版校验
perf.py --op <name>  (通用 CLI: 单一入口, 无算子专属代码)
```

## 文件分工

| 文件 | 改不改 | 作用 |
|---|---|---|
| `references/examples/<op>_flow.json` | **每算子一个** | 计算流定义：算子 = 多模版，每模版一条完整 flow。**唯一真源 + 符合性金标准**（人审对应真实设计文档） |
| `scripts/flow_runtime.py` | 共享，不改 | 路由（选模版）+ 解释（flow→case dict）。算子无关 |
| `scripts/perf.py` | 共享，不改 | 通用 CLI（`--op <name>`）：委托 `flow_runtime.build`。所有算子共用一个入口，无 per-op .py |
| `scripts/perf_testcase.py` | 契约/共享 | 输入 case json 的加载 + 零依赖校验（`validate()` 即唯一真契约） |
| `scripts/_engine_bridge.py` | 共享，不改 | 加载引擎/数据、逐 case 跑、打印；ctx 暴露 dtype 字节 + UB 容量 |
| `scripts/evaluate_a5_vector_roofline.py` + 两个 JSON | **绝不重写** | 指令 issue/cycle、overlap、带宽、tiling 修正——实测标定、有回归测试 |

## flow 文件格式

```json
{
  "operator": "layer_norm",
  "attrs": ["axis", "epsilon"],
  "reduce_axis_attr": "axis",          // 给定则可用 reduced 粒度（rows = n / 该轴长）
  "compute_dtype": "fp32_for_lowp",    // 低精度输入：op 步骤按 fp32 算；Cast 显式写成 when:"lowp" 步骤；或 "same"
  "gm_rw_contention": true,
  "gm_bandwidth_model": "l2_aware",   // 可选：L2 驻留感知 GM 带宽(标定曲线)，取代固定 chip_peak+α；
                                      // 重复 copy_in 同名张量→记 gm_reread(L2 复用、与 HBM 流重叠)。
                                      // 曲线按 1读1写 流式标定，多输入流算子(add=2读1写)先不开
  "launch_floor_model": "hardware",   // 设备启动下限 max(body, floor)；小 shape 抬到 ~1.6us，中大不变
  "templates": [
    {
      "name": "row_resident_1pass",
      "select": [],                    // 谓词全命中且 UB 可行才选；[]=无条件候选。无 tiling → 由 VF footprint 自动推
      "flow": [
        {"copy_in": "x"},
        {"vf": "mean", "live": [{"name":"x","extent":"row"}],
         "ops": [{"op":"ReduceSum","on":"full"}, {"op":"Muls","on":"reduced"}]},
        {"copy_in": "gamma", "when": "has:gamma"}, {"copy_in": "beta", "when": "has:beta"},
        {"vf": "normalize",
         "live": [{"name":"x","extent":"row"}, {"name":"gamma","extent":"row","when":"has:gamma"},
                  {"name":"beta","extent":"row","when":"has:beta"}, {"name":"y","extent":"row"}],
         "ops": [{"op":"Sub","on":"full"}, ..., {"op":"Mul","on":"full","when":"has:gamma"}, {"op":"Add","on":"full","when":"has:beta"}]},
        {"copy_out": "y", "like": "x"}
      ]
    },
    {"name": "row_split_2pass", "select": [], "tiling": {...feasible chunk...},
     "flow": [ {"copy_in":"x"}, {"vf":"pass1",...}, {"copy_in":"x"}, {"vf":"pass2",...}, {"copy_out":"y","like":"x"} ]}
  ]
}
```

三种步骤：
- `copy_in: <名>` → MTE2，该张量字节计入 `gm_read`；**同名出现两次 = 重读两次**（非全载多 VF）。`"*"` = 全部输入。`when` 条件包含。
- `vf` → 一段寄存器驻留矢量计算块；含 `ops`(微指令列表) + `live`(该 VF 期间常驻 UB 的张量, `{name, extent: row|scalar, dtype?, when?, db?}`)。VF 内不展开 UB 划分（寄存器级）。
  - `db`（默认 true）= 该 buffer 是否双缓冲：footprint 计 ×2 否则 ×1。**一般只对流式进出的 x/y 开 DB**；加载一次的 gamma/beta/weight 与临时量标 `db:false`。
- `op: <指令>` `on: full|reduced`（写在 vf.ops 里）→ Vector，elements = `n` 或 `rows`。可选 `count`。
- `copy_out: <名>` → MTE3，计入 `gm_write`；`like:"x"` 输出与某输入同形同类。
- 步骤/张量 `when`：`"has:<名>"`（输入存在）或 `"lowp"`（主输入低精度）。低精度 `Cast` 用 `when:"lowp"` 显式写在对应 VF 的 ops 里，`src/dst_dtype` 可用 `@in`/`fp32`；**每个读 x 的 VF 各 cast 一次**（寄存器不跨 VF）。cast 算子本身的 `Cast` 不写 src/dst（由输入/输出 dtype 推）。

**tiling**：默认**不写**，由各 VF 的 `live` UB footprint 自动推（峰值存活字节，每 buffer 按 `db` 计 ×2/×1 → `ub_bytes_per_unit`，`double_buffer=1`，unit=行/元素由 `reduce_axis_attr` 定，自动路径下 `core_split_units` 即取该 unit）；行切分 split 模版写**显式 `tiling`** 覆盖（chunk）。写显式 tiling 时**务必给 `core_split_units`**（核间分核单位：elementwise=`n`、行类=`rows`）——它和 `total_tile_units`（核内 UB tile）解耦，决定核间尾块均衡 core_balance；漏写则回退用 UB tile_count，会对均分整除的算子误罚（见 perf_case_schema.md「核间尾块负载均衡」）。表达式变量：`n / rows / cols(=H) / db / ub / nin`；支持 `+ - * / // %`、`ceil/floor/min/max`。

## 模版路由（flow_runtime）

- case 写 `"template":"<name>"` → 强制走该模版（用于对比不同模版性能）。
- 否则按 `templates` 顺序，选第一个：`select` 谓词全命中 **且** tiling UB 可行（`ub_bytes_per_unit*double_buffer ≤ ub`）。
- **全载/非全载自动落出**：resident 模版列前；一行放不下 UB（不可行）→ fall 到行切分模版（其 flow 显式重读 x）。
- 谓词类型：`{"attr":"<名>","eq":<值>}` / `{"attr":"<名>","in":[...]}` / `{"dtype":[...]}`。**attr 也能驱动模版**（不同 attr → 不同计算流）。

## 编写契约

1. **模版是完整无条件的流**：把"全载 vs 非全载""不同 attr"写成**不同模版**，而不是一条流里加条件。
2. **指令名/dtype 必须真实**：引擎强校验，不支持的 op/dtype 直接报错（能抓建模错）。例：bf16 reduce 不支持 → 用 `compute_dtype:"fp32_for_lowp"` 让内部按 fp32 算，并在对应 VF 显式写 when:"lowp" 的 Cast。
3. **元素粒度分清**：归约后的逐行标量运算用 `on:"reduced"`(=rows)，逐元素用 `on:"full"`(=n)。
4. **tiling 用真实工作集**，且必须有一个 UB 可行的模版兜底（大 H 给行切分模版），否则引擎报 UB 不可行。
5. **调优只动白名单旋钮**（case 的 `tuning`，见 `perf_testcase.TUNING_WHITELIST`），记 `tuning_reason`；**禁止**改引擎标定常量。

## 加一个新算子的步骤

1. 读设计文档 → 写 `references/examples/<op>_flow.json`（一个或多个模版）。
2. 写 `references/examples/<op>_cases.json`（小/中/大 + 典型 shape；需要时用 `"template"` 强制某模版对比）。
3. 不写任何 .py —— 通用 CLI 按名字处理所有算子。
4. 跑 `python3 scripts/perf.py --op <op> --cases references/examples/<op>_cases.json --show-case`（表格含路由到的模版），再跑 `tests/offline/test_design_conformance.py`。
5. 上板实测后对照 `cp4_target_time_us`，按需调白名单旋钮。

## 已有样例

- `add` / `cast` / `gelu` — 单模版（elementwise；gelu 低精度 Cast 为 when:"lowp" 显式步骤）。
- `softmax` / `rms_norm` / `layer_norm` — 双模版：`row_resident_1pass`（x 读一次）+ `row_split_2pass`（非全载、x 显式读两次）。layer_norm 还演示可选 affine 输入（gamma/beta 的 `when`）与强制模版对比。
