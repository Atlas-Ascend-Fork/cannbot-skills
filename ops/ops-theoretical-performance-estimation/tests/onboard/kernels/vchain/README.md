# vchain 直调 `<<<>>>` —— vector-bound 验证 ST（bf16 输入）

合成的**计算受限**算子，专门验证引擎 roofline 的 **vector 半边**(issue/cycle、超越函数
transcendental 子管线、低精度 Cast)，与访存半边(add/rms_norm)对称。

计算：`y = Σ_{k=0}^{K-1} sqrt(x+k)`，bf16 输入、**fp32 计算**。K 个 sqrt **各自独立**(不是依赖链)→
**吞吐受限**,理论(Sqrt 吞吐地板)≈实测。K 大 → `vector_time ≫ mte ≫ launch_floor`,理论 bound =
`vector_bound`;bf16 I/O 把访存减半→更 vector-bound。与 `vchain_flow.json` 逐 op 对齐(Cast + Adds×K +
Sqrt×K + Add×K + Cast;attr `K` 经 op `count` 引用)。

## 构建 / 运行
```bash
source /path/to/cann/set_env.sh
bash build_run.sh all          # build + 跑 vchain_cases.json 同源用例
./build/vchain_direct --rows R --cols H --K K [--tile 8192] [--iters 50]
```

## 关键设计
- **bf16↔fp32 Cast(VF 内)**：读 `DataCopy<bfloat16_t,DIST_UNPACK_B16>`+`Cast<float,bfloat16_t,..>`，
  写 `Cast<bfloat16_t,float,..>`+`DataCopy<..,DIST_PACK_B32>`。cast trait 必须命名空间作用域(模板引用
  参需静态存储)且只在 device 遍可见 → 用 `#if defined(__NPU_ARCH__)` 守起来，host 遍走空体桩(.asc
  host+device 两遍编译)。
- **吞吐最优(无依赖链)**：`y=Σ_k sqrt(x+k)` 的 K 个 sqrt 互相独立。4 路 k 交织(`k0..k3` 跑寄存器从
  `x+{0,1,2,3}` 起、每轮 `+4.0f` 常量步进)+ 4 个累加器 `a0..a3`,每轮先发 4 个 Sqrt 再 4 个 Add 再 4 个
  Adds → sqrt 之间无数据依赖,达指令吞吐峰值。**VF 内只用常量步进**(运行期 `(float)k` 标量会触发
  "Unsupported Inst must be hoisted")。

## 实测结论(bench-device-03, golden bit-exact maxRel=0)

| case | 实测us | 理论us(Sqrt t4谱) | eff | 判定 |
|---|---|---|---|---|
| m 4096×2048 K=64 | 168.3 | 158.9 | 0.94 | OK |
| m 4096×2048 K=128 | 327.3 | 317.8 | 0.97 | OK |
| l 8192×4096 K=64 | 667.8 | 635.6 | 0.95 | OK |

理论(sqrt t4 吞吐地板,ALU 完全重叠)≈实测,**3×OK**。这验证了引擎 vector 半边对一个吞吐最优 vector-bound
kernel 是准确的上限。

**这个结论是经过一次 tiling 建模修正才闭环的**:早期 core_balance(核间尾块均衡)误用 **UB tile_count**(填满 UB
的大 tile 数,265 个)当分核粒度 → `ceil(265/64)=5、eta=0.828` → 把有效地板 158.9 假性抬到 191.9 > 实测 → 小 shape
假 BELOW_FLOOR。**真实 kernel 是按元素静态均分到 64 核**(`elemsPerCore=8388608/64=131072` 整除,每核各 16 tile,
零核间不均)。修正:core_balance 改吃 flow 声明的 `core_split_units=n`(分核单位),`131072/131072=eta 1.0` → 理论
158.9 ≈ 实测。详见 `references/perf_case_schema.md`「核间尾块负载均衡」。

Sqrt fp32 = **t4 已确认**:`regbase-microbench` 微基准实测报告记录单核测量值 `issue_per_vector_cycle=0.4996≈0.5`、
`expected_class=four_cycle_dual_issue`、`class_match=true`(Sqrt/Exp/Log 一致)。

> ⚠️ **方法学(务必记住)**:
> 1. **吞吐只能用单核无依赖 microbench 标**,不能用 `t=op(t)` 依赖链 kernel(那测的是延迟,受 ILP 限)。
> 2. **microbench 口径**:用 `measured_cycles`(稳态吞吐真值),**不要减空循环基线**(body_cycles)——unroll
>    流水里循环开销与指令体完全重叠,再减 empty=双重扣除,会虚高 ~23%。本仓曾据 body_cycles 的 0.616 把 Sqrt
>    误判 t3.25,实为 measured_cycles 的 0.4996=t4。
> 3. **不为单算子改标定常量**:曾据依赖链 vchain 把 Sqrt 误标 t8;后又据吞吐最优 vchain 的 BELOW_FLOOR(其实
>    是 core_balance 喂错粒度)把 Sqrt 改 t3.25(且只改 Sqrt、Exp/Log 同样数据却没改=双标)——违铁律,已回退 t4。
>    BELOW_FLOOR 要回到 microbench/引擎分量去定位真因,而不是改常量去消除它。
> 4. **tiling 要在 flow 里如实表达**:核间分核(core_split_units)与核内 UB tile(total_tile_units)是两件事,
>    core_balance 必须吃分核单位。vchain 暴露了引擎曾用 UB tile_count 当分核粒度 → 对均分整除算子误罚。已修。
