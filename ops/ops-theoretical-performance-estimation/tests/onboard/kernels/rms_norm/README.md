# rms_norm 直调 `<<<>>>` 性能 ST

自包含的 AscendC 直调工程：实测值直接对应 `references/examples/rms_norm_flow.json` 的两个模版。
参考 ops-nn/norm/rms_norm 的 arch35 RegBase 计算，用高层 AscendC API 写（fp32）：

| kernel | 对应 flow 模版 | x 从 GM 读 |
|---|---|---|
| `rms_norm_full`  | `row_resident_1pass` | 一次（整行常驻 UB） |
| `rms_norm_split` | `row_split_2pass`    | 两次（按列切 chunk，pass1 求 Σx²、pass2 归一） |

计算：`s=Σxᵢ² ; rstd=1/sqrt(s/H+eps) ; yᵢ=xᵢ·rstd·gammaᵢ`。

实现照 ops-nn arch35 RegBase 的 idiom：
- 矢量计算全在 **VF**（`__VEC_SCOPE__` + MicroAPI `RegTensor/MaskReg/UpdateMask/Mul/Add/ReduceSum/...`），
  寄存器级横向归约（`ReduceSum(vMean, accReg, preg)`）；
- GM↔UB 同步**全交给 TQue EnQue/DeQue**，不手插 SetFlag/WaitFlag；
- rstd 留 UB，VF2 用 `DataCopy<LoadDist::DIST_BRC_B32>` 广播进寄存器，**不做 V→S 标量回读**。

## 构建 / 运行

前置：当前环境已有可用 Ascend950，执行 `source /path/to/cann/set_env.sh` 加载实际 Toolkit（已有 CANN 9.1.0 验证记录）。

```bash
bash build_run.sh build   # 只编译
bash build_run.sh run     # 在当前 950 上跑用例
bash build_run.sh all     # 编译 + 跑
```

`build_run.sh` 内置 4 个用例（shape 同源 `rms_norm_cases.json`：全载 s/m/l + 非全载 xl）。
单独跑：`./build/rms_norm_direct --variant {full|split} --rows R --cols H [--chunk C] [--iters N]`，
打印 `RESULT ... us=.. golden=PASS maxRel=..`。

## 接入对照

把各 `RESULT` 的 `us` 按 case_id 写成一份实测 CSV（存 skill 外，不提交），再
`python3 ../../compare.py --file <你的rms_norm.csv>` 与理论下界对照（CSV 格式见 collect_onboard.md）。

注意：UB≈248KB/core；split 的 `--chunk` 要让 3 个双缓冲队列+临时量放得下（H=16384 用 4096）。
实测（bench-device-01）：s/m 全载 eff≈0.98（贴理论下界）、xl 非全载 eff≈0.80、l(H=8192) eff≈0.53。
l 偏低是单累加器 reduce 的依赖链丢 ILP（ops-nn 用多累加器树形归约规避），属 kernel 优化余量、非模型 bug；
只要 golden PASS 且无 BELOW_FLOOR，理论下界即被验证为有效。详见 `../../collect_onboard.md`。
