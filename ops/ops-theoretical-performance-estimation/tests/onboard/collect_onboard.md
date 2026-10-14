# 上板实测采集

以下编译、运行和采集命令默认在**已配置好 Ascend950 / DAV_3510 的当前环境**中执行。
使用与理论用例相同的 shape，采集设备侧时延，整理为实测 CSV，再用 `compare.py` 与理论性能上限对照。

实测 CSV、运行日志保存在 skill 目录外；报告中记录设备匿名编号、CANN 版本、测量日期和计时方法。

## 环境准备

- 当前环境已有可用的 Ascend950、配套 CANN Toolkit 与算子包；直调工程还需要 CMake 和支持 ASC 的编译器。
- 在本 skill 根目录开始执行下列命令，将 `/path/to/...` 替换为实际路径；后续步骤沿用同一个 Bash 会话。
- 使用 TTK 路径时，先按 [ops-test-kit 项目说明](https://gitcode.com/cann/ops-test-kit)准备其源码及 Python 依赖。
- case shape 与 `references/examples/<op>_cases.json` 保持一致，才能逐 case 对照。

```bash
SKILL_DIR="$PWD"
source /path/to/cann/set_env.sh
RESULT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ops-perf.XXXXXX")"
```

## 使用 TTK 采集（以 add 为例）

从同源用例生成 CSV，然后在已准备好的 ops-test-kit 目录执行 Kernel 测试：

```bash
TTK_DIR="/path/to/ops-test-kit"
python3 "$SKILL_DIR/tests/onboard/gen_ttk_csv.py" --op add \
  --cases "$SKILL_DIR/references/examples/add_cases.json" \
  --out "$RESULT_DIR/add_ttk.csv"

(
  cd "$TTK_DIR" &&
  python3 -m ttk kernel -i "$RESULT_DIR/add_ttk.csv" -b=release \
    -o "$RESULT_DIR/add_ttk_out.csv"
)
```

查看 `$RESULT_DIR/add_ttk_out.csv` 及测试日志，确认 kernel 执行成功，并取**实际被测实现对应的设备侧 Task Duration（us）**。
保留 TTK 输出的性能列以区分不同执行路径；重复测量时记录采样次数和取值方法，不把主机总运行时间当作 kernel 时延。
精度状态和性能数据分别记录；缺少 golden 或未执行比较时，不能声称精度验证通过。

按 `testcase_name` 对应 `case_id`，将选定的时延列整理成 `$RESULT_DIR/add_measured.csv`。格式示例：

```csv
# operator: add
# cases: add_cases.json
# source: example; bench-device-01; replace with measurement date, CANN version and timing method
case_id,measured_us,note
add_m_4M,23.553,example value - replace with the current measurement
```

上面的时延仅用于展示文件格式，需替换为本轮实测值。一个算子一个 CSV，`case_id` 与 cases.json 保持一致。
整理完成后运行对照：

```bash
python3 "$SKILL_DIR/tests/onboard/compare.py" --file "$RESULT_DIR/add_measured.csv"
```

`compare.py --dir` 会读取目录中的所有 CSV；批量对照时应指定只存放上述标准实测 CSV 的目录，避免混入 TTK 输入及原始输出文件。

## 算子覆盖与扩展

- `gen_ttk_csv.py` 当前只有 **add** 的 adapter（2 输入 elementwise，已验证）。
- 加新算子：先在 950 确认其 kernel 可被 `ttk kernel -b=release` 测到（op_name / 多输入 / attr 如 axis·epsilon·gamma·beta），再给 gen_ttk_csv.py 加 adapter。softmax/layer_norm/rms_norm 属此类。
- **更优做法（推荐的 ST 终态）**：为这些算子自写 `<<<>>>` 直调 kernel（`ascendc-direct-invoke-template`），使**实测值直接对应我们 flow 描述的计算流**，不依赖 CANN 安装包里算子是否存在、其 kernel 是否与 flow 一致。这样上板 ST 自洽、可复现。

## 直调 `<<<>>>` 采集（推荐终态，已对 rms_norm 跑通）

`kernels/rms_norm/` 是自写 `<<<>>>` 直调 ST 工程：**实测值直接对应 flow 的两个模版**，不依赖
CANN 包里算子是否存在/是否与 flow 一致。参考 ops-nn/norm/rms_norm 的 arch35 RegBase 计算，
用高层 AscendC API 写两套 kernel（fp32）：
- `rms_norm_full`  ↔ `row_resident_1pass`（整行常驻 UB，x 读 **一次**）
- `rms_norm_split` ↔ `row_split_2pass`（按列切 chunk，x 读 **两次**）

host main 用 `aclrtEvent` 取设备耗时（warmup + iters 均值），并跑 CPU golden 自校验，
打印可解析的 `RESULT ... us=.. golden=PASS` 行。工程自包含（standalone CMake + `find_package(ASC)`），
已有 CANN 9.1.0 验证记录；先按上面的环境准备步骤加载当前 Toolkit。

```bash
# 编译并执行 rms_norm_cases.json 对应的 fp32 用例
set -o pipefail
bash "$SKILL_DIR/tests/onboard/kernels/rms_norm/build_run.sh" all \
  | tee "$RESULT_DIR/rms_norm.log"
```

日志中每条 `[case_id] RESULT ...` 对应一个用例。确认 `golden=PASS` 后，把 `us` 按 case_id 整理为
`$RESULT_DIR/rms_norm_measured.csv`，文件头使用 `# operator: rms_norm` 和 `# cases: rms_norm_cases.json`，
并填写本轮测量来源，再执行：

```bash
python3 "$SKILL_DIR/tests/onboard/compare.py" --file "$RESULT_DIR/rms_norm_measured.csv"
```

需要分步操作时，先运行 `build_run.sh build`，再运行 `build_run.sh run`；二者均在当前 950 环境执行。

实现照 ops-nn arch35 RegBase 的 idiom：矢量计算全在 **VF**（`__VEC_SCOPE__`+MicroAPI，寄存器级
`ReduceSum`）；GM↔UB 同步**全交给 TQue EnQue/DeQue**，不手插 SetFlag/WaitFlag；rstd 留 UB 用
`DIST_BRC_B32` 广播进寄存器（不做 V→S 标量回读）。注意 UB≈248KB/core，split 的 chunk 要让 3 个双缓冲
队列+临时量放得下（H=16384 用 chunk=4096）。

实测（bench-device-01）：s/m 全载 eff≈0.98 贴理论性能上限、xl 非全载 eff≈0.80 —— 模型在 kernel 高效时是
紧致的性能上限。l(H=8192) eff≈0.53 偏低是单累加器 reduce 的依赖链丢 ILP（ops-nn 用多累加器树形归约规避），
属 kernel 优化余量、非模型 bug；只要 golden PASS 且无 BELOW_FLOOR，理论性能上限即被验证为有效。

## 判定与处置（见 tests/README.md 验收口径）

| 判定 | 含义 | 处置 |
|---|---|---|
| `OK` | 实测落在 [理论, cp4] | 达标 |
| `BELOW_FLOOR` | 实测时延<理论时延(实测性能超过理论上限) = 模型把上限算低了 | **必修**：定位被低估的常量（多为 L2 驻留带宽 #1），修引擎/数据 |
| `LOW_EFF` | 实测时延远高于理论时延(实测性能远低于上限) | 区分：profiling 占比对得上 cases 的 bytes/指令→kernel 未优化；对不上→flow 漏建模 |

## 留痕（不提交进 skill）

实测 CSV 是 point-in-time 快照，**不提交进 skill**。要留痕就存进 PR 描述 / 评审记录 / 个人目录，
采集口径（设备/次数/取值）务必写进 `# source:` 头便于复核。改引擎后想看漂移，重跑 kernel
得新 CSV，再 `compare.py --file` 对照即可——可复现性由 kernel + 工具保证，不靠提交快照。
