# 阶段 3：编译与安装

**前置条件**：`[GATE-2] FILES_MODIFIED=[...]` 已输出。

## LOADED 检查点

| 序号 | MUST READ 文件 | LOADED Token |
|------|---------------|-------------|
| 1 | `references/m200-port-guide.md`（风险扫描程序；310P 构建注意事项见本文件） | `[LOADED] m200-port-guide` |

**3510 路线的 l2-guide 编译速查不适用本路线（RegBase 相关错误不会出现）。**

**LOADED Token 输出后，方可继续。**

## Step 3.1：准备构建安装脚本

**MUST 使用模板化脚本**，禁止自行拼编译命令。

### 3.1.1 填充 build_and_install.sh

读取 `../../references/build_system/build_and_install.sh.template`，将阶段 0 的环境变量填入：

| 模板变量 | 填入值 | 来源 |
|---------|--------|------|
| `{{CANN_SET_ENV}}` | CANN set_env.sh 完整路径 | 阶段 0 CANN 环境 |
| `{{CONDA_ENV}}` | 环境名 | 阶段 0 Python 环境 |
| `{{SOC}}` | `ascend310p` | 阶段 0 Step 0.8 实证落盘的 SOC_BUILD |
| `{{OP_NAME}}` | 算子名 | 阶段 0 算子信息 |
| `{{OP_DIR}}` | 算子工程根目录 | 阶段 0 算子信息 |

将填充变量以表格形式在对话中展示（供用户审阅留痕）：

```
构建配置：
| 变量 | 值 |
|------|-----|
| CANN 环境 | /path/to/set_env.sh |
| Conda 环境 | env_name |
| 目标 SoC | ascend310p |
| 算子名 | op_name |
| 工程目录 | /path/to/op_dir |
```

展示后写入脚本并直接执行；发现配置错误时修正后重新执行。

### 3.1.2 执行构建安装

```bash
bash build_and_install.sh
```

脚本会自动完成：编译 → 安装 → 验证安装。

## Step 3.2：全量仓编译参考（按需）

如果编译错误涉及 CMake 配置或编译选项，可查阅全量仓：

- **CMake 配置**：`$DEVKIT_PATH/cmake/` 目录下查找相关 CMake 模块
- **编译脚本**：`$DEVKIT_PATH/build.sh` 查看官方编译流程

读取全量仓文件后输出：`[LOADED] $DEVKIT_PATH/<相对路径>`

## Step 3.3：处理编译错误

如果编译失败：

1. 在日志中搜索错误：`grep -n "[Ee]rror" build.log`
2. 查阅下方 310P 编译错误速查表与 stage_1 Step 1.3.2 的 errata 处置卡定位修复
3. 修复后重新执行脚本
4. 最多 3 次排错循环，3 次后仍失败则停止并向用户报告详细错误信息

> **安装/构建类故障**（安装步路径拼接错误、runtime 回退内置 kernel、stub 库 `libopapi_math.so` 缺失、nm 符号不可见、依赖下载失败）见 `../../references/build_system/build-troubleshooting.md`；下表覆盖**编译期**常见错误。

### 常见编译错误速查（310P 版）

| 错误特征 | 根因 | 修复 |
|----------|------|------|
| `Exec format error: bisheng` | bisheng 脚本缺少 shebang | `sed -i '1i#!/bin/bash' build/gen_bisheng_dir/bisheng` |
| `Error 137` | OOM Kill | 限制并行度 `-j4` |
| `bisheng: error: Unsupported NPU architecture or soc` + `LLVM ERROR: PlugIn Error: no aicore arch specified` | bisheng `--npu-arch` 拼写错误（CANN 9.0 只接受 `dav-2002`；`dav-m200`/`dav-2201` 均被拒） | 310P 直调 bisheng 一律 `--npu-arch dav-2002`；标准 `build.sh --soc=ascend310p` 路线由框架推导 |
| `--ops= xxx`（ops 名为空） | `--ops=` 后有多余空格 | 修正为 `--ops=xxx`（无空格） |
| `(1UL<<D)-1` 移位告警/UB | D≥64 时 C++ 移位未定义行为（E8） | 改显式 FULL_MASK 或掩码表 |
| `sizeof(UbLayout)` 类断言 | UB 超预算 | 310P 按 ~240KB 级核算（m200-port-guide §4） |
| aclrtlaunch 头缺失（torch-extension 形态裁剪 fast path 后） | host 仍 include 已移除 kernel 的 aclrtlaunch_* 头 | 同步裁剪 host include（见下方工程形态分支） |

### ★ 静默缺陷警示（310P 门禁纪律）

E1（DataCopyPad/`Copy<float,false>` 静默 no-op）与 E10（CAST_RINT 垃圾）**编译完全无感**。本路线下 GATE-3 的 BUILD=PASS **不代表搬运/cast 正确**——只能靠阶段 4 精度用例暴露。禁止以编译通过推断功能正确。

### 工程形态分支（torch-extension raw-launch 工程）

本文件默认形态是 ops-repo vendor 包（`build_and_install.sh` → .run → OPP vendors）。若来源工程是 **torch-extension raw-launch 形态**（自带 setup.py/CMakeLists + tikcpp `ascendc_library` + ACLRT_LAUNCH_KERNEL 自动生成头，如 npubench/TileLang2AscendC 产出），改用工程自带构建：

```bash
cd kernel && rm -rf build && mkdir build && cd build
cmake .. -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH=${CANN_PATH} -DCMAKE_BUILD_TYPE=Release
make -j8   # 产物 build/<op>_ext*.so；aclrtlaunch_* 头由 tikcpp 按 SOC_VERSION 重新生成（EC-203 无 tilingKey 形参自动适配）
```

- `SOC_VERSION` 拼写 = `Ascend310P3`（tikcpp 形态；ops-repo 形态才用 `--soc=ascend310p`）
- 此形态下无 vendor 安装步，GATE-3 的验证改为：.so 存在 + python 侧 `torch.ops.load_library` + `torch.ops.npu.<op>` 可调用
- host 侧若引用了 fast-path kernel 的 aclrtlaunch 头，裁剪源列表时同步裁剪 include

### `--enable_host_tiling` 边界

该选项的必需性是 3510 regbase tiling 场景的实证结论；310P（MemBase tiling）是否必需**未实证（U3）**：先试不带，失败（tiling 注册缺失症状见公共 `../../references/build_system/build-troubleshooting.md` §2）再加，结果回写 `.env_info.md` 与 `references/m200-port-guide.md` §6。

## Step 3.4：验证安装

安装脚本会自动执行此步。手动验证：

```bash
source ${CANN_SET_ENV}
# 检查算子 .so 是否已安装
find ${CANN_PATH}/opp/vendors/ -name "libcust_opapi.so" -exec nm -D {} \; | grep -i "op_name"
```

## Gate 输出条件

- [ ] build_and_install.sh 执行成功
- [ ] 编译产物已生成
- [ ] 算子已安装到 CANN 运行环境
- [ ] libcust_opapi.so 中包含算子符号

**全部通过 → 输出 `[GATE-3] BUILD=PASS INSTALL=PASS` → 进入阶段 4**
