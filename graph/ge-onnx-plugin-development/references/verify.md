# 编译执行与调试

一键脚本模板：`templates/run.sh`；执行比对模板：`templates/run_model.py`。

## 1. 编译前检查

```bash
source /usr/local/Ascend/cann/set_env.sh   # CANN 环境（atc 可用）
python3 -V                                 # 执行侧 Python 版本需与 run 包构建版本一致
```

- Python 版本一致性：run 包按特定 Python 版本（如 cp312）产出插件桥接库，执行 atc 的
  Python 解释器版本必须与之一致，否则桥接产物不匹配（多版本支持开发中：GE 将提供
  cp310~cp314 桥接产物与不匹配时的现场编译兜底，合入后以 run 包发布说明为准）；
- Python 依赖：`pip3 install torch numpy onnx`（torch 2.7~2.8、onnx 1.21.0 已实测；
  `acl` 由 CANN toolkit 自带）。

## 2. atc 编译

```bash
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin"   # 指向插件目录（目录内只放插件文件）
atc --model=model.onnx \
    --framework=5 \
    --output=model \
    --soc_version=Ascend910_9362
```

- `--framework=5` 表示 ONNX 框架；
- `--soc_version` 用完整芯片名（`npu-smi info` 查询，如 Ascend910_9362、Ascend910B1；
  短名会被 rtSetSocVersion 拒绝）；
- 动态 shape 模型需 `--input_shape` 指定输入形状（当前链路不做动态 shape 自动推导）；
- `ASCEND_CUSTOM_OPP_PATH` 非空且指向目录内存在 .py 时才加载 Python 插件；为空或
  纯 C++ 自定义算子目录（无 .py）时不触发，不影响 C++ 流程；
  多个插件目录可用 `:` 分隔拼接。桥接加载失败（Python 运行时缺失、产物不匹配等）时
  根因详情（runtime 版本、可用产物清单、修复指引）直接透出 atc 控制台。

## 3. 执行与比对

用 `templates/run_model.py`：ACL 加载 OM、拷入输入、执行、拷回输出，并与 PyTorch
参考结果 `np.testing.assert_allclose` 比对。或直接用 `templates/run.sh` 一键跑通
导出 → 编译 → 执行。

## 4. 调试

### 4.1 看插件报错（日志可见性）

插件回调抛 Python 异常或插件文件加载失败时，默认 E19999 报错中会包含 Python 侧错误信息
（异常类型、错误语句、插件文件与行号）。需要更完整编译日志时：

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
atc ... --log=info    # info 级别即可屏显 traceback；完整 plog 落盘用 --log=debug
```

### 4.2 图 dump 验证接入结果

```bash
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/plugin"
mkdir -p graph_dump
DUMP_GE_GRAPH=3 DUMP_GRAPH_PATH="$(pwd)/graph_dump" \
atc --model=model.onnx --framework=5 --output=model --soc_version=<SOC>
```

编译后图文件保存在 `graph_dump/pid_*/`：`ge_onnx_*.pbtxt` 按 13 个编译阶段记录图结构
（PreRunBegin → ... → Build），打开可确认：

- 一对一映射：节点 type 变为 target 算子（如 MyElu→Elu），属性已转写；
- decompose：原节点被替换为子图算子（如 Threshold+Mul），子图节点携带
  `/原算子名` 原始类型标记，可追溯分解来源；
- `ge_proto_*.txt` 为文本格式，便于直接查看。

### 4.3 常见编译期症状

| 症状 | 检查方向 |
|------|---------|
| E13010/E16002 算子未注册（其后跟 E19999 诊断） | domain/opsets 两侧一致性（export.md §3 三段拆解；自定义域版本惯例登记 1；`ai.onnx` 允许作为插件 domain——此时 opsets 覆盖模型的 opset_version，但不要在 custom_opsets 给 ai.onnx 登记版本）；`ASCEND_CUSTOM_OPP_PATH` 是否指向插件目录。较新 GE 版本诊断信息会指明版本冲突的生效版本与被覆盖算子 |
| WARNING 指明 ai.onnx 域版本冲突 | custom_opsets 给 ai.onnx 登记了版本，与 opset_version 互相覆盖，删掉该登记 |
| 编译开始即报插件冲突/加载失败退出 | 不同插件 opsets 重叠（同一 origin type 注册两次）；或插件目录内有语法错误文件（fail-fast，用 §4.1 看具体文件与行号） |
| 属性转写后值不对 | parse_operator 仅 `f` 字段是字符串形式（`i`/`s` 为原生类型），`f` 取用时转数值；零值标量（0/0.0/""）在旧 run 包会丢失，升级 CANN 包 |
