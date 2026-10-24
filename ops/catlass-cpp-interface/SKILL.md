---
name: catlass-cpp-interface
description: CATLASS C++ 通用算子和已登记注意力家族五阶段工作流的 01 接口确认。触发：开始开发 CATLASS C++ 算子、恢复接口阶段或接口语义变更时，用于初始化工程、确认并冻结 operator contract，以及创建和校验 workflow.json。
---

# 01 接口确认

开始前根据已确认的数学语义选择 `algorithm_family`；普通 CATLASS 算子使用 `general`。新工程运行
`scripts/init_operator_project.sh <operator_name> [--algorithm-family <family>]`；继续工程先运行
`scripts/validate_workflow.py --workflow <operator>/docs/workflow.json`。

读取 `guides/interface_and_golden_contract.md`。已登记的注意力家族还要使用
`catlass-cpp-knowledge query --type operators --family <对应家族> --compact` 查询对应 Operator 卡；
普通算子按实际组件查询 `apis/`，必要时显式加入 draft 卡。知识是核对材料，不替代用户说明。

`sparse_flash_mla` 先完整读取 `operators/direct/transformer/attention/sparse_flash_mla_computation.md`、
`operators/direct/transformer/attention/sparse_flash_mla_interface.md`、
`operators/direct/transformer/attention/sparse_flash_mla_metadata.md`；普通 SWA 同时核对接口卡中的一般 SWA 合法域。
主算子与独立 metadata 的接口、输出、ValueDepend 和真实输入依赖分别冻结；用户已提供明确
golden/测试脚本时据此完成契约，无需重复征求同一授权。该家族目前只冻结
`target_architecture=atlas_a2_a3`。

确认目标设备代际，并把 `docs/workflow.json` 的 `target_architecture` 从 `pending` 设置为
`atlas_a2_a3`（A2/A3）或 `ascend950`（A5）。该字段是后续设计与实现的架构分支权威；
设备代际尚未确认时不得进入 reference；`block_sparse_attention` 的 Arch22 知识同样仅支持 `atlas_a2_a3`。

完成 `docs/api.md`，覆盖入口、输入输出顺序、shape、dtype、layout、state、属性、默认值、
mask、fixed/varlen、chunk、tail、padding、head ratio 和非法输入行为。分别记录
`operator_contract` 与 `golden_contract`；材料冲突按公开接口/用户明确说明、测试和示例、
参考实现、论文或相邻实现的优先级请求用户裁决。

只有用户确认接口后才把 `operator_contract` 设为 `frozen`，并原子更新状态为：
`stage=reference`、`golden_contract=provisional`、`issue_type=null`、
`resume_from=null`、`validation_scope=full`，并保留已确认的非 `pending`
`target_architecture`。更新后必须再次运行 validator。
