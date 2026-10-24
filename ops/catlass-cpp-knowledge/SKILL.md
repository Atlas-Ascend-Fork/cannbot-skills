---
name: catlass-cpp-knowledge
description: 从已安装的 cannbot-knowledge 查询 CATLASS C++ 接口、五阶段指南和注意力算子知识。触发：开发 CATLASS C++ 算子时需要查阅接口、阶段规则，或在设计阶段按 family 查询并读取 Operator 卡，包括显式家族查询返回的 draft 卡。
---

# CATLASS C++ Knowledge

知识位于 `cannbot-knowledge/knowledge/ops/catlass/`。安装 `catlass-op-generator` 后，依赖仓库位于项目 `.cannbot/dependencies/catlass-op-generator/cannbot-knowledge/`；也可通过 `CANNBOT_KNOWLEDGE_ROOT` 或 `--knowledge-root` 指定知识仓 checkout。卡片分布在 `apis/`、`guides/`、`operators/direct/transformer/attention/`，Operator 文件以家族名前缀区分。

先查询，再按返回的相对路径读取正文。命令从本 Skill 目录执行：

```bash
python scripts/record_knowledge.py initialize --project-root <workspace>
python scripts/record_knowledge.py query --project-root <workspace> --type guides --text Stage --compact
python scripts/record_knowledge.py query --project-root <workspace> --type apis --text BlockMmad --include-draft --compact
python scripts/record_knowledge.py query --project-root <workspace> --type operators --family sparse_flash_mla --compact
python scripts/record_knowledge.py get --project-root <workspace> --path operators/direct/transformer/attention/sparse_flash_mla_computation.md
```

选择家族别名、架构过滤或排查知识路径问题时，读取 [查询与读取规则](references/knowledge_lookup.md)。设计阶段应按已确认的数学家族查询，选择当前阶段所需卡片并完整读取；draft 内容需结合冻结的 contract、固定源码和实际证据判断。

通用设计必读 `guides/solution_design_reference.md`、`guides/stage_design_rules.md` 与 `guides/kernel_stage_sync_patterns.md`；具体家族从查询结果选择 Operator 卡。`validate` 只检查已安装知识路径和卡片存在。知识新增和治理在 cannbot-knowledge 仓库完成，本 Skill 不执行写入或重建索引。
