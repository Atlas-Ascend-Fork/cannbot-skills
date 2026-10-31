# 芯片→产品型号映射表

> **使用场景**：将代码中的内部芯片标识转换为最终文档使用的固定产品项。产品支持结论仍需结合目标 API 的有效代码链路判断，命中映射不等于支持。

## `__NPU_ARCH__` 值→产品项

| `__NPU_ARCH__` 值 | 映射到的固定产品项 |
|---|---|
| 3510、3101 | Ascend 950PR&950DT系列产品 |
| 2201 | Atlas A3系列产品 |
| 2201 | Atlas A2系列产品 |
| 3002 | Atlas 200I/500 A2推理产品 |
| 2002 | Atlas推理系列产品AI Core |
| 2002 | Atlas推理系列产品Vector Core |
| 1001 | Atlas训练系列产品 |

**判定规则：**
- `__NPU_ARCH__ == 2201` 的有效支持证据同时覆盖 A3 和 A2 两个固定产品项，两项支持状态必须一致。
- `__NPU_ARCH__ == 2002` 表示 310P 架构范围。若目标 API 的有效代码链路明确限制在 AIC/Cube 路径，只判定“Atlas推理系列产品AI Core”；明确限制在 AIV/Vector 路径，只判定“Atlas推理系列产品Vector Core”。
- 2002 的有效支持路径没有 AIC/AIV 执行单元限制时，AI Core 和 Vector Core 两项都判定为支持。
- 候选路径进入不支持报错、恒 false 断言、空实现或不可达分支时，不得判定为支持。
- 部分头文件可能使用旧数值或旧产品称谓，最终输出以本文件固定产品项为准。

## 芯片标识→产品项

| 代码中的芯片标识 | 映射到的固定产品项 | 匹配方式 |
|---|---|---|
| Ascend 910D、Ascend 910_95、Ascend 950 | Ascend 950PR&950DT系列产品 | 精确匹配 |
| Ascend 910C、Ascend 910_93 | Atlas A3系列产品 | 精确匹配 |
| Ascend 910B* | Atlas A2系列产品 | 前缀匹配 |
| Ascend 310B* | Atlas 200I/500 A2推理产品 | 前缀匹配 |
| Ascend 310P AIC | Atlas推理系列产品AI Core | 精确匹配 |
| Ascend 310P AIV | Atlas推理系列产品Vector Core | 精确匹配 |
| Ascend 310P* | 根据有效代码链路的 AIC/AIV 限制映射；无限制时同时映射两项 | 前缀匹配 |
| Ascend 910 | Atlas训练系列产品 | 精确匹配 |

精确匹配优先于前缀匹配，匹配时忽略大小写、空格和下划线差异。

## 固定产品项及顺序

产品支持情况必须按以下 7 项顺序输出，不得增删、合并或改写：

1. Ascend 950PR&950DT系列产品
2. Atlas A3系列产品
3. Atlas A2系列产品
4. Atlas 200I/500 A2推理产品
5. Atlas推理系列产品AI Core
6. Atlas推理系列产品Vector Core
7. Atlas训练系列产品

产品支持状态只能写“支持”或“不支持”。产品支持章节的固定注释、产品文本和顺序以 `tasks/generate.md` 模板为准。

## 禁止作为最终产品项的内容

| 禁止内容 | 处理方式 |
|---|---|
| `__NPU_ARCH__` 数值、`__CCE_AICORE__`、`__DAV_*` | 转换为对应固定产品项，不在最终正文泄露内部标识 |
| ascend910b、ASCEND910_95、ASCEND910_93、SOC_xxx、DAV_xxx | 按映射转换 |
