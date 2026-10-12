# 错误码来源与维护规则

## 1. 基准范围

本目录收录CANN算子开发和错误路径治理所需的公开错误码资料：

- `documented-error-code-catalog.md`：供人工检索的精简清单。
- `documented-error-codes.json`：供问题定位、静态检查和审计脚本消费的完整机器可读清单。
- `development-standard.md`：跨Skill共享的开发使用规范。
- `component-error-code-source-index.md`：文档清单无匹配项时，按组件继续核对源码注册文件的索引。

当前快照固定为华为昇腾CANN `920beta2`官方公开错误码资料，共334条。每条记录的`url`是该条目的来源证据。

## 2. 允许的来源

按以下优先级采集和核验：

1. 华为昇腾CANN官方公开错误码资料页面。
2. `gitcode.com/cann`公开仓库中与目标版本对应的错误码注册表、公开头文件和实现。
3. 用户明确指定的CANN安装包中注册表与公开头文件；这类证据用于验证目标环境，不直接写入通用基准。

分析产物可以引用本基准，但不能反向覆盖真源。

## 3. 不同来源回答的问题

| 来源 | 可以证明 | 不能单独证明 |
|---|---|---|
| 官方资料页 | 编号已公开、面向用户的语义、原因与解决方向 | 目标版本已注册、调用宏存在 |
| 开源注册表/实现 | 目标源码分支中的模板、Arglist、映射与入口 | 用户安装包包含同一实现 |
| 目标安装包 | 交付版本的注册项和头文件 | 调用点实际include到了该头 |
| CMake/编译命令 | 调用点实际头文件、宏和条件编译分支 | 运行时一定产生预期首错 |
| 负向运行日志 | 实际错误码、字段和首错唯一性 | 其他构建分支行为相同 |

## 4. JSON数据契约

根对象字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | 机器可读契约版本 |
| `source` | 来源类型，必须指向CANN官方公开资料 |
| `documentation_version` | 文档快照版本 |
| `scope` | 码表适用边界；明确资料收录不等于目标版本注册 |
| `provenance` | 条目来源声明 |
| `count` | `codes`数组长度 |
| `governance_deprecated` | 当前治理禁止新增并要求替换的编号列表 |
| `codes` | 错误码记录数组 |

每条`codes[]`至少包含`code/prefix/module/title/error_message/causes/solutions/deprecated_in_document/version/url`。下游工具应按字段名读取，不依赖JSON对象字段顺序。

`codes[].code`命中只代表资料收录。静态检查应另设“目标版本注册已验证”和“Arglist已验证”状态，不能合并成单一布尔值。

## 5. 更新流程

1. 锁定一个明确的CANN官方资料版本，不把多个版本无标记合并。
2. 从官方页面取得编号、模块、标题、错误信息、原因、解决方案和URL。
3. 更新JSON后，以同一数据生成或同步Markdown清单；不得只改其中一份。
4. 检查编号唯一、URL非空、`count`等于数组长度、每条`version`与快照一致。
5. 对治理状态与官方资料“已废弃”字段的差异单独记录，不篡改官方字段。
6. 用目标版本公开注册表抽样核对模板和Arglist，并在变更说明中记录无法核对的条目。

建议的一致性检查：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("ops/cann-error-code-standard/references/documented-error-codes.json")
data = json.loads(path.read_text(encoding="utf-8"))
codes = data["codes"]
names = [item["code"] for item in codes]
assert data["count"] == len(codes)
assert len(names) == len(set(names))
assert all(item.get("url", "").startswith("https://www.hiascend.com/") for item in codes)
assert all(item.get("version") == data["documentation_version"] for item in codes)
print(f"validated {len(codes)} documented error codes")
PY
```
