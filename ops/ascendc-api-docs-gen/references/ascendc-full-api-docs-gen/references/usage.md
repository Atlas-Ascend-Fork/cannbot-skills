> 从父 Skill `ascendc-api-docs-gen` 进入；本页命令先以 `references/ascendc-full-api-docs-gen/` 为工作目录。

# AscendC 接口文档生成

这个 skill 根据指定的 AscendC API、代码仓和参考资料生成接口文档。它会先收集资料，再完成代码取证、资料对齐、用户确认和最终校验。

两种用法最终都会在 API 对应的输出目录中使用同一种 `manifest.json`，后续流程相同。

符号运算符 API 的目录和最终文档文件名前缀均使用可读名称，例如 `operator+` 使用 `outputs/operator_add/operator_add_YYYYMMDDHHmm.md`；清单中仍保存原始接口名 `operator+`。

## 1. 直接在对话中使用

不需要预先创建 `manifest.json`。例如，直接向 skill 发送：

```text
请为 asc_copy_gm2l1_align 生成 AscendC 接口文档。
代码仓根目录：<repo-root>
接口文件：<repo-root>/include/c_api/cube_datamove/cube_datamove.h
```

skill 会只解析指定的接口文件，自动确定同一篇文档需要覆盖的成员，并创建：

```text
<repo-root>/outputs/asc_copy_gm2l1_align/manifest.json
```

创建或读取清单后，skill 会先展示本次最终要生成文档的接口名称、类别、调用方式、成员和接口文件，请用户确认；确认后才会询问是否有“内部脱敏资料”“仓内已有人工文档”，以及是否还有其他资料。内部脱敏资料只能人工提供路径。尚未给出仓内已有人工文档时，skill 会在代码仓的 `docs` 目录中寻找当前 API 的精确候选，并给出最多三份同类 API 文档作为示例工程结构参考。直接下层 API 文档逐份由用户确认；宏表映射先按同一实现分组，由 skill 结合当前代码选择至多三份可能相关的文档，再逐份让用户确认。宏表中的候选映射不会自动成为当前 API 的行为或资料事实。候选不正确时再输入路径。没有某份资料时可以明确回答不存在；也可以补充任意名称的其他资料和路径。

调用示例先从代码仓 `examples/` 中寻找当前 API 的直接调用，先取 `Compute()`，没有可用结果再取 `Process()`。若脚本只定位到 Helper 中的直接调用，skill 会仅在该候选 `.asc` 文件内核验其到入口的调用链；仓内仍没有可用示例时，才检查用户提供的同 API 资料；仍没有则依据当前 API 实现代码构造，并只借用同类 API 文档的工程、Host、Kernel 或 CMake 结构。同类 API 文档不能迁移其 API 调用、参数、输入、结果或技术结论。下层 API 文档中的每一条约束或行为都要结合当前 API 的成员、条件和调用位置复核：只有代码证明当前 API 对用户实际使用该行为，或用户明确确认时，才会写入当前 API 文档。

## 2. 先生成 manifest.json，再运行 skill

需要先生成清单时，可运行：

```text
node tools/ascendc-create-manifest.mjs --repo "<repo-root>" --interface-file "<repo-root>/include/c_api/cube_datamove/cube_datamove.h" --api-name "asc_copy_gm2l1_align"
```

然后把下面的提示词发送给 skill：

```text
manifest.json 文件路径为：
<repo-root>/outputs/asc_copy_gm2l1_align/manifest.json

请完整读取该文件，然后开始生成 AscendC 接口文档。
```

如果插件已经完成接口检测和用户选择，也可把选择结果传给同一工具创建清单：

```text
node tools/ascendc-create-manifest.mjs --api "<接口检测和用户选择结果的 JSON 文件路径>" --selected-interface "<所选接口 ID>" --repo "<repo-root>" --output-dir "<repo-root>/outputs/asc_copy_gm2l1_align"
```

将生成的 `manifest.json` 路径按上面的提示词交给 skill 即可。若该清单已经登记了资料，skill 会直接沿用；尚未登记的资料会在代码取证前询问。



## 3. 插件调用示例

以用户在插件中选择 <code>asc_copy_gm2l1_align</code> 为例：用户先选择接口文件，插件运行接口检测脚本并展示候选；用户选定接口后，插件可选择自己创建 <code>manifest.json</code>，或直接启动 skill。

```text
node scripts/detect-interfaces.js "<repo-root>/include/c_api/cube_datamove/cube_datamove.h" "<repo-root>" --output "<repo-root>/outputs/interfaces.json"
```

### 插件创建 manifest.json

插件把检测结果文件和用户选中的接口 ID 传给创建工具：

```text
node tools/ascendc-create-manifest.mjs --api "<repo-root>/outputs/interfaces.json" --selected-interface "c_api|__aicore__|asc_copy_gm2l1_align（用户选择的接口id）" --repo "<repo-root>" --output-dir "<repo-root>/outputs/asc_copy_gm2l1_align"
```

随后启动 skill，并将以下内容作为本次 skill 的输入：

```text
manifest.json 文件路径为：
<repo-root>/outputs/asc_copy_gm2l1_align/manifest.json

请完整读取该文件，然后开始生成 AscendC 接口文档。
```

### 插件直接启动 skill

插件不必先运行创建工具；它从用户选中的候选中取出 API 名称和接口文件路径，并将下面的内容作为本次 skill 的输入：

```text
请为 asc_copy_gm2l1_align 生成 AscendC 接口文档。
代码仓根目录：<repo-root>
接口文件：<repo-root>/include/c_api/cube_datamove/cube_datamove.h
```

skill 会自行创建 <code><repo-root>/outputs/asc_copy_gm2l1_align/manifest.json</code>，再继续后续流程。

### skill 内部的用户交互

下表仅列 skill 在执行中发起的用户交互。插件应将 skill 发出的 <code>question</code> 原样展示给用户，并将所选项或用户自行输入的内容回传。

| 时机 | 是否发生 | 用户需要提供或确认的内容 |
| --- | --- | --- |
| 确认最终文档接口 | 清单尚未确认时 | 确认本次最终生成文档的接口名称、类别、调用方式、成员和接口文件；下层 API 与同类 API 文档不会替代该接口。 |
| 收集默认资料 | 按资料状态 | 为“内部脱敏资料”提供路径或确认不存在；未提供“仓内已有人工文档”时，确认当前 API、同类 API、直接下层 API 或经代码审阅后推荐的宏映射组文档。候选不正确再输入路径或确认不存在。路径不可读时，提供替代路径或确认不存在。 |
| 收集其他资料 | 每次任务一次 | 补充任意数量的“资料名称：文件路径”，或确认没有其他资料。 |
| 资料与当前 API 的关系不清 | 按需 | 选择它是当前 API、同类 API、调用链可达的下层 API、仅作一般参考，或明确指出其中可用于当前 API 的章节或事实。 |
| 下层 API 资料的行为是否适用于当前 API | 按需 | 当代码无法确认当前 API 是否对用户使用某条下层约束或行为时，选择写入、仅作实现说明或暂缓。 |
| 代码与资料存在冲突 | 按需 | 在代码结论、资料结论或用户自定义结论之间确认最终采用的具体事实。 |
| 多份资料一致但代码未证实 | 按需 | 选择保留该共同事实，或暂缓为 <code>【待开发补齐】</code>。 |
| 用户自行输入事实 | 按需 | 确认 skill 整理后的单条事实表述，避免插件或模型误解用户原意。 |
| 新代码证据改变已确认事实 | 按需 | 对受影响的已确认事实再次确认。 |

资料收集完成、所有事实已确认或暂缓后，生成、审阅和最终校验不再需要用户交互；skill 会直接输出最终文档路径。
会话创建与资料登记入口以 [WORKFLOW.md](../WORKFLOW.md) 为准。
