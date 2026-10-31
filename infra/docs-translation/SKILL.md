---
name: docs-translation
description: 将本地 CANN 仓库中的中文 Markdown 文档（含 API 文档）翻译为英文，按路径类型生成 _en.md 或 en/ 目录下的同名文件，支持 repoName 排除配置、增量翻译与提交到远程仓库。当用户需要翻译文档、生成或更新英文文档、准备英文版 PR，或提及 docs-translation 时使用。
license: CANN-2.0
---

# PR文档翻译Skill

## 功能描述

检测用户本地仓库的md文档，自动翻译成英文文档。支持：

- 非docs目录下的所有md文档
- docs目录下，排除列表外的md文档

使用方法和排除配置说明见 [references/usage.md](references/usage.md)。

## 触发条件

满足任一条件即触发本skill：

- 用户要求翻译文档、生成或更新英文文档、准备英文版PR
- 用户提及 docs-translation
- 用户指定的md文档不存在对应英文翻译（`_en.md`或en目录同名文件），或原文档相比已有翻译发生变更

## 执行流程

### 1. 交互式获取本地仓库路径（必须首先执行）

**不通过远程地址推断仓库名，不自行克隆仓库**，本地仓库路径一律由用户提供。

**场景A：用户仅提供远程地址或仓库名（未提供本地路径）**

> **🚩 红线（不可妥协）：必须实际发起一次交互式提问工具（question）调用向用户索取本地仓库路径。**
>
> - **禁止仅以普通文本追问路径**（例如"请告诉我你的仓库路径"这类纯文本回复）
> - **禁止从远程地址推断本地仓库路径**
> - **禁止自行克隆仓库**
> - 未发起 question 工具调用，即视为本步骤未完成，不得进入第2步

**必须执行的动作：**

1. 立即发起一次 question 工具调用，要求用户提供已 clone 到本地的仓库绝对路径。
2. 在发起 question 调用的同一条回复中，必须明确写出"将使用交互式提问工具（question）"（不能只写"请提供路径"而不提工具）。

**question 工具调用示例（必须照此形式实际发起，而不是在文本里复述这段 JSON）：**

```json
{
  "questions": [
    {
      "question": "请提供已 clone 到本地的仓库绝对路径（例如 D:\\WorkSpace\\metadef_trunk）。我不会通过远程地址推断路径，也不会自行克隆仓库。",
      "header": "本地仓库路径",
      "options": [
        {"label": "我直接输入路径", "description": "选择此项后输入本地仓库的绝对路径"}
      ]
    }
  ]
}
```

**答复必须包含以下四要素：**

1. 不会通过远程地址推断本地仓库路径，也不会自行克隆
2. 将使用交互式提问工具（question）向用户索取clone到本地的仓库路径
3. 拿到路径后会校验：路径存在且为git仓库（路径下存在`.git`目录），校验失败则重新提示输入
4. 后续扫描、翻译、提交等所有操作均以该路径为仓库根目录

**首次回复口径模板（可直接套用，四要素缺一不可）：**

```text
我不会通过远程地址推断本地仓库路径，也不会自行克隆仓库。
接下来我会调用交互式提问工具（question），请你提供已 clone 到本地的仓库路径。
拿到路径后我会校验：路径存在且为 git 仓库（存在 .git 目录），校验失败会重新提示输入。
后续扫描、翻译、提交等所有操作，均以该路径为仓库根目录。
```

**场景B：用户已提供本地路径（如`D:\WorkSpace\metadef_trunk`）**

直接校验路径存在且为git仓库（路径下存在`.git`目录）；校验失败则使用question工具重新提示输入；校验通过后，后续扫描、翻译、提交等所有操作均以该路径为仓库根目录（在该目录下执行扫描、git等命令）。

### 2. 扫描所有md文档并统计

扫描用户本地仓库的所有`.md`文档（排除.git目录），**必须先统计排除路径文件数量，再计算实际需翻译数量**（避免误导用户认为需要翻译所有文件）。

**扫描流程（以下命令需在bash中执行，工作目录为仓库根目录）：**

1. **统计总md文件数量**

   ```bash
   find . -name "*.md" -not -path "./.git/*" -type f | wc -l
   ```

2. **读取排除列表配置**（`<skill_dir>`为SKILL.md所在目录，与待翻译仓库位置无关；bash命令中写作`${skill_dir}`）

   ```bash
   cat "${skill_dir}/exclude_docs.json"
   ```

3. **统计排除路径文件数量**：根据excludeDocsPaths的**实际配置动态构造**命令（不得照抄示例中的路径），例如配置为`docs/dflow_api`时：

   ```bash
   find . -name "*.md" -not -path "./.git/*" -path "*/docs/dflow_api/*" | wc -l
   ```

4. **计算实际需要翻译的文件数量**

   ```text
   实际需翻译数量 = 总文件数 - 排除路径文件数
   ```

**统计报告格式（向用户输出）：**

```text
扫描结果：
- 总md文件：1495个
- 排除路径（跳过）：1345个
  - docs/dflow_api：401个
  - docs/graph_engine_api：843个
  - docs/llm_datadist_api：101个
- 实际需翻译：150个
```

### 3. 判断文档是否需要翻译

**步骤1：将本地仓库路径与repoName模糊匹配**

- 匹配规则：本地仓库路径包含repoName关键字即匹配成功（如`D:\WorkSpace\ge_test`包含`ge`匹配成功，`metadef_trunk`匹配`metadef`）
- 匹配到多个repoName时，取关键字最长的配置项
- 匹配成功 → 读取该配置项的excludeDocsPaths；未匹配 → 无文件排除，整个仓库进行翻译

**步骤2：逐个文档判断**

```text
1. 文档是否在docs目录下？
   → 否：需要翻译
   → 是：继续
2. 已匹配repoName时，文档路径是否在excludeDocsPaths中？
   → 是：跳过（该路径由专业翻译人员负责）
   → 否：需要翻译
3. 是否已存在对应翻译文件？
   → 否：需要完整翻译
   → 是：检查原文档是否有变更（见第5步）
      → 有变更：需要增量翻译
      → 无变更：跳过
```

**回答口径（凡涉及排除判断的答复，必须包含以下三点，缺一不可）：**

1. **匹配结果**：本地仓库路径与repoName做模糊匹配，路径中包含repoName关键字即匹配成功（给出本次具体匹配说明）
2. **命中跳过**：在excludeDocsPaths中的docs子路径跳过翻译（该路径由专业翻译人员负责）
3. **其余照常**：**不在排除列表中的其他docs路径仍照常翻译**

**示例：**

| 本地仓库路径 | repoName | docs路径 | excludeDocsPaths | 是否翻译 |
| ----------- | -------- | ----------------------------- | ----------------- | --------- |
| `D:\WorkSpace\ge_test` | ge | `docs/graph_engine_api/xx.md` | `["docs/graph_engine_api"]` | 跳过（在排除路径） |
| `D:\WorkSpace\ge_test` | ge | `docs/architecture/xx.md` | `["docs/graph_engine_api"]` | 翻译（不在排除路径） |
| `D:\WorkSpace\pypto` | pypto | `docs/api/xx.md` | `["docs"]` | 跳过（整个docs排除） |
| `D:\WorkSpace\custom_repo` | 未匹配 | `docs/any.md` | - | 翻译（无排除，整个仓库翻译） |

### 4. 确定翻译文件存放路径

**非docs目录文档：**

- 存放规则：与原文件同级目录
- 文件命名：`<原文件名>_en.md`
- 示例：`README.md` → `README_en.md`

**docs目录文档：**

**步骤1：检查文档是否位于docs内某个zh文件夹下**（zh可出现在docs内任意层级，如`docs/zh/`、`docs/design/modules/zh/`）

```bash
find docs -type d -name "zh"    # docs内是否存在zh文件夹（判断是否为情况B）
echo $filepath | grep "/zh/"    # 文档是否位于某个zh文件夹下
```

**步骤2：按情况确定存放规则**

| 情况 | 存放规则 | 文件命名 | 示例 |
| --- | --- | --- | --- |
| 文档位于某个zh文件夹下（非排除列表） | 在该zh同级目录新建en文件夹，en目录层级与zh完全一致 | `<原文件名>.md`（不添加`_en`后缀） | `docs/zh/api/quantize.md` → `docs/en/api/quantize.md`<br>`docs/design/zh/modules/feature.md` → `docs/design/en/modules/feature.md`<br>`docs/design/modules/zh/sub/overview.md` → `docs/design/modules/en/sub/overview.md` |
| docs下不在任何zh文件夹内的文档（非排除列表） | 与原文件同级目录 | `<原文件名>_en.md` | `docs/README.md` → `docs/README_en.md`<br>`docs/design/overview.md` → `docs/design/overview_en.md` |
| docs目录内不存在任何zh文件夹 | 与原文件同级目录 | `<原文件名>_en.md` | `docs/api/quantize.md` → `docs/api/quantize_en.md`<br>`docs/guide/install.md` → `docs/guide/install_en.md` |

**为什么zh下文档不加`_en`后缀：** zh和en目录已区分中英文，无需后缀；保持目录结构一致便于对照管理。不在zh内的docs文档使用`_en`后缀，因为同一目录下需要区分中英文。

**PR模板文件（特殊规则）：**

- 识别规则：文件名包含`PULL_REQUEST_TEMPLATE`且后缀为语言代码（如`.zh-CN.md`）
- 存放规则：与原文件同级目录（`.gitcode/`目录）
- 文件命名：将语言代码替换为对应英文代码，不使用`_en`后缀
- 示例：`.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` → `.gitcode/PULL_REQUEST_TEMPLATE.en-US.md`；其他平台模板同理（`ISSUE_TEMPLATE.zh-CN.md` → `ISSUE_TEMPLATE.en-US.md`）

### 5. 检查翻译文件是否存在并检测变更

**检查翻译文件是否存在**（按第4步的存放规则确定目标路径后检查）：

- 非docs、docs内不在zh下的文档：检查同级`<原文件名>_en.md`
- zh下的文档：检查`<zh父路径>/en/<zh后的路径>/<原文件名>`

```bash
ls README_en.md                  # 非docs文档
ls docs/en/api/quantize.md       # zh下文档
ls docs/design/overview_en.md    # docs内非zh文档
```

**检测原文档变更（翻译文件已存在时）：**

```bash
# 方式1：与上一次提交比较
git diff HEAD~1 -- <文件>

# 方式2：检查工作区状态（覆盖未提交改动）
git status <文件>

# 方式3：获取变更统计
git diff HEAD~1 --stat -- <文件>
```

> **注意**：`git diff HEAD~1`仅覆盖最近一次提交。若原文档在更早提交中已修改而翻译文件未同步，需结合修改时间判断：分别用`git log --format="%ci" -1 -- <原文件>`和`git log --format="%ci" -1 -- <翻译文件>`获取最后修改时间，原文件修改时间晚于翻译文件时按有变更处理；无法确定时按有变更处理（宁多勿漏）。

### 6. 读取翻译标准（必须执行）

> **⚠️ 重要警告：翻译前必须先读取翻译标准！遗漏此步骤会导致翻译不符合规范，需要返工修正！**

**必须在翻译任何内容之前**，阅读 [references/translation-style-guide.md](references/translation-style-guide.md) 全文并逐条遵守（基于英文风格指南 <https://developers.google.com/style> 整理，含详细规则、示例与翻译质量检查清单）。

**核心红线（详见参考文档）：**

- **语态时态**：主动语态优先，操作指引一律祈使句（"Select..."而非"You can select..."）；一般现在时为主，操作后的瞬时结果不用将来时
- **句子结构**：每个句子必须有明确主语和动词；反直译，译文须能脱离中文原文自然成立；每句不超过25个单词
- **词汇**：禁用 via、e.g.、i.e.、etc.、can't、it's、don't、thus、figure out 等；缩略词写全称（cannot、do not）
- **冠词**：正文句子中单数可数名词前必须有冠词；特指用the，泛指用a/an（表格单元格名词词组除外）
- **术语**：中文"接口"一律译为API，**译文绝对禁止出现interface**；术语表命中即用且全文一致（样例→sample、算子→operator、张量→tensor、开发者→you等）
- **固定句式**："是否XXX"→Whether XXX（禁止Indicates whether/Specifies whether）；"返回XXX"→Returns XXX.；"无"→None（不加句号）；列举清单最后一项前统一加and
- **标题**：API文档章节标题命中固定映射表即替换（功能说明→Description、参数说明→Parameters、示例→Examples等），复数形态不可更改，标题层级不变
- **格式镜像**：中文源文是什么格式，英文就是什么格式，逐处一致、一字不差；译文中每个`**`/`` ` ``/`\_`必须能在中文源文对应位置找到依据，找不到立即删除；禁止自行加粗/加code、禁止格式升级/降级/迁移
- **代码块**：整体保持原样，仅翻译其中的注释；代码样式与字符不可改动
- **文件末尾**：必须保留一行空行（以单个换行符结尾，MD047）；增量编辑后同样保证
- **非断行连字符**：U+2011统一替换为ASCII连字符`-`

### 7. 执行翻译

**不存在翻译文件（完整翻译）：**

1. 使用read工具读取中文文档完整内容
2. 按翻译标准进行完整翻译
3. 根据文档路径类型确定存放位置并创建翻译文件

**已存在翻译文件且有变更（增量翻译）：**

1. 使用bash工具获取中文文档的变更内容（diff）
2. 分析变更点
3. 将变更点翻译成英文
4. 使用edit工具更新对应的英文文档（更新位置由路径类型决定，见第4步），**仅更新变更部分，不重写整个文件**

**其他要求：**

- 中文链接：引用其他中文文档的链接保持原路径不变，待对应文档翻译后再更新
- 翻译完成后执行 [references/translation-style-guide.md](references/translation-style-guide.md) 中的"翻译质量检查清单"，重点包括：行数比对（英文行数少于中文超过30%视为漏翻译，需逐段对照补充；代码块、空白行差异可忽略）、格式镜像逐处对应、正文标题无残留中文字符

### 8. 使用todowrite跟踪进度

创建任务列表跟踪翻译进度，**必须包含统计步骤**（避免误导用户）：

```json
[
  {"content": "交互式获取用户输入的本地仓库路径", "status": "in_progress", "priority": "high"},
  {"content": "扫描所有md文档并统计总数", "status": "pending", "priority": "high"},
  {"content": "统计排除路径文件数量", "status": "pending", "priority": "high"},
  {"content": "计算实际需翻译数量（总数-排除数）", "status": "pending", "priority": "high"},
  {"content": "读取翻译标准（必须执行）", "status": "pending", "priority": "high"},
  {"content": "分批翻译各目录文档", "status": "pending", "priority": "high"},
  {"content": "执行翻译质量检查（行数比对、格式镜像等）", "status": "pending", "priority": "high"},
  {"content": "提交翻译变更到远程仓库", "status": "pending", "priority": "high"}
]
```

**并行提效：** 并行读取多个相关文件（中文原文、英文版本、风格指南）；并行执行多个bash命令。若待翻译文件超过50个，建议每30个分批处理；若单文件超过500行，建议每200行分批翻译。

### 9. 提交变更到远程仓库

翻译完成后，将所有变更的中英文md文档提交到远程仓库：

```bash
# 1. 查看变更的md文档（含中文原文和英文翻译）
git status -- "*.md"

# 2. 添加变更文件（中英文成对添加，路径按第4步存放规则）
git add README.md README_en.md                              # 非docs文档：译文加_en后缀
git add docs/zh/api/quantize.md docs/en/api/quantize.md     # zh下文档：中英文件名一致
git add docs/build.md docs/build_en.md                      # docs非zh文档：译文加_en后缀

# 3. 创建提交
git commit -m "docs: add/update English translation for markdown files"

# 4. 推送到远程仓库（新分支使用 git push -u origin <branch_name>）
git push origin <branch_name>
```

**注意事项：**

- 提交前确保所有翻译文件都已创建或更新
- 提交信息应清晰说明翻译的内容
- 中文原文和英文翻译一起提交，保持同步；多个翻译文件建议批量提交

## 配置文件说明

### 排除文档配置

- **路径**：`<skill_dir>/exclude_docs.json`（与SKILL.md同级存放；`<skill_dir>`指skill文件所在目录，与待翻译仓库位置无关；bash命令中写作`${skill_dir}`）
- **作用**：定义公开产品文档仓库及其需要排除翻译的 docs 子路径；配置文件不存在时视为无排除项，整个仓库进行翻译
- **格式**：JSON格式，包含 mapping 数组，每个元素包含：
  - `repoName`: 仓库名（用于与用户输入的本地仓库路径进行模糊匹配）
  - `excludeDocsPaths`: 需要跳过翻译的 docs 子路径列表

  ```json
  {
      "mapping": [
        {
             "repoName": "ge",
             "excludeDocsPaths": ["docs/dflow_api", "docs/graph_engine_api", "docs/llm_datadist_api"]
        },
        {
             "repoName": "pypto",
             "excludeDocsPaths": ["docs"]
        }
     ]
  }
  ```

- **匹配规则**：
  1. 用户交互式输入本地仓库路径（见执行流程第1步）
  2. 将本地仓库路径与 repoName 进行模糊匹配（路径中包含 repoName 关键字即匹配成功；匹配到多个时取关键字最长的配置项）
  3. 匹配成功：该仓库 docs 目录下 excludeDocsPaths 中列出的子路径跳过翻译，其他docs路径照常翻译
  4. 未匹配：无文件排除，整个仓库进行翻译

### 翻译标准文档

- **参考文档**：[references/translation-style-guide.md](references/translation-style-guide.md)（基于 <https://developers.google.com/style> 整理）

## 目录结构示例

```tree
仓库根目录/
├── README.md                  # 原始中文文档（非docs）
├── README_en.md               # 翻译后英文文档（同级目录）
├── CONTRIBUTING.md            # 原始中文文档（非docs）
├── CONTRIBUTING_en.md         # 翻译后英文文档（同级目录）
├── docs/                      # docs目录
│   ├── README.md              # 原始文档（docs根目录，不在任何zh文件夹内）
│   ├── README_en.md           # 翻译后英文文档（同级目录，命名<原文件名>_en.md）
│   ├── build.md               # 原始文档（docs根目录，不在任何zh文件夹内）
│   ├── build_en.md            # 翻译后英文文档（同级目录，命名<原文件名>_en.md）
│   ├── zh/                    # 中文文档目录（docs根层级）
│   │   ├── api/
│   │   │   ├── README.md      # 原始文档（zh下，非排除列表）
│   │   │   └── quantize.md    # 原始文档（zh下，非排除列表）
│   │   └── guide/
│   │       └── install.md     # 原始文档（zh下，非排除列表）
│   │       └── excluded.md    # 原始文档（zh下，在排除列表中，不翻译）
│   ├── en/                    # 英文文档目录（与zh同级，目录结构一致）
│   │   ├── api/
│   │   │   ├── README.md      # 翻译后英文文档（文件名与原文一致）
│   │   │   └── quantize.md    # 翻译后英文文档（文件名与原文一致）
│   │   └── guide/
│   │       └── install.md     # 翻译后英文文档（文件名与原文一致）
│   └── design/                # docs子目录
│       ├── overview.md        # 原始文档（不在任何zh文件夹内）
│       ├── overview_en.md     # 翻译后英文文档（同级目录，命名<原文件名>_en.md）
│       └── modules/           # 更深层级子目录
│           ├── overview.md    # 原始文档（不在任何zh文件夹内）
│           ├── overview_en.md # 翻译后英文文档（同级目录）
│           ├── zh/            # 中文文档目录（modules层级）
│           │   └── feature.md # 原始文档（zh下，非排除列表）
│           └── en/            # 英文文档目录（与zh同级）
│               └── feature.md # 翻译后英文文档（文件名与原文一致）
├── examples/
│   └── README.md              # 原始中文文档（非docs）
│   └── README_en.md           # 翻译后英文文档（同级目录）
└── npu_ops/
    └── README.md              # 原始中文文档（非docs）
    └── README_en.md           # 翻译后英文文档（同级目录）
```

**说明：**

- zh下的文件如果在排除列表中，则跳过不翻译（如`zh/guide/excluded.md`）
- zh下的文件如果不在排除列表中，则翻译到该zh同级的en目录对应位置，文件名保持一致
- zh可以出现在docs目录内的任意层级（如`docs/zh/`、`docs/design/modules/zh/`等），en目录始终创建在对应zh的同级
- 不在任何zh文件夹内的docs文档（如`docs/README.md`、`docs/design/overview.md`）翻译到同级目录，命名为`<原文件名>_en.md`

## 注意事项

 1. **路径获取（重要）**：本地仓库路径一律交互式获取。用户只给远程地址或仓库名时，**必须实际发起 question 工具调用**索取路径，禁止仅以普通文本追问、禁止从远程地址推断、禁止自行克隆；校验路径存在且含`.git`目录，失败重新提示（见执行流程第1步）
 2. **扫描范围与统计（重要）**：扫描所有md文档（docs与非docs）；必须先统计排除路径文件数量，再计算实际需翻译数量，并向用户输出统计信息（见执行流程第2步）
 3. **排除判断**：docs目录下的文档需检查是否在排除列表中；凡涉及排除判断的答复，按第3步"回答口径"完整输出三点（匹配结果、命中跳过、其余照常）
 4. **必须先读取翻译标准**：翻译任何内容前，必须先阅读[references/translation-style-guide.md](references/translation-style-guide.md)全文，翻译完成后执行其中的翻译质量检查清单（重点：行数比对±30%、格式镜像）
 5. **文件存放路径与命名**：严格按执行流程第4步的规则（zh下文档同名进en目录，其余同级加`_en`后缀；PR模板替换语言代码）
 6. **增量更新**：已存在翻译文件时仅翻译变更部分，避免重复翻译（见执行流程第5、7步）
 7. **格式一致性**：保持Markdown格式、表格结构、链接引用一致；格式镜像逐处对应源文
 8. **提交变更**：翻译完成后将所有变更的中英文md文档提交到远程仓库（见执行流程第9步）
