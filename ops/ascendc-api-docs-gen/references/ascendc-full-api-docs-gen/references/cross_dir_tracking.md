# 跨目录引用追踪规则（强制）

本文件定义跨目录引用追踪的完整执行流程。阶段 1/2 必须在代码分析前执行此流程。

## 为什么需要追踪

AscendC 头文件代码中，一个 API 的头文件可能通过 `#include` 引用外部共享头文件，这些外部头文件中可能包含关键的：
- 数据类型约束（`__NPU_ARCH__` 条件编译中的类型特化逻辑）
- 结构体定义（tiling 参数结构、工具结构体）
- 计算公式（常量定义、内联函数）
- 芯片差异分支（`#if defined(__NPU_ARCH__)` 条件编译）
- 校验逻辑（kernel_check 中的参数约束）

头文件 `#include` 引用的外部文件可能还有对应的 `.cpp` 实现文件，实现文件中包含：
- Tiling 参数校验逻辑（参数取值范围、合法组合）
- 具体的条件分支实现（不同芯片的计算路径差异）
- 宏展开后的实际代码逻辑

若不追踪这些外部引用，会导致遗漏关键约束、芯片差异分支和计算逻辑。

## 代码库目录结构（参考）

```
├── cmake                               # Ascend C 构建源代码
├── docs                                # 项目文档介绍
├── examples                            # Ascend C API样例工程
├── impl                                # Ascend C API接口实现源代码
│   ├── adv_api                         # Ascend C 高阶API实现源代码
│   ├── aicpu_api                       # Ascend C AI CPU API实现源代码
│   ├── basic_api                       # Ascend C 基础API实现源代码
│   ├── c_api                           # Ascend C 语言扩展层C API实现源代码
│   ├── experimental                   # Ascend C TENSOR API实现源代码
│   ├── simt_api                        # Ascend C SIMT API实现源代码
│   └── utils                           # Ascend C 工具类实现源代码
├── include                             # Ascend C API接口声明源代码
│   ├── adv_api                         # Ascend C 高阶API声明源代码
│   ├── aicpu_api                       # Ascend C AI CPU API声明源代码
│   ├── basic_api                       # Ascend C 基础API声明源代码
│   ├── c_api                           # Ascend C 语言扩展层C API声明源代码
│   ├── experimental                   # Ascend C TENSOR API声明源代码
│   ├── simt_api                        # Ascend C SIMT API声明源代码
│   └── utils                           # Ascend C 工具类声明源代码
```

**要点**：
- `include/` 与 `impl/` 的子目录结构一一对应，头文件路径中的 `include/` 替换为 `impl/` 通常能定位到实现文件
- 不同 API 类型（`adv_api`、`basic_api`、`c_api`、`simt_api` 等）之间可能存在跨目录引用

## 执行流程（4 步，必须按序执行）

### 步骤 1：扫描 #include 指令

对用户提供的 API 目录下（包括 `include/`、`impl/`）**每一个** `.h`/`.hpp`/`.cpp` 文件，提取所有 `#include "..."` 指令（忽略 `#include <...>` 系统头文件）。

**扫描命令**（使用 Grep 工具）：
```
pattern: #include\s+"([^"]+)"
path: <API目录>
glob: *.{h,hpp,cpp}
```

### 步骤 2：定位外部引用的实际路径

对每条 `#include "path"` 指令，按以下规则解析实际文件路径：

1. **相对路径**：以包含该 `#include` 的文件所在目录为基准解析
   - 例：`include/{api_type}/{module}/{api}.h` 中的 `#include "{api}_utils.h"` → 解析为同目录文件（内部引用）
   - 例：`include/{api_type}/{module}/{api}.h` 中的 `#include "../common/types.h"` → 解析为 `include/{api_type}/common/types.h`（同模块内引用）
   - 例：`impl/{api_type}/detail/{api}.cpp` 中的 `#include "../../other_module/bar.h"` → 解析为外部引用
2. **同目录引用**：`#include "xxx.h"` → 同目录下的 `xxx.h`（属于内部引用，不需额外追踪但需正常读取）
3. **子目录引用**：`#include "sub/impl.h"` → 当前目录下的子目录（通常属于内部引用）

**判断是否为外部引用**：
- 解析后的实际路径**超出**当前 API 所属的功能目录 → **外部引用，必须追踪**
- 解析后的路径仍在当前 API 目录内 → 内部引用，正常读取即可

**"当前 API 所属的功能目录"的确定方法**：
- 找到用户提供的 API 主头文件所在目录（如 `include/{api_type}/{module}/`）
- 该目录及其上级模块目录（如 `include/{api_type}/`）视为内部
- 超出此范围的为外部引用

**常见外部引用模式**（AscendC 代码库高频出现）：

| 引用模式 | 典型路径 | 包含的关键内容 |
|---------|---------|--------------|
| 共享数据结构 | `../common/`、`../../common/` | 公共结构体定义、枚举、常量 |
| 公共工具头文件 | `../../utils/`、`../shared/` | 通用工具函数、类型特征（type traits） |
| 跨 API 类型引用 | 从一类 API 目录引用另一类 API 目录的头文件 | 被调用的基础/高阶 API 的头文件和 check 代码 |
| 跨模块引用 | 从一个模块引用另一个模块的定义 | 跨模块的共享定义 |
| 平台/架构定义 | `arch/`、`platform/` | 芯片架构相关宏定义、条件编译辅助 |

### 步骤 3：读取外部文件、定位对应 .cpp 实现文件并递归追踪

对每个外部引用的头文件：

**3a. 读取外部头文件**：
1. 根据步骤 2 解析的路径，使用 Glob 或 Read 工具定位并读取文件
2. 判断是否包含关键实现（结构体定义、宏定义、模板实现、`__NPU_ARCH__` 条件编译、`ASCEND_IS_AIC` 检查、数据类型约束等）
3. 若被引用头文件自身也有 `#include "..."` 外部引用，继续追踪（最大深度：3 层）

**3b. 定位并读取对应的 .cpp 实现文件**：

对外部头文件，**必须同时定位并读取其对应的 `.cpp` 实现文件**（如果存在）。实现文件中通常包含：
- Tiling 参数校验逻辑
- 条件分支的具体实现
- 函数模板的实例化代码
- 芯片差异的具体计算路径

**定位方法**（按优先级依次尝试）：

1. **同名路径替换**：将头文件路径中的 `include/` 替换为 `impl/`，后缀 `.h`/`.hpp` 替换为 `.cpp`
   - 例：`include/{api_type}/{module}/{name}_tiling.h` → 尝试 `impl/{api_type}/{module}/{name}_tiling.cpp`
2. **同目录查找**：在与头文件相同的目录下查找同名 `.cpp`
   - 例：`common/{name}_types.h` → 尝试 `common/{name}_types.cpp`
3. **impl 目录下查找**：在 `impl/` 对应路径下查找
   - 例：`include/{api_type}/detail/{name}_utils.h` → 尝试 `impl/{api_type}/detail/{name}_utils.cpp`
4. **api_check 目录查找**：对 API 头文件，尝试在 `impl/{api_type}/detail/api_check/` 或 `impl/{api_type}/detail/api_check/kernel_check/` 下查找对应的 check 文件

**若找到对应的 .cpp 文件**：
- 将该 .cpp 文件纳入分析范围
- 对该 .cpp 文件同样执行 `#include "..."` 扫描（最大深度：2 层）

**【常见错误】**：仅读取外部 `.h` 头文件而跳过对应的 `.cpp` 实现文件，会导致 tiling 参数校验、芯片差异分支的实现逻辑全部丢失。

**递归深度限制**：头文件 `#include` 最多追踪 3 层；`.cpp` 实现文件内的 `#include` 最多追踪 2 层。

### 步骤 4：将外部文件纳入分析范围

追踪到的所有外部文件（头文件 + 对应 .cpp 实现文件）**必须与用户提供的 API 目录内文件同等对待**——在后续的芯片差异分析、参数提取中，外部文件中的代码逻辑必须被纳入分析。

## 代码分析清单格式要求

每个阶段报告末尾的 `## 代码分析清单` 必须包含追踪到的外部文件，并在"分析原因"列中注明引用链：

| 文件路径 | 分析原因 |
|---------|---------|
| /abs/path/include/{api_type}/{module}/{api}.h | API 主头文件 |
| /abs/path/include/{api_type}/{module}/{api}_utils.h | 工具头文件（API 同目录） |
| /abs/path/include/{api_type}/{module}/{api}_tiling.h | Tiling 参数定义头文件（API 同目录） |
| /abs/path/impl/{api_type}/{module}/{api}_tiling.cpp | **由 {api}_tiling.h 定位对应的 .cpp 实现文件**，包含 tiling 参数校验逻辑 |
| /abs/path/include/{api_type}/common/common_types.h | **由 {api}.h #include "../common/common_types.h" 引入**，包含公共结构体定义 |
| /abs/path/impl/{api_type}/detail/api_check/kernel_check/{module}_check.h | **由 {api}.h 定位对应的 api_check 文件**，包含参数硬约束 |
| /abs/path/impl/{api_type}/detail/api_check/kernel_check/{module}_check.cpp | **由 {module}_check.h 定位对应的 .cpp 实现文件**，包含约束校验实现 |
| /abs/path/include/{other_api_type}/{module}/utils.h | **由 common_types.h #include "..." 引入**（第2层追踪），包含基础工具定义 |
| /abs/path/xxx.h | **由 xxx.h #include 引入，文件不存在**，跳过 |

**强制要求**：
- `#include` 引入的外部头文件，"分析原因"以加粗的"**由 xxx #include 引入**"开头，注明引入链路和层数
- 对应的 `.cpp` 实现文件，"分析原因"以加粗的"**由 xxx.h 定位对应的 .cpp 实现文件**"开头，注明定位方式
- 对应的 api_check 文件，"分析原因"以加粗的"**由 xxx.h 定位对应的 api_check 文件**"开头
- 找不到对应 `.cpp` 文件的头文件，不强制要求记录缺失项
- `#include` 引入但文件不存在的，记录并标注"文件不存在"

## 验证清单（阶段输出前必须逐条检查）

在写入阶段输出文件前，必须对照以下清单验证：

- [ ] 已扫描 API 目录下所有 `.h`/`.hpp`/`.cpp` 文件的 `#include "..."` 指令
- [ ] 已识别出所有指向 API 目录外部的 #include 引用
- [ ] 已读取每个外部引用头文件的内容
- [ ] 已对外部头文件尝试定位对应的 `.cpp` 实现文件，找到的已纳入分析
- [ ] 已对外部头文件尝试定位对应的 `api_check` 文件（适用于 API 头文件），找到的已纳入分析
- [ ] 已对外部文件递归追踪 `#include`（头文件最多 3 层，.cpp 文件最多 2 层）
- [ ] 代码分析清单中包含所有追踪到的外部文件，并注明引用链
- [ ] 分析结论中已纳入外部文件和对应 .cpp 文件中的关键逻辑（校验、计算、分支等）
- [ ] **外层守卫传递性检查**：若 API 主头文件存在外层 `#if __NPU_ARCH__` 守卫，已确认被 `#include` 引入的 check/impl 文件内的 arch 守卫分支是否都在该外层守卫范围内；超出外层守卫范围的 arch 值（即使在内层 check 文件中出现）不作为支持产品的证据

若某条未通过，必须补充执行后再输出。
