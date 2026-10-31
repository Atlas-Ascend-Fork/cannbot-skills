# 源码信息提取指南

本文档详细说明如何从Ascend CANN算子源码中提取文档所需信息，以 [`example-aclnn-md.md`](example-aclnn-md.md) 和 [`example-readme.md`](example-readme.md) 为参考范例。

**MANDATORY - READ ENTIRE FILE**：阶段1源码分析时，必须完整阅读本指南。

---

## 使用边界

输入状态、资料读取顺序和冲突采用策略统一见[父入口的资料来源与冲突处理说明](../SKILL.md#资料来源与冲突处理说明)。解释冲突处理方法时，先说明“未提供规格、需求或设计文档时直接依据源码生成”和“有资料时先完整读取并提取设计意图，再完整扫描源码核验”两条路径，再解释当前冲突；已声明提供但不可读的资料不能当作未提供。本指南负责“源码线索如何核验为文档事实”，不另设咨询或交付流程。

全文代码、参数值、shape 和产品结论都是提取方法的示例。生成或解释规则时，区分格式与事实：模板决定怎样写，目标接口的证据决定能写什么；示例本身不证明其他算子的支持度、默认值或约束。

## 证据文件覆盖

按下表核对目标接口的证据类别。文件路径是常见布局下的定位线索，不是排除其他目录、共享文件或依赖的过滤条件；实际范围由接口声明和可达调用链确定。

| 证据类别 | 常见文件或目录 | 核验用途 |
|----------|----------------|----------|
| 接口声明 | `op_host/op_api/aclnn_*.h` | 递归扫描全部 `ACLNN_API` 声明，确认接口数量、实际函数签名和参数顺序 |
| 接口实现 | `op_host/op_api/aclnn_*.cpp` | 参数校验、空指针语义、调用链及实际错误码分支 |
| 算子注册 | `op_host/*_def.cpp` | Input/Output/Attr、ParamType、dtype、format 和 AddConfig；注册不单独证明运行时限制 |
| 公共定义 | `op_host/*_base.h` | 枚举、维度常量、参数索引及其实际使用含义 |
| Shape 推导 | `op_host/*_infershape.cpp` | 输出 shape、条件输出和推导条件 |
| Tiling 头文件 | `op_host/*_tiling.h` | 数据结构、字段类型、常量以及 Host/Kernel 间参数传递；不能用读取 cpp 代替头文件核验 |
| Tiling 实现 | `op_host/*_tiling.cpp`、可用的 `op_host/*_tiling_arch35.cpp` | shape/attr 校验、平台分支、workspace 和确定性调度路径 |
| Kernel 实现与分派 | `op_kernel/*.cpp`、`op_kernel/*_tiling_key.h` | 实际执行分支、类型转换、访存及数值约束；核对 tiling key 与实现的对应关系，不能由 Host 注册代替 |
| 平台配置 | `op_host/config/*/` | 与 AddConfig、接口及实现交叉核验逐平台支持情况，目录存在或缺失不独立证明支持度 |
| 调用与测试 | `examples/test_aclnn_*.cpp`、`tests/st/` | 目标接口的实际调用、参数构造和已有测试范围；不将示例取值泛化为约束 |

在可用范围内完整读取目标证据文件，并沿调用链读取必要定义；缺失文件或未解析依赖记录其影响和替代证据，不能默认为对应约束不存在。Tiling 头文件、Tiling 实现和 Kernel 是不同证据类别，合并说明时仍须保留各自的文件范围与核验用途。

---

## 1. 从 `*_def.cpp` 提取

### 1.1 输入参数信息

```cpp
this->Input("grad_output")
    .ParamType(REQUIRED)                              // → 输入/输出列取值
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})  // → 数据类型
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})  // → 数据格式
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

**提取映射**：

| 源码 | aclnn MD | README |
|------|----------|--------|
| `ParamType(REQUIRED)` | 输入/输出列="输入"，使用说明列按需填写 | 输入/输出/属性列="输入" |
| `ParamType(OPTIONAL)` | 输入/输出列="可选输入"；空指针语义还须核验接口校验和实际分支 | 输入/输出/属性列="可选输入" |
| `ge::DT_FLOAT16` | FLOAT16 | FLOAT16 |
| `ge::DT_BF16` | BFLOAT16 | BFLOAT16 |
| `ge::DT_FLOAT` | FLOAT32 | FLOAT32 |
| `ge::FORMAT_ND` | ND | ND |
| `ge::FORMAT_NCHW` | NCHW | NCHW |
| `.AutoContiguous()` 或接口中的连续化处理 | 作为支持非连续 Tensor 的核验线索，沿目标接口调用链确认后填表 | 不涉及 |
| 未发现连续化配置 | 不能直接标×；继续检查接口校验和实现，缺证时待确认 | 不涉及 |

**注意**：def.cpp中Input名称使用snake_case（如`grad_output`），对应README参数名；aclnn接口参数名使用camelCase（如`gradOutput`），对应aclnn MD参数名。

### 1.2 输出参数信息

```cpp
this->Output("grad_input")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
```

输出类型按目标输出的注册、接口校验及实现分别核验；不能根据 mean/rstd 等参数名推断固定类型，也不能默认与输入类型一致。

### 1.3 属性信息

```cpp
this->Attr("output_size").AttrType(REQUIRED).ListInt();    // → 属性名、类型ListInt、无默认值
this->Attr("input_size").AttrType(REQUIRED).ListInt();     // → 属性名、类型ListInt、无默认值
this->Attr("align_corners").AttrType(OPTIONAL).Bool(false); // → 属性名、类型bool、默认值false
this->Attr("scales_h").AttrType(OPTIONAL).Float();          // → 属性名、类型float、无默认值（空）
this->Attr("scales_w").AttrType(OPTIONAL).Float();          // → 属性名、类型float、无默认值（空）
```

**Attr类型映射示例**（aclnn 类型须读取实际函数声明，不能仅由注册属性类型推导）：

| 源码Attr类型 | aclnn接口类型 | aclnn MD参数名写法 | README数据类型 | README输入/输出/属性 |
|-------------|-------------|-------------------|---------------|---------------------|
| `.Int(x)` | `int64_t` | `attrName（int64_t）` | INT64 | 属性/可选属性 |
| `.Float(x)` | `double` | `attrName（double）` | FLOAT32 | 属性/可选属性 |
| `.Bool(x)` | `bool` | `attrName（bool）` | BOOL | 属性/可选属性 |
| `.ListInt()` | `aclIntArray*` | `attrName（aclIntArray*）` | INT64 | 属性/可选属性 |

**核验要点**：
- 注册中的属性类型用于 README；aclnn 参数的类型、指针和 const 以实际原型为准，上表只演示两层类型可能不同
- 列表参数的 size、元素范围和空值语义分别检查，不能由列表类型补造约束
- `AttrType(REQUIRED)` → README中标"属性"
- `AttrType(OPTIONAL)` → README中标"可选属性"，描述列用`<ul><li>`包裹默认值

### 1.4 平台配置

```cpp
this->AICore().AddConfig("ascend910b");    // → Atlas A2系列产品，仍需核验接口校验与发布支持
this->AICore().AddConfig("ascend910_93");  // → Atlas A3系列产品，仍需核验接口校验与发布支持

OpAICoreConfig regbaseConfig;
regbaseConfig.Input("grad_output")
    .ParamType(REQUIRED)
    .DataType(xDtype)
    .Format(xFormat)
    .UnknownShapeFormat(xFormat);
regbaseConfig.DynamicCompileStaticFlag(true)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .ExtendCfgInfo("opFile.value", "upsample_bicubic2d_aa_grad_apt");
this->AICore().AddConfig("ascend950", regbaseConfig);  // → Ascend 950PR&950DT系列产品，仍需核验接口与发布支持
```

**注意**：950平台可能有独立的`OpAICoreConfig`配置，含DynamicCompileStaticFlag等扩展信息，且对应独立的tiling文件（`*_tiling_arch35.cpp`）。

`AddConfig` 是平台注册证据，不是完整发布支持结论。逐平台交叉核验 config、接口检查及实际实现；未找到注册不等于已证明不支持，证据不足时记录“待确认”。

---

## 2. 从 `*_base.h` 提取

### 2.1 枚举定义

```cpp
enum class NormType : int64_t {
    NONE = 0,
    LAYER_NORM,              // = 1
    LAYER_NORM_AFFINE,       // = 2
    RMS_NORM,                // = 3
    RMS_NORM_AFFINE,         // = 4
};
```

**转换为文档描述**：

| 枚举值 | 数值 | aclnn MD使用说明列 | README描述列 |
|--------|------|-------------------|-------------|
| NONE | 0 | 枚举值支持0，其中0表示不做正则化 | 0表示不做正则化 |
| LAYER_NORM | 1 | 1表示LayerNorm | 1表示LayerNorm |
| LAYER_NORM_AFFINE | 2 | 2表示LayerNormAffine | 2表示LayerNormAffine |
| RMS_NORM | 3 | 3表示RmsNorm | 3表示RmsNorm |
| RMS_NORM_AFFINE | 4 | 4表示RmsNormAffine | 4表示RmsNormAffine |

仅文档化目标 aclnn 接口实际暴露并接受的枚举值。是否属于内部枚举须核验接口与参数校验，不能依据名称名单判断。

### 2.2 维度常量

```cpp
constexpr size_t INPUT_DIM_NUM = 4;   // B, S, H, D
```

检查常量在目标参数校验或 shape 推导中的实际用途，再记录维度；常量存在本身不能证明所有参数具有该维度。

### 2.3 参数索引枚举

```cpp
enum class InputIndexForward : size_t {
    QUERY_INDEX = 0,
    KEY_INDEX,
    ...
};
```

用于定位内部参数绑定；对外参数顺序和数量仍以实际 aclnn 声明及转发关系核对。

**注意**：部分算子（如upsample_bicubic2d_aa_grad）不存在`*_base.h`文件，此时从def.cpp和aclnn接口头文件提取参数信息即可。

---

## 3. 从 `*_infershape.cpp` 提取

### 3.1 输出shape推导

```cpp
*queryOutputShape = {batch, head, querySeq + encoderQuerySeq, dim}; // B, H, S, D
```

**上述代码的示例转换**（实际维度含义与顺序须由目标实现确认）：
- 输入shape为(B, S, N, D)即BSND
- 输出shape为(B, N, S, D)即BNSD（注意维度顺序变化！）
- 拼接场景：S = querySeq + encoderQuerySeq

### 3.2 条件输出

```cpp
if (*isTraining) {
    if (*normType == static_cast<int64_t>(NormType::LAYER_NORM) ||
        *normType == static_cast<int64_t>(NormType::LAYER_NORM_AFFINE)) {
        *normQueryMeanShape = {batch, querySeq, head, 1};
        *normQueryRstdShape = {batch, querySeq, head, 1};
    }
}
```

**上述分支可确认的信息**：
- isTraining=true 且 normType 为示例中两个值时，该分支为 mean/rstd 设置 `[batch, querySeq, head, 1]`
- 其他模式是否输出、是否允许空指针及输出 shape，仍需读取完整推导与接口使用路径；不能从未展示的分支补出结论

**aclnn MD使用说明列写法**：
```html
<ul><li>normType=1或2且isTraining=true时有效。</li></ul>
```

**README描述列写法**：
```
LayerNorm中的query均值输出，用于反向
```

### 3.3 空infershape文件

infershape.cpp 为空或不存在时，沿实际调用链查找输出构造及框架推导证据，结合接口注释和注册核验。不能仅因文件为空就认定由框架自动处理；无法确认的 shape 保留待确认。

---

## 4. 从 `aclnn_*.cpp/h` 提取

### 4.1 函数签名

```cpp
aclnnStatus aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(
    const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize,
    bool alignCorners, double scalesH, double scalesW, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

**提取规则**：
- 参数顺序必须与函数声明完全一致
- 参数类型映射为参数名列，保留原型中的 `const`：`gradOutput（const aclTensor*）`、`outputSize（const aclIntArray*）`、`alignCorners（bool）`、`scalesH（double）`
- 最后两个参数固定为 `workspaceSize（uint64_t*）` 和 `executor（aclOpExecutor**）`

### 4.2 内部接口转发

```cpp
aclnnStatus aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(...) {
    // 参数检查
    auto checkRet = CheckParams(gradOutput, outputSize, inputSize, scalesH, scalesW, out);
    // Contiguous处理
    auto selfContiguous = l0op::Contiguous(gradOutput, uniqueExecutor.get());
    // 调用L0算子
    const aclTensor *upsampleOut = l0op::UpsampleBicubic2dAAGrad(
        selfContiguous, outputSize, inputSize, out, alignCorners, realScalesH, realScalesW, uniqueExecutor.get());
    // ViewCopy
    auto viewCopyResult = l0op::ViewCopy(upsampleOut, out, uniqueExecutor.get());
}
```

**提取规则**：
- 文档描述目标外部接口；沿转发链核对类型转换、参数重排、默认值和输出复用，不假定内外语义完全相同
- `l0op::Contiguous` 是非连续输入处理的线索，须确认目标参数可达该调用且未被前置校验拒绝
- CheckParams逻辑是错误码描述的主要来源

### 4.3 参数注释

```cpp
* @param [in] gradOutput: Device侧的aclTensor, 数据类型支持FLOAT32,FLOAT16,BFLOAT16,
 * 支持非连续的Tensor,数据格式支持NCHW，shape仅支持四维Tensor。数据类型与出参out的数据类型一致。
 * @param [in] outputSize: Host侧的aclIntArray，数据类型支持INT64，size大小为2。
 * 表示输入gradOutput在H和W维度上的空间大小。
```

**提取规则**：
- 注释中的描述是文档参数描述的主要来源
- "支持非连续的Tensor" → 非连续Tensor列标√
- "数据格式支持NCHW" → 记录 NCHW；只有目标实现另有证据时才增加 ND 或其他格式
- "shape仅支持四维Tensor" → 维度列填4
- "数据类型与出参out的数据类型一致" → 只记录数据类型与 out 一致；数据格式是否一致另行核验，不扩展原注释的含义
- "size大小为2" → 记录 size 为 2；各元素范围须另查校验分支，不能自行补充正数限制

### 4.4 CheckParam逻辑提取错误码

```cpp
static bool CheckNotNull(const aclTensor *gradOutput, const aclIntArray *inputSize, const aclTensor *out)
{
    OP_CHECK_NULL(gradOutput, return false);
    OP_CHECK_NULL(inputSize, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor *gradOut, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(gradOut, out->GetDataType(), return false);
    return true;
}

static bool CheckShape(...)
{
    OP_CHECK_WRONG_DIMENSION(gradOutput, DIM_LIMIT, return false);
    OP_CHECK(inputSizeNum == EXPECT_SIZE, ..., return false);
    OP_CHECK(outputSizeNum == EXPECT_OUTPUTSIZE, ..., return false);
    return true;
}

static bool CheckScalesValid(const double width, const double high)
{
    if ((width < 0) || (high < 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scales_w and scales_h cannot be negative ...");
        return false;
    }
    return true;
}
```

**转换为错误码表格**：

| 返回码 | 错误码 | 描述 |
|--------|--------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 传入的gradOutput、inputSize或out是空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | gradOutput或out的数据类型不在支持的范围之内。 |
| | | gradOutput与out的数据类型不一致。 |
| | | gradOutput的shape不是4维。 |
| | | inputSize的size与目标源码中已核验的EXPECT_SIZE不一致。 |
| | | outputSize的size与目标源码中已核验的EXPECT_OUTPUTSIZE不一致。 |
| | | scalesH或scalesW的值为负数。 |

---

## 5. 从 `*_tiling.cpp` 提取约束

### 5.1 参数校验

```cpp
if (headDim_ < 1 || headDim_ > 1024) {
    OP_LOGE(ctx, "headDim must be in [1, 1024].");
    return ge::GRAPH_FAILED;
}
```

**转换为约束说明**：
- "headDim长度在[1~1024]之间。"

### 5.2 依赖关系约束

```cpp
if (ropeSeq_ > std::min(querySeq_ + encoderQuerySeq_, keySeq_ + encoderKeySeq_)) {
    OP_LOGE(ctx, "seqRope must <= min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey).");
}
```

**转换为约束说明**：
- "seqRope 不超过 Min(seqQuery+seqEncoderQuery, seqKey+seqEncoderKey)。" 下界须由其他校验分支另行确认。

### 5.3 平台差异化约束

tiling.cpp中常通过`socVer_`分支实现平台特定校验，必须提取到文档中。

```cpp
if (socVer_ == SOC_VER_950_CODE) {
    if (innerPrecise_ != BsaInnerCalcPrec::LOW_HIGH_MIXED) {
        OP_LOGE(bsaContext->GetNodeName(), "On chip 950, only innerPrec = 4 is supported, "
            "but got %u.", innerPrecise_);
        return ge::GRAPH_FAILED;
    }
} else {
    if (innerPrecise_ != BsaInnerCalcPrec::ALL_HIGH && innerPrecise_ != BsaInnerCalcPrec::ALL_LOW) {
        OP_LOGE(bsaContext->GetNodeName(), "On chip 910 & 910_93, only innerPrec = 0 or 1 is supported, "
            "but got %u.", innerPrecise_);
        return ge::GRAPH_FAILED;
    }
}
```

**转换为参数表下方的产品差异化描述**（仅对上述已核验分支适用）：
```markdown
<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：innerPrecise仅支持配置为4。
<!-- end id1 -->
<!-- npu="910b,A3" id2 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：innerPrecise仅支持配置为0或1。
<!-- end id2 -->
```

**平台分支核验**：读取目标代码中的枚举定义和所有可达分支，再映射公开产品名。不能把示例常量的数值或 else 分支当成跨算子通用的产品映射。

### 5.4 RegBase平台判断

```cpp
bool regBase = Ops::Cv::OpTiling::IsRegbaseSocVersion(tilingContext);
if (regBase) {
    OP_LOGI(tilingContext->GetNodeName(), "enter Tiling4UpsampleBicubic2dAAGradRegbase");
    return Tiling4UpsampleBicubic2dAAGradRegbase(tilingContext);
}
```

**提取规则**：读取目标版本 `IsRegbaseSocVersion` 的定义及调用分支，确认实际覆盖的平台和 tiling 路径。函数名或 `arch35` 文件后缀只用于定位，不独立证明产品支持或 else 分支覆盖范围。

---

## 6. 从 `*_tiling_arch35.cpp` 提取确定性计算信息

### 6.1 确定性计算分支

```cpp
baseTiling_.isDeterministic = context_->GetDeterministic();

// CalTilingData中的分支
if (isDataCopy) {
    baseTiling_.schId = CONST_0; // 纯copy模板
    baseTiling_.isDeterministic = 1;  // DataCopy场景强制确定性
} else if (baseTiling_.isDeterministic) {
    baseTiling_.schId = CONST_1; // nchw 分线程 deterministic
} else {
    baseTiling_.schId = CONST_1; // nchw 分线程 non-deterministic
}
```

**转换为约束说明中的确定性计算描述**：分别核验各平台的默认模式、是否支持切换和实际实现。下例只有在这些事实全部证实时才可采用，不能据产品名选择默认值：
```markdown
- 确定性计算：

  <!-- npu="A3,910b" id1 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：{各平台已核验的默认行为与切换能力；不同时分开描述}。
  <!-- end id1 -->
  <!-- npu="950" id2 -->
  - <term>Ascend 950PR&950DT系列产品</term>：{该平台已核验的默认行为与切换能力}。
  <!-- end id2 -->
```

**判断规则**：
- `isDeterministic` 是模式相关分支的线索；默认值、实际模式差异和是否支持切换仍须核验配置及 Kernel。已确认支持手动开启时配置 API 使用 `aclrtSetSysParamOpt`，但不能因此断言默认非确定性。
- 逐个平台核验默认确定性行为，不因平台是910b/910_93就推断为确定性；缺少证据时标注“待确认”。
- 上述示例仅在 `isDataCopy` 分支显式设置了确定性标志；其他算子仍须检查实际分支。

---

## 7. 从 `examples/test_aclnn_*.cpp` 提取调用示例

### 7.1 提取要点

1. ACL初始化代码（Init函数）
2. 输入Tensor构造函数名和格式
   - `CreateAclTensor` → `ACL_FORMAT_ND`
   - `CreateAclNchTensor` → `ACL_FORMAT_NCHW`
3. aclIntArray构造方式（`aclCreateIntArray`）
4. 两段式接口调用
5. workspace申请和释放
6. 结果同步和拷贝
7. 资源释放

### 7.2 示例不存在时

没有现成样例时，只有在原型、参数约束和调用语义已核验后才能构造调用示例；关键事实缺失时按父入口保留草稿，不用常见值填空。完整示例必须包含：
- CHECK_RET/LOG_PRINT宏定义
- GetShapeSize辅助函数
- Init辅助函数
- CreateAclTensor或CreateAclNchTensor模板函数（根据def.cpp的Format选择）
- aclCreateIntArray用法（若接口含aclIntArray参数）
- 7步流程完整呈现
- 开头固定话术（以下相对链接属于生成文档，不相对于本指南）：`示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。`

### 7.3 CreateAclTensor变体选择

| def.cpp Format | 使用函数 | aclFormat |
|---------------|---------|-----------|
| `FORMAT_ND` | `CreateAclTensor` | `ACL_FORMAT_ND` |
| `FORMAT_NCHW` | `CreateAclNchTensor` | `ACL_FORMAT_NCHW` |
| `FORMAT_ND` + `FORMAT_NCHW` | `CreateAclNchTensor` | `ACL_FORMAT_NCHW` |

---

## 8. 从 `config/` 目录提取平台信息

### 8.1 目录结构

```
config/
├── ascend910b/          → Atlas A2系列的配置线索
│   ├── *_binary.json
│   └── *_simplified_key.ini
├── ascend910_93/        → Atlas A3系列的配置线索
│   ├── *_binary.json
│   └── *_simplified_key.ini
└── ascend950/           → Ascend 950系列的配置线索
    ├── *_binary.json
    └── *_simplified_key.ini
```

目录名仅提供平台配置线索。读取配置内容，与 AddConfig、接口限制、实现和目标发布版本交叉核验；有目录不直接等于支持，缺目录也不直接等于不支持。分别记录支持、不支持和证据不足，不能照抄模板中的 √/× 或支持度。

---

## 9. 参数名映射：def.cpp → aclnn接口 → 文档

def.cpp中的Input/Output名称（snake_case）与aclnn接口参数名（camelCase）的映射关系：

| def.cpp名称 | aclnn接口参数名 | aclnn MD参数名列 | README参数名列 |
|-------------|----------------|-----------------|---------------|
| `grad_output` | `gradOutput` | `gradOutput（const aclTensor*）` | `grad_output` |
| `grad_input` | `out` | `out（aclTensor*）` | `grad_input` |
| `output_size` | `outputSize` | `outputSize（const aclIntArray*）` | `output_size` |
| `input_size` | `inputSize` | `inputSize（const aclIntArray*）` | `input_size` |
| `align_corners` | `alignCorners` | `alignCorners（bool）` | `align_corners` |
| `scales_h` | `scalesH` | `scalesH（double）` | `scales_h` |
| `scales_w` | `scalesW` | `scalesW（double）` | `scales_w` |

**映射规则**：
- def.cpp名称用snake_case
- aclnn接口参数名用camelCase
- aclnn MD参数名列用camelCase + 中文括号类型
- README参数名列用def.cpp的snake_case（不含类型）

**特殊情况**：某些参数在def.cpp中声明为Input（如output_size、input_size），但在aclnn接口中为`aclIntArray*`类型。在README中按def.cpp声明归类：
- def.cpp `this->Attr("output_size")` → README标"属性"
- def.cpp `this->Input("blockShape")` 且 `.ParamType(OPTIONAL)` → README标"可选输入"；只有 Input 声明不能推出可选

---

## 10. 反向算子信息提取

### 10.1 前向算子关联

反向算子的前向关联须由接口语义、需求或实现确认。名称的 Grad 后缀只能作为查找线索，不能独立证明关联关系。以下是已确认关联后的路径示例：

```
算子名: UpsampleBicubic2dAAGrad
前向算子名: UpsampleBicubic2dAA
前向算子路径: ../upsample_bicubic2d_aa/
```

**提取规则**：
1. 用接口语义和目标实现确认前向算子及实际接口名
2. 在已提供的项目范围内核对目标目录和文档是否存在；缺少目标时记录缺口，不构造虚假链接
3. 按实际目录生成交叉引用链接，以下仅为该示例的写法：
   - aclnn MD: `[aclnnUpsampleBicubic2dAA](../../upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)`
   - README: `[UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README.md)`

### 10.2 功能说明写法

- aclnn MD: `接口功能：[aclnnUpsampleBicubic2dAA](../../upsample_bicubic2d_aa/docs/aclnnUpsampleBicubic2dAA.md)的反向传播。如果输入张量的shape为(N, C, H, W)，则输出张量的shape为(N, C, inputSize[2], inputSize[3])。`
- README: `算子功能：[UpsampleBicubic2dAA](../upsample_bicubic2d_aa/README.md)的反向传播。如果输入张量的shape为(N, C, H, W)，则输出张量的shape为(N, C, inputSize[2], inputSize[3])。`
