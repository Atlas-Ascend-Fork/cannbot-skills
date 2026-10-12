# 组件错误码源码索引

## 1. 使用时机

仅在官方文档错误码清单没有与根因和字段同时匹配的候选时，使用本索引继续检查目标组件的注册与实现。

索引命中不等于该错误码可直接用于算子开发。继续核对：

1. 调用点根因是否确实归属该组件，禁止因文案相似而跨模块借码。
2. 目标分支或目标安装包是否存在该注册项、模板和Arglist。
3. 该码是否面向用户公开，还是组件内部诊断码。
4. 是否已有下层首错；当前层不重复上报。
5. 资料未收录但源码已注册时，明确标记“源码注册、资料缺失”，评估补资料或正式治理，不把它误标为文档正式码。

仍无合适语义时，按[开发使用规范](development-standard.md)形成结构化正式码类别提案，编号分配前不得编造。

## 2. 来源索引

| errClass | 归属组件 | 错误码所在仓 | 错误码所在文件 | 可访问性 |
|---|---|---|---|---|
| AI CPU Errors | AI CPU | runtime | [`src/dfx/error_manager/error_code.json`](https://gitcode.com/cann/runtime/blob/master/src/dfx/error_manager/error_code.json) | 公开 |
| RTS Errors | Runtime | runtime | [`src/dfx/error_manager/error_code.json`](https://gitcode.com/cann/runtime/blob/master/src/dfx/error_manager/error_code.json) | 公开 |
| ACL Errors | Runtime | runtime | [`src/dfx/error_manager/error_code.json`](https://gitcode.com/cann/runtime/blob/master/src/dfx/error_manager/error_code.json) | 公开 |
| HCCL Errors | HCCL | hcomm | [`src/legacy/ascend910/common/error_code/error_code.cc`](https://gitcode.com/cann/hcomm/blob/master/src/legacy/ascend910/common/error_code/error_code.cc) | 公开 |
| Profiling Errors | Profiling | runtime | [`src/dfx/error_manager/error_code.json`](https://gitcode.com/cann/runtime/blob/master/src/dfx/error_manager/error_code.json) | 公开 |
| Driver Errors | Driver | driver-dev | [`pkg_inc/hal_error_code/drv_error_code.h`](https://gitcode.com/cann/driver-dev/blob/master/pkg_inc/hal_error_code/drv_error_code.h) | 公开 |
| Queue Schedule Errors | AI CPU | runtime | [`src/dfx/error_manager/error_code.json`](https://gitcode.com/cann/runtime/blob/master/src/dfx/error_manager/error_code.json) | 公开 |
| DVPP Errors | DVPP | dvpp-dev | [`src/dvpp_op_base/base/dvpp_err_msg_code.cpp`](https://gitcode.com/cann/dvpp-dev/src/dvpp_op_base/base/dvpp_err_msg_code.cpp) | 公开 |
| AMCT Errors | AMCT | amct-dev | [`amct_acl/src/error_code/amct_error_code.cpp`](https://gitcode.com/cann/amct-dev/amct_acl/src/error_code/amct_error_code.cpp) | 公开 |
| DUMP Errors | DUMP | runtime | [`src/dfx/error_manager/error_code.json`](https://gitcode.com/cann/runtime/blob/master/src/dfx/error_manager/error_code.json) | 公开 |
| Operator Errors | aclnn | opbase | [`src/op_common/log/log.cpp`](https://gitcode.com/cann/opbase/blob/master/src/op_common/log/log.cpp) | 公开 |
| Nnopbase Errors | aclnn | opbase | [`src/nnopbase/composite_op/log/op_error_manager.cpp`](https://gitcode.com/cann/opbase/blob/master/src/nnopbase/composite_op/log/op_error_manager.cpp) | 公开 |

## 3. 检索结果记录格式

每次使用索引，记录以下字段，供开发、定位和静态检查Skill复用：

| 字段 | 内容 |
|---|---|
| `errClass/component` | 目标错误类别和归属组件 |
| `repository/ref` | 仓库与已核对的分支、tag或commit |
| `registry_path` | 实际注册文件路径 |
| `code/template/arglist` | 候选编号、消息模板和参数列表 |
| `public_documentation` | 官方资料URL；没有则明确“资料缺失” |
| `callable_from_target` | 当前模块是否能通过公开头和链接依赖调用 |
| `decision` | 复用现有码、下层首错透传、补资料或新增正式码类别 |
| `remaining_evidence` | 尚缺的安装包、include链、编译或负向日志证据 |
