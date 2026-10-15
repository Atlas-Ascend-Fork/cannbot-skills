# Contraction AscendC 运行时入口绑定

本文件维护 AscendC 工程的注册 ABI、编译期分派和构建绑定。共享设计结果和参数消费者见 [公共适配契约](../../generic/common-contract.md#7-设计与接口适配契约)。

## 1 Kernel 入口

### 1.1 注册 ABI 与存储绑定

注册 kernel 按公开顺序声明输入、输出、workspace、tiling 的 GM_ADDR；dtype 来自注册编译参数，模板形参只描述 TilingKey 策略。Host 和 Device 使用同一定义读取 TilingData，不手写不一致的结构或数值 key。

下面是入口职责伪码，不是可直接编译的 API 声明；真实宏、参数类型和 Params 赋值从目标工程及所选 adapter 取得。直接 launch 的官方样例可带类型模板，但不能把该测试入口原样当成注册算子 ABI。

当前 AscendC adapter 不使用 TPipe/TBuf；入口按模板确定的静态局部存储方案绑定片上地址，槽所有权由显式流水同步管理。AIC-only、MIX 任务类型与 InitSocState 等初始化必须匹配所选特化，不把某个 Basic 实例的 AIV return 假设泛化到其他 kernel。

### 1.2 入口职责伪码

```text
registered_entry(inputs..., outputs..., workspace, tiling):
    read tiling using the registered definition
    dispatch A/B ND/DN layout and strategy using the project's template mechanism
    bind static local storage according to the selected implementation mapping
    map every input/output address and tiling field into same-version Kernel::Params
    Kernel kernel
    kernel(params)
```

GM_ADDR 必须收到正确设备地址。各输入指针按其坐标映射、真实物理布局和所选 Params 的消费者绑定。

直接测试工程中若 GM_ADDR 为 uint8_t*，使用匹配的设备指针类型，不把 void* 未转换地传给 CCE kernel；使用 memcpy 时包含其声明头。注册入口不承接测试工具的固定文件名、参数顺序或目录约定。

## 2 编译期分派

注册算子按工程的 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 机制定义场景、输入方向及确实改变代码的策略维度；禁止另加 `TILING_KEY_IS` 分发。dtype 使用编译期 `DTYPE_IN0` 等注册类型，不再复制成 runtime key。

A/B 的 ND/DN、执行组织和可选 `hasBias` 维度由 [模板总览](../../generic/template-overview.md#3-tilingkey) 定义。adapter 消费同一组编译期选择，再以逻辑 M/N/K 生成相应地址与操作参数。复用工程已有模板参数及其方向对应关系；每个合法组合绑定唯一入口，未支持组合不能落入默认成功分支。

## 3 调度与构建绑定

launch 核数、任务总数、SWAT 映射及尾部拆分参数来自同一模板策略，按 [公共适配契约](../../generic/common-contract.md#7-设计与接口适配契约) 绑定到工程已有 Params/TilingData 及对应消费者。入口传递并执行这些结果，保留每核任务循环、任务基址和跨任务槽轮转。

编译器、头文件和链接依赖采用匹配版本；实际重载、同步机制及完成边界按目标 SDK 的公开声明核对。张量和操作映射见 [AscendC adapter](capability-mapping.md)；同步使用所选映射引用的完整协议。

语义 Golden、地址覆盖、资源生命周期及实现身份遵循 [公共验收契约](../../generic/common-contract.md#5-产出与验收)，具体覆盖场景由所选模板提供。构建通过不能代替设备精度和性能证据。
