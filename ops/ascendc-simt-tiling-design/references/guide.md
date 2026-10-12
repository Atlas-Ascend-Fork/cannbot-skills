# SIMT 算子切分设计指南

SIMT 算子的切分设计与 SIMD 有本质差异：不按 UB 容量切分数据、不做 inQueue/outQueue 流水，而是以核数切分 + 线程数设置为核心；但仍需在 UB 中分配 Data Cache（及核内共享 buffer）空间。

## SIMT vs SIMD 切分差异

| 要素 | SIMD 方式 | SIMT 方式 |
|------|----------|----------|
| 多核切分 | 按 UB 单次处理量切分 | 按元素总量切分（ceil(总量/单核最少元素数)） |
| 单核并行 | UB Buffer + 向量指令 | 线程数（`Dim3` 可运行时收缩，上限 `LAUNCH_BOUND` 1024/2048） |
| 数据搬运 | 需显式 Load/Store | 支持直接读写 GM |
| UB 使用 | 全量使用 | 仅核内共享场景使用，DCache ≥32KB |
| Buffer 规划 | inQueue/outQueue/tmpBuf | TBuf（仅在需要共享内存时使用） |

## 核数切分

```
需求核数   = ceil(输出元素总数 / 每核元素数)
每核元素数 = max(ceil(输出元素总数 / 物理核数), 1024)，并对 warp(32) 对齐
下发 blockDim = clamp(需求核数, 1, 物理核数)
```

注意区分两个概念：**需求核数**由数据量与每核下限推出；**下发 blockDim** 是 `SetBlockDim` 的实参，必须封顶物理核数（越大越无益，只增调度开销）。

- 通过 tiling 侧 `SetBlockDim` 设置核数
- 合理设置 `perCoreElements` 避免负载不均
- 避免总数据量少但启动核数多的场景
- **gridDim建议不超过物理核数 `coreNum`**。由于硬件调度差异，昇腾上暂未支持单 core 多 block 能力，启动的 block 数大于 `core_num` 时会循环调度 core，带来大量额外调度开销，故不宜随元素数膨胀。当最终索引需 64 位（大 shape）时仍须先转宽再乘。

> 进阶分核策略（双任务分核取 max、线程数自适应收缩、按最大工作量子循环估核、平台参数动态化）见 [tiling-advanced.md](tiling-advanced.md)。

## 线程数设置

- 默认值: 1024，最大值: 2048
- **`LAUNCH_BOUND(N)` 必须编译期常量**：它决定每线程寄存器分配上限
- **`Simt::Dim3(N)` 可为运行时值**：如 tiling 自适应收缩后的 `threadNum`，但必须 **≤ `LAUNCH_BOUND`**
- 二者取同一常量最安全；运行时收缩时以编译期上限作 `LAUNCH_BOUND`，以收缩值作 `Dim3`
- 实例：声明 `LAUNCH_BOUND(2048)` 作编译期上限，启动 `Dim3(tilingData.threadNum)`，tiling 侧按数据量逐次减半

### 按算子类型选择

| 算子类型 | 建议线程数 | 原因 |
|---------|-----------|------|
| 搬运类算子 | 2048 / 1024 | 更多线程隐藏内存延迟 |
| 计算类算子 | 512 / 1024 | 寄存器压力大，需平衡 |

> 档位参考：2048 纯搬运/少变量；1024 坐标分解多变量（默认）；512 寄存器密集 / 64 位索引（512 = 64 寄存器/线程）；256/128/64 高维累加 / int64 归约（随维度递减控寄存器）。可按索引位宽自动分档：int32 索引 1024、int64 砍半。

### 正确写法

```cpp
constexpr uint32_t THREAD_NUM = 512;

// VF 函数声明
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void OpComputeSimt(...);

// VF 调用
Simt::VF_CALL<OpComputeSimt<T>>(Simt::Dim3(THREAD_NUM), args...);
```

### 运行时收缩写法（允许，须 ≤ LAUNCH_BOUND）

```cpp
// LAUNCH_BOUND 固定为编译期上限；Dim3 可传 tiling 自适应后的线程数（不得超上限）
constexpr uint32_t THREAD_NUM_BOUND = 2048;   // 寄存器分配上限
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_BOUND) inline void OpComputeSimt(...);

int32_t threadNum = static_cast<int32_t>(tilingData_->threadNum);   // tiling 侧已收缩
Simt::VF_CALL<OpComputeSimt<T>>(Simt::Dim3(threadNum), args...);
```

> 实例：`THREAD_NUM_LAUNCH_BOUND = 2048` + tiling 侧线程数逐次减半。收缩只影响并行度，不影响寄存器上限。

> 如果数据量较小导致线程空转过多，应通过调整**核数**（`SetBlockDim`）来适配。

## DCache 与 UB 空间设置

SIMT 线程访问 GM 需经 UB 内划出的 **Data Cache** 中转。UB 总 256KB，从低到高划分为**静态内存 → 动态内存 → 预留空间 → Data Cache** 四区，共用同一块 UB。

官方规则：

- Data Cache 可配置范围 **最小 32KB、最大 128KB**，实际大小由用户申请的动/静态内存**反向推导**：
  `Data Cache = min(256KB − 静态内存 − 动态内存, 128KB)`
- 预留空间已自动扣除，因此 host 侧要设置的是**动态 UB 大小**（`SetDynUBufSize`），给 Data Cache 留出目标空间：

```cpp
constexpr uint64_t DCACHE_SIZE = 128 * 1024;                        // 目标 Data Cache，32KB~128KB
uint64_t ubSize = 0;
platformAsc.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);   // 平台 UB，禁硬编码
context->SetDynUBufSize(ubSize - DCACHE_SIZE);        // 再扣静态 TPipe buffer 预算
```

- **取值思路**：流式访问、DCache 需求小的算子可多给动态 UB（DCache 取 32KB 下限）；随机访存/查表热点多的算子给足 DCache（上限 128KB）。
- 可用 UB 示例：256KB − 8KB 预留 − 32KB DCache = **216KB**。
- **接口名口径**：当前接口为 `gert::TilingContext::SetDynUBufSize`（默认 0 = 不声明），可用 `GetDynUBufSize` 查询。
- **host 私有化决策与 device TPipe 分配必须用同一个数**，否则 UB 越界。
- **平台差异**：`GetRuntimeUBSize()` 在 SIMD/SIMT 混合场景返回上限 **216KB**，非混合场景返回固定值 **248KB**（官方 `GetRuntimeUBSize` 文档）。

## 索引位宽自适应（默认 32 位，按需升 64 位）

把索引位宽做成 tiling 决策、编码进 tilingKey，kernel 侧用模板参数 `ADDR_T` 编译期分发（kernel 侧写法见 BP `index-calculation.md`）：

```cpp
// 最大索引中间量 或 总元素数 超 int32 → 升 64 位；仅 SIMT 路径提供该分支
uint32_t sizeAddrType = ((innerSize * outerCount > INT32_MAX) || (totalSize > INT32_MAX)) ? 1 : 0;
if (!isSimt) { sizeAddrType = 0; }              // 非 SIMT 路径无 64 位分支
tilingKey = GET_TILINGKEY(routeFlags..., sizeAddrType, ...);
```

- 判定须含**中间乘积**（`shape[i]*stride[i]`、`尾轴 × 行数`、`isize*osize`），不能只看元素总数。
- 最终索引/地址需 64 位时（大 shape），核内乘法（如 `blockIdx × perCoreElements`、`i × stride`）须先转宽再乘（操作数来自返回 uint32 的 API，见 BP `index-calculation.md`）。

## Tiling 侧审查清单

上线前逐项自查：

- [ ] 网格块数是否封顶物理核数且不超 65535（防 blockIdx 越界引发 507035）？索引计算在需 64 位时是否先转宽再乘？索引位宽是否按需选 32/64 位并编入 tilingKey？
- [ ] 小 shape 是否做每核最小工作量钳制 + 32 对齐？
- [ ] 多阶段算子（清零+计算）是否两套分核取 max？
- [ ] kernel 有 `SyncAll()` 时是否 `SetScheduleMode(1)`？
- [ ] workspace 大小含系统区？分段公式与 kernel 逐段一致？有无 uint64 溢出？
- [ ] SIMT 模板 host tiling 是否已声明 Dynamic Unified Buffer（`SetDynUBufSize`）且大小与 DCache/静态预留一致？
- [ ] tilingKey 与 kernel 侧 `TILING_KEY_IS` / `if constexpr` 一一对应？
- [ ] deterministic 分支：dtype 白名单 + workspace 追加 + 模板选择是否齐备？
- [ ] 核数与 UB 是否取自平台 API（未硬编码）？
- [ ] compileInfo 为 null 时是否 fallback 直读 `context->GetPlatformInfo()`？

## 相关文档

- 进阶分核策略：[tiling-advanced.md](tiling-advanced.md)
- kernel 侧编程：`ascendc-simt-best-practices` skill
