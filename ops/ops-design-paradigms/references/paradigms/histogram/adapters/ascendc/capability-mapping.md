<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式 Ascend C SIMT 能力映射（混合 VF）

> 本文件把 [../../generic/capability-contract.md](../../generic/capability-contract.md) 的能力谓词映射为 Ascend C SIMT 的真值与 API 实体。
> SIMT 内核 API 存在性以 `$ASC_DEVKIT_DIR` 头文件（CANN 9.2.0-beta.1 的 `x86_64-linux/asc/include/simt_api/`）与 `simt_histogram_story` 样例（SIMT 内核参考）源码为准；regbase 工程框架 API 以 `include/register/`、`asc/impl/`、`include/exe_graph/runtime/` 头文件与 `examples/add_example/` 源码为准；未能溯源者标「待验证」。

## 1 能力谓词真值

| 谓词 | 真值 | API 实体 | 依据 |
|------|------|----------|------|
| `WARP_SIZE` | 32 | `constexpr int32_t WARP_SIZE = 32` | `sample_common.h` L48 |
| `THREAD_COUNT_PER_BLOCK` | ✔ 512（默认，按性能测试结果调整） | `constexpr int32_t SIMT_THREADS_PER_BLOCK = 512` + `__launch_bounds__(512)`；`asc_vf_call(dim3(512))` 与其保持一致 | `src/3_advanced_tuning.asc`；`examples/add_example/op_kernel/arch35/add_example.h` |
| `GRID_DIM` | 运行时（= core 数） | `SetBlockDim(needCoreNum)` + `gridDim.x`（SIMT VF 内即 core 号/总 core 数） | `add_example_tiling.cpp` `SetBlockDim`；`add_example.h` grid-stride |
| `LOCAL_HIST_ATOMIC` | ✔（UB 原子，收敛核内多线程同 bin 并发写） | `asc_atomic_add(__ubuf__ int32_t*, int32_t)` | `simt_api/device_atomic_functions.h` L25 |
| `LOCAL_HIST_CLEAR` | ✔（SIMD 批量清零） | `Duplicate(localHist, static_cast<OutType>(0), batchBins)`（regbase `__aicore__` 上下文） | `basic_api/kernel_operator_vec_duplicate_intf.h` L53 |
| `LOCAL_HIST_FLUSH` | ✔（SIMD DMA 原子归并 UB→GM） | `SetAtomicAdd<T>()` + `DataCopyPad(gm, local, …)`（MTE3）+ `SetAtomicNone()`/`DisableDmaAtomic()` | `basic_api/kernel_operator_set_atomic_intf.h` L39/L41/L44；`basic_api/kernel_operator_data_copy_intf.h` L379；`three_interpolate_backward.cpp` L318/L328/L338 |
| `GLOBAL_MERGE_ATOMIC` | ✔（GM 逐元素原子，int32/int64/float，收敛跨核同 bin 并发写） | `asc_atomic_add(__gm__ int32_t*, int32_t)` 等 | `simt_api/device_atomic_functions.h` L31/L37/L39 |
| `THREAD_BARRIER` | ✔ | `asc_syncthreads()` | `simt_api/device_sync_functions.h` L20 |
| `MEMORY_FENCE` | ✔（可选） | `asc_threadfence()` / `asc_threadfence_block()` | `simt_api/device_sync_functions.h` L22/L24 |
| `VECTOR_MTE3_SYNC` | ✔（跨引擎：VECTOR 写 UB 完成 → MTE3 读 UB 归并） | `` `TQue<TPosition::VECOUT>` `` 的 `EnQue`/`DeQue`（携带 VECTOR→MTE3 同步语义，无需显式 `SetFlag`） | `basic_api/kernel_tpipe.h` L60-82/L188 |
| `VECTOR_LOAD_WIDTH` | 4（float4） | `float4` + `asc_ldcg(__gm__ float4*)` | `simt_api/device_functions.h` L188；`src/2_mem_optimized.asc` |
| `ONCHIP_HIST_CAPACITY` | `ubSize − DCACHE_SIZE`（`SetLocalMemorySize` 传入值；DCache ≥32KB 最小，官方 guide.md 示例 128KB）——DCache=32KB 时 ≈216KB（256KB − 8KB − 32KB）。约束局部直方图**单批** bins 数量（分批加载突破总 nbins 限制） | `__ubuf__[]` 静态数组（编译器分配） | `ascendc-simt-tiling-design/guide.md` L63-71（`SetLocalMemorySize(ubsize − DCACHE_SIZE)`、`DCACHE_SIZE=128*1024`、可用 UB=216KB）；`examples/add_example/op_host/arch35/add_example_tiling.cpp` L36/L198（`DCACHE_SIZE=32*1024`、`SetLocalMemorySize(ubSize − DCACHE_SIZE − STATIC_UB_ESTIMATE)`） |
| `ALIGN_GRANULARITY` | 32 字节（UB block）；填充元素数 = 32 / sizeof(HistT) 随 dtype 变化（int32/float→8、fp16→16、int64→4、int8→32） | 平台接口 `GetUbBlockSize`（禁止写死 32）；`src/1_basic_simt.asc` 的 `ALIGNED_NUM = 8` 仅 int32 计数特例 | `op_common/op_kernel/platform_util.h` L30（`return 32U`）；`src/1_basic_simt.asc` L22、L57 |
| `PREFIX_MINMAX_REDUCE` | ✔（`asc_atomic_min/asc_atomic_max(__gm__ float*)`，已实测） | 缺省 min/max 归约：每线程局部 min/max + `` `asc_atomic_min/asc_atomic_max(__gm__ float*)` `` 归约到 workspace 暂存（清 0 段初始化 +min/-max）；histogram 算子默认 range 分支已上板实测（多核跨核原子收敛、min==max 特判均通过） | `simt_api/device_atomic_functions.h` L83/L99 |
| `FLAT_INDEX_CALC` | ✔（纯标量整数乘加，无专用 API） | `idx = ((i0 * nbins1 + i1) * nbins2 + i2) …` 的 C++ 整数表达式（行优先 Horner 递推） | 行优先布局约定；不涉及 SIMT 原子/向量原语 |

## 2 SIMT 内核 API 实体（可溯源）

### 2.1 原子操作 `asc_atomic_add`

头文件：`simt_api/device_atomic_functions.h`。签名（节选）：

```cpp
// UB（片上）原子累加
inline int32_t asc_atomic_add(__ubuf__ int32_t* address, int32_t val);
inline uint32_t asc_atomic_add(__ubuf__ uint32_t* address, uint32_t val);
inline float    asc_atomic_add(__ubuf__ float* address, float val);
// GM（全局）原子累加
inline int32_t asc_atomic_add(__gm__ int32_t* address, int32_t val);
inline uint32_t asc_atomic_add(__gm__ uint32_t* address, uint32_t val);
inline float    asc_atomic_add(__gm__ float* address, float val);
inline int64_t  asc_atomic_add(__gm__ int64_t* address, int64_t val);
inline uint64_t asc_atomic_add(__gm__ uint64_t* address, uint64_t val);
// 无地址空间限定（泛型指针）
inline int32_t asc_atomic_add(int32_t* address, int32_t val);
```

原子族另有 `sub/exch/max/min/inc/dec/cas/and/or/xor` 与对应 `asc_atomic_*` 命名。局部直方图用 UB 重载（`__ubuf__ int32_t*`），跨 block 归并用 GM 重载（`__gm__ int32_t*`）。

### 2.2 同步与栅栏

SIMT 线程栅栏（计数段内，`simt_api/device_sync_functions.h`）：

```cpp
inline void asc_syncthreads();          // block 内线程栅栏
inline void asc_threadfence();          // 全局线程/内存栅栏
inline void asc_threadfence_block();    // block 内栅栏
```

SIMD 跨引擎同步（`` `VECTOR_MTE3_SYNC` ``，计数段 VECTOR → 归并段 MTE3，regbase `` `__aicore__` `` 上下文，**非 SIMT VF 内**）：

- 本范式局部直方图模板用 `` `TQue<TPosition::VECOUT>` `` 队列管理 UB，`` `EnQue` ``/`` `DeQue` `` **自带 VECTOR→MTE3 流水同步语义**，因此计数段（SIMT VF）写 UB 后**无需显式 `SetFlag`/`WaitFlag`**（见 §2.9）。
- 显式硬件事件同步（`SetFlag`/`WaitFlag`/`FetchEventID`）为底层机制，供不用队列的形态参考。头文件：`basic_api/kernel_operator_block_sync_intf.h`、`basic_api/kernel_tpipe.h`：

```cpp
// basic_api/kernel_tpipe.h L474
template <HardEvent evt> __aicore__ inline TEventID FetchEventID();   // 事件 id（TPipe）
// basic_api/kernel_operator_block_sync_intf.h L43/L46/L98/L99
__aicore__ inline void SetFlag(int32_t eventID);                      // VECTOR 写 UB 完成，通知 MTE3
__aicore__ inline void WaitFlag(int32_t eventID);                     // 等待 MTE3 归并完成
template <HardEvent evt> __aicore__ inline void SetFlag(TEventID id);
template <HardEvent evt> __aicore__ inline void WaitFlag(TEventID id);
```

```cpp
// 不用 TQue 队列时的显式写法
asc_vf_call<HistogramSimtCount<...>>(...);                         // SIMT VF 计数（VECTOR）
event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(eventId);                                // VECTOR → MTE3 跨引擎同步（显式兜底）
WaitFlag<HardEvent::V_MTE3>(eventId);                               // 等待 MTE3 归并完成
AtomicAddToGm(ubLoop, yLocalNum);                                   // MTE3 SetAtomicAdd+DataCopyPad 归并
```

- `` `asc_syncthreads()` ``/`` `asc_threadfence()` `` 是 SIMT 线程/内存栅栏，**不能替代**跨引擎同步。

### 2.3 地址空间限定符

`__ubuf__`（片上 UB）、`__gm__`（全局内存）用于指针地址空间标注；局部直方图用 `__ubuf__`，输入/输出用 `__gm__`（或裸指针，由指针类型推导）。泛型指针重载已兼容无限定指针。

### 2.4 线程内建变量与启动

```
threadIdx.x / blockDim.x / blockIdx.x / gridDim.x   // 编译器内建（CUDA 风格，SIMT VF 内可用）
```

`__global__`/`__simt_vf__`/`__aicore__` 为编译器关键字（`tools/impl/utils/sys_macros.h` 注明「For ascc preprocess: __global__ should not be replaced」）；`asc_vf_call` 为 SIMT VF 启动原语（见 §4）。这些是编译器/语言构造，非头文件符号。

### 2.5 向量类型与向量化加载

头文件：`simt_api/device_functions.h`、`simt_api/vector_functions.h`。

```cpp
inline float4 asc_ldcg(__gm__ float4* address);   // 带全局缓存提示的向量加载
inline float4 make_float4(float x, float y, float z, float w);
```

`float4/float2/int4/uint4/uchar4` 为内建向量类型；缓存提示族另有 `asc_ldcg/asc_ldca/asc_stcg/asc_stwt`。样例用 `*reinterpret_cast<float4*>(x + i)` 直接向量化解引用。

### 2.6 寄存器与编译调优

```cpp
__simt_vf__ __aicore__ __launch_bounds__(N) inline void VF(...);   // N = 每 block 线程数，锁定每线程寄存器
// 编译器选项：--npu-arch=dav-3510、-O3
// 寄存器分析：--cce-res-usage
```

- `__launch_bounds__(N)`：约束每线程寄存器上限（N=512 → 64 寄存器/线程，样例零溢出），可用在 `__simt_vf__` 函数上（`add_example.h` L29）。
- `--cce-res-usage`：产生寄存器/栈用量报告，用于选择 N。

### 2.7 warp 函数（可选原语）

头文件：`simt_api/device_warp_functions.h`。`asc_shfl/asc_shfl_down/asc_shfl_up/asc_shfl_xor/asc_reduce_add/asc_ballot/asc_all/asc_any/asc_activemask`。本范式骨架不强制使用，但在自定义 min/max 前置归约（`PREFIX_MINMAX_REDUCE`）中可作 warp 归约备选，标「待验证」。

### 2.8 SIMD DMA 原子归并（`SetAtomicAdd` / `DataCopyPad` / `SetAtomicNone`）

局部直方图模板的归并段（`` `LOCAL_HIST_FLUSH` ``）用 SIMD DMA 原子把 UB 局部直方图累加到 GM 输出。头文件：`basic_api/kernel_operator_set_atomic_intf.h`、`basic_api/kernel_operator_data_copy_intf.h`。签名（节选）：

```cpp
// basic_api/kernel_operator_set_atomic_intf.h
template <typename T> __aicore__ inline void SetAtomicAdd();   // L39：置 DMA 原子加模式，后续 DataCopy(UB→GM) 累加而非覆盖
__aicore__ inline void DisableDmaAtomic();                     // L41：关闭 DMA 原子（SetAtomicNone 的替代）
__aicore__ inline void SetAtomicNone();                        // L44：已废弃，建议改用 DisableDmaAtomic
// basic_api/kernel_operator_data_copy_intf.h
template <typename T, PaddingMode mode = PaddingMode::Normal>
__aicore__ inline __inout_pipe__(MTE3) void DataCopyPad(       // L379：UB→GM 搬移（带 padding）
    const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams);
```

语义与使用（权威样例 `three_interpolate_backward.cpp` L318/L328/L338）：

```cpp
SetAtomicAdd<HistT>();                                // 置原子加模式
DataCopy(hist_gm[batchStart], localHist[0], params);  // GM[batchStart..] += localHist[0..]（原子累加）
SetAtomicNone();                                      // 关闭原子模式（DisableDmaAtomic 等价）
```

- 归并把 `` `BATCH_BINS` `` 个 UB 计数原子累加到 `hist[batchStart .. batchStart+BATCH_BINS)`，跨核收敛由 DMA 原子的硬件原子性保证。
- `SetAtomicAdd<T>()` 以模板指定累加 dtype（int32/int64/float 依 `HistT`）；`SetAtomicNone` 已标废弃，新代码优先 `DisableDmaAtomic`。
- 该归并段为 SIMD（MTE3 管线），位于 regbase `__aicore__` 上下文，非 SIMT VF 内；计数段（SIMT VF / VECTOR）结束到归并段之间的跨引擎同步由 `` `TQue<TPosition::VECOUT>` `` 的 `EnQue`/`DeQue` 携带（§2.9），**无需显式 `SetFlag<HardEvent::V_MTE3>`/`WaitFlag`**；`` `THREAD_BARRIER` ``/`` `MEMORY_FENCE` ``（SIMT 栅栏）不足以跨引擎同步（见 [kernel-skeleton.md](kernel-skeleton.md) §2 与 [../../generic/sync-model.md](../../generic/sync-model.md) §2/§3）。

### 2.9 UBUF 队列（`TQue` + DOUBLE BUFFER）

局部直方图模板的 UB 局部直方图用 `` `TQue` ``（QUEUE）+ 双缓冲（depth 2），配合 `` `AllocTensor` ``/`` `EnQue` ``/`` `DeQue` ``/`` `FreeTensor` `` 让下一批的 SIMT VF 计数与本批的 MTE3 归并重叠。头文件：`basic_api/kernel_tpipe.h`。签名（节选）：

```cpp
// basic_api/kernel_tpipe.h L188：TPosition=TPosition::VECOUT（VECTOR 生产 → MTE3 消费），depth=双缓冲深度
template <TPosition pos, int32_t depth, ...> class TQue;
// basic_api/kernel_tpipe.h L60/L62/L65/L67
template <typename T> __aicore__ inline LocalTensor<T> AllocTensor();   // 取一块空闲 UB
// basic_api/kernel_tpipe.h L72/L75
template <typename T> __aicore__ inline bool EnQue(const LocalTensor<T>& tensor);  // 生产完成入队
// basic_api/kernel_tpipe.h L77/L79/L82
template <typename T> __aicore__ inline LocalTensor<T> DeQue();         // 取待消费 buffer（MTE3 读）
// basic_api/kernel_tpipe.h L70
template <typename T> __aicore__ inline void FreeTensor(LocalTensor<T>& tensor);   // 归还 UB
// basic_api/kernel_tpipe.h（TPipe）
__aicore__ inline void InitBuffer(TQue& que, int32_t depth, int64_t perBufSize);   // 初始化队列缓冲
```

语义与使用（本范式双缓冲取 depth 2）：

```cpp
TQue<TPosition::VECOUT, 2> histQue;                       // QUEUE + DOUBLE BUFFER
InitBuffer(histQue, 2, BATCH_BINS * sizeof(OutType));
// 每批：AllocTensor → asc_vf_call（SIMT VF 计数） → EnQue → DeQue → 归并 → FreeTensor
```

- `` `TPosition::VECOUT` `` 表示生产者是 VECTOR、消费者是 MTE3，`` `EnQue` ``/`` `DeQue` `` **携带 VECTOR→MTE3 的流水同步语义**，故 SIMT VF（`` `asc_vf_call` ``）写 UB 后**无需再显式 `SetFlag<HardEvent::V_MTE3>`/`WaitFlag`**。
- 双缓冲（depth 2）下 `` `AllocTensor` `` 与上一批 `` `FreeTensor` `` 成对，使 SIMT VF 计数与 MTE3 归并重叠，减少批量重扫的串行等待。

## 3 待验证项与验证义务

| 待验证项 | 原因 | 验证义务 |
|---------|------|---------|
| `PREFIX_MINMAX_REDUCE` 的落地 | ✔ 已实测 | 「每线程 `` `asc_atomic_min/max(__gm__ float*)` `` 归约到 workspace 暂存（清 0 段初始化 +min/-max）」已上板实测通过（histogram 算子默认 range 分支，CANN 9.2.0 + Ascend950，含多核跨核收敛）。优化可改用 warp 归约（§2.7）+ 每核一次原子 |
| 二维/多维直方图的落地 | `simt_histogram_story` 样例仅覆盖 1D；D 特征读入、展平原子累加与展平分批 DMA 归并无样例直接覆盖 | 依 [kernel-skeleton.md](kernel-skeleton.md) §6 骨架落地后，上板验证展平 index 与分批归并；离散类标轴无 range/边界计算，越界类标丢弃 |
| 局部直方图模板的 SIMD DMA 归并（`` `LOCAL_HIST_FLUSH` ``）落地 | `simt_histogram_story` 仅覆盖纯 SIMT（GM 直接原子）；`SetAtomicAdd`+`DataCopyPad`+`SetAtomicNone` 的 UB→GM 原子归并只有 `three_interpolate_backward` 等 SIMD 样例间接覆盖，直方图混合形态（SIMT 计数 + SIMD 归并）无样例直接覆盖 | 依 [kernel-skeleton.md](kernel-skeleton.md) §2 骨架落地后，上板验证：UB 分批 bins 计数、SIMT VF 计数结束到 DMA 归并之间的可见性栅栏、跨核原子累加正确性 |
| DCache ≥ 32KB 的配置入口 | 已解：样例 `add_example_tiling.cpp` 有显式 DCache 配置（`DCACHE_SIZE=32*1024`，L36/L198） | 混合 VF 下 DCache 由 host 侧 TilingFunc 经 `SetLocalMemorySize(ubSize − DCACHE_SIZE)` 预留；**必须传 `ubSize − DCACHE_SIZE`，禁止 `SetLocalMemorySize(0)`**（置 0 = DCache 预留 0，`__ubuf__` 直方图与 DCache 抢占片上空间有越界风险） |
| 自定义边界二分 `lower_bound` 的向量化/性能 | 样例未覆盖 CustomEdges | 逐线程标量二分，上板验证性能 |
| 双输出 `bin_edges` 的 host 生成 | 样例只输出 hist | bin_edges 由 host 按 linspace/自定义边界生成，非 kernel 职责 |
| 混合 VF 下 `GET_TILING_DATA_WITH_STRUCT` 的静态头文件 | 该宏为 codegen 注入（`asc_op_compiler/get_op_tiling.py` 生成），非静态头文件符号 | 按 `examples/add_example/op_kernel/add_example_apt.cpp` 用法落地，上板确认 tiling 解析正确 |

## 4 混合 SIMT VF API 真值

> 混合 SIMT VF = regbase 工程框架 + SIMT VF 内核，是本范式唯一形态。下表把该形态的 API 映射到具体实体与依据；实体以 CANN 9.2.0-beta.1 头文件与 `examples/add_example/` 源码为准，未能溯源者标「待验证」。

| API / 构造 | 侧 | 语义 | 实体与依据 |
|------|----|------|-----------|
| `__simt_vf__` | 设备 | SIMT VF 函数标记（编译器关键字，非头文件符号） | `add_example.h` L29 `__simt_vf__ __aicore__ __launch_bounds__(512)` |
| `asc_vf_call<F>(dim3, args...)` | 设备 | 启动 SIMT VF；仅 `__NPU_ARCH__==3510/5102` 且混合编译路径可用 | `simt_api/common_functions.h` L36-42（`cce::async_invoke<funcPtr>`） |
| `dim3` | 设备 | 线程块维度类型 | `simt_api/common_functions.h` L33（`using dim3 = cce::dim3`） |
| `AscendC::Simt::VF_CALL` / `Dim3` | 设备 | 与 `asc_vf_call`/`dim3` 等价的辅助命名空间形式 | `simt_api/cpp/kernel_simt_utils.h` L21-28 |
| `GetThreadIdx/GetThreadNum/GetBlockIdx/GetBlockNum/GetWarpSize` | 设备 | CUDA 内建的辅助函数形式（等价 `threadIdx/blockIdx`） | `simt_api/cpp/kernel_simt_common_intf.h` L26-36 |
| `__launch_bounds__(N)` | 设备 | 每 block 线程数上限，锁定每线程寄存器 | `add_example.h` L29（`__launch_bounds__(512)`） |
| `GM_ADDR` | 设备 | regbase 入口参数类型（通用 64 位地址） | `add_example_apt.cpp` L25 |
| `REGISTER_TILING_DEFAULT(T)` | 设备 | 声明默认 tiling 结构 | `asc/impl/basic_api/utils/kernel_utils_macros.h` L385 |
| `GET_TILING_DATA_WITH_STRUCT(T, data, arg)` | 设备 | 从 tiling GM_ADDR 解析结构体（codegen 注入宏） | `python/site-packages/asc_op_compile_base/asc_op_compiler/get_op_tiling.py` L703（编译期注入，非静态头文件，标「待验证」） |
| `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)` | 设备 | 声明混合 SIMT VF 核类型（SIMT VF 访问 UB 时**必需**，缺失则 `asc_vf_call` 编译为空体 → `GetPhyAddr()` UB 越界 error 271） | `asc/impl/basic_api/utils/kernel_utils_macros.h` L376/L381、L91 |
| `g_coreType` / `AIC` | 设备 | split-core 子核判断，`if (g_coreType == AIC) { return; }` 让 AIC 子核早退（与上条配套） | `asc/include/adv_api/activation/reglu.h` L51 等 |
| `AscendC::TPipe` | 设备 | TPipe 类实例，**必须显式构造**后 `InitBuffer`/`FetchEventID`/`GetTPipePtr()` 才有效（构造函数 `g_vecTPipePtr = this`） | `impl/basic_api/dav_3510/kernel_tpipe_impl_c310.h` L154 |
| `IMPL_OP_OPTILING(op)` | 主机 | 注册 TilingFunc/TilingParse | `include/register/op_impl_registry.h` L156（`gert::OpImplRegisterV2`，链式 `.Tiling(...)/.TilingParse<...>(...)`） |
| `SetBlockDim(n)` | 主机 | 设 grid（core 数） | `include/exe_graph/runtime/tiling_context.h` L254（deprecated → `SetSimdNumBlocks`） |
| `SetLocalMemorySize(n)` | 主机 | 设 dyn UB 预算：**必须传 `n = ubSize − DCACHE_SIZE`（预留 DCache），禁止置 0** | `include/exe_graph/runtime/tiling_context.h` L502（deprecated → `SetDynUBufSize`）；`add_example_tiling.cpp` L36/L198（`SetLocalMemorySize(ubSize − DCACHE_SIZE − STATIC_UB_ESTIMATE)` + L195 校验 `ubSize > DCACHE_SIZE + STATIC_UB_ESTIMATE`） |
| `SetTilingKey(k)` | 主机 | 设 tiling key（模板分派） | `include/exe_graph/runtime/tiling_context.h` L202 |
| `ASCENDC_TPL_*`（`ARGS_DECL/UINT_DECL/UI_LIST/ARGS_SEL/UINT_SEL/SEL`） | 主机 | tiling key 模板声明/选择宏 | `asc/include/utils/tiling/template_argument.h` L67-95、L259-275 |
| `GET_TPL_TILING_KEY(...)` | 主机 | 由模板实参计算 tiling key | `asc/include/utils/tiling/template_argument.h` L287 |
| `OpDef` / `OP_ADD` / `AICore().AddConfig(...)` | 主机 | 算子定义注册 | `include/register/op_def_registry.h`；`add_example_def.cpp` |

**关键约束**：

- **混合 SIMT VF + TQue 的两条硬性要求（缺一不可，否则 error 271）**：(1) regbase 入口函数体开头 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);` + `if (g_coreType == AIC) { return; }`；(2) 用 TQue/InitBuffer/FetchEventID 前**显式构造 `AscendC::TPipe`**（`GetTPipePtr()` 返回的全局指针仅由 TPipe 构造赋值）。
- `asc_vf_call`（及 `AscendC::Simt::VF_CALL`）仅在 `__NPU_ARCH__==3510 || 5102` 的混合编译路径（`!__NPU_COMPILER_INTERNAL_PURE_SIMT__`）可用（`common_functions.h` L32-44）。当前环境 Ascend950PR 即 `__NPU_ARCH__ == 3510`（arch35）。
- `SetBlockDim`/`SetLocalMemorySize` 已标 deprecated，新代码建议 `SetSimdNumBlocks`/`SetDynUBufSize`；`add_example` 沿用旧接口，语义等价。
- `GET_TILING_DATA_WITH_STRUCT` 是 codegen 注入宏（非静态头文件），随 `arch35` 算子编译注入，源码不可见，按「待验证」级对待，使用方式以 `add_example_apt.cpp` 为准。
- 混合 VF 的 `__ubuf__` 局部直方图、`asc_atomic_add`、`asc_syncthreads()` 与 §1/§2 SIMT 内核能力真值**完全一致**，不重复列出。
- 局部直方图模板的归并段（`` `LOCAL_HIST_FLUSH` ``）使用 SIMD `basic_api` 符号（`SetAtomicAdd`/`DataCopyPad`/`SetAtomicNone`，见 §2.8），经 `kernel_operator.h` 在 regbase `__aicore__` 上下文可用，非 SIMT VF 内符号。
