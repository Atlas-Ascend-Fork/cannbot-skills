# 分核策略进阶

## 1. 双任务分核取 max（两阶段并行度需求不同）

很多算子有两个阶段的并行度需求（如"输出清零"+"数据累加"、"argmax 前向"+"梯度回传"、"默认值填充"+"稀疏写"），tiling 分别计算各阶段核数后取最大值作为 blockDim：

```cpp
formerLength_ = CeilDiv(totalLength_, coreNum_);          // 阶段1: 输入侧均分
needXCoreNum_ = CeilDiv(totalLength_, formerLength_);
clearYFactor_ = CeilDiv(bins_, coreNum_);                 // 阶段2: 清零侧独立分核
clearYCoreNum_ = CeilDiv(bins_, clearYFactor_);
needCoreNum_ = std::max(needXCoreNum_, clearYCoreNum_);   // 取 max
```

## 2. 线程数自适应收缩（防"核多用不满"）

SIMT 每核线程数不是固定的。当 `CeilDiv(workSize, threadNum) < aivNum/2`（核数利用不足一半）时逐次减半线程数，保证每个线程分到足够元素：

```cpp
uint32_t threadNum = MAX_THREAD_NUM /*2048*/;
while (threadNum >= 2 * MIN_THREAD && CeilDiv(ySize, threadNum) < aivNum / 2) {
    threadNum /= 2;                       // 每线程分不到足够元素就折半
}
perCoreElements = AlignUp(CeilDiv(ySize, needCoreNum), threadNum);   // 对齐到线程数整数倍
```

> 收缩后的 `threadNum` 作为运行时 `Dim3` 下发，**不受 `constexpr` 限制**；VF 侧 `LAUNCH_BOUND` 仍取编译期上限（如 2048），保证寄存器分配不超限。

## 3. 按"最大工作量子循环"估算核数

单线程内包含多个工作量悬殊的阶段（如帧内清零/中心力/邻居力），取三者最大值作为工作量基准，防止按单一阶段估核导致负载高估或低估：

```cpp
int64_t maxWorkload = max({initTotal, centerTotal, neighborTotal});
int64_t perCoreElements = std::max(CeilDiv(maxWorkload, aivCoreNum), PER_CORE_MIN);
int32_t needCoreNum = CeilDiv(maxWorkload, perCoreElements);
```

## 4. 平台参数动态化

核数与 UB 大小必须来自平台 API，禁止硬编码（如直接写 `48` / `256*1024`）：

```cpp
uint32_t aivNum = platformAsc.GetCoreNumAiv();                 // 物理 AIV 核数
uint64_t ubSize = 0;
platformAsc.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
```

## 选择指南

| 场景 | 策略 |
|------|------|
| 通用逐元素 / 小 shape | 见 [guide.md](guide.md)「核数切分」|
| 两阶段算子（清零+累加） | 两套分核取 max |
| 核数利用不足 | 线程数自适应减半 |
| 单线程多阶段工作量悬殊 | 按最大工作量基准估核 |
| 任意场景 | 核数/UB 取自平台 API，禁硬编码 |