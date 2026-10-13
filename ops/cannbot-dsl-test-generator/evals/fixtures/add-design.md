# add 设计

## Algorithm-Add

公开接口为 `run(x, y)`，输出 `z`。两个输入必须同形状、同为 float32，逐元素计算 `z = x + y`；形状不一致在计算前抛出 ValueError，错误语义为 shape_mismatch。

## Stage-Elementwise

独立 CPU 参考使用两个输入的加法，输出类型和形状与输入一致。NaN 按加法传播。比较使用规格中 float32 的 rtol、atol 和 metric，不调整阈值。
