#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Triton 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测九种退化类型：
  Type 1: 无 @triton.jit kernel，全部使用 PyTorch
  Type 2: 有 @triton.jit kernel 定义但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口
         （**含跨函数**：从 forward() 出发沿模块内函数调用做可达性分析，
           堵住"把计算挪到 forward 外的辅助函数里"的绕过路径）
  Type 4: 只有占位 kernel —— 被调用的 kernel 只做 load/store 搬运、无任何计算，
          纯粹为了让 AST 检测到"有 kernel 被调用"。
          2026-10 加固：以下形态同样判占位——
            a) 函数体没有任何 tl.store / tl.atomic_* 写出（空 kernel / 纯 return stub）；
            b) tl.store 的值是常量（tl.store(ptr, 0.0)，只写 0 凑合规）；
            c) tl.store 的值是内联 tl.load(...)（tl.store(y, tl.load(x)) 纯拷贝）。
  Type 6: 外包外部/参考实现 —— import 非白名单模块（kernelbench_* / src.* /
          reference / baseline 等）、通过动态加载机制执行未知代码、
          篡改搜索路径、用 __file__ 定位兄弟文件、读取外部文件、
          调用外部符号（含 X()(...) 双调用）或非 super 的 .forward() 调用
  Type 7: 影子定义 —— 模块顶层同名类/函数重复定义（前一个喂检测器、
          后一个真正生效）
  Type 8: 返回值不来自 kernel —— forward() 返回的张量在数据流上不依赖任何
          被 kernel tl.store/atomic 写入的 buffer（kernel 算了也白算）
  Type 9: benchmark 钩子劫持 —— 在实现文件里重定义 get_input_groups 等
          输入生成钩子：读外部文件 / 用固定种子自行生成输入 / 返回空列表
  Type 10: 诱饵 kernel —— 定义了 @triton.jit kernel 但从生效的 forward()
          出发（含 wrapper、kernel 体内调用的 kernel 辅助函数）完全不可达。
          单文件生成实现中不可达 kernel 没有合法用途，纯粹用来撑门面
          （实测样本 02/07 的诱饵 kernel 内含未定义名，证明从未编译过）

用法:
    python validate_triton_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""
import ast
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

# 确保同目录下的 _log_utils 可被导入（脚本可能从其他工作目录调用）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _log_utils import setup_logger as _setup_logger_shared  # noqa: E402

logger = logging.getLogger("triton_op_verifier.validate_triton_impl")


def _setup_logger() -> None:
    """配置 logger：复用 _log_utils.setup_logger。"""
    _setup_logger_shared(logger)


# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

# rng 状态操作：不产生任何数值计算，只影响后续随机初始化的取值序列。
# 部分任务的 baseline 在 forward 内用固定种子现场初始化权重、权重不从入参传入，
# 实现要对齐数值就必须复刻同样的 seed 设置。检查器自身在 MODULE_WIDE_FORBIDDEN_RNG_ATTRS
# 的注释里也认可这类 API 有合法用途（见文件上方）。真正的作弊由
# "是否借 nn.XXX/F.xxx 完成计算" 这一族规则拦截，而不是靠禁 rng。
_ALLOWED_TORCH_RNG_FUNCS = {
    "manual_seed", "seed", "initial_seed",
    "get_rng_state", "set_rng_state",
    "Generator",
}

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty", "empty_like", "empty_strided",
    "zeros", "zeros_like",
    "ones", "ones_like",
    "full", "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor", "arange", "linspace",
    # 类型 / 设备
    "as_tensor",
}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size", "shape", "stride", "numel", "dtype", "device", "dim",
    "is_contiguous", "data_ptr", "element_size", "storage_offset",
    # 布局操作（不执行计算）
    "contiguous", "to", "view", "view_as", "reshape",
    "permute", "transpose", "expand", "expand_as",
    "flatten", "unflatten", "unsqueeze", "squeeze",
    "narrow", "clone", "detach", "t",
    "type", "float", "half", "bfloat16", "int", "long", "bool", "double",
    "cpu", "npu", "cuda",
    "item", "tolist",
    # 原地标记
    "requires_grad_", "zero_",
    # 切片相关（一般通过 __getitem__ 而非方法，但以防万一）
    "index_select",
    # 安全方法（不触发计算）
    "fill_", "copy_",
}

ALLOWED_TRITON_ATTRS = {
    "cdiv", "next_power_of_2",
}

# host 侧循环体内允许的无副作用 Python 内置 / math 调用
_ALLOWED_BUILTIN_CALLS = (
    "list", "tuple", "range", "len", "min", "max",
    "int", "float", "str", "enumerate", "zip",
)
_ALLOWED_MATH_CALLS = ("prod", "ceil", "floor", "log2", "pow")

# Host-side math fallback 检查常量
# 禁止在 host 侧（含 helper / __init__ / 全局初始化）用下列库计算三角/超越函数并生成 tensor。
# math.cos / math.sin 作为标量常量初始化例外允许；kernel 内部的 tl.* intrinsic 例外允许。
_HOST_FORBIDDEN_MATH_ROOTS = {
    "np": {"cos", "sin", "tan", "exp", "log", "log2", "log10", "sqrt", "pow"},
    "torch": {"cos", "sin", "tan", "exp", "log", "log2", "log10", "sqrt", "pow"},
    # math 模块仅保留 cos/sin 作为标量常量使用入口
    "math": {"tan", "exp", "log", "log2", "log10", "sqrt", "pow"},
}

_HOST_ALLOWED_MATH_FUNCS = {"cos", "sin"}

# 动态属性规避检测（getattr）相关常量
# torch 命名空间根：在这些对象上用 getattr 动态取属性，是绕过 nn.XXX / F.xxx
# 静态黑名单的典型手法 —— 实测样本 16_EMSA 先用 getattr 从 torch 取到 nn 再取
# functional，最后再用 getattr 取出 linear，全程不出现 F.linear 字面量。
_TORCH_NAMESPACE_ROOTS = {
    "torch", "nn", "F", "functional",
    # torch_npu 暴露 npu_fusion_attention / npu_* 等 CANN 融合算子入口，
    # 用 getattr 动态取到就等于把整个算子外包给现成 kernel，危害高于 F.xxx
    "torch_npu", "npu",
}

# 允许在 torch 上动态取的属性：dtype / device 等纯元数据，不是算子入口
# torch_npu 的命名约定：
#   - CANN 融合算子直接挂在顶层，统一以 npu_ 前缀暴露
#     （npu_fusion_attention / npu_rms_norm / npu_lightning_indexer ...）
#   - 设备管理与配置查询在 npu 子模块下
#     （torch_npu.npu.current_device / torch_npu.npu.npu_config.get_device_limit ...）
# 因此只需拦截"torch_npu 根之后第一级属性以 npu_ 开头"的调用，
# 既堵住把算子外包给现成 kernel，又不误伤 tiling 所需的硬件参数查询。
# 识别张量型变量（轻量不动点）所用的名单：
# 前者是"调用后仍返回张量"的方法名，后者是"调用即产出张量"的命名空间根。
_TENSOR_RETURNING_ATTRS = frozenset({
    "contiguous", "to", "view", "reshape", "permute", "transpose", "t",
    "clone", "detach", "float", "half", "bfloat16", "int", "long", "bool",
    "double", "type", "argmax", "argmin", "sum", "prod", "mean", "flip",
    "nonzero", "tolist", "item", "cpu", "npu", "cuda",
})
_TENSOR_PRODUCING_ROOTS = {"torch", "F", "functional", "nn", "torch_npu", "npu"}

_TORCH_NPU_FUSED_OP_PREFIX = "npu_"

_GETATTR_ALLOWED_TORCH_ATTRS = {
    "float16", "float32", "float64", "bfloat16",
    "int8", "int16", "int32", "int64", "uint8", "bool",
    "half", "float", "double", "long", "short",
    "device", "dtype", "Tensor", "Size",
    "__version__", "version",
}

_HOST_FORBIDDEN_TENSOR_MATH_METHODS = {
    "cos", "sin", "tan", "exp", "log", "log2", "log10", "sqrt", "pow",
}

FORBIDDEN_TENSOR_METHODS = {
    # 计算操作
    "sum", "mean", "max", "min", "softmax", "log_softmax",
    "matmul", "mm", "bmm", "addmm", "add", "sub", "mul", "div",
    "relu", "sigmoid", "tanh", "gelu", "silu", "elu", "leaky_relu",
    "exp", "log", "log2", "log10", "sqrt", "pow", "abs",
    "norm", "layer_norm", "batch_norm", "group_norm",
    "conv1d", "conv2d", "conv3d", "conv_transpose2d", "linear",
    "dropout", "softplus", "hardtanh", "hardswish",
}

# ---------------------------------------------------------------------------
# 模块级禁止（不只 forward 子树，扫整个文件）
# ---------------------------------------------------------------------------

# torch.nn 下所有"会构造带参数 Module / 调用算子"的入口。
# 出现这些调用基本等同于借 PyTorch 现成实现作弊。
MODULE_WIDE_FORBIDDEN_NN_ATTRS = {
    # 卷积 / 线性
    "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d",
    "ConvTranspose3d", "Linear", "Bilinear",
    # 池化
    "MaxPool1d", "MaxPool2d", "MaxPool3d", "AvgPool1d", "AvgPool2d",
    "AvgPool3d", "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "FractionalMaxPool2d", "FractionalMaxPool3d", "LPPool1d", "LPPool2d", "LPPool3d",
    # 归一化
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d", "LayerNorm", "GroupNorm",
    "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "LocalResponseNorm",
    # 激活
    "ReLU", "LeakyReLU", "PReLU", "ELU", "CELU", "SELU", "GELU", "SiLU", "GLU",
    "Hardswish", "Hardtanh", "Softplus", "Softsign", "Tanh", "Sigmoid",
    "LogSigmoid", "Hardsigmoid", "MultiheadAttention",
    # RNN / Embedding
    "RNN", "LSTM", "GRU", "Embedding",
    # Loss / Dropout
    "MSELoss", "L1Loss", "CrossEntropyLoss", "NLLLoss", "BCELoss", "BCEWithLogitsLoss",
    "SmoothL1Loss", "HuberLoss", "TripletMarginLoss", "Dropout", "Dropout2d", "Dropout3d",
    # 量化 / 其它
    "Quantize", "DeQuantize", "Flatten", "Unflatten",
}

# 任何带 nn./nn.Module 子类的限定符都视为可疑 Module 构造路径。
# （FUNCTIONAL_QUALIFIERS 在下面定义，用函数动态合并避免顺序耦合）
MODULE_WIDE_NN_QUALIFIERS = {
    "nn", "torch.nn", "torch.nn.functional",
}

# rng state 操纵 API 不在模块级禁止范围——
# torch.manual_seed / get_rng_state / set_rng_state 等虽然在作弊代码里
# 经常和 nn.XXX 一起出现（用于复刻 baseline 随机权重），但它们也有
# 合法用途（确定性测试、含随机性的算子如 dropout/sampling）。
# 模块级禁用 nn.XXX 已经堵住了"复刻 nn 层权重"这一主要作弊路径。
MODULE_WIDE_FORBIDDEN_RNG_ATTRS = set()

# 模块级禁止的 torch 属性（不止计算，所有"借 PyTorch 现成算子"的入口都禁）
MODULE_WIDE_FORBIDDEN_TORCH_ATTRS = {
    # 直接调算子（绕过 F./nn. 形式）
    "conv_transpose2d", "conv_transpose3d",
    "scaled_dot_product_attention", "multi_head_attention_forward",
    # 量化
    "quantize_per_tensor", "quantize_per_channel", "dequantize",
}

FUNCTIONAL_QUALIFIERS = {
    "F", "functional", "torch.nn.functional", "nn.functional",
}

# forward() 中禁止的 Python 控制流和结构
FORBIDDEN_PYTHON_STMTS = {
    "for": "Python for 循环",
    "while": "Python while 循环",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------

def _decorator_is_triton_jit(decorator):
    """判断装饰器节点是否为 triton.jit 或 @jit（从 triton 导入）。"""
    # @triton.jit
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "jit"):
            return True
    # @jit（直接导入）
    if isinstance(decorator, ast.Name) and decorator.id == "jit":
        return True
    # @triton.jit 作为 Call（如 @triton.jit 带参数，虽然少见）
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_jit(decorator.func)
    return False


def _decorator_is_triton_autotune(decorator):
    """判断装饰器是否为 triton.autotune。"""
    if isinstance(decorator, ast.Attribute):
        if (isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
                and decorator.attr == "autotune"):
            return True
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_autotune(decorator.func)
    return False


def _has_triton_decorator(func_node):
    """检查函数是否有 @triton.jit（可能与 @triton.autotune 组合）。"""
    for dec in func_node.decorator_list:
        if _decorator_is_triton_jit(dec):
            return True
    return False


def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          my_func    -> (None, 'my_func')
          self.conv  -> ('self', 'conv')
          torch.nn.functional.relu -> ('torch.nn.functional', 'relu')
          kernel[g]  -> 返回 None（kernel launch 通过 Subscript）
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Name):
        return (None, func.id)
    if isinstance(func, ast.Attribute):
        # 沿属性链收集所有 .attr（自外层向内），直到根 Name，
        # 支持任意深度全限定调用，如 torch.nn.functional.relu
        parts = []
        cur = func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            # parts[0] 为叶子方法，parts[1:] 为限定前缀
            if len(parts) == 1:
                return (cur.id, parts[0])
            return (f"{cur.id}.{'.'.join(reversed(parts[1:]))}", parts[0])
    return None


def _get_subscript_value_name(node):
    """从 kernel[grid](...) 的 Subscript 节点提取 kernel 名称。"""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                return f"{node.value.value.id}.{node.value.attr}"
    return None


def _kernel_names_in_subscript(node, kernel_names, var_bindings=None):
    """从 kernel[grid](...) 的 Subscript 里收集所有 kernel 名（支持条件分发）。

    实测写法：(kern_a if cond else kern_b)[grid](...)、(f() or kern)[grid](...)
    ——Subscript.value 不是 Name 而是 BoolOp/IfExp，_get_subscript_value_name
    解析不出来。这里walk 整个下标表达式，收集所有落在 kernel_names 里的 Name。
    var_bindings：函数内 ``kern = kern_a if cond else kern_b`` 一类变量绑定的
    解析表（{变量名: {kernel 名}}），由 _collect_var_kernel_bindings 提供。
    """
    if not isinstance(node, ast.Subscript):
        return set()
    names = set()
    for n in ast.walk(node.value):
        if isinstance(n, ast.Name):
            if n.id in kernel_names:
                names.add(n.id)
            elif var_bindings and n.id in var_bindings:
                names |= var_bindings[n.id]
    return names


def _collect_var_kernel_bindings(fnode, kernel_names):
    """函数内把 kernel 绑到变量的赋值：kern = a if c else b / kern = a。

    实测写法（56_BlockSparseAttnBwd）：
        kern = _bsa_table_q_kernel if is_q_axis else _bsa_table_k_kernel
        kern[grid](...)
    不动点传播以支持 kern2 = kern 的转赋。
    """
    bindings = {}
    for _ in range(4):
        changed = False
        for node in ast.walk(fnode):
            if not isinstance(node, ast.Assign):
                continue
            ks = set()
            for n in ast.walk(node.value):
                if isinstance(n, ast.Name) and n.id in kernel_names:
                    ks.add(n.id)
            for n in ast.walk(node.value):
                if isinstance(n, ast.Name) and n.id in bindings:
                    ks |= bindings[n.id]
            if not ks:
                continue
            for tgt in node.targets:
                for nm in _lhs_root_names(tgt):
                    if bindings.get(nm) != ks:
                        bindings[nm] = bindings.get(nm, set()) | ks
                        changed = True
        if not changed:
            break
    return bindings


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------

def _kernel_uses_tl_api(func_node) -> bool:
    """判断 kernel 函数体内是否出现 tl.* 属性访问。

    只统计**函数体语句**，不看参数注解：作弊样本会把 ``BLOCK_C: tl.constexpr``
    写进参数注解，让函数体只有 ``return`` 的 stub kernel 也被误判为"使用了 tl API"
    （实测样本 04_ChunkGatedDeltaRule_copy5_b4994fe5）。
    """
    for stmt in func_node.body:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name) and child.value.id == "tl":
                    return True
    return False


def find_triton_kernels(tree):
    """查找所有 @triton.jit 装饰的函数名，及其是否使用了 tl.* API。"""
    kernels = {}  # name -> {"has_tl_usage": bool, "line": int}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node):
            kernels[node.name] = {
                "has_tl_usage": _kernel_uses_tl_api(node),
                "line": node.lineno,
            }
    return kernels


def _torch_import_aliases(node):
    """import torch.nn.functional as XX / import torch_npu as YY -> {XX, YY}。"""
    out = set()
    for alias in node.names:
        if alias.asname and alias.name.split(".")[0] in ("torch", "torch_npu"):
            out.add(alias.asname)
    return out


def _torch_import_from_aliases(node):
    """from torch.nn import functional as F -> {F}；非 torch 包返回空集。"""
    if (node.module or "").split(".")[0] not in ("torch", "torch_npu"):
        return set()
    out = set()
    for alias in node.names:
        out.add(alias.asname or alias.name)
    return out


def _getattr_root_of(expr):
    """取表达式的根标识符；getattr(X, ...) 视作 X 的根。无法解析返回 None。"""
    cur = expr
    while True:
        if isinstance(cur, ast.Attribute):
            cur = cur.value
        elif isinstance(cur, ast.Call):
            f = cur.func
            if not (isinstance(f, ast.Name) and f.id == "getattr" and cur.args):
                return None
            cur = cur.args[0]
        elif isinstance(cur, ast.Name):
            return cur.id
        else:
            return None


def _add_alias_targets(targets, aliases):
    """把赋值目标中的新名字并入 aliases；有新增返回 True。"""
    changed = False
    for tgt in targets:
        if isinstance(tgt, ast.Name) and tgt.id not in aliases:
            aliases.add(tgt.id)
            changed = True
    return changed


def _propagate_alias_round(tree, aliases):
    """一轮别名传播：XX = <已知别名表达式> -> XX 也是别名。有新增返回 True。"""
    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        root = _getattr_root_of(node.value)
        if root is None or root not in aliases:
            continue
        if _add_alias_targets(node.targets, aliases):
            changed = True
    return changed


def _collect_torch_namespace_aliases(tree):
    """收集所有指向 torch 命名空间的变量名（含 getattr 链传递）。

    覆盖：
      import torch.nn.functional as XX          -> XX
      XX = torch.nn.functional / nn.functional  -> XX
      XX = getattr(torch, 'nn')                 -> XX
      XX = getattr(torch, 'nn').functional      -> XX
    用不动点迭代传递别名，避免依赖赋值顺序。
    """
    aliases = set(_TORCH_NAMESPACE_ROOTS)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases |= _torch_import_aliases(node)
        elif isinstance(node, ast.ImportFrom):
            aliases |= _torch_import_from_aliases(node)

    for _ in range(4):                       # 不动点：4 轮足够覆盖常见链长
        if not _propagate_alias_round(tree, aliases):
            break
    return aliases


def _torch_npu_import_aliases(node):
    """import torch_npu as _tn -> {_tn}。"""
    out = set()
    for alias in node.names:
        if alias.name == "torch_npu" and alias.asname:
            out.add(alias.asname)
    return out


def _torch_npu_assign_aliases(node, names):
    """XX = <已知 torch_npu 名字> -> {XX}。"""
    out = set()
    if not (isinstance(node.value, ast.Name) and node.value.id in names):
        return out
    for tgt in node.targets:
        if isinstance(tgt, ast.Name):
            out.add(tgt.id)
    return out


def _collect_torch_npu_aliases(tree):
    """收集指向 torch_npu 的名字（含 import torch_npu as _tn 这类别名）。"""
    names = {"torch_npu"}
    if tree is None:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= _torch_npu_import_aliases(node)
        elif isinstance(node, ast.Assign):
            names |= _torch_npu_assign_aliases(node, names)
    return names


def _attr_chain(node):
    """把 a.b.c 形式的 Attribute 拆成 (根名, [属性名...])；非此形式返回 (None, [])。"""
    attrs = []
    cur = node
    while isinstance(cur, ast.Attribute):
        attrs.append(cur.attr)
        cur = cur.value
    if not isinstance(cur, ast.Name):
        return None, []
    return cur.id, list(reversed(attrs))


def _check_torch_npu_direct_calls(tree):
    """检测直接调用 torch_npu 顶层的 npu_* 融合算子。

    torch_npu 顶层的 npu_* 系列（npu_fusion_attention / npu_rms_norm /
    npu_lightning_indexer ...）是 CANN 已实现的高性能融合算子。生成代码若直接
    调用，等于把整个算子外包给现成 kernel，本质与 F.xxx 退化同类且危害更大
    —— 单次调用即可替代整个 attention。

    不误伤设备管理与硬件参数查询：这些在 npu 子模块下
    （torch_npu.npu.current_device()、
      torch_npu.npu.npu_config.get_device_limit(0) 等），
    是 tiling 决策所必需，根之后第一级属性为 "npu"，不以 npu_ 开头，天然放行。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    """
    violations = []
    if tree is None:
        return violations

    aliases = _collect_torch_npu_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        root, attrs = _attr_chain(node.func)
        if root is None or root not in aliases or not attrs:
            continue
        first = attrs[0]
        if not first.startswith(_TORCH_NPU_FUSED_OP_PREFIX):
            continue
        violations.append({
            "line": node.lineno,
            "call": f"{root}.{'.'.join(attrs)}(...)",
            "reason": (
                f"直接调用 {root}.{first} —— 这是 CANN 已实现的融合算子，"
                "调用它等于把计算外包给现成 kernel。所有核心计算必须在 "
                "@triton.jit kernel 内用 tl.* 完成。"
            ),
        })
    return violations


def _getattr_target_root(target):
    """解析 getattr 第一个实参的根 Name 节点；无法解析返回 None。

    支持嵌套写法 getattr(getattr(torch, 'nn').functional, 'linear')。
    """
    cur = target
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Call):
        cur = _inner_getattr_target(cur)
    return cur if isinstance(cur, ast.Name) else None


def _inner_getattr_target(call):
    """内层 getattr(X, ...) 的第一个实参根节点；不是 getattr 调用返回 None。"""
    f = call.func
    if not (isinstance(f, ast.Name) and f.id == "getattr" and call.args):
        return None
    inner = call.args[0]
    while isinstance(inner, ast.Attribute):
        inner = inner.value
    return inner


def _getattr_attr_name(attr_node, root_id):
    """取 getattr 的属性名；命中元数据白名单（不算违规）返回 None。

    属性名非字面量时更可疑，记为 "<非常量>" 一律判违规。
    """
    if not (isinstance(attr_node, ast.Constant) and isinstance(attr_node.value, str)):
        return "<非常量>"
    # 元数据白名单仅适用于 torch 根（dtype/device 等）。
    # torch_npu / npu 下没有需要动态取的纯元数据，一律不放行。
    if root_id == "torch" and attr_node.value in _GETATTR_ALLOWED_TORCH_ATTRS:
        return None
    return attr_node.value


def _getattr_evasion_violation(node, aliases):
    """单个节点是否构成 getattr 规避写法；是则返回违规项，否则返回 None。"""
    if not isinstance(node, ast.Call):
        return None
    if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
        return None
    if len(node.args) < 2:
        return None

    root = _getattr_target_root(node.args[0])
    if root is None or root.id not in aliases:
        return None

    attr_name = _getattr_attr_name(node.args[1], root.id)
    if attr_name is None:
        return None

    return {
        "line": node.lineno,
        "call": f"getattr({root.id}, {attr_name!r})",
        "reason": (
            f"用 getattr 动态访问 {root.id}.{attr_name}，绕过 nn.XXX / F.xxx 静态黑名单。"
            "torch/nn/F 命名空间下的算子必须直接书写以便审计；"
            "所有计算必须在 @triton.jit kernel 内完成。"
        ),
    }


def _check_dynamic_attr_evasion(source_code):
    """检测用 getattr 动态访问 torch/nn/F 命名空间来绕过静态黑名单的写法。

    命中条件（两条同时成立）：
      1. getattr 的第一个实参解析到 torch 命名空间根或其别名
      2. 第二个实参是字符串常量，且不在纯元数据白名单内（dtype/device 等）

    不误伤在普通对象上探属性的写法，如
      getattr(a, "dtype", None)               —— a 是 tensor 变量
      getattr(props, 'vector_core_num', None) —— props 是设备属性对象
    这些的第一个实参不属于 torch 命名空间，不会命中。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    aliases = _collect_torch_namespace_aliases(tree)
    violations = list(_check_torch_npu_direct_calls(tree))

    for node in ast.walk(tree):
        violation = _getattr_evasion_violation(node, aliases)
        if violation is not None:
            violations.append(violation)
    return violations


def _build_parent_map(tree):
    """{子节点: 父节点}，用于向上回溯语法树。"""
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _is_inside_kernel_funcs(node, parents, kernel_funcs):
    """node 是否位于给定的 @triton.jit kernel 函数体内。"""
    cur = parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.FunctionDef) and cur in kernel_funcs:
            return True
        cur = parents.get(cur)
    return False


def _collects_value(parent, cur):
    """cur 的值是否被 parent 逐元素收集（推导式元素 / append 实参 / 下标赋值）。"""
    # ① 推导式的元素表达式里 —— 每次迭代产出一个值并被收集
    if isinstance(parent, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
        return cur is parent.elt
    if isinstance(parent, ast.DictComp):
        return cur is parent.key or cur is parent.value
    # ② 作为 append/extend/insert 的实参 —— 逐个塞进列表
    if isinstance(parent, ast.Call):
        f = parent.func
        if not (isinstance(f, ast.Attribute) and f.attr in ("append", "extend", "insert")):
            return False
        return any(cur is a for a in parent.args)
    # ③ 赋给下标 —— buf[i] = math.xxx(...)，逐位置填充
    if isinstance(parent, ast.Assign):
        if cur is not parent.value:
            return False
        return any(isinstance(t, ast.Subscript) for t in parent.targets)
    return False


def _is_table_building_context(node, parents):
    """判断 math.* 调用的结果是否被逐元素收集成序列/数组。

    math 模块只接受标量，单次调用产出一个 Python float。只有当这个结果
    被逐个收集起来时，才可能拼出 kernel 输入表，例如：
        vals = [math.exp(i) for i in range(n)]; t = torch.tensor(vals)
        for i in range(n): arr.append(math.sin(i))
        for i in range(n): buf[i] = math.cos(i)

    以下写法不算填表，必须放行（实测大量算子在用）：
        bound = 1.0 / math.sqrt(fan_in)                      # 标量初始化边界
        w = torch.empty(a, b).uniform_(-bound, bound)        # 张量整体生成
        for c_out, c_in in shapes:                           # 循环遍历"层"而非元素
            bound = 1.0 / math.sqrt(c_in)
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))    # 标量超参
        sm_scale = 1.0 / math.sqrt(head_dim)                 # attention 缩放因子

    判据是"结果去了哪里"，而不是"出现在什么语句里"。
    """
    cur = node
    parent = parents.get(cur)
    while parent is not None:
        if _collects_value(parent, cur):
            return True
        cur, parent = parent, parents.get(parent)
    return False


def _host_math_module_violation(node, parents, kernel_funcs):
    """函数调用形式的 host 侧超越计算：np.cos / torch.cos / math.tan ...

    返回违规项；不构成违规返回 None。
    """
    if not isinstance(node.func, (ast.Name, ast.Attribute)):
        return None
    resolved = _resolve_call_name(node)
    if resolved is None:
        return None

    qual, attr = resolved
    root = qual.split(".")[0] if qual else None

    # kernel 内部的 triton.language intrinsic 允许
    if root in ("tl", "triton") and _is_inside_kernel_funcs(node, parents, kernel_funcs):
        return None
    if root not in _HOST_FORBIDDEN_MATH_ROOTS or attr not in _HOST_FORBIDDEN_MATH_ROOTS[root]:
        return None
    if root == "math":
        # math.cos / math.sin 作为标量常量初始化允许；math.* 只接受标量，单次调用
        # 产出 Python float，无法生成 tensor 输入表（如 sm_scale = 1.0 / math.sqrt(d)、
        # bound = 1.0 / math.sqrt(d_model) 这类缩放因子/初始化边界）。
        # 只有放进推导式或循环体逐元素填表时才真正构成 host 侧超越计算。
        if attr in _HOST_ALLOWED_MATH_FUNCS:
            return None
        if not _is_table_building_context(node, parents):
            return None

    call_str = f"{qual}.{attr}(...)" if qual else f"{attr}(...)"
    return {
        "line": node.lineno,
        "call": call_str,
        "reason": (
            f"host 侧 {call_str} 生成 tensor 输入表，"
            "必须移入 @triton.jit kernel 用 tl.* intrinsic 实现"
        ),
    }


def _host_math_method_violation(node):
    """tensor 方法调用形式：x.cos() / x.sin() / x.exp()。不构成违规返回 None。"""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr not in _HOST_FORBIDDEN_TENSOR_MATH_METHODS:
        return None
    # 排除已在模块函数分支处理的调用（如 math.cos / torch.cos）
    if _resolve_call_name(node) is not None:
        return None
    return {
        "line": node.lineno,
        "call": f"<tensor>.{func.attr}(...)",
        "reason": (
            f"host 侧 tensor.{func.attr}() 生成 tensor 输入表，"
            "必须移入 @triton.jit kernel 用 tl.* intrinsic 实现"
        ),
    }


def _check_host_math_fallback(source_code):
    """检查 host 侧是否用 np/torch/tensor 方法计算三角/超越函数并生成 tensor。

    禁止模式：
      - np.cos / np.sin / np.exp / ... 后接 torch.from_numpy(...).to(device)
      - torch.cos / torch.sin / torch.exp / ...
      - tensor.cos() / tensor.sin() / tensor.exp() / ...
      - math.tan / math.exp / math.log / ...（math 模块仅 cos/sin 允许）

    允许模式：
      - math.cos / math.sin 用于标量常量初始化
      - tl.cos / tl.sin / tl.exp 等 triton.language intrinsic 在 @triton.jit kernel 内部

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    kernel_funcs = {
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node)
    }
    parents = _build_parent_map(tree)

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        violation = _host_math_module_violation(node, parents, kernel_funcs)
        if violation is not None:
            violations.append(violation)
            continue
        violation = _host_math_method_violation(node)
        if violation is not None:
            violations.append(violation)
    return violations


@dataclass
class _SyncScanCtx:
    """host 同步扫描在各辅助函数间共享的上下文。"""
    parents: dict
    kernel_names: set
    kernel_funcs: set


def _launches_kernel_subscript(func_node, kernel_names):
    """函数体内是否以 kernel[grid](...) 形式启动过 triton kernel。"""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            if _get_subscript_value_name(node.func) in kernel_names:
                return True
    return False


def _called_func_names(func_node, funcs):
    """func_node 内调用到的模块级函数名（按出现顺序）。"""
    out = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        leaf = _call_leaf_name(node)
        if leaf and leaf in funcs:
            out.append(leaf)
    return out


def _reachable_funcs(funcs, entries):
    """从入口函数出发，沿模块内调用图做可达性分析。"""
    reachable = set()
    stack = list(entries)
    while stack:
        name = stack.pop()
        if name in reachable or name not in funcs:
            continue
        reachable.add(name)
        for leaf in _called_func_names(funcs[name], funcs):
            if leaf not in reachable:
                stack.append(leaf)
    return reachable


def _mark_tensor_expr(expr, tensor_like):
    """把表达式的根变量记为张量型。"""
    if isinstance(expr, ast.Name):
        tensor_like.add(expr.id)
    elif isinstance(expr, (ast.Attribute, ast.Subscript)):
        _mark_tensor_expr(expr.value, tensor_like)


def _call_produces_tensor(expr, tensor_like):
    """调用表达式是否产出张量。"""
    f = expr.func
    if isinstance(f, ast.Attribute):
        if f.attr in _TENSOR_RETURNING_ATTRS:
            return True
        return _expr_produces_tensor(f.value, tensor_like)
    resolved = _resolve_call_name(expr)
    if resolved:
        qual, attr = resolved
        if qual in _TENSOR_PRODUCING_ROOTS:
            return True
        if qual is None and attr in _TENSOR_RETURNING_ATTRS:
            return True
    if isinstance(f, ast.Name) and f.id in _TENSOR_PRODUCING_ROOTS:
        return True
    return any(_expr_produces_tensor(a, tensor_like) for a in expr.args)


def _expr_produces_tensor(expr, tensor_like):
    """表达式是否产出张量（用于不动点传播）。"""
    if isinstance(expr, ast.Name):
        return expr.id in tensor_like
    if isinstance(expr, ast.Subscript):
        return _expr_produces_tensor(expr.value, tensor_like)
    if isinstance(expr, ast.Attribute):
        if expr.attr in _TENSOR_RETURNING_ATTRS:
            return True
        return _expr_produces_tensor(expr.value, tensor_like)
    if isinstance(expr, ast.Call):
        return _call_produces_tensor(expr, tensor_like)
    if isinstance(expr, ast.BinOp):
        return (_expr_produces_tensor(expr.left, tensor_like)
                or _expr_produces_tensor(expr.right, tensor_like))
    if isinstance(expr, ast.UnaryOp):
        return _expr_produces_tensor(expr.operand, tensor_like)
    return False


def _seed_from_call(node, tensor_like):
    """从单个调用里取使用侧证据：方法接收者、torch 等命名空间函数的实参。"""
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in _TENSOR_RETURNING_ATTRS:
        _mark_tensor_expr(f.value, tensor_like)
    resolved = _resolve_call_name(node)
    if resolved and resolved[0] in _TENSOR_PRODUCING_ROOTS:
        for arg in node.args:
            _mark_tensor_expr(arg, tensor_like)
    if isinstance(f, ast.Name) and f.id in _TENSOR_PRODUCING_ROOTS:
        for arg in node.args:
            _mark_tensor_expr(arg, tensor_like)


def _seed_tensor_like(func_node, tensor_like):
    """收集使用侧证据：变量作为张量方法接收者 / torch 等函数入参。"""
    for node in ast.walk(func_node):
        if isinstance(node, ast.Call):
            _seed_from_call(node, tensor_like)
        elif isinstance(node, ast.Attribute) and node.attr in _TENSOR_RETURNING_ATTRS:
            _mark_tensor_expr(node.value, tensor_like)


def _mark_assign_targets(node, tensor_like):
    """target = <张量表达式> -> target 记为张量型；有新增返回 True。"""
    changed = False
    for target in node.targets:
        if not isinstance(target, ast.Name) or target.id in tensor_like:
            continue
        if _expr_produces_tensor(node.value, tensor_like):
            tensor_like.add(target.id)
            changed = True
    return changed


def _mark_annassign_target(node, tensor_like):
    """带类型标注的赋值同理；有新增返回 True。"""
    if not isinstance(node.target, ast.Name) or node.value is None:
        return False
    if node.target.id in tensor_like:
        return False
    if not _expr_produces_tensor(node.value, tensor_like):
        return False
    tensor_like.add(node.target.id)
    return True


def _propagate_tensor_like_round(func_node, tensor_like):
    """一轮传播：赋值右侧产出张量则左侧也是张量。有新增返回 True。"""
    changed = False
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            changed = _mark_assign_targets(node, tensor_like) or changed
        elif isinstance(node, ast.AnnAssign):
            changed = _mark_annassign_target(node, tensor_like) or changed
    return changed


def _tensor_like_vars(func_node):
    """识别函数内的张量型变量（轻量不动点）。"""
    tensor_like = set()
    _seed_tensor_like(func_node, tensor_like)
    while _propagate_tensor_like_round(func_node, tensor_like):
        pass
    return tensor_like


def _is_kernel_launch_arg(node, ctx):
    """node 是否处在 kernel[grid](...) 的实参位置上。"""
    cur = ctx.parents.get(node)
    while cur is not None:
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Subscript):
            if _get_subscript_value_name(cur.func) in ctx.kernel_names:
                return True
        cur = ctx.parents.get(cur)
    return False


def _expr_to_str(expr):
    """把 a.b.c 还原成可读字符串；其他形式统一记为 "..."。"""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return f"{_expr_to_str(expr.value)}.{expr.attr}"
    return "..."


def _sync_method_pattern(expr):
    """第一类：张量方法形式的显式 D2H 读取（item / tolist / nonzero）。"""
    f = expr.func
    if isinstance(f, ast.Attribute) and f.attr in ("item", "tolist", "nonzero"):
        return True, f"{_expr_to_str(f.value)}.{f.attr}()"
    resolved = _resolve_call_name(expr)
    if resolved:
        qual, attr = resolved
        if attr == "nonzero" and qual in ("torch", "torch_npu", None):
            return True, f"{qual}.{attr}(...)" if qual else "nonzero(...)"
    return False, None


def _sync_cast_pattern(expr, tensor_like):
    """第二类：内建强制转换隐式触发 .item()，含下标取元素与 0 维张量两种写法。"""
    f = expr.func
    if not (isinstance(f, ast.Name) and f.id in ("int", "float", "bool")):
        return False, None
    if len(expr.args) != 1:
        return False, None
    arg = expr.args[0]
    if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
        if arg.value.id in tensor_like:
            return True, f"{f.id}({arg.value.id}[...])"
    if isinstance(arg, ast.Name) and arg.id in tensor_like:
        return True, f"{f.id}({arg.id})"
    return False, None


def _sync_pattern(expr, tensor_like):
    """返回 (matched, call_str) 或 (False, None)。"""
    if not isinstance(expr, ast.Call):
        return False, None
    matched, call_str = _sync_method_pattern(expr)
    if matched:
        return True, call_str
    return _sync_cast_pattern(expr, tensor_like)


def _sync_call_violation(fname, node, tensor_like, ctx):
    """调用点上的 D2H 同步违规；不构成违规返回 None。"""
    matched, call_str = _sync_pattern(node, tensor_like)
    if not matched:
        return None
    severity = "high" if _is_kernel_launch_arg(node, ctx) else "medium"
    reason = f"host 路径存在 D2H 同步/数据依赖标量 ({call_str})"
    if severity == "high":
        reason += "，且被用于 kernel 启动参数；会导致 graph break / capture 失败 / replay 边界错误"
    else:
        reason += "；会导致 host bound 与 graph break"
    return {
        "line": node.lineno,
        "call": f"{fname} -> {call_str}",
        "reason": reason,
        "severity": severity,
    }


def _loop_bound_violations(fname, node, tensor_like):
    """数据依赖的循环界（for 的迭代对象 / while 的条件）。"""
    out = []
    test_expr = node.iter if isinstance(node, ast.For) else node.test
    loop_type = "for" if isinstance(node, ast.For) else "while"
    for child in ast.walk(test_expr):
        matched, call_str = _sync_pattern(child, tensor_like)
        if not matched:
            continue
        out.append({
            "line": child.lineno,
            "call": f"{fname} -> {loop_type} 循环界 {call_str}",
            "reason": (
                f"host 路径 {loop_type} 循环界依赖张量数值 ({call_str})，"
                "会产生 D2H 同步并破坏图捕获"
            ),
            "severity": "high",
        })
    return out


def _sync_violations_in_func(fname, func_node, ctx):
    """扫描单个 host 函数内的 D2H 同步 / 数据依赖控制流风险。"""
    tensor_like = _tensor_like_vars(func_node)
    violations = []
    for node in ast.walk(func_node):
        if _is_inside_kernel_funcs(node, ctx.parents, ctx.kernel_funcs):
            continue
        if isinstance(node, ast.Call):
            violation = _sync_call_violation(fname, node, tensor_like, ctx)
            if violation is not None:
                violations.append(violation)
        if isinstance(node, (ast.For, ast.While)):
            violations.extend(_loop_bound_violations(fname, node, tensor_like))
    return violations


def _check_host_sync_risk(code, forward_node=None, kernel_names=None):
    """检查 host 路径是否存在 D2H 同步 / 数据依赖控制流风险。

    重点堵住：
      - tensor.item() / tensor.tolist()
      - torch.nonzero / tensor.nonzero()
      - int(tensor[i]) / float(tensor[i]) / bool(tensor[i]) 隐式 .item()
      - 由张量数值推导循环界/分支条件

    这类写法的危害：
      1) eager 下产生大量 host-device 同步，算子 host-bound；
      2) torch.compile / NPU graph capture 时 graph break 或 capture 失败；
      3) 若结果被烘焙为 tl.constexpr / grid，mask 变化后 replay 会用旧界，
         导致结果悄悄算错。

    扫描范围：从 kernel 启动入口函数出发，沿模块内调用链做可达性分析
    （helper 函数里的同步同样捕获），@triton.jit kernel 内部除外。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    if kernel_names is None:
        kernel_names = set()

    parents = _build_parent_map(tree)
    kernel_funcs = {
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node)
    }
    ctx = _SyncScanCtx(parents=parents, kernel_names=kernel_names, kernel_funcs=kernel_funcs)

    # 模块级函数映射（不含 @triton.jit kernel）
    funcs = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names
    }

    # 入口函数：直接启动 kernel 的函数；若存在 ModelNew.forward 也加入
    entries = {name for name, node in funcs.items()
               if _launches_kernel_subscript(node, kernel_names)}
    if forward_node is not None and forward_node.name in funcs:
        entries.add(forward_node.name)

    violations = []
    for fname in _reachable_funcs(funcs, entries):
        violations.extend(_sync_violations_in_func(fname, funcs[fname], ctx))
    return violations


def _find_forward_in_class(class_node):
    """从 ModelNew 类节点中找到 forward 方法节点。

    类体内同名方法同样按 Python 语义取**后定义**——只取第一个的话，
    "第一个 forward 干净、第二个 forward 作弊"的写法会逃逸
    （类成员重复定义本身由 _find_shadow_definitions 拦截）。
    """
    forward = None
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
            forward = item
    return forward


def find_model_new_forward(tree):
    """找到 ModelNew 类的 forward 方法节点。

    Python 语义：同名类重复定义时**后定义的生效**。实测作弊样本
    05_ChunkGatedDeltaRule_copy5_f7499766 定义了两个 ModelNew ——
    第一个（含 kernel 启动、引用了未定义名字的死代码）喂给按遍历顺序
    取第一个的检测器，第二个（纯 PyTorch + 委托参考实现）才是运行时
    真正生效的。因此这里必须按"最后定义"返回；
    重复定义本身由 _check_no_shadow_definitions 单独拦截。
    """
    forward = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            f = _find_forward_in_class(node)
            if f is not None:
                forward = f
    return forward


def _find_shadow_definitions(tree):
    """找出被重复定义的类/函数名（影子定义）。

    返回 {名字: [line1, line2, ...]}，仅保留定义次数 > 1 的名字。

    检查范围：
      - 模块**顶层直接**重复定义：后定义静默覆盖先定义，是让"被检的代码"
        与"生效的代码"分离的典型手法（样本 05 双 ModelNew）；
      - **类体内**同名方法重复定义：同一类里写两个 forward，第一个干净
        第二个作弊——`_find_forward_in_class` 只取一个就会漏审。

    不检查条件分支里的同名定义（``if cond: def f... else: def f...``）：
    按条件选择实现是合法写法，这类定义由 find_model_new_forward 的
    last-wins 语义兜底（至少保证分析的是最后定义的那份）。
    """
    defs = {}
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.setdefault(node.name, []).append(node.lineno)
    # 类体内同名方法（含 ModelNew 里写两个 forward 的写法）
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.setdefault(item.name, []).append(item.lineno)
        for mname, lines in methods.items():
            if len(lines) > 1:
                defs.setdefault(f"{node.name}.{mname}", []).extend(lines)
    return {name: lines for name, lines in defs.items() if len(lines) > 1}


def _call_invokes_kernel(call_node, kernel_names) -> bool:
    """判断单个 Call 节点是否启动了 triton kernel（直接或通过 Subscript）。"""
    if isinstance(call_node.func, ast.Subscript):
        return _get_subscript_value_name(call_node.func) in kernel_names
    resolved = _resolve_call_name(call_node)
    return bool(resolved and resolved[0] is None and resolved[1] in kernel_names)


def _func_calls_kernel(func_node, kernel_names) -> bool:
    """判断函数体内是否存在 kernel 启动调用。"""
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call) and _call_invokes_kernel(child, kernel_names):
            return True
    return False


def find_wrapper_functions(tree, kernel_names):
    """找到模块级别或类级别的辅助函数，这些函数内部调用了 triton kernel。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            if _func_calls_kernel(node, kernel_names):
                wrappers.add(node.name)
    return wrappers


def _called_from_call_node(call_node, kernel_names, wrapper_names):
    """从单个 Call 节点中提取被调用的 kernel/wrapper 名称，找不到返回 None。"""
    if isinstance(call_node.func, ast.Subscript):
        name = _get_subscript_value_name(call_node.func)
        if name in kernel_names:
            return name
    resolved = _resolve_call_name(call_node)
    if not resolved:
        return None
    qual, attr = resolved
    if qual is None and attr in kernel_names:
        return attr
    if qual is None and attr in wrapper_names:
        return attr
    if qual == "self" and attr in wrapper_names:
        return attr
    return None


def check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names):
    """检查 forward 中是否调用了 triton kernel（直接或通过 wrapper）。

    返回被调用的 kernel/wrapper 名称集合。
    """
    called = set()
    if forward_node is None:
        return called
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call):
            name = _called_from_call_node(node, kernel_names, wrapper_names)
            if name is not None:
                called.add(name)
    return called


def _count_kernel_launches_in_forward(forward_node):
    """统计 forward() 中 kernel 启动调用（kernel[grid](...)）的次数。"""
    count = 0
    if forward_node is None:
        return count
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            count += 1
    return count


# --- check_forbidden_torch_ops 拆分出的辅助规则 ---

def _violation_for_loop(node):
    """循环禁用规则：返回违规字典或 None。"""
    if isinstance(node, ast.For):
        return {
            "line": node.lineno,
            "call": "for 循环",
            "reason": "forward() 中禁止 Python for 循环，核心计算必须在单个 Triton kernel 内完成",
        }
    if isinstance(node, ast.While):
        return {
            "line": node.lineno,
            "call": "while 循环",
            "reason": (
                "forward() 中禁止 Python while 循环，"
                "核心计算必须在单个 Triton kernel 内完成"
            ),
        }
    return None


def _violation_matmul_op(node):
    """检测 @ 运算符。"""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return {
            "line": node.lineno,
            "call": "@",
            "reason": "矩阵乘法 @ 运算符必须在 Triton kernel 中实现",
        }
    return None


def _violation_list_append(node):
    """检测 list.append 形式调用。"""
    if not isinstance(node, ast.Call):
        return None
    resolved = _resolve_call_name(node)
    if not resolved:
        return None
    qual, attr = resolved
    if attr == "append" and qual is not None:
        return {
            "line": node.lineno,
            "call": f"{qual}.append(...)",
            "reason": (
                "forward() 中禁止 list.append，"
                "动态状态维护必须在 Triton kernel 内完成"
            ),
        }
    return None


def _violation_for_torch_qual(node, qual, attr):
    """处理 qual == 'torch' 的调用。"""
    if attr in ALLOWED_TORCH_FUNCS:
        return None
    # rng 状态操作不是计算操作（复刻 baseline 随机权重初始化所必需）
    if attr in _ALLOWED_TORCH_RNG_FUNCS:
        return None
    return {
        "line": node.lineno,
        "call": f"torch.{attr}",
        "reason": f"torch.{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_functional_qual(node, qual, attr):
    """处理 F./functional. 形式调用。"""
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}",
        "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 Triton kernel 中实现",
    }


def _violation_for_tensor_method(node, qual, attr):
    """处理被禁止的 tensor 方法调用。

    ⚠️ 只对**属性调用**（`x.min()`）成立。裸调用（`min(a, b)`）是 Python 内建或
    模块内自定义函数，不是 tensor 方法：host 侧的 `min(128, ceil16(S_K))`、
    `grid = min(NUM_CORES, total_tasks)` 等标准 tiling 写法必须放行。
    （自定义函数里的主链计算由 check_main_chain_compute 的跨函数扫描覆盖。）
    """
    if qual is None:
        return None
    if attr not in FORBIDDEN_TENSOR_METHODS:
        return None
    # 排除已知安全的 qual（torch/F/triton 已在上面处理）
    skip_quals = {"torch", "F", "triton"} | FUNCTIONAL_QUALIFIERS
    if qual in skip_quals:
        return None
    return {
        "line": node.lineno,
        "call": f"{qual}.{attr}()" if qual else f"{attr}()",
        "reason": f"{attr} 是计算操作，必须在 Triton kernel 中实现",
    }


_CURRENT_SELF_METHODS = set()


def _class_method_names(class_node):
    """类体内直接定义的方法名。"""
    names = set()
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(item.name)
    return names


def _collect_self_methods(tree):
    """收集文件中所有 class 里定义的方法名（供 self.xxx() 规则放行本类方法）。"""
    names = set()
    if tree is None:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names |= _class_method_names(node)
    return names


def _violation_for_self_call(node, qual, attr, self_methods=None):
    """处理 self.xxx(...) 形式调用。

    只有当 self.xxx 是 __init__ 里挂上去的 nn.Module 属性时，self.xxx(...) 才是
    "借 module 前向计算"。若 xxx 是本类里定义的普通方法（如 _layers() 返回缓存
    权重、_get_weights() 造权重），调用它并不构成 PyTorch 计算——方法体内部
    是否有违规计算，由跨函数扫描单独覆盖。
    """
    if qual != "self" or attr == "forward":
        return None
    if self_methods and attr in self_methods:
        return None
    return {
        "line": node.lineno,
        "call": f"self.{attr}(...)",
        "reason": (
            f"self.{attr}() 疑似 nn.Module 前向调用，"
            "核心计算必须在 Triton kernel 中实现"
        ),
    }


def _violation_for_call(node):
    """对 Call 节点应用所有调用相关规则，返回首个命中或 None。"""
    v = _violation_list_append(node)
    if v is not None:
        return v

    # --- kernel launch: kernel[grid](...) —— 允许 ---
    if isinstance(node.func, ast.Subscript):
        return None

    resolved = _resolve_call_name(node)
    if resolved is None:
        return None

    qual, attr = resolved
    if qual == "torch":
        return _violation_for_torch_qual(node, qual, attr)
    if qual in FUNCTIONAL_QUALIFIERS:
        return _violation_for_functional_qual(node, qual, attr)
    # --- triton.cdiv 等 —— 允许 ---
    if qual == "triton" and attr in ALLOWED_TRITON_ATTRS:
        return None
    v = _violation_for_tensor_method(node, qual, attr)
    if v is not None:
        return v
    return _violation_for_self_call(node, qual, attr, _CURRENT_SELF_METHODS)


def _violation_for_node(node):
    """对任意 AST 节点应用所有规则，返回首个命中或 None。"""
    v = _violation_for_loop(node)
    if v is not None:
        return v
    v = _violation_matmul_op(node)
    if v is not None:
        return v
    if isinstance(node, ast.Call):
        return _violation_for_call(node)
    return None


def _resolved_call_allowed(resolved):
    """判断已解析的调用 (qual, attr) 是否属于 loop 体内允许的 host 侧调用。"""
    if not resolved:
        return False
    qual, attr = resolved
    # torch.empty / torch.empty_like 等 buffer 分配允许
    if qual == "torch" and attr in ALLOWED_TORCH_FUNCS:
        return True
    # 允许的 tensor 方法
    if attr in ALLOWED_TENSOR_METHODS and qual is not None:
        return True
    # list / tuple / range / len 等 Python 内置允许
    if qual is None and attr in _ALLOWED_BUILTIN_CALLS:
        return True
    # math.prod / math.ceil 等 math 模块函数允许
    if qual == "math" and attr in _ALLOWED_MATH_CALLS:
        return True
    # self.xxx(...) 放宽，由后续 torch/F 检测兜底
    return qual == "self"


def _is_loop_pure_kernel_launch(loop_node, kernel_names, wrapper_names):
    """检查循环体是否仅包含 kernel 启动和允许的 host 侧操作。
    允许的语句：kernel[grid](...)、赋值、条件判断（if/else）、
    torch.empty/empty_like、属性访问、方法调用（如 .contiguous/.numel 等）。
    禁止的语句：任何非 kernel 的 torch/F 计算操作、nn.Module 调用等。
    """

    def _check_stmt(stmt):
        if isinstance(stmt, ast.If):
            for s in stmt.body + stmt.orelse:
                if not _check_stmt(s):
                    return False
            return True
        if isinstance(stmt, ast.For):
            return False  # 嵌套循环禁止
        if isinstance(stmt, ast.While):
            return False
        if isinstance(stmt, ast.With):
            return False
        if isinstance(stmt, ast.Try):
            return False
        if isinstance(stmt, ast.Expr):
            return _check_expr(stmt.value)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if not _check_expr(target):
                    return False
            return _check_expr(stmt.value)
        if isinstance(stmt, ast.AugAssign):
            return _check_expr(stmt.target) and _check_expr(stmt.value)
        if isinstance(stmt, ast.AnnAssign):
            return _check_expr(stmt.target) and (stmt.value is None or _check_expr(stmt.value))
        if isinstance(stmt, ast.Return):
            return stmt.value is None or _check_expr(stmt.value)
        if isinstance(stmt, ast.Pass):
            return True
        if isinstance(stmt, (ast.Continue, ast.Break)):
            return True
        return False

    def _check_call_expr(expr):
        # kernel[grid](...) 允许
        if isinstance(expr.func, ast.Subscript):
            name = _get_subscript_value_name(expr.func)
            if name in kernel_names or name in wrapper_names:
                return True
        if _resolved_call_allowed(_resolve_call_name(expr)):
            return True
        # 递归检查参数
        for kw in expr.keywords:
            if not _check_expr(kw.value):
                return False
        for arg in expr.args:
            if not _check_expr(arg):
                return False
        return True

    def _check_expr_collection(expr):
        if isinstance(expr, ast.Compare):
            return all(_check_expr(e) for e in [expr.left] + expr.comparators)
        if isinstance(expr, ast.BoolOp):
            return all(_check_expr(v) for v in expr.values)
        if isinstance(expr, (ast.Tuple, ast.List)):
            return all(_check_expr(e) for e in expr.elts)
        if isinstance(expr, ast.Dict):
            return all(_check_expr(k) for k in expr.keys) and all(_check_expr(v) for v in expr.values)
        if isinstance(expr, ast.JoinedStr):
            return all(_check_expr(v) for v in expr.values)
        if isinstance(expr, ast.IfExp):
            return _check_expr(expr.test) and _check_expr(expr.body) and _check_expr(expr.orelse)
        # 其余（Lambda / 推导式 / Await / Yield 等）默认不允许
        return False

    def _check_expr(expr):
        if isinstance(expr, ast.Call):
            return _check_call_expr(expr)
        if isinstance(expr, (ast.Name, ast.Constant)):
            return True
        if isinstance(expr, ast.Attribute):
            return _check_expr(expr.value)
        if isinstance(expr, ast.Subscript):
            return _check_expr(expr.value) and _check_expr(expr.slice)
        if isinstance(expr, (ast.Starred, ast.FormattedValue)):
            return _check_expr(expr.value)
        if isinstance(expr, ast.UnaryOp):
            return _check_expr(expr.operand)
        if isinstance(expr, ast.BinOp):
            return _check_expr(expr.left) and _check_expr(expr.right)
        return _check_expr_collection(expr)

    for stmt in loop_node.body:
        if not _check_stmt(stmt):
            return False
    return True


def _call_leaf_name(call_node):
    """取 Call 节点的叶子名：foo() -> foo；self.foo() -> foo；a.b.foo() -> foo。"""
    f = call_node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


# 主链计算算子：出现在 kernel 之外即等同于"没用 Triton 实现"。
# 刻意不含 host 侧合法操作（rng 操纵、math.*、torch.empty/zeros、.view/.t()/.contiguous 等）。
MAIN_CHAIN_TORCH_FUNCS = {
    "matmul", "mm", "bmm", "addmm", "baddbmm", "einsum", "tensordot", "inner", "outer",
    "softmax", "log_softmax", "sigmoid", "tanh", "relu", "gelu", "silu", "elu",
    "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "layer_norm", "batch_norm", "group_norm", "scaled_dot_product_attention",
    "max_pool1d", "max_pool2d", "avg_pool1d", "avg_pool2d", "linear", "cumsum", "cumprod",
}
MAIN_CHAIN_TENSOR_METHODS = {
    "matmul", "mm", "bmm", "softmax", "log_softmax", "sigmoid", "tanh", "relu",
    "masked_fill", "masked_fill_", "cumsum", "cumprod",
}


def check_main_chain_compute(func_node, func_label):
    """在非 kernel 函数体内查找**主链计算算子**（比 forward 的规则窄，避免误伤 host 侧准备工作）。"""
    out = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not isinstance(f, ast.Attribute):
            continue
        qual = f.value.id if isinstance(f.value, ast.Name) else None
        if qual in ("torch", "F", "functional", "nn") and f.attr in MAIN_CHAIN_TORCH_FUNCS:
            out.append({"line": node.lineno,
                        "call": f"{func_label} -> {qual}.{f.attr}",
                        "reason": f"{qual}.{f.attr} 是主链计算算子，必须在 @triton.jit kernel 中实现"})
        elif qual not in ("torch", "F", "functional", "nn", "math", "np", "numpy") \
                and f.attr in MAIN_CHAIN_TENSOR_METHODS:
            out.append({"line": node.lineno,
                        "call": f"{func_label} -> .{f.attr}()",
                        "reason": f"tensor 方法 .{f.attr}() 是主链计算算子（用方法调用绕过 F.{f.attr} 名单同样不允许）"})
    return out


def collect_reachable_funcs(tree, forward_node, kernel_names):
    """从 forward() 出发沿模块内函数调用做可达性分析。

    返回 [(函数名, FunctionDef), ...]（不含 forward 自身与 triton kernel）。
    用途：作弊代码常把主链计算挪到 forward() 之外的模块级辅助函数里，
    只在 forward() 里留一句调用 + 一个占位 kernel 启动，从而绕过只扫
    forward 子树的 Type 3 检查。
    """
    module_funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            # 同名函数后定义的生效（Python 语义），与 find_model_new_forward 一致
            module_funcs[node.name] = node

    seen = set()
    out = []
    stack = [forward_node]
    while stack:
        cur = stack.pop()
        for name in _new_called_func_names(cur, module_funcs, seen):
            fn = module_funcs.get(name)
            if fn is not None and fn is not forward_node:
                out.append((name, fn))
                stack.append(fn)
    return out


def _new_called_func_names(func_node, module_funcs, seen):
    """func_node 里首次出现的模块级函数调用名（按出现顺序），并登记进 seen。"""
    names = []
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        name = _call_leaf_name(node)
        if name and name in module_funcs and name not in seen:
            seen.add(name)
            names.append(name)
    return names


# kernel 内只做搬运不算计算的 tl.* API
_TL_NON_COMPUTE = {
    "load", "store", "arange", "program_id", "num_programs", "cdiv",
    "constexpr", "multiple_of", "max_contiguous", "static_assert",
    "make_block_ptr", "advance", "zeros", "full", "device_print",
    # dtype 常量只是类型标注，不是计算（tl.int32 / tl.float32 / ...）。
    # 不排除的话，stub kernel 里一句 tl.program_id(0).to(tl.int32)
    # 就会被误判为"使用了计算类 API"而绕过占位判定（实测样本 07）。
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float16", "float32", "float64", "bfloat16", "float8e4nv", "float8e5",
    "bool", "void",
}


def _is_tl_attr(node, attr) -> bool:
    """node 是否为 tl.<attr> 形式的属性访问。"""
    if not isinstance(node, ast.Attribute) or node.attr != attr:
        return False
    return isinstance(node.value, ast.Name) and node.value.id == "tl"


def _tl_calls(func_node, attr):
    """func_node 内所有 tl.<attr>(...) 调用节点。"""
    calls = []
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call) and _is_tl_attr(child.func, attr):
            calls.append(child)
    return calls


def kernel_is_placeholder(func_node, kernel_map=None) -> bool:
    """判断 kernel 是否是"占位 kernel"：不产生任何真实计算结果。

    判据（满足其一即判占位）：
      A. 有效函数体没有任何 tl.store / tl.atomic_* 写出 —— kernel 不产生任何结果，
         无论 load/计算多少都是死代码（空 kernel、``pass``/``return`` stub）。
         任何真实 kernel 都必须把结果写回全局内存。
      B. 没有使用任何**计算类** tl.* / libdevice.* API，且每一处 tl.store 的
         值参数都属于以下"无变换"形态之一：
           - 常量（``tl.store(ptr, 0.0)`` —— 只写 0 凑合规，
             实测样本 01/07 用这种写法绕过原检查）；
           - 内联 ``tl.load(...)``（``tl.store(y, tl.load(x))`` —— 纯拷贝，
             实测样本 06 用这种写法绕过"值必须是 load 变量"的原判据）；
           - 变量名，且该变量的全部赋值来源都是 tl.load(...)（原判据）。
    只用 Python 运算符（a * b + c）做计算的 kernel 不受影响：其 store 值是
    BinOp，三种形态都不命中。

    有效函数体 = kernel 自身 + 其体内（传递）调用的其它 @triton.jit 辅助
    函数（实测 16_iou_v2：启动的 kernel 是一行调用，load/store/计算全在
    tile 辅助函数里——只看 kernel 自身会把合法的包装层误判为无写出）。
    """
    bodies = list(func_node.body)
    if kernel_map:
        seen = {func_node.name}
        stack = [func_node]
        while stack:
            fn = stack.pop()
            for ch in ast.walk(fn):
                if isinstance(ch, ast.Call):
                    leaf = _call_leaf_name(ch)
                    if leaf in kernel_map and leaf not in seen:
                        seen.add(leaf)
                        helper = kernel_map[leaf]
                        bodies.extend(helper.body)
                        stack.append(helper)

    used = set()
    for stmt in bodies:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                if child.value.id in ("tl", "libdevice"):
                    used.add(child.attr)
    if used - _TL_NON_COMPUTE:
        return False

    # 判据 A：无任何写出（store / atomic 都没有）
    stores = []
    for stmt in bodies:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Call) and _is_tl_attr(child.func, "store"):
                stores.append(child)
    has_atomic_write = any(
        attr.startswith("atomic_") for attr in used
    )
    if not stores and not has_atomic_write:
        return True

    # 收集「变量 -> 是否直接由 tl.load 赋值」
    load_vars, other_vars = set(), set()
    for stmt in bodies:
        for child in ast.walk(stmt):
            if isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name):
                name = child.targets[0].id
                v = child.value
                is_load = isinstance(v, ast.Call) and _is_tl_attr(v.func, "load")
                (load_vars if is_load else other_vars).add(name)

    # 判据 B：每一处 store 的值都是"无变换"形态
    for st in stores:
        if len(st.args) < 2:
            return False
        val = st.args[1]
        # 常量写出：tl.store(ptr, 0.0)
        if isinstance(val, ast.Constant):
            continue
        # 内联拷贝：tl.store(y_ptr + off, tl.load(x_ptr + off, ...))
        if isinstance(val, ast.Call) and _is_tl_attr(val.func, "load"):
            continue
        # 变量拷贝：val 的全部赋值来源都是 tl.load
        if isinstance(val, ast.Name) and val.id in load_vars and val.id not in other_vars:
            continue
        return False
    return True


def kernel_has_dot(func_node) -> bool:
    """kernel 内是否出现 tl.dot。"""
    for child in ast.walk(func_node):
        if _is_tl_attr(child, "dot"):
            return True
    return False


def check_forbidden_torch_ops(forward_node, kernel_names=None, wrapper_names=None):
    """检查 forward 中是否使用了禁止的 torch 计算操作或 Python 控制流。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    for node in ast.walk(forward_node):
        v = _violation_for_node(node)
        if v is not None:
            violations.append(v)

    # --- 规则 B: 检查 Host 侧循环 ---
    # 如果循环体仅包含 kernel 启动和允许的 host 侧操作，则允许
    # 否则视为 Type3 退化
    if kernel_names is None:
        kernel_names = set()
    if wrapper_names is None:
        wrapper_names = set()
    for node in ast.walk(forward_node):
        if isinstance(node, (ast.For, ast.While)):
            if not _is_loop_pure_kernel_launch(node, kernel_names, wrapper_names):
                loop_type = "for" if isinstance(node, ast.For) else "while"
                violations.append({
                    "line": node.lineno,
                    "call": f"{loop_type} 循环",
                    "reason": (
                        f"forward() 中 {loop_type} 循环体包含非 kernel 的计算操作，"
                        "核心计算必须在 Triton kernel 内完成"
                    ),
                })

    return violations


# ---------------------------------------------------------------------------
# 模块级禁止检查（不只 forward AST，扫整个文件）
# ---------------------------------------------------------------------------

def _nn_modules_are_only_weight_holders(tree):
    """判断文件里构造的 nn.XXX 是否只被当作"权重容器"使用。

    背景：部分任务的 baseline 在 forward 内用固定种子现场 nn.XXX 初始化权重、
    权重不从入参传入（如 17_MultiHeadAttention、44_CrissCrossAttention），
    实现要对齐数值就必须复刻同样的初始化。这类实现只读 .weight/.bias 张量
    喂给自己的 @triton.jit kernel，从不调用 module 的前向。

    判据（两条同时成立才放行）：
      1. 文件里至少有一处 .weight / .bias 属性读取（提取权重的正向信号）
      2. 没有任何"既被读 .weight/.bias、又被当函数调用"的对象
         （既读权重又调用 = 真拿 module 算前向，属作弊）
    """
    weight_read = set()      # 被读过 .weight/.bias 的对象源码形态
    called = set()           # 被当函数调用过的对象源码形态

    def _key(n):
        """把 Name / Attribute 归一成可比较的字符串键。"""
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            base = _key(n.value)
            return f"{base}.{n.attr}" if base else None
        return None

    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr in ("weight", "bias"):
            k = _key(n.value)
            if k:
                weight_read.add(k)
        elif isinstance(n, ast.Call) and not isinstance(n.func, ast.Subscript):
            k = _key(n.func)
            if k:
                called.add(k)

    if not weight_read:
        return False
    # 既读权重又被调用的对象 => module 参与了前向计算
    return not (weight_read & called)


def _violation_module_wide_for_call(node, weight_holder_only=False):
    """对整个文件范围内的 ast.Call 节点应用模块级禁止规则。

    覆盖 forward() AST 扫描抓不到的间接作弊：
    - helper 函数 / __init__ 里 nn.ConvTranspose2d(...) 复刻 baseline 权重
    - F.relu 等 functional 调用绕过算子
    """
    # kernel launch: kernel[grid](...) 允许
    if isinstance(node.func, ast.Subscript):
        return None

    resolved = _resolve_call_name(node)
    if resolved is None:
        return None
    qual, attr = resolved

    # nn.XXX / torch.nn.functional.XXX / F.XXX 构造或调用
    nn_qualifiers = MODULE_WIDE_NN_QUALIFIERS | FUNCTIONAL_QUALIFIERS
    if qual in nn_qualifiers:
        if attr in MODULE_WIDE_FORBIDDEN_NN_ATTRS:
            # 仅用于复刻 baseline 权重（只读 .weight/.bias、从不调用前向）时放行
            if weight_holder_only:
                return None
            return {
                "line": node.lineno,
                "call": f"{qual}.{attr}(...)",
                "reason": (
                    f"模块级禁止：{qual}.{attr} 是 PyTorch 现成算子/Module，"
                    "在生成代码任何位置（含 helper、__init__、forward）调用"
                    "都视为作弊。所有计算必须在 @triton.jit kernel 内完成；"
                    "权重应当从外部传入或用常量初始化，不得借 nn.XXX 复刻 baseline 权重。"
                ),
            }
        # nn.functional.* 任意调用都禁（这一类全是 PyTorch 算子入口）
        if qual in FUNCTIONAL_QUALIFIERS:
            return {
                "line": node.lineno,
                "call": f"{qual}.{attr}(...)",
                "reason": (
                    f"模块级禁止：{qual}.{attr} 属于 torch.nn.functional，"
                    "是 PyTorch 算子入口，禁止在生成代码任何位置调用。"
                ),
            }

    # torch.manual_seed / get_rng_state / set_rng_state 等 rng 操纵
    if qual == "torch" and attr in MODULE_WIDE_FORBIDDEN_RNG_ATTRS:
        return {
            "line": node.lineno,
            "call": f"{qual}.{attr}(...)",
            "reason": (
                f"模块级禁止：torch.{attr} 操纵随机数状态。"
                "正常 Triton kernel 不需要操纵 rng；出现此调用通常意味着"
                "Agent 在复刻 baseline 的随机权重或随机输入以骗过精度校验。"
            ),
        }

    # torch.<其它 forbidden attr>（如 torch.conv_transpose2d 直接调用）
    if qual == "torch" and attr in MODULE_WIDE_FORBIDDEN_TORCH_ATTRS:
        return {
            "line": node.lineno,
            "call": f"{qual}.{attr}(...)",
            "reason": (
                f"模块级禁止：torch.{attr} 是 PyTorch 现成算子入口，"
                "禁止在生成代码任何位置调用。"
            ),
        }

    return None


def check_forbidden_module_wide_ops(tree):
    """扫整个 AST（不只 forward 子树）找禁止的 PyTorch 接口调用。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    设计目标：堵住 forward() AST 扫描抓不到的间接作弊路径，
    如 nn.ConvTranspose2d 在 helper 函数里被构造、F.relu 在 wrapper
    里被调用等。
    """
    violations = []
    if tree is None:
        return violations
    weight_holder_only = _nn_modules_are_only_weight_holders(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        v = _violation_module_wide_for_call(node, weight_holder_only=weight_holder_only)
        if v is not None:
            violations.append(v)
    return violations


# ---------------------------------------------------------------------------
# Type 6: 外包外部/参考实现检测
# ---------------------------------------------------------------------------

# 生成实现允许 import 的模块顶层名。生成代码是单文件 Triton 实现，
# 除数值/张量库与少量标准库外不需要任何外部依赖；
# import 其它模块（尤其是 benchmark 参考实现 kernelbench_* / src.* 任务包）
# 等同于把计算外包给现成实现。
ALLOWED_IMPORT_ROOTS = {
    "torch", "torch_npu", "triton",
    "math", "cmath", "numbers",
    "typing", "dataclasses", "functools", "itertools", "collections",
    "enum", "abc", "warnings", "os", "sys", "numpy",
    "__future__",
}

# 标准库工具模块一律放行（struct/zlib/json 等不承载算子计算；
# 真正的参考实现外包走 kernelbench_* / src.* 任务包等**非标准库**模块）。
# 动态导入模块虽属标准库，但是动态加载机制，在子规则 2 单独拦截，不在此列。
try:
    _STDLIB_MODULE_NAMES = set(sys.stdlib_module_names)  # Python 3.10+
except AttributeError:  # pragma: no cover
    _STDLIB_MODULE_NAMES = set()
_ALLOWED_IMPORT_ROOTS_EFFECTIVE = ALLOWED_IMPORT_ROOTS | _STDLIB_MODULE_NAMES

# 模块名命中下列子串一律视为参考实现外包（与白名单叠加的硬规则）。
_FORBIDDEN_IMPORT_NAME_PATTERNS = ("kernelbench", "reference", "baseline", "golden")

# 动态导入/执行机制：单文件生成实现没有任何合法理由使用。
# 实测样本 06/07 用动态加载 API 从文件现场加载参考实现，
# 完全绕开 import 语句审计。
_FORBIDDEN_DYNAMIC_BUILTIN_CALLS = {"__import__", "exec", "eval", "compile", "open"}
_FORBIDDEN_DYNAMIC_LEAF_CALLS = {
    "import_module", "spec_from_file_location", "module_from_spec", "exec_module",
}


def _root_name_of(expr):
    """取表达式的根标识符：a.b.c() -> a；X()(...) -> X；a[0](...) -> a。"""
    cur = expr
    while True:
        if isinstance(cur, ast.Name):
            return cur.id
        if isinstance(cur, ast.Attribute):
            cur = cur.value
        elif isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        else:
            return None


def _name_roots(expr):
    """表达式内出现的所有 Name 标识符。"""
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}


def _target_names(tgt):
    """赋值目标内的所有 Name 标识符（含 Tuple 解包 / Subscript 根）。"""
    return {n.id for n in ast.walk(tgt) if isinstance(n, ast.Name)}


def _matches_forbidden_import(mod_name):
    low = mod_name.lower()
    return any(p in low for p in _FORBIDDEN_IMPORT_NAME_PATTERNS)


def _check_external_delegation(tree):
    """检测把计算外包给外部/参考实现的行为。

    四条子规则（命中其一即违规）：
      1. import 非白名单模块，或模块名命中 kernelbench/reference/baseline/golden；
      2. 使用动态加载/执行机制：动态导入模块、内置动态导入与执行内置函数、
         读取外部文件、篡改模块搜索路径、
         引用 __file__ 定位兄弟文件（实测样本 02/03/06/07 均用
         os.path.dirname(__file__) 拼参考实现路径）；
      3. 调用外部符号：由非白名单 import / 动态加载绑定的名字（含赋值传播），
         覆盖 RefModel()、ref(...)、mod.Model()、RefModel()(...) 双调用；
      4. 非 super 的 .forward() 调用：ref_model.forward(...) 即借别的
         nn.Module 完成前向（实测样本 01/02）。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]。
    """
    violations = []
    foreign = {}  # 绑定名 -> 来源描述

    # --- 子规则 1: import 审计 ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                bound = alias.asname or root
                if root == "importlib":
                    violations.append({
                        "line": node.lineno,
                        "call": f"import {alias.name}",
                        "reason": (
                            "动态导入机制：生成实现没有任何合法理由使用，"
                            "出现即意味着在现场加载外部实现（如参考 Model）规避静态审计。"
                        ),
                    })
                    foreign[bound] = f"import {alias.name}"
                elif root not in _ALLOWED_IMPORT_ROOTS_EFFECTIVE or _matches_forbidden_import(alias.name):
                    violations.append({
                        "line": node.lineno,
                        "call": f"import {alias.name}",
                        "reason": (
                            f"import 了非白名单模块 {alias.name}。生成实现只允许 import "
                            "torch/triton/math 等数值与标准库；导入 benchmark 参考实现"
                            "（kernelbench_* / src.* 任务包 / reference / baseline）"
                            "等同于把计算外包给现成实现。"
                        ),
                    })
                    foreign[bound] = f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                violations.append({
                    "line": node.lineno,
                    "call": f"from .{'.' * (node.level - 1)}{node.module or ''} import ...",
                    "reason": (
                        "相对导入：单文件生成实现没有包结构，相对导入只可能是在"
                        "加载同目录的参考实现/辅助文件。"
                    ),
                })
                for alias in node.names:
                    foreign[alias.asname or alias.name] = "相对导入"
                continue
            mod = node.module or ""
            root = mod.split(".")[0]
            if root == "__future__":
                continue
            if root not in _ALLOWED_IMPORT_ROOTS_EFFECTIVE or _matches_forbidden_import(mod):
                violations.append({
                    "line": node.lineno,
                    "call": f"from {mod} import ...",
                    "reason": (
                        f"从非白名单模块 {mod} 导入符号。生成实现只允许 import "
                        "torch/triton/math 等数值与标准库；导入 benchmark 参考实现"
                        "（kernelbench_* / src.* 任务包 / reference / baseline）"
                        "等同于把计算外包给现成实现。"
                    ),
                })
                for alias in node.names:
                    foreign[alias.asname or alias.name] = f"from {mod} import"

    # --- 子规则 2: 动态导入/执行机制 + 搜索路径篡改 + __file__ + 读文件 ---
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_DYNAMIC_BUILTIN_CALLS:
                violations.append({
                    "line": node.lineno,
                    "call": f"{f.id}(...)",
                    "reason": (
                        f"{f.id}() 属于动态执行/外部文件读取机制，单文件生成实现"
                        "没有合法用途；出现即意味着在加载外部代码或外部数据"
                        "（参考实现、预计算答案、私有输入集）。"
                    ),
                })
            elif isinstance(f, ast.Attribute):
                leaf = f.attr
                root = _root_name_of(f.value)
                if root == "importlib" or leaf in _FORBIDDEN_DYNAMIC_LEAF_CALLS:
                    violations.append({
                        "line": node.lineno,
                        "call": f"{_expr_to_str(f)}(...)",
                        "reason": (
                            f"{leaf} 属于动态导入机制：生成实现没有合法理由使用，"
                            "出现即意味着在现场加载外部实现规避静态审计。"
                        ),
                    })
                q = _expr_to_str(f.value)
                if q.startswith("sys.path") and leaf in ("insert", "append", "extend", "remove", "pop"):
                    violations.append({
                        "line": node.lineno,
                        "call": f"{q}.{leaf}(...)",
                        "reason": (
                            "篡改模块搜索路径：唯一用途是让解释器找到"
                            "实现文件之外的模块（参考实现/私有数据文件所在目录）。"
                        ),
                    })
        elif isinstance(node, ast.Name) and node.id == "__file__" and isinstance(node.ctx, ast.Load):
            violations.append({
                "line": node.lineno,
                "call": "__file__",
                "reason": (
                    "引用 __file__ 定位实现文件旁边的其它文件——生成实现必须是"
                    "自包含单文件，出现 __file__ 即意味着在读取/加载兄弟文件"
                    "（参考实现或私有输入数据）。"
                ),
            })

    # --- 子规则 3: 外部符号传播与调用 ---
    # 赋值传播：x = <根在外部集合里的表达式> -> x 也是外部对象
    # （ref_model = Model()、mod = importlib.util.module_from_spec(spec) 等）
    ext = set(foreign) | {"importlib"}
    for _ in range(4):  # 不动点：4 轮足够覆盖常见链长
        changed = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _root_name_of(node.value) in ext:
                for tgt in node.targets:
                    for nm in _target_names(tgt):
                        if nm not in ext:
                            ext.add(nm)
                            changed = True
        if not changed:
            break
    ext.discard("importlib")  # 动态导入名已在子规则 2 拦截

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        root = _root_name_of(f)
        if root in ext:
            violations.append({
                "line": node.lineno,
                "call": f"{root}...(...)",
                "reason": (
                    f"调用外部符号 {root}（来源：{foreign.get(root, '动态加载/传播')}）。"
                    "借外部模块的现成实现完成计算属于作弊；"
                    "所有核心计算必须在本文件的 @triton.jit kernel 内完成。"
                ),
            })
            continue
        # --- 子规则 4: 非 super 的 .forward() 调用 ---
        if isinstance(f, ast.Attribute) and f.attr == "forward":
            if _root_name_of(f.value) != "super":
                violations.append({
                    "line": node.lineno,
                    "call": f"{_expr_to_str(f.value)}.forward(...)",
                    "reason": (
                        "调用其它对象的 .forward() 即借外部 nn.Module 完成前向计算。"
                        "ModelNew.forward() 必须自己启动 @triton.jit kernel 完成计算。"
                    ),
                })
    return violations


# ---------------------------------------------------------------------------
# Type 9: benchmark 输入钩子劫持检测
# ---------------------------------------------------------------------------

_BENCHMARK_HOOK_NAMES = {"get_input_groups", "get_inputs"}

_HOOK_RNG_CALLS = {
    "manual_seed", "Generator", "randn", "rand", "randint",
    "randn_like", "rand_like",
}


def _check_hook_hijack(tree):
    """检测实现文件重定义 benchmark 输入生成钩子（get_input_groups/get_inputs）。

    背景：benchmark 的输入必须由任务文件的 get_input_groups() 生成。实现文件
    自带钩子会劫持输入来源——实测样本 02 从私有 JSON 读 case 并用固定种子
    重新生成全部输入张量（连 raw_gate 语义都换掉），样本 04 返回空列表
    让验证"零用例通过"。

    判据（命中其一即违规）：
      a) 钩子或其可达辅助函数里读外部文件（open / json.load[s]）；
      b) 钩子里用固定种子 RNG 自行生成输入张量
         （torch.manual_seed / torch.Generator / torch.randn 等）；
      c) 钩子直接返回空列表/空 tuple（验证没有任何用例可跑）。
    """
    violations = []
    module_funcs = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            module_funcs[n.name] = n
    for hook in sorted(_BENCHMARK_HOOK_NAMES):
        fn = module_funcs.get(hook)
        if fn is None:
            continue
        reachable = _reachable_funcs(module_funcs, [hook])
        nodes = [fn] + [module_funcs[name] for name in reachable if name in module_funcs]
        # c) 空返回
        for child in ast.walk(fn):
            if (isinstance(child, ast.Return)
                    and isinstance(child.value, (ast.List, ast.Tuple))
                    and not child.value.elts):
                violations.append({
                    "line": child.lineno,
                    "call": f"{hook}() 返回空列表",
                    "reason": (
                        f"{hook}() 返回空序列：验证/测速将没有任何用例可跑，"
                        "等于 vacuous pass。输入必须由任务文件的钩子生成，"
                        "实现文件不得重定义。"
                    ),
                })
        # a)/b)
        for fnode in nodes:
            for child in ast.walk(fnode):
                if not isinstance(child, ast.Call):
                    continue
                root = _root_name_of(child.func)
                leaf = _call_leaf_name(child)
                if leaf == "open" or (root == "json" and leaf in ("load", "loads")):
                    violations.append({
                        "line": child.lineno,
                        "call": f"{hook}() -> {leaf}(...)",
                        "reason": (
                            f"{hook}() 读取外部文件生成输入，劫持 benchmark 输入来源。"
                            "输入必须由任务文件的钩子生成，实现文件不得重定义。"
                        ),
                    })
                elif root == "torch" and leaf in _HOOK_RNG_CALLS:
                    violations.append({
                        "line": child.lineno,
                        "call": f"{hook}() -> torch.{leaf}(...)",
                        "reason": (
                            f"{hook}() 用固定种子 RNG 自行生成输入张量，"
                            "劫持 benchmark 输入来源（可换成对实现更有利的分布）。"
                            "输入必须由任务文件的钩子生成，实现文件不得重定义。"
                        ),
                    })
    return violations


# ---------------------------------------------------------------------------
# Type 8: 返回值必须来自 kernel 写出（数据流检查）
# ---------------------------------------------------------------------------

# 共享底层存储的 layout 方法：kernel 写进 x.view(...) 等价于写进 x。
# .contiguous()/.to()/.float() 可能拷出新对象，不在此列。
_STORAGE_SHARING_METHODS = {
    "view", "view_as", "reshape", "flatten", "unflatten",
    "transpose", "permute", "t", "squeeze", "unsqueeze", "expand", "expand_as",
}


def _lhs_root_names(tgt):
    """赋值目标的"持有名"：Name 自身；Tuple/List 逐元素；Subscript/Attribute 取根名。

    （不含下标里的索引变量——buf[i] 被写不等于 i 被写。）
    """
    if isinstance(tgt, ast.Name):
        return {tgt.id}
    if isinstance(tgt, (ast.Tuple, ast.List)):
        out = set()
        for e in tgt.elts:
            out |= _lhs_root_names(e)
        return out
    if isinstance(tgt, (ast.Attribute, ast.Subscript)):
        r = _root_name_of(tgt)
        return {r} if r else set()
    if isinstance(tgt, ast.Starred):
        return _lhs_root_names(tgt.value)
    return set()


def _kernel_write_params(func_node):
    """kernel 中被 tl.store / tl.atomic_* 写入的指针参数名集合 + 全部参数名（按位次）。

    实测中 kernel 很少直接 tl.store(out_ptr + ...)，而是先算局部别名
    （out_base = out_ptr + b * stride / o_ptrs = out_ptr + offs）再经别名写出。
    因此先做**指针别名不动点传播**：局部变量 = 含参数的表达式 -> 该局部变量
    视为这些参数的写出通道。
    """
    arg_nodes = func_node.args.posonlyargs + func_node.args.args + func_node.args.kwonlyargs
    params = [a.arg for a in arg_nodes]
    pset = set(params)

    alias_of = {}  # 局部变量名 -> 其表达式（传递地）引用的参数名集合
    for _ in range(6):  # 不动点：6 轮足够覆盖常见别名链长
        changed = False
        for node in ast.walk(func_node):
            if not isinstance(node, ast.Assign):
                continue
            roots = set()
            for n in ast.walk(node.value):
                if isinstance(n, ast.Name):
                    if n.id in pset:
                        roots.add(n.id)
                    elif n.id in alias_of:
                        roots |= alias_of[n.id]
            if not roots:
                continue
            for tgt in node.targets:
                for nm in _lhs_root_names(tgt):
                    if nm in pset:
                        continue
                    merged = alias_of.get(nm, set()) | roots
                    if merged != alias_of.get(nm):
                        alias_of[nm] = merged
                        changed = True
        if not changed:
            break

    written = set()
    for child in ast.walk(func_node):
        if not (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)):
            continue
        f = child.func
        if not (isinstance(f.value, ast.Name) and f.value.id == "tl"):
            continue
        if f.attr != "store" and not f.attr.startswith("atomic_"):
            continue
        if not child.args:
            continue
        for n in ast.walk(child.args[0]):
            if isinstance(n, ast.Name):
                if n.id in pset:
                    written.add(n.id)
                elif n.id in alias_of:
                    written |= alias_of[n.id]
    return written, params, alias_of


def _compute_kernel_write_info(kernels):
    """{kernel 名: (被写参数集合, 参数位次列表)}，支持 kernel 调用 kernel 辅助函数。

    kernel 体内调用其它 @triton.jit 辅助函数时（16_iou_v2 的 tile 包装层
    写法），辅助函数里的 tl.store 写出沿调用点实参映射回调用方的参数。
    不动点迭代以支持多级包装。
    """
    info = {}
    for name, node in kernels.items():
        w, p, alias = _kernel_write_params(node)
        info[name] = [set(w), p, alias]
    for _ in range(6):
        changed = False
        for name, node in kernels.items():
            wset, params, alias_of = info[name]
            for ch in ast.walk(node):
                if not isinstance(ch, ast.Call):
                    continue
                leaf = _call_leaf_name(ch)
                if leaf not in kernels or leaf == name:
                    continue
                c_w, c_params, _ = info[leaf]
                for i, arg in enumerate(ch.args):
                    if i >= len(c_params) or c_params[i] not in c_w:
                        continue
                    for r in _name_roots(arg):
                        targets = set()
                        if r in params:
                            targets.add(r)
                        elif r in alias_of:
                            targets |= alias_of[r]
                        for t in targets:
                            if t not in wset:
                                wset.add(t)
                                changed = True
                for kw in ch.keywords:
                    if kw.arg in c_w:
                        for r in _name_roots(kw.value):
                            targets = set()
                            if r in params:
                                targets.add(r)
                            elif r in alias_of:
                                targets |= alias_of[r]
                            for t in targets:
                                if t not in wset:
                                    wset.add(t)
                                    changed = True
        if not changed:
            break
    return {name: (w, p) for name, (w, p, _a) in info.items()}


def _mark_written_roots(arg_expr, out):
    """kernel 启动实参 => 被写入的 host 侧变量名（就地写入语义）。"""
    if isinstance(arg_expr, ast.Name):
        out.add(arg_expr.id)
    elif isinstance(arg_expr, (ast.Attribute, ast.Subscript)):
        root = _root_name_of(arg_expr)
        if root:
            out.add(root)
    elif isinstance(arg_expr, ast.Call):
        # kernel 写进 x.view(...) 等价于写进 x（共享存储）；
        # x.contiguous()/.to() 可能拷出新对象，kernel 写不到原 x，不算
        f = arg_expr.func
        if isinstance(f, ast.Attribute) and f.attr in _STORAGE_SHARING_METHODS:
            root = _root_name_of(f.value)
            if root:
                out.add(root)


def _func_positional_params(fnode):
    """函数的按位次参数名列表（posonly + args + kwonly）。"""
    return [a.arg for a in (
        fnode.args.posonlyargs + fnode.args.args + fnode.args.kwonlyargs)]


def _resolve_starred_args(fnode, star_name, launch_lineno):
    """把 kernel[grid](*args) 里的 *args 解析回元组/列表字面量的元素。

    实测样本（35_QuantMatmulReduceSum）的写法是
        args = (x1r, x2t, s1c, s2c, out, B, M, N, ...)
        _qmmrs_nz_kernel[grid](*args)
    取该函数内 launch 行之前最后一次对该名字的元组/列表赋值。
    找不到返回 None（放弃映射，不误报）。
    """
    best = None
    for node in ast.walk(fnode):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == star_name for t in node.targets):
            continue
        if isinstance(node.value, (ast.Tuple, ast.List)) and node.lineno < launch_lineno:
            if best is None or node.lineno > best.lineno:
                best = node
    return list(best.value.elts) if best is not None else None


def _compute_kernel_written(tree, kernel_names):
    """不动点计算每个 host 函数内"被 kernel 写入"的变量集合。

    返回 (written, returns_written)：
      written[fname]         —— 函数 fname 内被 kernel 写入的变量名集合
      returns_written[fname] —— fname 的某个 return 是否携带被写入的变量

    传播规则：
      1. kernel[grid](..., out, ...) 中处于 tl.store/atomic 参数位次的实参被写；
      2. t.copy_(s) 且 s 已写 -> t 已写；
      3. 调用模块内函数 g，g 内部写了第 i 个参数 -> 调用点第 i 个实参被写；
      4. LHS = <含已写变量的表达式> 或 <returns_written 的函数调用> -> LHS 已写。
    """
    kernels = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in kernel_names:
            kernels[node.name] = node
    write_info = _compute_kernel_write_info(kernels)

    # host 侧函数表（不含 kernel；同名后定义生效，与 Python 语义一致）
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name not in kernel_names:
            funcs[node.name] = node

    written = {name: set() for name in funcs}
    returns_written = {name: False for name in funcs}

    for _ in range(12):  # 不动点：12 轮足够覆盖常见调用链
        changed = False
        for fname, fnode in funcs.items():
            w = written[fname]
            var_bindings = _collect_var_kernel_bindings(fnode, kernel_names)
            for node in ast.walk(fnode):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                    # 规则 1：kernel 启动（含 (a if c else b)[grid](...) 条件分发、
                    # kern = a if c else b 变量绑定，候选 kernel 的写出位次取并集）
                    pos_args = list(node.args)
                    starred = any(isinstance(a, ast.Starred) for a in pos_args)
                    for kname in _kernel_names_in_subscript(node.func, kernel_names, var_bindings):
                        if kname not in write_info:
                            continue
                        wparams, kparams = write_info[kname]
                        # kernel[grid](*args)：把 Starred 解析回元组字面量
                        cur_args = pos_args
                        if starred:
                            expanded = []
                            for a in pos_args:
                                if isinstance(a, ast.Starred) and isinstance(a.value, ast.Name):
                                    elts = _resolve_starred_args(fnode, a.value.id, node.lineno)
                                    if elts is None:
                                        expanded = None
                                        break
                                    expanded.extend(elts)
                                else:
                                    expanded.append(a)
                            cur_args = expanded if expanded is not None else []
                        for i, arg in enumerate(cur_args):
                            if i < len(kparams) and kparams[i] in wparams:
                                before = len(w)
                                _mark_written_roots(arg, w)
                                changed = changed or len(w) != before
                        for kw in node.keywords:
                            if kw.arg in wparams:
                                before = len(w)
                                _mark_written_roots(kw.value, w)
                                changed = changed or len(w) != before
                    continue
                if isinstance(node, ast.Call):
                    f = node.func
                    # 规则 2：copy_ 传播
                    if isinstance(f, ast.Attribute) and f.attr == "copy_" and node.args:
                        if _name_roots(node.args[0]) & w:
                            r = _root_name_of(f.value)
                            if r and r not in w:
                                w.add(r)
                                changed = True
                    # 规则 3：跨函数参数写入映射
                    leaf = _call_leaf_name(node)
                    if leaf in funcs and leaf != fname:
                        gparams = _func_positional_params(funcs[leaf])
                        if gparams and gparams[0] == "self":
                            gparams = gparams[1:]
                        gw = written[leaf]
                        for i, arg in enumerate(node.args):
                            if i < len(gparams) and gparams[i] in gw:
                                before = len(w)
                                _mark_written_roots(arg, w)
                                changed = changed or len(w) != before
                # 规则 4：赋值传播
                if isinstance(node, ast.Assign):
                    rhs = node.value
                    hit = bool(_name_roots(rhs) & w)
                    if not hit and isinstance(rhs, ast.Call):
                        leaf = _call_leaf_name(rhs)
                        if leaf in funcs and returns_written[leaf]:
                            hit = True
                    if hit:
                        for tgt in node.targets:
                            for nm in _target_names(tgt):
                                if nm not in w:
                                    w.add(nm)
                                    changed = True
                    # 规则 4b：共享存储的反向传播——outf = out.reshape(...)
                    # 且 outf 已被 kernel 写 -> out 也被写（view 与源共享存储）。
                    # 只认 _STORAGE_SHARING_METHODS；.contiguous()/.to() 可能
                    # 拷出新对象，不做反向传播（样本 01 靠这个区分 Of 与 output）。
                    if isinstance(rhs, ast.Call) and isinstance(rhs.func, ast.Attribute) \
                            and rhs.func.attr in _STORAGE_SHARING_METHODS:
                        src = _root_name_of(rhs.func.value)
                        if src and src not in w:
                            lhs_names = set()
                            for tgt in node.targets:
                                lhs_names |= _target_names(tgt)
                            if lhs_names & w:
                                w.add(src)
                                changed = True
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    if _name_roots(node.value) & w and isinstance(node.target, ast.Name) \
                            and node.target.id not in w:
                        w.add(node.target.id)
                        changed = True
            # returns_written：本函数的 return 是否携带已写变量
            for node in ast.walk(fnode):
                if isinstance(node, ast.Return) and node.value is not None:
                    if _name_roots(node.value) & w and not returns_written[fname]:
                        returns_written[fname] = True
                        changed = True
        if not changed:
            break
    return written, returns_written


def _const_placeholder_tensors(forward_node):
    """forward 中由 torch.empty/zeros/ones/full 分配的**零元素**占位张量名。

    认两种写法：形状参数全是字面常量（torch.empty(0, ...)），或形状元组里
    含常量 0（torch.empty(0, H, P, N)——任一维为 0 即零元素张量，其余维度
    是什么都无所谓）。用于放行 FA 类算子返回值里的 API 兼容空张量
    （reserved / softmax_lse）。
    形状全由变量构成且不含 0 的分配（torch.zeros(B, T, ...)）不在此列——
    把变量形状的常量填充 buffer 当输出返回正是作弊形态（样本 04）。
    """
    def _zero_numel_shape(e):
        if isinstance(e, ast.Constant):
            return e.value == 0
        if isinstance(e, (ast.Tuple, ast.List)):
            return any(_zero_numel_shape(x) for x in e.elts)
        return False

    def _all_const(e):
        if isinstance(e, ast.Constant):
            return True
        if isinstance(e, (ast.Tuple, ast.List)):
            return all(_all_const(x) for x in e.elts)
        return False

    def _is_placeholder_call(v):
        if not isinstance(v, ast.Call):
            return False
        resolved = _resolve_call_name(v)
        if not resolved or resolved[0] != "torch" \
                or resolved[1] not in ("empty", "zeros", "ones", "full"):
            return False
        return bool(v.args) and (_all_const(v.args[0]) or _zero_numel_shape(v.args[0]))

    names = set()
    for node in ast.walk(forward_node):
        if not isinstance(node, ast.Assign):
            continue
        v = node.value
        is_ph = _is_placeholder_call(v)
        if not is_ph and isinstance(v, ast.IfExp):
            # dz = torch.empty_like(z) if has_z else torch.empty(0, ...)
            is_ph = _is_placeholder_call(v.body) or _is_placeholder_call(v.orelse)
        if is_ph:
            for tgt in node.targets:
                names |= _lhs_root_names(tgt)
    return names


def _meta_like_names(forward_node, fwd_params):
    """forward 中的标量/元信息变量名（形状 int、device、dtype、bool 开关等）。

    种子：常数、对入参的元信息访问（.shape/.size()/.stride()/.device/.dtype/
    .numel()/.dim() 等）、int/float/bool/str/len 强转；传播：这些名字的
    算术/比较/解包赋值。用于放行早退分支（if B == 0: return value.view(
    batch_size, ...)）和零维占位张量里的形状变量——它们是元信息不是数据。
    """
    meta = set()
    for _ in range(8):  # 不动点
        changed = False
        for node in ast.walk(forward_node):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if value is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            hit = False
            if isinstance(value, ast.Constant):
                hit = True
            elif isinstance(value, ast.Attribute):
                # x.shape / x.device / x.dtype（x 为入参或元信息根）
                r = _root_name_of(value)
                if r in fwd_params or r in meta:
                    hit = True
            elif isinstance(value, ast.Call):
                f = value.func
                if isinstance(f, ast.Name) and f.id in ("int", "float", "bool", "str", "len"):
                    hit = True
                elif isinstance(f, ast.Attribute) and f.attr in (
                        "size", "numel", "dim", "stride", "shape", "element_size",
                        "is_contiguous", "get_device"):
                    r = _root_name_of(f.value)
                    if r in fwd_params or r in meta:
                        hit = True
                else:
                    # triton.cdiv / next_power_of_2 / min / max 等作用于元信息的调用
                    r = _root_name_of(f)
                    if r in ("triton", "math", "builtins"):
                        hit = True
                    elif isinstance(f, ast.Name) and f.id in ("min", "max", "abs", "round"):
                        hit = True
            elif isinstance(value, ast.Subscript):
                # x.shape[0] / q_st[1]（q_st 已是元信息）
                r = _root_name_of(value)
                if r in meta:
                    hit = True
                elif r in fwd_params and isinstance(value.value, ast.Attribute) \
                        and value.value.attr in ("shape",):
                    hit = True
            elif isinstance(value, (ast.BoolOp, ast.Compare, ast.UnaryOp)):
                # 比较/布尔/取负的结果是 bool/int 标量，与张量还是元信息无关
                roots = _name_roots(value)
                if roots <= (meta | fwd_params | {"True", "False"}) or not roots:
                    hit = True
            elif isinstance(value, (ast.BinOp, ast.IfExp)):
                # 算术/三元表达式只传播元信息——操作数含入参张量时结果是张量，
                # 不能标成元信息（否则 host 侧 x * 2 后 return 会逃逸）
                roots = _name_roots(value)
                if roots and roots <= meta:
                    hit = True
                elif not roots:
                    hit = True
            if hit:
                for tgt in targets:
                    for nm in _lhs_root_names(tgt):
                        if nm not in meta:
                            meta.add(nm)
                            changed = True
        if not changed:
            break
    return meta


def _check_output_from_kernel(tree, forward_node, kernel_names):
    """数据流检查：forward() 返回的每个张量必须（传递地）来自被 kernel 写入的 buffer。

    这是"外包参考实现"类作弊的语义兜底：无论作弊者把真实计算藏在 import、
    helper 还是 host torch 调用里，只要 forward() 返回的张量在数据流上不依赖
    任何被 kernel tl.store/atomic 写入的 buffer，kernel 就是纯摆设。

    放行规则（避免误伤）：
      - 返回元素为常量/None（无变量根）——跳过；
      - 返回元素至少一个变量根 ∈ 被 kernel 写入集合——放行；
      - 返回元素是（或包含）对模块内 helper 的调用，且 helper 的返回值
        携带 kernel 写出（return _lstm_forward(...) 形态）——放行；
      - 返回元素的全部变量根都是 forward 入参（如 use_initial_state=False 时
        直接回传 initial_state）——放行；
      - 返回元素的未知根全是常数形状占位张量（torch.empty(0) 等 API 兼容位），
        且同一条 return 里已有 kernel 写出的主输出——放行；
      - 其余（根变量既不是 kernel 写出、也不是入参，如 ref 调用的结果、
        host torch 计算结果、变量形状的 torch.zeros 常量填充）——违规。

    已知盲区（由 Type 4/6/10 兜底）：返回 f(q) 且 q 同时被占位 kernel 写过
    （实测样本 06 的 _pkernel 往输入 q 里拷贝自身）。
    """
    if forward_node is None:
        return []
    written, returns_written = _compute_kernel_written(tree, kernel_names)
    fwd_written = written.get(forward_node.name, set())
    fwd_params = set(_func_positional_params(forward_node)) - {"self"}
    const_placeholders = _const_placeholder_tensors(forward_node)
    meta_like = _meta_like_names(forward_node, fwd_params)

    # 模块名不是数据根：torch.empty_like(x) 里的 torch 不该参与判定
    module_roots = {"torch", "triton", "tl", "math", "np", "numpy", "os", "sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_roots.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                module_roots.add(alias.asname or alias.name)

    def _elt_has_kernel_output(elt):
        if _name_roots(elt) & fwd_written:
            return True
        for n in ast.walk(elt):
            if isinstance(n, ast.Call):
                leaf = _call_leaf_name(n)
                if leaf and returns_written.get(leaf):
                    return True
        return False

    violations = []
    for node in ast.walk(forward_node):
        if not (isinstance(node, ast.Return) and node.value is not None):
            continue
        elts = node.value.elts if isinstance(node.value, ast.Tuple) else [node.value]
        has_kernel_main = any(_elt_has_kernel_output(e) for e in elts)
        for elt in elts:
            roots = _name_roots(elt) - {"self"} - module_roots
            if not roots:
                continue  # 常量 / None / 内联 torch.empty(0) 等纯模块调用
            if _elt_has_kernel_output(elt):
                continue
            unknown = roots - fwd_params
            if not unknown:
                continue  # 全部来自入参（如回传 initial_state / 早退分支回传 value 的变形）
            if unknown <= meta_like:
                # 未知根全是形状 int/device/bool 等元信息，且张量主体来自入参或
                # kernel（roots 里已有 param/written），如早退分支
                # return value.transpose(...).view(batch_size, ...)
                if roots & (fwd_params | fwd_written):
                    continue
                # 纯元信息元素只有在其同一条 return 已有 kernel 主输出时才放行
                if has_kernel_main:
                    continue
            if has_kernel_main and unknown <= const_placeholders:
                continue  # API 兼容的空占位张量（reserved / softmax_lse）
            violations.append({
                "line": getattr(elt, "lineno", node.lineno),
                "call": f"return 元素 ({', '.join(sorted(unknown))})",
                "reason": (
                    "forward() 的返回值在数据流上不依赖任何被 kernel 写入的 "
                    "buffer——kernel 的计算结果没有进入返回值，真实结果来自 "
                    "kernel 之外的路径（参考实现外包 / host 侧 torch 计算 / "
                    "常量填充）。forward() 必须返回由 @triton.jit kernel "
                    "写出的张量。"
                ),
            })
    return violations


# ---------------------------------------------------------------------------
# Type 10: 诱饵 kernel 检测（定义了但从生效的 forward 不可达）
# ---------------------------------------------------------------------------

def _reachable_kernels(tree, forward_node, kernel_names, wrapper_names):
    """从生效的 forward() 出发可达的 kernel 集合。

    可达路径：
      - forward() 及其**整条 host 侧调用链**（wrapper 调 wrapper 也算）里的
        kernel[grid](...) 启动，包括 `(kern_a if cond else kern_b)[grid](...)`
        条件分发写法；
      - kernel 体内以普通调用形式使用其它 @triton.jit 函数（Triton 的
        kernel 辅助函数惯用法），沿 kernel→kernel 调用链传播。
    """
    if forward_node is None:
        return set()
    # host 侧：forward + 全部可达辅助函数里的 kernel 启动
    host_funcs = [forward_node] + [
        fnode for _, fnode in collect_reachable_funcs(tree, forward_node, kernel_names)]
    stack = []
    for fnode in host_funcs:
        bindings = _collect_var_kernel_bindings(fnode, kernel_names)
        for node in ast.walk(fnode):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                stack.extend(_kernel_names_in_subscript(node.func, kernel_names, bindings))

    # kernel -> kernel 调用链（kernel 辅助函数）
    funcs = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
    reachable = set()
    while stack:
        name = stack.pop()
        if name in reachable or name not in kernel_names:
            continue
        reachable.add(name)
        knode = funcs.get(name)
        if knode is None:
            continue
        for node in ast.walk(knode):
            if isinstance(node, ast.Call):
                leaf = _call_leaf_name(node)
                if leaf in kernel_names and leaf not in reachable:
                    stack.append(leaf)
    return reachable


def _check_no_decoy_kernels_filled(result, tree, forward_node, kernel_names, wrapper_names):
    """填充 no_decoy_kernels 检查（Type 10）；返回 passed。

    单文件生成实现中，任何 @triton.jit kernel 都应服务于 forward 的计算路径。
    定义了却完全不可达的 kernel 只有两种可能：迭代残留（应删除），或故意
    放的诱饵——让文件"看起来很 Triton"，骗过人工审查与只看"是否存在像样
    kernel"的静态检查（实测样本 02/07：诱饵 kernel 内含未定义名
    b_off_none / v_cont，证明从未真正编译运行）。
    """
    reachable = _reachable_kernels(tree, forward_node, kernel_names, wrapper_names)
    decoys = sorted(kernel_names - reachable)
    check = result["checks"]["no_decoy_kernels"]
    check["decoys"] = decoys

    if not decoys:
        check["passed"] = True
        return True

    check["error"] = (
        f"发现 {len(decoys)} 个从 forward() 不可达的 kernel（疑似诱饵）: {decoys}"
    )
    result["regression_type"] = 10
    result["suggestion"] = (
        f"kernel {decoys} 已定义但从生效的 forward() 出发（含 wrapper 与 "
        "kernel 体内的辅助函数调用）完全不可达。单文件生成实现中不可达 kernel "
        "没有合法用途：要么是迭代残留（请删除），要么是用来撑门面的诱饵。"
        "保留任何 kernel 都必须让它真正参与 forward 的计算路径。"
    )
    return False


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------

def _empty_result(filepath):
    return {
        "valid": False,
        "filepath": filepath,
        "checks": {
            "triton_kernel_exists": {"passed": False, "kernels": [], "error": None},
            "no_shadow_definitions": {"passed": False, "duplicates": {}, "error": None},
            "kernel_called_from_forward": {"passed": False, "called": [], "error": None},
            "no_forbidden_module_wide_ops": {"passed": False, "violations": [], "error": None},
            "kernel_not_placeholder": {"passed": False, "placeholders": [], "error": None},
            "no_decoy_kernels": {"passed": False, "decoys": [], "error": None},
            "no_external_delegation": {"passed": False, "violations": [], "error": None},
            "no_hook_hijack": {"passed": False, "violations": [], "error": None},
            "no_dynamic_attr_evasion": {"passed": False, "violations": [], "error": None},
            "no_host_math_fallback": {"passed": False, "violations": [], "error": None},
            "no_host_sync_risk": {"passed": False, "violations": [], "error": None},
            "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
            "output_from_kernel": {"passed": False, "violations": [], "error": None},
        },
        "regression_type": None,
        "suggestion": "",
    }


def _check_kernel_exists(result, tree):
    """填充 triton_kernel_exists 检查；返回 (passed, kernel_names)。"""
    kernels = find_triton_kernels(tree)
    kernel_names = set(kernels.keys())
    result["checks"]["triton_kernel_exists"]["kernels"] = [
        {"name": k, "line": v["line"], "has_tl_usage": v["has_tl_usage"]}
        for k, v in kernels.items()
    ]

    if not kernel_names:
        result["checks"]["triton_kernel_exists"]["error"] = "未找到任何 @triton.jit 装饰的 kernel 函数"
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有 Triton kernel。必须创建至少一个 @triton.jit 装饰的函数，"
            "在其中使用 tl.load/tl.store 实现核心计算逻辑。"
        )
        return False, kernel_names

    kernels_without_tl = [k for k, v in kernels.items() if not v["has_tl_usage"]]
    if len(kernels_without_tl) == len(kernels):
        result["checks"]["triton_kernel_exists"]["error"] = (
            f"kernel 函数 {kernels_without_tl} 未使用任何 tl.* API，"
            "可能是空壳 kernel"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "虽然存在 @triton.jit 装饰的函数，但没有使用 triton.language (tl) API。"
            "kernel 必须使用 tl.load/tl.store 等进行显式内存操作和计算。"
        )
        return False, kernel_names

    result["checks"]["triton_kernel_exists"]["passed"] = True
    return True, kernel_names


def _check_forward_calls_kernel(result, tree, kernel_names):
    """填充 kernel_called_from_forward 检查；返回 (passed, forward_node, wrapper_names)。"""
    forward_node = find_model_new_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            "未找到 ModelNew.forward() 方法"
        )
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew 类或 forward 方法。"
        return False, None, set()

    wrapper_names = find_wrapper_functions(tree, kernel_names)
    called = check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names)
    result["checks"]["kernel_called_from_forward"]["called"] = list(called)

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"@triton.jit kernel {list(kernel_names)} 已定义但 forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        wrapper_hint = (
            f"也存在 wrapper 函数 {list(wrapper_names)} 但 forward 也未调用它们。"
            if wrapper_names else ""
        )
        result["suggestion"] = (
            f"已定义 kernel {list(kernel_names)} 但 ModelNew.forward() 中未调用。"
            "forward() 必须通过 kernel_name[grid](...) 形式启动 kernel。"
            f"{wrapper_hint}"
        )
        return False, forward_node, wrapper_names

    result["checks"]["kernel_called_from_forward"]["passed"] = True
    return True, forward_node, wrapper_names


def _check_no_forbidden_ops(result, forward_node, kernel_names, wrapper_names, tree=None):
    """填充 no_forbidden_torch_ops 检查；返回 passed。

    除 forward() 子树外，**还扫描从 forward() 可达的模块内辅助函数**——
    把主链计算挪到 forward 外的函数里是常见的绕过手法。
    """
    violations = check_forbidden_torch_ops(forward_node, kernel_names, wrapper_names)
    if tree is not None:
        for fname, fnode in collect_reachable_funcs(tree, forward_node, kernel_names):
            violations.extend(check_main_chain_compute(fnode, f"{fname}()"))
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if not violations:
        result["checks"]["no_forbidden_torch_ops"]["passed"] = True
        return True

    result["checks"]["no_forbidden_torch_ops"]["error"] = (
        f"forward() 及其可达辅助函数中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"forward() 调用了 Triton kernel 但仍使用 PyTorch 进行部分计算: "
        f"{violation_details}。"
        "所有核心计算必须在 @triton.jit kernel 中完成，"
        "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
    )
    return False


def _check_no_forbidden_module_wide_ops(result, tree):
    """填充 no_forbidden_module_wide_ops 检查；返回 passed。

    扫整个文件（不只 forward AST）找 nn.XXX 构造、rng 操纵、F.xxx 调用等
    间接作弊路径。
    """
    violations = check_forbidden_module_wide_ops(tree)
    result["checks"]["no_forbidden_module_wide_ops"]["violations"] = violations

    if not violations:
        result["checks"]["no_forbidden_module_wide_ops"]["passed"] = True
        return True

    result["checks"]["no_forbidden_module_wide_ops"]["error"] = (
        f"模块级发现 {len(violations)} 处禁止的 PyTorch 接口调用（不只 forward）"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"生成代码借 PyTorch 接口作弊: {violation_details}。"
        "禁用 nn.XXX 构造、F.xxx 调用；"
        "所有计算必须在 @triton.jit kernel 内完成，"
        "权重应从外部传入或用常量初始化，不得借 nn.XXX 复刻 baseline 权重。"
    )
    return False


def _check_kernel_not_placeholder(result, tree, kernel_names, called_names):
    """Type 4/5：被调用的 kernel 是否只是占位；矩阵乘类算子是否有 tl.dot。"""
    kernels = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in kernel_names:
            kernels[node.name] = node

    called = [n for n in (called_names or kernel_names) if n in kernels] or list(kernels)
    placeholders = []
    for name in called:
        node = kernels.get(name)
        if node is not None and kernel_is_placeholder(node, kernel_map=kernels):
            placeholders.append(name)
    result["checks"]["kernel_not_placeholder"]["placeholders"] = placeholders

    if placeholders and len(placeholders) == len(called):
        result["checks"]["kernel_not_placeholder"]["error"] = (
            f"被调用的 kernel 全部是占位 kernel（只做 load/store 搬运，无任何计算）: {placeholders}"
        )
        result["regression_type"] = 4
        result["suggestion"] = (
            f"检测到占位 kernel {placeholders}：它只把数据从输入拷到输出，不参与任何计算，"
            "真实计算仍在 PyTorch 侧。这属于为通过 AST 检测而伪造 Triton 调用。"
            "所有核心计算必须在 @triton.jit kernel 内用 tl.* 完成。"
        )
        return False

    result["checks"]["kernel_not_placeholder"]["passed"] = True
    return True


def _check_no_dynamic_attr_evasion(result, code):
    """填充 no_dynamic_attr_evasion 检查；返回 passed。

    扫整个文件（不只 forward）检测用 getattr 动态访问 torch/nn/F 命名空间
    以规避 nn.XXX / F.xxx 静态黑名单的作弊写法。
    """
    violations = _check_dynamic_attr_evasion(code)
    result["checks"]["no_dynamic_attr_evasion"]["violations"] = violations

    if not violations:
        result["checks"]["no_dynamic_attr_evasion"]["passed"] = True
        return True

    result["checks"]["no_dynamic_attr_evasion"]["error"] = (
        f"检测到 {len(violations)} 处绕过 Triton 实现的 PyTorch/CANN 算子入口"
    )
    details = "; ".join(f"第{v['line']}行 {v['call']}" for v in violations[:5])
    result["regression_type"] = 3
    result["suggestion"] = (
        f"生成代码绕过 Triton 实现直接使用 PyTorch/CANN 算子入口: {details}。"
        "包括用 getattr 动态访问 torch/nn/F/torch_npu 命名空间以规避静态黑名单，"
        "以及直接调用 torch_npu 的 npu_* 融合算子；"
        "所有核心计算必须在 @triton.jit kernel 内用 tl.* 完成。"
    )
    return False


def _check_no_host_math_fallback(result, code):
    """填充 no_host_math_fallback 检查；返回 passed。

    扫整个文件（不只 forward）检测 host 侧是否用 np/torch/tensor 方法
    计算三角/超越函数并作为 Triton kernel 输入表。
    """
    violations = _check_host_math_fallback(code)
    result["checks"]["no_host_math_fallback"]["violations"] = violations

    if not violations:
        result["checks"]["no_host_math_fallback"]["passed"] = True
        return True

    result["checks"]["no_host_math_fallback"]["error"] = (
        f"检测到 {len(violations)} 处 host 侧三角函数/超越函数调用"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"host 侧使用非 Triton 数学库生成 kernel 输入表: {violation_details}。"
        "所有 sin/cos/exp/log 等超越计算必须在 @triton.jit kernel 内用 tl.* intrinsic 实现；"
        "host 侧仅允许 math.cos/math.sin 标量常量初始化。"
    )
    return False


def _check_no_host_sync_risk(result, code, forward_node, kernel_names):
    """填充 no_host_sync_risk 检查；返回 passed。

    从 kernel 启动入口出发，沿模块内调用链扫描 host 侧 D2H 同步风险
    （helper 函数不逃逸），包括 .item()、int(tensor[i])、torch.nonzero、
    数据依赖循环界等。
    """
    violations = _check_host_sync_risk(code, forward_node=forward_node, kernel_names=kernel_names)
    result["checks"]["no_host_sync_risk"]["violations"] = violations

    if not violations:
        result["checks"]["no_host_sync_risk"]["passed"] = True
        return True

    result["checks"]["no_host_sync_risk"]["error"] = (
        f"检测到 {len(violations)} 处 host 侧 D2H 同步/数据依赖标量风险"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']} ({v.get('severity', 'medium')})" for v in violations[:5]
    )
    result["regression_type"] = 3
    result["suggestion"] = (
        f"host 路径存在 D2H 同步或数据依赖控制流: {violation_details}。"
        "张量数值必须在 @triton.jit kernel 内消费；"
        "如需从 mask 恢复结构参数，应由调用方显式传入或一次性 host 预计算并 cache，"
        "禁止在每次调用中通过 .item()/int(tensor[i]) 同步读取。"
    )
    return False


def _check_no_shadow_definitions_filled(result, tree):
    """填充 no_shadow_definitions 检查（Type 7）；返回 passed。"""
    duplicates = _find_shadow_definitions(tree)
    result["checks"]["no_shadow_definitions"]["duplicates"] = duplicates

    if not duplicates:
        result["checks"]["no_shadow_definitions"]["passed"] = True
        return True

    details = "; ".join(
        f"{name} 定义于第 {', '.join(str(line) for line in lines)} 行"
        for name, lines in sorted(duplicates.items())
    )
    result["checks"]["no_shadow_definitions"]["error"] = (
        f"模块顶层存在 {len(duplicates)} 个重复定义: {details}"
    )
    result["regression_type"] = 7
    result["suggestion"] = (
        f"检测到影子定义: {details}。同名类/函数重复定义时 Python 只有后定义的生效——"
        "这是让'被检测的代码'（前一个，含 kernel 启动）与'生效的代码'"
        "（后一个，纯 PyTorch / 委托参考实现）分离的典型作弊手法。"
        "每个类/函数只保留一个定义。"
    )
    return False


def _check_no_external_delegation_filled(result, tree):
    """填充 no_external_delegation 检查（Type 6）；返回 passed。"""
    violations = _check_external_delegation(tree)
    result["checks"]["no_external_delegation"]["violations"] = violations

    if not violations:
        result["checks"]["no_external_delegation"]["passed"] = True
        return True

    result["checks"]["no_external_delegation"]["error"] = (
        f"检测到 {len(violations)} 处外部/参考实现外包（动态导入、非白名单 import、"
        "外部符号调用、非 super 的 .forward() 调用）"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 6
    result["suggestion"] = (
        f"生成实现把计算外包给外部/参考实现: {violation_details}。"
        "实现必须是自包含单文件：只允许 import torch/triton/math 等白名单模块，"
        "禁止通过动态加载机制执行未知代码，禁止篡改搜索路径或 "
        "用 __file__ 定位兄弟文件，禁止调用参考实现的 Model/forward。"
        "所有核心计算必须在本文件的 @triton.jit kernel 内完成。"
    )
    return False


def _check_no_hook_hijack_filled(result, tree):
    """填充 no_hook_hijack 检查（Type 9）；返回 passed。"""
    violations = _check_hook_hijack(tree)
    result["checks"]["no_hook_hijack"]["violations"] = violations

    if not violations:
        result["checks"]["no_hook_hijack"]["passed"] = True
        return True

    result["checks"]["no_hook_hijack"]["error"] = (
        f"检测到 {len(violations)} 处 benchmark 输入钩子劫持"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 9
    result["suggestion"] = (
        f"实现文件重定义了 benchmark 输入钩子: {violation_details}。"
        "输入必须由任务文件的 get_input_groups()/get_inputs() 生成；"
        "实现文件不得自行读外部文件、用固定种子 RNG 重新生成输入，"
        "也不得返回空用例列表。请从实现文件中删除这些钩子。"
    )
    return False


def _check_output_from_kernel_filled(result, tree, forward_node, kernel_names):
    """填充 output_from_kernel 检查（Type 8）；返回 passed。"""
    violations = _check_output_from_kernel(tree, forward_node, kernel_names)
    result["checks"]["output_from_kernel"]["violations"] = violations

    if not violations:
        result["checks"]["output_from_kernel"]["passed"] = True
        return True

    result["checks"]["output_from_kernel"]["error"] = (
        f"forward() 有 {len(violations)} 个返回元素在数据流上不依赖任何 "
        "被 kernel 写入的 buffer"
    )
    violation_details = "; ".join(
        f"第{v['line']}行 {v['call']}" for v in violations[:5]
    )
    result["regression_type"] = 8
    result["suggestion"] = (
        f"kernel 的计算结果没有进入返回值: {violation_details}。"
        "forward() 必须返回由 @triton.jit kernel 通过 tl.store/atomic 写出的张量"
        "（或其 view/reshape 等共享存储的变换）。"
        "参考实现外包、host 侧 torch 计算、返回常量填充的 buffer 都会被此检查拦截。"
    )
    return False


def validate(code, filepath="<unknown>"):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。
    """
    result = _empty_result(filepath)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["triton_kernel_exists"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    # 记录本文件定义的类方法名，供 self.xxx() 规则区分"本类方法"与"nn.Module 前向"
    global _CURRENT_SELF_METHODS
    _CURRENT_SELF_METHODS = _collect_self_methods(tree)

    ok, kernel_names = _check_kernel_exists(result, tree)
    if not ok:
        return result

    # 影子定义检查（Type 7）必须先于 forward 分析：重复定义会让
    # "被检测的代码"与"生效的代码"分离（find_model_new_forward 已按
    # Python 语义取后定义，但重复定义本身就是作弊信号，直接判失败）。
    if not _check_no_shadow_definitions_filled(result, tree):
        return result

    ok, forward_node, wrapper_names = _check_forward_calls_kernel(result, tree, kernel_names)
    if not ok:
        return result

    # 模块级检查放在 forward 级之前：模块级作弊（nn.XXX 复刻权重等）
    # 比单纯的 forward 内 PyTorch 退化更严重，应当优先暴露。
    if not _check_no_forbidden_module_wide_ops(result, tree):
        return result

    if not _check_kernel_not_placeholder(result, tree, kernel_names,
                                         result["checks"]["kernel_called_from_forward"].get("called")):
        return result

    # 诱饵 kernel 检查（Type 10）：定义了但从生效 forward 不可达的 kernel
    # 是"被检代码 ≠ 生效代码"的帮凶，紧接占位检查之后拦截。
    if not _check_no_decoy_kernels_filled(result, tree, forward_node, kernel_names, wrapper_names):
        return result

    # 外部/参考实现外包检查（Type 6）：作弊样本的主路径——import 参考实现、
    # 动态加载、调用外部 Model/forward。早于 forward 级检查暴露。
    if not _check_no_external_delegation_filled(result, tree):
        return result

    # benchmark 输入钩子劫持检查（Type 9）。
    if not _check_no_hook_hijack_filled(result, tree):
        return result

    # 动态属性规避检查：必须早于其它 PyTorch 入口检查——被 getattr 混淆过的
    # 调用在后续按 qual/attr 匹配的规则里全部解析不出来，会整片漏检。
    if not _check_no_dynamic_attr_evasion(result, code):
        return result

    # Host-side math fallback 检查：必须在 forward 级 PyTorch 退化检查之前
    # 拦截 np.cos / torch.cos / tensor.cos() 等 host 侧生成 kernel 输入表的行为。
    if not _check_no_host_math_fallback(result, code):
        return result

    # Host-side D2H 同步风险检查：在 forward 级退化检查之前拦截，避免 host
    # bound / graph break / replay 正确性风险从 helper 函数逃逸。
    if not _check_no_host_sync_risk(result, code, forward_node, kernel_names):
        return result

    if not _check_no_forbidden_ops(result, forward_node, kernel_names, wrapper_names, tree):
        return result

    # 数据流检查（Type 8）放在最后：它是语义兜底——无论真实计算被藏在
    # import、helper 还是 host torch 调用里，只要返回值不来自 kernel 写出
    # 就判失败。放在最后保证它看到的是全部前置检查都"合规"的代码。
    if not _check_output_from_kernel_filled(result, tree, forward_node, kernel_names):
        return result

    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_code(path, want_json):
    """读取代码文件；失败时返回 None，由调用方决定退出。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        if want_json:
            logger.info("%s", json.dumps({"valid": False, "error": f"文件不存在: {path}"}))
        else:
            logger.error("[ERROR] 文件不存在: %s", path)
        return None


def _emit_pass(result):
    kernels = result["checks"]["triton_kernel_exists"]["kernels"]
    called = result["checks"]["kernel_called_from_forward"]["called"]
    logger.info("[PASS] Triton 实现验证通过")
    logger.info(
        "  - 发现 %d 个 @triton.jit kernel: %s",
        len(kernels),
        ", ".join(k["name"] for k in kernels),
    )
    logger.info("  - forward() 调用: %s", ", ".join(called))
    logger.info("  - forward() 中无禁止的 PyTorch 计算操作")
    logger.info("  - 无外部/参考实现外包（import 白名单、无动态导入、无外部符号调用）")
    logger.info("  - 无 benchmark 输入钩子劫持")
    logger.info("  - 无 host 侧三角函数/超越函数调用")
    logger.info("  - 无 host 侧 D2H 同步/数据依赖标量风险")
    logger.info("  - 无诱饵 kernel（所有 kernel 均可达）")
    logger.info("  - forward() 返回值均来自 kernel 写出的 buffer")


# _emit_fail 中需要逐条展开违规详情的检查项（同时决定展开顺序）
_VIOLATION_DETAIL_CHECKS = (
    "no_forbidden_module_wide_ops",
    "no_external_delegation",
    "no_hook_hijack",
    "no_host_math_fallback",
    "no_host_sync_risk",
    "no_forbidden_torch_ops",
    "output_from_kernel",
)


def _emit_fail(result):
    rtype = result["regression_type"]
    type_desc = {
        1: "完全无 Triton kernel（纯 PyTorch）",
        2: "有 Triton kernel 但 forward() 未调用",
        3: "部分计算使用 PyTorch（需全部移入 Triton kernel；含 forward 可达的辅助函数）",
        4: "只有占位 kernel（无写出 / 写常量 / 纯 load-store 搬运，为通过检测而伪造 Triton 调用）",
        6: "外包外部/参考实现（非白名单 import、动态导入、调用参考 Model/forward）",
        7: "影子定义（同名类/函数重复定义，被检代码与生效代码分离）",
        8: "forward() 返回值不来自 kernel 写出（kernel 结果未进入返回值）",
        9: "benchmark 输入钩子劫持（重定义 get_input_groups/get_inputs）",
        10: "诱饵 kernel（定义了但从生效的 forward() 不可达）",
    }
    logger.info("[FAIL] 检测到 PyTorch 退化 — Type %s: %s", rtype, type_desc.get(rtype, "未知"))

    for check_name, check_result in result["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        logger.info("  [%s] %s", status, check_name)
        if check_result["error"]:
            logger.info("         %s", check_result["error"])

    if any(result["checks"][name]["violations"] for name in _VIOLATION_DETAIL_CHECKS):
        logger.info("  违规详情:")
        for check_name in _VIOLATION_DETAIL_CHECKS:
            vs = result["checks"][check_name]["violations"]
            if not vs:
                continue
            logger.info("    [%s] 共 %d 处:", check_name, len(vs))
            for v in vs:
                logger.info("      第 %s 行: %s — %s", v["line"], v["call"], v["reason"])

    logger.info("\n  修复建议: %s", result["suggestion"])


def main():
    _setup_logger()
    parser = argparse.ArgumentParser(
        description="检查生成代码是否退化为 PyTorch 原生实现（AST 静态分析）"
    )
    parser.add_argument("file", help="要检查的 Python 文件路径")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出")
    args = parser.parse_args()

    code = _load_code(args.file, args.json)
    if code is None:
        sys.exit(1)
    result = validate(code, filepath=args.file)

    if args.json:
        logger.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
    elif result["valid"]:
        _emit_pass(result)
    else:
        _emit_fail(result)

    sys.exit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
