#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""离线 ST：引擎建模旋钮回归 —— 逐个验证引擎各档位的数值行为符合其公式。

覆盖 overlap 标定(test_overlap_model.py)之外的引擎逻辑：GM 带宽口径、
读写争用、显式固定开销、流水填充/排空、尾块负载均衡、bound 分类。
参考值从 ascend950pr.json 自推（不写魔数），data 变了断言自动跟随。

用法: python3 tests/offline/test_engine_knobs.py
"""

from __future__ import annotations

import math
import importlib
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL_DIR / "scripts"))
engine = importlib.import_module("evaluate_a5_vector_roofline")

HW = engine.load_json(SKILL_DIR / "references/hardware/ascend950pr.json")
IDB = engine.load_json(SKILL_DIR / "references/vector_instruction_types.json")
V = HW["variants"]["ascend950pr_9589"]
GM = float(V["peak"]["gm_bandwidth_bytes_per_s"])
GMC = float(V["derived_from_ini"]["vector_core_ddr_bandwidth_bytes_per_s"])
ALPHA = float(HW["common"]["gm_bandwidth_measured"]["rw_contention_alpha"])
OVERHEADS = HW["common"]["overheads"]
FLOOR = float(OVERHEADS["min_launch_floor_us"])
LA = HW["common"]["gm_bandwidth_l2_aware"]
LA_CURVE = sorted(
    (float(b), float(bw)) for b, bw in LA["curve_combined_bytes_to_copy_bps"]
)
LA_L2BW, LA_HBMBW = (
    LA_CURVE[0][1],
    LA_CURVE[-1][1],
)  # 小工作集 L2 高带宽 / 大工作集 HBM 带宽
LA_REREAD = float(LA["l2_reread_bandwidth_bytes_per_s"])

fails = []
checks = 0


def approx(a, b, tol=2e-3):
    return abs(a - b) <= tol * max(1.0, abs(b))


def ck(label, cond):
    global checks
    checks += 1
    if not cond:
        fails.append(label)


def run(**kw):
    cd = {
        "case_id": "t",
        "active_vector_cores": 64,
        "memory": kw.pop("memory", {}),
        "vector": kw.pop(
            "vector", {"instructions": [{"op": "Add", "dtype": "fp32", "elements": 1}]}
        ),
    }
    cd.update(kw)
    return engine.evaluate_case(cd, HW, IDB)


TINY = {"instructions": [{"op": "Add", "dtype": "fp32", "elements": 1}]}  # vector≈0
BIG = {
    "instructions": [{"op": "Add", "dtype": "fp32", "elements": 1000000000}]
}  # vector 主导
B = 1.6e9  # 选定使 mte 在 chip_peak 下恰为 1000us

# 1) GM 带宽口径：chip_peak vs vector_core_conservative
rp = run(memory={"gm_read_bytes": B}, vector=TINY)
rc = run(
    memory={"gm_read_bytes": B},
    gm_bandwidth_model="vector_core_conservative",
    vector=TINY,
)
ck("gm chip_peak mte2 = B/GM", approx(rp["mte2_time_us"], B / GM * 1e6))
ck(
    "gm conservative 比值 = GM/GMC",
    approx(rc["mte2_time_us"] / rp["mte2_time_us"], GM / GMC),
)

# 2) 默认 GM 时间仅由流量和所选带宽决定，零流量不产生开销
for size in (0, 1e5, 8e6):
    r = run(memory={"gm_read_bytes": size}, gm_rampup_model="none", vector=TINY)
    ck(f"GM 无爬坡 size={size:g}", approx(r["mte2_time_us"], size / GM * 1e6))

# 3) 读写争用：max_pipe 下 mte2==mte3 时 body ×(1+alpha)
base = run(memory={"gm_read_bytes": B, "gm_write_bytes": B}, vector=TINY)
cont = run(
    memory={"gm_read_bytes": B, "gm_write_bytes": B}, gm_rw_contention=True, vector=TINY
)
ck("contention alpha 字段", approx(cont["gm_rw_contention_alpha"], ALPHA))
ck(
    "contention body ×(1+alpha)",
    approx(cont["kernel_body_time_us"] / base["kernel_body_time_us"], 1 + ALPHA),
)

# 4) 固定开销默认 0；显式值/表达式按加性开销计入总时延
r0 = run(memory={"gm_read_bytes": B}, vector=TINY)
ck(
    "fixed none=0",
    r0["fixed_overhead_us"] == 0.0 and r0["fixed_overhead_source"] == "none",
)
for overhead in (2.5, "T/2"):
    r = run(
        memory={"gm_read_bytes": B},
        fixed_overhead_us=overhead,
        fixed_overhead_model="none",
        variables={"T": 5},
        vector=TINY,
    )
    ck(
        f"fixed 显式开销 {overhead}",
        approx(r["fixed_overhead_us"], 2.5)
        and r["fixed_overhead_source"] == "explicit"
        and approx(r["theoretical_time_us"] - r0["theoretical_time_us"], 2.5),
    )

# 5) 流水填充/排空（max_pipe + tile_count）：tile=1 退化 serial；关闭则取 max；tiles 大→趋近 max
t = B / GM * 1e6  # 单条 mem 流水耗时 ≈1000us
mem2 = {"gm_read_bytes": B, "gm_write_bytes": B}
r1 = run(memory=mem2, tile_count=1, vector=TINY)
ck("fill_drain tile=1 → serial(2t)", approx(r1["kernel_body_time_us"], 2 * t))
r1off = run(memory=mem2, tile_count=1, fill_drain=False, vector=TINY)
ck("fill_drain 关闭 → max(t)", approx(r1off["kernel_body_time_us"], t))
rN = run(memory=mem2, tile_count=1000, vector=TINY)
ck("fill_drain tiles 大 → fill_drain≈t/tiles", approx(rN["fill_drain_us"], t / 1000))

# 5b) launch floor：max(body, floor)，小 body 抬到 floor，大 body 不变
rsmall = run(memory={"gm_read_bytes": 1000}, launch_floor_model="hardware", vector=TINY)
ck("launch floor 抬小 body 到 floor", approx(rsmall["kernel_body_time_us"], FLOOR))
ck("launch floor 输出字段", approx(rsmall["launch_floor_us"], FLOOR))
rbig = run(
    memory={"gm_read_bytes": B, "gm_write_bytes": B},
    launch_floor_model="hardware",
    vector=TINY,
)
ck("launch floor 不动大 body", approx(rbig["kernel_body_time_us"], t))
ck(
    "launch floor 默认 none=0",
    approx(run(memory={"gm_read_bytes": B}, vector=TINY)["launch_floor_us"], 0.0),
)

# 5c) l2_aware：有效带宽随工作集(小→L2高带宽, 大→HBM), 重读字节走L2不计入HBM流
big = 8 * LA_CURVE[-1][0]  # 远大于曲线上界 → 取 HBM 带宽
rL = run(
    memory={"gm_read_bytes": big, "gm_write_bytes": big},
    gm_bandwidth_model="l2_aware",
    vector=TINY,
)
ck(
    "l2_aware 大工作集→HBM copy_bw",
    approx(rL["kernel_body_time_us"], (2 * big) / LA_HBMBW * 1e6),
)
small = LA_CURVE[0][0] / 4  # 远小于下界 → 取 L2 带宽
rS = run(
    memory={"gm_read_bytes": small, "gm_write_bytes": small},
    gm_bandwidth_model="l2_aware",
    vector=TINY,
)
ck(
    "l2_aware 小工作集→L2 copy_bw",
    approx(rS["kernel_body_time_us"], (2 * small) / LA_L2BW * 1e6),
)
# 重读：gm_reread 从 HBM 流剔除（按 L2、与流重叠），故 body 由 (read-reread)+write 决定
rR = run(
    memory={"gm_read_bytes": 2 * big, "gm_write_bytes": big, "gm_reread_bytes": big},
    gm_bandwidth_model="l2_aware",
    vector=TINY,
)
ck(
    "l2_aware 重读剔除HBM流",
    approx(rR["kernel_body_time_us"], (2 * big) / LA_HBMBW * 1e6),
)
ck("l2_aware 输出 gm_reread_bytes", approx(rR["memory"]["gm_reread_bytes"], big))

# 6) bound 分类
ck(
    "mte2_bound",
    run(memory={"gm_read_bytes": B, "gm_write_bytes": 1000}, vector=TINY)["bound"]
    == "mte2_bound",
)
ck(
    "mte3_bound",
    run(memory={"gm_read_bytes": 1000, "gm_write_bytes": B}, vector=TINY)["bound"]
    == "mte3_bound",
)
ck(
    "vector_bound",
    run(memory={"gm_read_bytes": 1000}, vector=BIG)["bound"] == "vector_bound",
)
ck(
    "contention 合并后 bound 归读侧",
    run(
        memory={"gm_read_bytes": B, "gm_write_bytes": 1000},
        gm_rw_contention=True,
        vector=TINY,
    )["bound"]
    == "mte2_bound",
)

# 7) 尾块负载均衡 eta：显式 tile_count=65（无 core_split_units→回退 tile_count）, cores=64
eta = (65 / 64) / math.ceil(65 / 64)
rb = run(memory={"gm_read_bytes": 1000}, tile_count=65, vector=BIG)
ck("core_balance eta", approx(rb["core_balance"], eta))
ck(
    "vector_time = pre/eta",
    approx(rb["vector_time_us"], rb["vector_time_pre_balance_us"] / eta),
)

# 7b) core_split_units（分核单位）覆盖 tile_count 决定 core_balance：
# tiling 给 tile_count=1（若误用→eta≈0.016），但分核单位=96 → eta=(96/64)/ceil(96/64)=0.75。
til = {
    "core_split_units": 96,
    "total_tile_units": 1,
    "ub_bytes_per_unit": 1,
    "double_buffer": 1,
}
rcs = run(memory={"gm_read_bytes": 1000}, tiling=til, vector=BIG)
ck(
    "core_split_units 覆盖 tile_count",
    approx(rcs["core_balance"], (96 / 64) / math.ceil(96 / 64)),
)

# 8) l2_capacity：reread/write 按「工作集 vs C_eff = C/l2_concurrent_streams」阶跃切 L2/HBM。
GBM = HW["common"]["gm_bandwidth_measured"]
CAP = float(GBM["l2_residency_knee_bytes"])
L2R = float(GBM["l2_read_bandwidth_bytes_per_s"])
L2W = float(GBM["l2_write_bandwidth_bytes_per_s"])
HBR = float(GBM["hbm_sustained_read_bytes_per_s"])


def _l2cap(reread=0, write=0, streams=1):
    mem = {
        "gm_read_bytes": reread,
        "gm_write_bytes": write,
        "gm_reread_bytes": reread,
        "reuse_working_set": reread,
        "l2_concurrent_streams": streams,
    }
    return run(memory=mem, gm_bandwidth_model="l2_capacity", vector=TINY)


def us(b, bw):
    return b / bw * 1e6


# streams=1, C_eff=C: reread=C 全命中 L2 / reread=1.5C miss HBM（cold=0 因 gm_read==gm_reread）
ck(
    "l2cap reread≤C 命中L2",
    approx(_l2cap(reread=int(CAP))["mte2_time_us"], us(CAP, L2R)),
)
ck(
    "l2cap reread>C miss HBM",
    approx(_l2cap(reread=int(CAP * 1.5))["mte2_time_us"], us(CAP * 1.5, HBR)),
)
# streams=2, C_eff=C/2: W=C/2 命中 / W=C(>C/2) miss → 证 C_eff=C/streams 生效
ck(
    "l2cap streams=2 W=C/2 命中",
    approx(_l2cap(reread=int(CAP // 2), streams=2)["mte2_time_us"], us(CAP // 2, L2R)),
)
ck(
    "l2cap streams=2 W=C miss(>C_eff)",
    approx(_l2cap(reread=int(CAP), streams=2)["mte2_time_us"], us(CAP, HBR)),
)
# 写阈值：y≤C_eff 走 L2_write / y>C_eff 走 HBM
ck("l2cap write≤C 走L2", approx(_l2cap(write=int(CAP))["mte3_time_us"], us(CAP, L2W)))
ck(
    "l2cap write>C 走HBM",
    approx(_l2cap(write=int(CAP * 1.5))["mte3_time_us"], us(CAP * 1.5, HBR)),
)


def main():
    n = checks
    if fails:
        print(f"FAIL  ({len(fails)} 处 / {n} 项)")
        for f in fails:
            print("  -", f)
        return 1
    print(
        f"PASS  ({n} 项引擎旋钮：带宽口径/读写争用/显式固定开销/launch floor/l2_aware/填充排空/bound/尾块均衡)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
