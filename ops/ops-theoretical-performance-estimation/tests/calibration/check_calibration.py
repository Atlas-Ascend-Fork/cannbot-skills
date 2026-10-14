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

"""离线 ST：标定一致性检查 —— 把引擎/JSON 里的标定常量「机械地」绑定到上板 bench 实测。

目的：把「不许为迎合算子凑常量」从口头铁律变成门禁。每个被标定的常量都必须能由
bench_results/ 里的上板实测推出来；若有人改了常量却没有对应 bench 数据背书，本检查挂。

注意分工：
- bench（bench_results/*.csv）是常量的**来源/真值**（上板生成，仪器见 tests/onboard/kernels/）。
- 算子 onboard（tests/onboard/compare.py 的 add/rms_norm/vchain）是**独立验证集**，不参与标定。
- 本检查只验「JSON 常量 ⟷ bench 实测」一致，不碰算子。

不需要板子（读已记录的 bench CSV + 跑引擎即可），故纳入离线门禁。
用法: python3 tests/calibration/check_calibration.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[2]
BENCH = Path(__file__).resolve().parent / "bench_results"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
engine = importlib.import_module("evaluate_a5_vector_roofline")

HW = engine.load_json(SKILL_DIR / "references/hardware/ascend950pr.json")
IDB = engine.load_json(SKILL_DIR / "references/vector_instruction_types.json")

fails: list[str] = []


def ck(label, cond, detail=""):
    if not cond:
        fails.append(f"{label}  {detail}")


def read_csv(name):
    rows = []
    for line in (BENCH / name).read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(s)
    hdr = rows[0].split(",")
    return [dict(zip(hdr, r.split(","))) for r in rows[1:]]


def eval_case(instrs, *, mem, gm_model=None, cores=64):
    cd = {
        "case_id": "calib",
        "active_vector_cores": cores,
        "tiling": {
            "total_tile_units": 1 << 24,
            "ub_bytes_per_unit": 8,
            "double_buffer": 2,
        },
        "memory": mem,
        "vector": {"instructions": instrs},
    }
    if gm_model:
        cd["gm_bandwidth_model"] = gm_model
    return engine.evaluate_case(cd, HW, IDB)


# ── 检查 1：l2_aware GM 带宽曲线 ⟷ membw 拷贝扫描 ────────────────────────────
# 引擎在 l2_aware 下对「纯拷贝(读+写各 combined/2)」的 GM 时间，应复现 bench 实测 copy_us。
# 这把 ascend950pr.json 的 l2_aware 曲线机械绑定到 membw 扫描；曲线被改而无 bench 背书即挂。
def check_l2_aware():
    sweep = read_csv("membw_copy_sweep.csv")
    tiny_vector = [{"op": "Add", "dtype": "fp32", "elements": 1}]
    checked = 0
    for row in sweep:
        combined = float(row["combined_MB"]) * 1024 * 1024
        meas = float(row["copy_us"])
        if (
            combined < 100 * 1024 * 1024
        ):  # <~L2: 受 warmup/开销影响大，扫描点不稳，跳过(曲线只用 ≥128MB 锚点)
            continue
        r = eval_case(
            tiny_vector,
            mem={"gm_read_bytes": combined / 2, "gm_write_bytes": combined / 2},
            gm_model="l2_aware",
        )
        pred = r["kernel_body_time_us"]
        rel = abs(pred - meas) / meas
        ck(
            "l2_aware ⟷ membw",
            rel <= 0.15,
            f"combined={row['combined_MB']}MB 引擎={pred:.1f} 实测={meas:.1f} 偏差={rel * 100:.0f}%>15%",
        )
        checked += 1
    return checked


# ── 检查 2：单指令吞吐 ⟷ regbase-microbench(单核 cycle 计数) ──────────────
def check_insn_throughput():
    # 引擎单核(active_cores=1, count=iters*unroll, elements=elements_per_inst)矢量时延，应 ≈
    # regbase-microbench 单核实测 microbench_us(=measured_cycles/1000,稳态吞吐真值口径)。
    # 正确口径下引擎与 microbench 近乎一致(t4 transcendental 0.999、single_issue reduce 0.957)，
    # 故容差收紧到 [0.9, 1.1]：能锁死 throughput_cycles(t4→0.999 过；误标 t3.25→0.81 挂、t8→2.0 挂)。
    # ⚠ microbench_us 必须用 measured_cycles 而非 body_cycles：减空循环会虚高 ~23%、把 t4 误判成 t3.25。
    lower_bound, upper_bound = 0.9, 1.1
    rows = read_csv("insn_throughput.csv")
    for row in rows:
        op = row["instruction"]
        cnt = int(row["count"])
        ne = int(row["elements_per_inst"])
        meas = float(row["microbench_us"])
        res = eval_case(
            [{"op": op, "dtype": row["dtype"], "elements": ne, "count": cnt}],
            mem={"gm_read_bytes": 0, "gm_write_bytes": 0},
            cores=1,
        )
        pred = res["vector_time_us"]
        ratio = pred / meas if meas > 0 else 0.0
        ck(
            f"{op} 吞吐 ⟷ microbench",
            lower_bound <= ratio <= upper_bound,
            f"引擎单核={pred:.1f} microbench={meas:.1f} 比值={ratio:.2f}∉[{lower_bound},{upper_bound}]"
            "（throughput_cycles 可能漂了）",
        )
    return len(rows)


# ── 检查 3：HBM 持续带宽 ⟷ membw read/write/copy(@512MB，远超 L2) ──────────────
# 引擎对纯访存的 GM 时间应复现 membw 实测：read/write 用 chip_peak(peak.gm_bandwidth)，
# copy 用 l2_aware(HBM 平台 = l2_aware 曲线尾点)。把 GM 带宽常量绑定到 membw bench。
def check_hbm():
    rows = [r for r in read_csv("membw_bandwidth.csv") if r["regime"] == "HBM"]
    tiny_vector = [{"op": "Add", "dtype": "fp32", "elements": 1}]
    n = 0
    for r in rows:
        byte_count = float(r["mb"]) * 1024 * 1024
        meas = float(r["us"])
        mode = r["mode"]
        if mode == "read":
            res = eval_case(
                tiny_vector, mem={"gm_read_bytes": byte_count, "gm_write_bytes": 0}
            )
        elif mode == "write":
            res = eval_case(
                tiny_vector, mem={"gm_read_bytes": 0, "gm_write_bytes": byte_count}
            )
        else:  # copy: 读+写，l2_aware
            res = eval_case(
                tiny_vector,
                mem={"gm_read_bytes": byte_count, "gm_write_bytes": byte_count},
                gm_model="l2_aware",
            )
        pred = res["kernel_body_time_us"]
        rel = abs(pred - meas) / meas
        ck(
            f"HBM {mode} ⟷ membw",
            rel <= 0.10,
            f"@{r['mb']}MB 引擎={pred:.1f} 实测={meas:.1f} 偏差={rel * 100:.0f}%>10%",
        )
        n += 1
    return n


# ── 检查 4：l2_capacity 容量模型 ⟷ membw read/write ───────────────────────────
# l2_capacity 模型按「复用工作集 W vs L2 容量 C」阶跃切 L2/HBM 带宽。用 membw read/write 锚点验证:
# 接近 C 的 L2 点(112/128)走 L2 带宽、HBM 点(256/512)走 HBM,引擎预测 ≈ membw。
# 小 size(<112)L2 区随尺寸爬坡(ramp,另由 gm_rampup 管)不在此严格校验;160/192=knee(过渡)跳过。
def check_l2_capacity():
    rows = [
        r
        for r in read_csv("membw_bandwidth.csv")
        if r["mode"] in ("read", "write")
        and r["regime"] in ("L2", "HBM")
        and float(r["mb"]) >= 112
    ]
    tiny_vector = [{"op": "Add", "dtype": "fp32", "elements": 1}]
    n = 0
    for r in rows:
        byte_count = float(r["mb"]) * 1024 * 1024
        meas = float(r["us"])
        mode = r["mode"]
        if mode == "read":  # 全 reread(cold=0),W=B → 阈值切 L2/HBM 读
            res = eval_case(
                tiny_vector,
                mem={
                    "gm_read_bytes": byte_count,
                    "gm_write_bytes": 0,
                    "gm_reread_bytes": byte_count,
                    "reuse_working_set": byte_count,
                },
                gm_model="l2_capacity",
            )
        else:  # 写 y=B → 阈值切 L2/HBM 写
            res = eval_case(
                tiny_vector,
                mem={"gm_read_bytes": 0, "gm_write_bytes": byte_count},
                gm_model="l2_capacity",
            )
        pred = res["kernel_body_time_us"]
        rel = abs(pred - meas) / meas
        ck(
            f"l2_capacity {mode}@{r['mb']}MB ⟷ membw",
            rel <= 0.12,
            f"引擎={pred:.1f} 实测={meas:.1f} 偏差={rel * 100:.0f}%>12%",
        )
        n += 1
    return n


def main():
    n1 = check_l2_aware()
    n2 = check_insn_throughput()
    n3 = check_hbm()
    n4 = check_l2_capacity()
    total = n1 + n2 + n3 + n4
    if fails:
        print(f"FAIL  ({len(fails)} 处 / {total} 项标定一致性)")
        for f in fails:
            print("  -", f)
        return 1
    print(
        f"PASS  ({total} 项标定一致性：l2_aware 曲线 {n1} 点 ⟷ membw；"
        f"单指令吞吐 {n2} 项 ⟷ regbase-microbench；HBM 带宽 {n3} 项 ⟷ membw；"
        f"l2_capacity {n4} 点 ⟷ membw)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
