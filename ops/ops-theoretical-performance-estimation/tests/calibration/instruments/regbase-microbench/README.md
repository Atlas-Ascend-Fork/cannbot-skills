# RegBase Microbenchmark

This directory is an independent Ascend950 / DAV_3510 RegBase micro-instruction benchmark. It does not update the existing `ops-theoretical-performance-estimation` skill data.

The first version measures a representative calibration set with known throughput classes. After the calibration set matches the expected class, extend the manifest and kernel with missing RegBase document APIs.

## What It Measures

The kernel records `AscendC::GetSystemCycle()` before and after an empty VF loop and a measured VF loop. Each block writes raw cycle counters to GM. The host prints JSONL records, and `scripts/analyze_results.py` computes:

- `body_cycles = measured_cycles - empty_cycles`
- `issue_per_cycle = iterations * unroll / body_cycles`
- `elements_per_cycle_per_core = issue_per_cycle * elements_per_inst`
- expected class match from `config/calibration_cases.json`

The timing protocol is documented in `docs/regbase_microbench_method.md`.

## Run

```bash
cd regbase-microbench
bash run.sh add_fp32_independent 20000 16 1
```

Arguments:

```text
run.sh <case_name> [iterations] [unroll] [block_dim]
```

Defaults are `iterations=20000`, `unroll=16`, `block_dim=1`.

The host prints raw JSONL to `output/raw_<case>.jsonl`; the analyzer writes `output/report_<case>.json`.

## Calibration Cases

Implemented kernel cases:

- `add_fp32_independent`
- `mul_fp32_independent`
- `cast_f32_to_f16`
- `muladddst_fp32`
- `sqrt_fp32`
- `reducesum_fp32`

The manifest also records planned cases such as low-precision Cast and the document APIs that still need measured data.

## Notes

- Ascend950 `GetSystemCycle()` uses a 1 GHz cycle basis, so `time_us = cycles / 1000`.
- Use `block_dim=1` for single-core instruction throughput calibration.
- Multi-core runs are useful for core-to-core variance, not for replacing single-core issue-rate classification.
