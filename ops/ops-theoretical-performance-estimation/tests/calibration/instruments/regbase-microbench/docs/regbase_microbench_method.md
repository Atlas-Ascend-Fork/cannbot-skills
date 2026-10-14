# RegBase Microbenchmark Method

## Goal

Measure RegBase micro-instruction throughput on Ascend950 / DAV_3510 and use known instruction classes to validate the measurement model before measuring instructions without existing throughput data.

## Timing Protocol

The benchmark uses public `AscendC::GetSystemCycle()` from the aicore side. The measured RegBase operations run inside `__simd_vf__` helpers. Because `GetSystemCycle()` is not placed inside the VF helper, each case runs two VF calls:

1. `empty` VF loop: same load/store, loop, and unroll structure but no target micro-instruction.
2. `measured` VF loop: same structure plus the target micro-instruction.

Both calls are wrapped with `PipeBarrier<PIPE_ALL>()` before reading the end cycle.

```text
empty_cycles = empty_end - empty_start
measured_cycles = measured_end - measured_start
body_cycles = measured_cycles - empty_cycles
```

The kernel writes raw counters to GM. The host prints JSONL records, and the analyzer computes the final metrics.

## Metrics

For each block:

```text
instruction_count = iterations * unroll
issue_per_cycle = instruction_count / body_cycles
elements_per_cycle_per_core = issue_per_cycle * elements_per_inst
```

For fp32 256B RegBase instructions, `elements_per_inst = 64`. A dual-issue fp32 instruction should therefore approach `128 elements/cycle/core`.

## Calibration Acceptance

The first pass uses class matching, not strict peak matching:

| Class | Expected issue/cycle |
| --- | ---: |
| dual_issue | 1.6 - 2.2 |
| single_issue | 0.75 - 1.25 |
| two_cycle_dual_issue | 0.75 - 1.25 |
| four_cycle_dual_issue | 0.35 - 0.65 |

If a known calibration case misses its class, adjust the benchmark template before using it to measure unknown instructions.

## Controls

- Prefer `block_dim=1` when classifying single-core issue rate.
- Increase `iterations` before changing the kernel if cycle counts are too small.
- Keep input load and sink store outside the inner measured loop; the empty loop subtracts their residual cost.
- Keep a GM-written sink value so the compiler has a visible output dependency.

## Limitations

- The first kernel covers only representative fp32/fp16 calibration cases.
- Low-precision Cast and document APIs such as `Gather`, `Pack`, `Interleave`, and `Histograms` are listed in the manifest but need dedicated VF templates before measurement.
- Results are measurement inputs for model review; they are not automatically written back into `vector_instruction_types.json`.
