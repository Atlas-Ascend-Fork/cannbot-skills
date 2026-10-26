# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Verify skipped executions from receipts, including the real CLI archive path."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from test_develop_state import (
    approved_contract,
    benchmark_result,
    package_solution,
    result,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def project_with_context(tmp_path, strict=False):
    module, path, root, head = approved_contract(tmp_path, performance=True)
    state = module.start_state_at(module.parse_contract(path), "focused_test")
    config = state["config"]
    reference = {}
    for key, name in (
        ("reference", "reference.py"),
        ("definition", "definition.json"),
        ("workload", "workload.jsonl"),
    ):
        file = root / name
        file.write_bytes(
            ("def run(x):\n    return x\n" if key == "reference" else "{}\n").encode()
        )
        reference[key] = name
        reference[key + "_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    config["reference_validation"] = reference
    config["required_cases"].append(
        dict(config["required_cases"][0], case_id="boundary-tail", category="boundary")
    )
    config["fusion"] = {"mode": "single_kernel" if strict else "performance"}
    config["approval"]["config_digest"] = module.config_digest(config)
    state["config_digest"] = module.config_digest(config)
    env = {
        "device": "npu:0",
        "arch": "test",
        "catlass": "test",
        "cann": "test",
        "torch": "test",
        "python": "test",
        "seed": 1,
        "warmup": 1,
        "trials": 2,
        "anti_hack": "single-fused" if strict else "off",
    }
    context = module.verification.build_context(config, env)
    return module, root, head, state, env, context


def report(root, context, head, path=None):
    return write_json(
        path or root / "reports/full-test.json",
        {
            "schema": "catlass.dsl.test-report.v1",
            "status": "passed",
            "exit_code": 0,
            "kernel_sha256": head,
            "context_sha256": hashlib.sha256(
                json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "command_id": "full-test",
            "argv": context["commands"]["full-test"],
            "cases": [
                {"case_id": row["case_id"], "status": "passed"}
                for row in context["required_cases"]
            ],
        },
    )


def complete_benchmark(root, context, head, directory):
    value = benchmark_result(head, target_met=True)
    value["correctness"] = {"status": "passed"}
    value["performance"]["candidate"]["std_ms"] = 0.01
    value["environment"] = dict(
        context["environment"], config=str(root / "workload.jsonl")
    )
    value["definition"] = {"sha256": context["files"]["definition"]["sha256"]}
    value["workload"] = {"sha256": context["files"]["workload"]["sha256"]}
    value["profiling"]["candidate"]["artifact"] = str(directory / "profile")
    strict = context["environment"]["anti_hack"] == "single-fused"
    for index, (workload, required) in enumerate(
        zip(value["workloads"], context["required_cases"])
    ):
        workload["uuid"] = required["case_id"]
        trace = directory / "profile" / ("case-{:04d}".format(index))
        trace.mkdir(parents=True)
        (trace / "kernel_details.csv").write_bytes(
            b"Name,Type,OP State,Duration(us)\n"
            b"fused_kernel,fused_kernel,N/A,100\n"
            b"fused_kernel,fused_kernel,N/A,100\n"
        )
        workload["performance"]["candidate"]["trace_dir"] = str(trace)
        if strict:
            iterations = []
            for iteration in range(2):
                path = (
                    trace
                    / "anti_hack"
                    / ("iteration-{:04d}".format(iteration))
                    / "kernel_details.csv"
                )
                path.parent.mkdir(parents=True)
                path.write_bytes(
                    b"Name,Type,OP State,Duration(us)\nfused_kernel,fused_kernel,N/A,100\n"
                )
                iterations.append(
                    {
                        "iteration": iteration,
                        "kernel_details": path.relative_to(trace).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "observed_kernel_names": ["fused_kernel"],
                        "observed_launches": 1,
                    }
                )
            write_json(
                trace / "anti_hack_manifest.json",
                {
                    "schema_version": 1,
                    "policy": "single-fused-catlass-kernel-v1",
                    "profiled_iterations": 2,
                    "iterations": iterations,
                },
            )
        else:
            workload["anti_hack"] = {"status": "disabled", "policy": "off"}
    if not strict:
        value["anti_hack"] = {"status": "disabled", "policy": "off"}
    return write_json(directory / "result.json", value)


def final_acceptance(module, root, context, head):
    run = root / ".catlass-dsl/optimize-runs/reuse-20261022-000000"
    final = run / "final"
    test_report = report(root, context, head, final / "raw/test.json")
    benchmark = complete_benchmark(root, context, head, final / "raw/benchmark")
    acceptance = module.verification.export_acceptance(
        context, root, head, [test_report], benchmark, final / "acceptance"
    )
    payload = {"acceptance": acceptance}
    path = write_json(final / "result.json", payload)
    write_json(
        run / "state.json",
        {
            "schema": "catlass.dsl.optimization.v5",
            "status": "passed",
            "next_action": "complete",
            "finalization": {"kernel_sha256": head, "verification": payload},
        },
    )
    return path


def advance(module, state, env, head, stage=None, **extra):
    return module.advance_state(
        state["config"],
        state,
        result(
            stage or state["next_action"], head, verification_environment=env, **extra
        ),
    )


def test_export_acceptance_rejects_tampered_strict_manifest(tmp_path):
    module, root, head, _state, _env, context = project_with_context(
        tmp_path, strict=True
    )
    final = root / "final"
    test_report = report(root, context, head, final / "raw/test.json")
    benchmark = complete_benchmark(root, context, head, final / "raw/benchmark")
    trace = Path(
        json.loads(benchmark.read_text())["workloads"][0]["performance"]["candidate"][
            "trace_dir"
        ]
    )
    manifest_path = trace / "anti_hack_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    manifest_path.write_text(json.dumps(dict(manifest, policy="off")), encoding="utf-8")
    with pytest.raises(ValueError, match="anti_hack manifest"):
        module.verification.export_acceptance(
            context, root, head, [test_report], benchmark, final / "acceptance-policy"
        )

    manifest_path.write_text(
        json.dumps(dict(manifest, profiled_iterations=3)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="anti_hack manifest"):
        module.verification.export_acceptance(
            context, root, head, [test_report], benchmark, final / "acceptance-trials"
        )

    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    snapshot = trace / manifest["iterations"][0]["kernel_details"]
    snapshot.write_bytes(
        b"Name,Type,OP State,Duration(us)\nfused_kernel,fused_kernel,N/A,200\n"
    )
    with pytest.raises(ValueError, match="摘要不一致"):
        module.verification.export_acceptance(
            context, root, head, [test_report], benchmark, final / "acceptance-sha"
        )


def test_complete_focused_execution_skips_duplicate_full_test(tmp_path):
    module, root, head, state, env, context = project_with_context(tmp_path)
    calls = Counter()
    for _ in range(4):
        stage = state["next_action"]
        if stage == "benchmark":
            break
        if stage in {"focused_test", "full_test"}:
            calls["full_test"] += 1
            state = advance(
                module,
                state,
                env,
                head,
                test_reports=[str(report(root, context, head))],
            )
        else:
            state = advance(module, state, env, head)
    assert state["next_action"] == "benchmark"
    assert calls["full_test"] == 1
    assert [event["stage"] for event in state["events"]] == [
        "focused_test",
        "final_review",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "skipped",
        "not_run",
        "failed",
        "argv",
        "exit_code",
        "source",
        "no_report",
    ],
)
def test_focused_claims_cannot_replace_complete_executor_reports(tmp_path, mutation):
    module, root, head, state, env, context = project_with_context(tmp_path)
    path = report(root, context, head)
    payload = json.loads(path.read_text())
    if mutation == "partial":
        payload["cases"].pop()
    elif mutation in {"skipped", "not_run", "failed"}:
        payload["cases"][0]["status"] = mutation
    elif mutation == "argv":
        payload["argv"].append("-k partial")
    elif mutation == "exit_code":
        payload["exit_code"] = 1
    elif mutation == "source":
        payload["kernel_sha256"] = "0" * 64
    write_json(path, payload)
    extra = {} if mutation == "no_report" else {"test_reports": [str(path)]}
    state = advance(module, state, env, head, cases=payload["cases"], **extra)
    state = advance(module, state, env, head)
    assert state["next_action"] == "full_test"
    assert not state["tests"].get("validation")


@pytest.mark.parametrize("strict", [False, True])
def test_optimize_handoff_requires_only_review_and_finish(tmp_path, strict):
    module, root, head, state, env, context = project_with_context(tmp_path, strict)
    path = final_acceptance(module, root, context, head)
    calls = Counter(full_test=1, benchmark=1)  # optimize finalization's actual gates.
    state["next_action"] = "optimize"
    state = advance(module, state, env, head, optimize_result=str(path))
    assert state["next_action"] == "final_review"
    for _ in range(8):
        stage = state["next_action"]
        if stage == "complete":
            break
        if stage == "full_test":
            calls["full_test"] += 1
        elif stage == "benchmark":
            calls["benchmark"] += 1
            payload = benchmark_result(head, target_met=True)
            payload["verification_environment"] = env
            state = module.advance_state(state["config"], state, payload)
            continue
        state = advance(module, state, env, head)
    assert state["next_action"] == "complete"
    assert calls == {"full_test": 1, "benchmark": 1}
    assert [event["stage"] for event in state["events"]] == [
        "optimize",
        "final_review",
        "finish",
    ]
    assert state["tests"]["reused_from"]["path"].endswith("final/result.json")


@pytest.mark.parametrize(
    "mutation",
    [
        "source",
        "reference",
        "definition",
        "workload",
        "command",
        "cases",
        "device",
        "arch",
        "cann",
        "seed",
        "trials",
        "policy",
        "profile",
        "report",
        "missing_context",
        "not_final",
    ],
)
def test_handoff_falls_back_when_any_binding_changes(tmp_path, mutation):
    module, root, head, state, env, context = project_with_context(tmp_path)
    path = final_acceptance(module, root, context, head)
    config = state["config"]
    if mutation == "source":
        kernel = root / config["kernel_path"]
        kernel.write_bytes(b"def run(x):\n    return x + 1\n")
        package_solution(kernel)
        head = hashlib.sha256(kernel.read_bytes()).hexdigest()
    elif mutation in {"reference", "definition", "workload"}:
        (root / context["files"][mutation]["path"]).write_bytes(b"changed")
    elif mutation == "command":
        config["commands"][0]["argv"] = '["python3","-m","pytest","-q"]'
    elif mutation == "cases":
        config["required_cases"].append(
            dict(config["required_cases"][0], case_id="new-case")
        )
    elif mutation in {"device", "arch", "cann"}:
        env[mutation] += "changed"
    elif mutation in {"seed", "trials"}:
        env[mutation] += 1
    elif mutation == "policy":
        env["anti_hack"] = "single-fused"
    elif mutation in {"profile", "report"}:
        acceptance = json.loads(path.read_text())["acceptance"]
        record = (
            acceptance["artifacts"][0]
            if mutation == "profile"
            else acceptance["tests"]["reports"][0]
        )
        (root / record["path"]).write_bytes(b"tampered")
    elif mutation == "missing_context":
        env = {}
    elif mutation == "not_final":
        path = write_json(
            path.parent.parent / "traces/iter-001/result.json",
            json.loads(path.read_text()),
        )
    config["approval"]["config_digest"] = module.config_digest(config)
    state["config_digest"] = module.config_digest(config)
    state["next_action"] = "optimize"
    state = advance(module, state, env, head, optimize_result=str(path))
    assert state["next_action"] == "benchmark"
    assert state["events"][-1]["reuse"]["status"] == "not_applicable"


def test_finish_rechecks_receipts_and_reschedules_changed_profile(tmp_path):
    module, root, head, state, env, context = project_with_context(tmp_path)
    path = final_acceptance(module, root, context, head)
    state["next_action"] = "optimize"
    state = advance(module, state, env, head, optimize_result=str(path))
    state = advance(module, state, env, head)
    assert state["next_action"] == "finish"
    artifact = json.loads(path.read_text())["acceptance"]["artifacts"][0]
    (root / artifact["path"]).unlink()
    state = advance(module, state, env, head)
    assert state["status"] == "not_run" and state["next_action"] == "benchmark"


def test_cli_archive_preserves_report_receipt_after_pruning(tmp_path):
    module, root, head, state, env, context = project_with_context(tmp_path)
    run = root / state["config"]["delivery"]["evidence_directory"]
    trace = run / "traces/iter-001-focused_test"
    test_report = report(root, context, head, trace / "raw/test.json")
    payload = result(
        "focused_test",
        head,
        verification_environment=env,
        test_reports=[str(test_report)],
    )
    payload["evidence"] = [test_report.relative_to(root).as_posix()]
    submission = write_json(trace / "submission.json", payload)
    state = module.advance_state(state["config"], state, payload)
    module._record_stage(
        state["config"],
        {"events": [], "next_action": "focused_test"},
        submission,
        payload,
        state,
    )
    assert not test_report.exists()
    assert (trace / "validation/command-0000.json").is_file()
    state = advance(module, state, env, head)
    assert state["next_action"] == "benchmark"


@pytest.mark.parametrize("strict", [False, True])
def test_finish_archive_keeps_reused_workload_profiles(tmp_path, strict):
    module, root, head, state, env, context = project_with_context(tmp_path, strict)
    source = final_acceptance(module, root, context, head)
    state["next_action"] = "optimize"
    state = advance(module, state, env, head, optimize_result=str(source))
    state = advance(module, state, env, head)
    run = root / state["config"]["delivery"]["evidence_directory"]
    trace = run / "traces/iter-003-finish"
    evidence = write_json(trace / "finish.json", {"status": "passed"})
    payload = result("finish", head, verification_environment=env)
    payload["evidence"] = [evidence.relative_to(root).as_posix()]
    submission = write_json(trace / "submission.json", payload)
    previous = {"events": list(state["events"]), "next_action": "finish"}
    for index, event in enumerate(state["events"], 1):
        event["sequence"] = index
    state = module.advance_state(state["config"], state, payload)
    module._record_stage(state["config"], previous, submission, payload, state)
    assert len(list((run / "final/profile").glob("case-*/kernel_details.csv"))) == 2
    assert not list((run / "final/profile").rglob("anti_hack_manifest.json"))
    final = json.loads((run / "final/result.json").read_text(encoding="utf-8"))
    assert final["benchmark"]["benchmark"]["reused_from"]["path"].endswith(
        "final/result.json"
    )


def test_changed_test_harness_invalidates_focused_receipt(tmp_path):
    module, root, head, state, env, _context = project_with_context(tmp_path)
    test = root / "tests/test_operator.py"
    test.parent.mkdir()
    test.write_text("def test_operator():\n    assert True\n", encoding="utf-8")
    context = module.verification.build_context(state["config"], env)
    state = advance(
        module, state, env, head, test_reports=[str(report(root, context, head))]
    )
    test.write_text("def test_operator():\n    assert False\n", encoding="utf-8")
    state = advance(module, state, env, head)
    assert state["next_action"] == "full_test"


def test_export_context_cli_and_old_states_remain_compatible(tmp_path, capsys):
    module, root, _head, state, env, context = project_with_context(tmp_path)
    state_path = write_json(root / "state.json", state)
    env_path = write_json(root / "environment.json", env)
    output = root / "verification-context.json"
    assert (
        module.main(
            [
                "export-verification-context",
                "--state",
                str(state_path),
                "--environment",
                str(env_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert json.loads(output.read_text()) == context
    assert json.loads(state_path.read_text()) == state  # Export does not mutate a run.


@pytest.mark.parametrize(
    "focused", [[], ["unknown"], ["correctness-main", "correctness-main"]]
)
def test_design_rejects_invalid_focused_case_selection(tmp_path, focused):
    module, _root, _head, state, _env, _context = project_with_context(tmp_path)
    state["config"]["focused_cases"] = focused
    state["config"]["approval"]["config_digest"] = module.config_digest(state["config"])
    with pytest.raises(module.ContractError, match="focused_cases"):
        module.validate_contract(state["config"], require_approved=True)
