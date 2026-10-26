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

"""Portable validation receipts shared by develop and optimize (no device imports)."""

import hashlib
import json
import math
import shutil
from pathlib import Path


CONTEXT_SCHEMA = "catlass.dsl.verification.v1"
REPORT_SCHEMA = "catlass.dsl.test-report.v1"
ANTI_HACK_POLICY = "single-fused-catlass-kernel-v1"
ENVIRONMENT_FIELDS = (
    "device",
    "arch",
    "catlass",
    "cann",
    "torch",
    "python",
    "seed",
    "warmup",
    "trials",
    "anti_hack",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_record(path, root):
    root = Path(root).resolve()
    path = Path(path)
    path = root / path if not path.is_absolute() else path
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("验证证据越出项目目录")
    current = path.absolute()
    while current != root:
        if current.is_symlink():
            raise ValueError("验证证据路径不得经过符号链接")
        current = current.parent
    if not resolved.is_file():
        raise ValueError("验证证据文件不存在")
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def checked_record(record, root):
    if not isinstance(record, dict) or file_record(record["path"], root) != record:
        raise ValueError("验证证据摘要已变化")
    return Path(root) / record["path"]


def environment(value):
    if not isinstance(value, dict) or any(
        key not in value for key in ENVIRONMENT_FIELDS
    ):
        raise ValueError("复用要求完整的当前设备、环境和测量参数")
    result = {key: value[key] for key in ENVIRONMENT_FIELDS}
    if result["anti_hack"] not in ("off", "single-fused"):
        raise ValueError("验证环境 anti_hack 非法")
    integers = all(type(result[key]) is int for key in ("seed", "warmup", "trials"))
    if not integers or result["warmup"] < 0 or result["trials"] < 1:
        raise ValueError("验证环境采样参数非法")
    if not isinstance(result["device"], str) or not result["device"]:
        raise ValueError("验证环境缺少 device")
    versions = ("arch", "catlass", "cann", "torch", "python")
    if any(not isinstance(result[key], str) or not result[key] for key in versions):
        raise ValueError("复用要求已确认的实际架构及运行环境版本")
    return result


def build_context(config, current_environment):
    root = config["repository"]["repository_root"]
    reference = config.get("reference_validation", {})
    files = {}
    for key in ("reference", "definition", "workload"):
        record = file_record(reference[key], root)
        if record["sha256"] != reference[key + "_sha256"]:
            raise ValueError("reference/definition/workload 与批准版本不一致")
        files[key] = record
    env = environment(current_environment)
    policy = (
        "single-fused"
        if config.get("fusion", {}).get("mode") == "single_kernel"
        else "off"
    )
    if env["anti_hack"] != policy:
        raise ValueError("当前环境与批准 anti-hack 策略不一致")
    command_rows = config["commands"]
    commands = {row["command_id"]: json.loads(row["argv"]) for row in command_rows}
    project = Path(root)
    dependencies = {
        project / "env.sh",
        project / "pytest.ini",
        project / "pyproject.toml",
    }
    for argv in commands.values():
        for token in argv:
            path = Path(token.split("::", 1)[0])
            path = project / path if not path.is_absolute() else path
            if path.suffix in {".py", ".sh"}:
                dependencies.add(path)
        if any(Path(token).name in {"pytest", "pytest.exe"} for token in argv):
            dependencies.update(project.glob("test_*.py"))
            dependencies.update((project / "tests").rglob("*.py"))
    kernel = (project / config["kernel_path"]).resolve()
    command_files = [
        file_record(path, project)
        for path in sorted(dependencies)
        if path.is_file()
        and path.resolve().is_relative_to(project.resolve())
        and path.resolve() != kernel
    ]
    return {
        "schema": CONTEXT_SCHEMA,
        "config_digest": config["approval"]["config_digest"],
        "files": files,
        "commands": commands,
        "command_files": command_files,
        "required_cases": [
            dict(row)
            for row in config["required_cases"]
            if row["category"] != "performance"
        ],
        "environment": env,
    }


def validate_context(context, root):
    if context.get("schema") != CONTEXT_SCHEMA or not context.get("required_cases"):
        raise ValueError("verification context 非法")
    environment(context["environment"])
    for record in context["files"].values():
        checked_record(record, root)
    for record in context.get("command_files", []):
        checked_record(record, root)
    if set(context["files"]) != {"reference", "definition", "workload"}:
        raise ValueError("verification context 缺少输入摘要")
    if any(
        row["command_id"] not in context["commands"]
        for row in context["required_cases"]
    ):
        raise ValueError("verification context required command 缺失")


def test_receipt(context, kernel_sha256, reports, root, origin):
    """Derive coverage from executor reports on disk, never submission case claims."""
    validate_context(context, root)
    if not isinstance(reports, list) or not reports:
        raise ValueError("缺少测试执行器生成的逐 case 报告")
    covered = set()
    records = []
    for value in reports:
        record = file_record(value, root)
        path = checked_record(record, root)
        report = json.loads(path.read_text(encoding="utf-8"))
        command = report.get("command_id")
        if (
            report.get("schema") != REPORT_SCHEMA
            or report.get("kernel_sha256") != kernel_sha256
            or report.get("context_sha256") != digest(context)
            or report.get("status") != "passed"
            or type(report.get("exit_code")) is not int
            or report["exit_code"] != 0
            or command not in context["commands"]
            or report.get("argv") != context["commands"][command]
            or not isinstance(report.get("cases"), list)
            or not report["cases"]
        ):
            raise ValueError("测试报告与当前源码、命令或验证上下文不一致")
        ids = set()
        for case in report["cases"]:
            case_id = case.get("case_id")
            if (
                not isinstance(case_id, str)
                or case_id in ids
                or case.get("status") != "passed"
            ):
                raise ValueError("测试报告包含缺失、重复或未通过的 case")
            ids.add(case_id)
            for required in context["required_cases"]:
                if (
                    case_id == required["case_id"]
                    and report["argv"] == context["commands"][required["command_id"]]
                ):
                    covered.add(case_id)
        if record in records:
            raise ValueError("测试报告重复")
        records.append(record)
    if covered != {row["case_id"] for row in context["required_cases"]}:
        raise ValueError("测试执行证据未覆盖全部 required cases 和获批命令")
    return {
        "status": "passed",
        "kernel_sha256": kernel_sha256,
        "context": context,
        "fingerprint": digest({"context": context, "kernel_sha256": kernel_sha256}),
        "reports": records,
        "origin": origin,
    }


def validate_receipt(receipt, context, kernel_sha256, root):
    if (
        receipt.get("context") != context
        or receipt.get("kernel_sha256") != kernel_sha256
    ):
        raise ValueError("验证指纹与当前源码、配置或环境不一致")
    paths = [checked_record(record, root) for record in receipt["reports"]]
    regenerated = test_receipt(context, kernel_sha256, paths, root, receipt["origin"])
    if regenerated != receipt:
        raise ValueError("验证记录已变化")


def validate_benchmark(benchmark, context, kernel_sha256):
    if (
        benchmark.get("status") != "passed"
        or benchmark.get("correctness", {}).get("status") != "passed"
    ):
        raise ValueError("最终 benchmark/correctness 未通过")
    if benchmark.get("solution", {}).get("source_sha256") != kernel_sha256:
        raise ValueError("最终 benchmark 与 best 源码不一致")
    if environment(benchmark.get("environment")) != context["environment"]:
        raise ValueError("最终 benchmark 环境或采样参数不一致")
    for key in ("definition", "workload"):
        if benchmark.get(key, {}).get("sha256") != context["files"][key]["sha256"]:
            raise ValueError("最终 benchmark 输入摘要不一致")
    if benchmark.get("profiling", {}).get("status") != "passed":
        raise ValueError("最终 profiling 未通过")
    workloads = benchmark.get("workloads")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("最终 benchmark 缺少 workloads")
    required_ids = {row["case_id"] for row in context["required_cases"]}
    observed = set()
    for row in workloads:
        if row.get("status") != "passed" or row.get("uuid") in observed:
            raise ValueError("最终 benchmark workload 未通过或重复")
        observed.add(row.get("uuid"))
        for side in ("candidate", "reference"):
            value = row.get("performance", {}).get(side, {}).get("mean_ms")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("最终 workload 时延非法")
    if not required_ids.issubset(observed):
        raise ValueError("最终 benchmark 未覆盖 required cases")


def _anti_hack_iterations(source, context):
    """Validate the strict-mode manifest and return its iteration evidence paths."""
    manifest_path = source / "anti_hack_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("最终 workload 缺少 anti_hack_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("anti_hack manifest 非法") from exc
    if not isinstance(manifest, dict):
        raise ValueError("anti_hack manifest 非法")
    trials = context["environment"]["trials"]
    entries = manifest.get("iterations")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("policy") != ANTI_HACK_POLICY
        or manifest.get("profiled_iterations") != trials
        or not isinstance(entries, list)
        or len(entries) != trials
    ):
        raise ValueError("anti_hack manifest 与验证上下文不一致")
    paths = []
    for expected, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("iteration") != expected:
            raise ValueError("anti_hack manifest iteration 顺序非法")
        relative = entry.get("kernel_details")
        pattern = (
            Path("anti_hack")
            / "iteration-{:04d}".format(expected)
            / "kernel_details.csv"
        )
        if not isinstance(relative, str) or Path(relative) != pattern:
            raise ValueError("anti_hack manifest kernel_details 路径非法")
        snapshot = source / pattern
        if not snapshot.is_file() or snapshot.is_symlink():
            raise ValueError("anti_hack 逐轮 kernel_details 缺失")
        if hashlib.sha256(snapshot.read_bytes()).hexdigest() != entry.get("sha256"):
            raise ValueError("anti_hack 逐轮 kernel_details 摘要不一致")
        paths.append(relative)
    return paths


def export_acceptance(
    context, root, kernel_sha256, reports, benchmark_path, destination
):
    """Keep portable reports, full benchmark and required profiler audit evidence."""
    destination = Path(destination)
    benchmark_path = checked_record(file_record(benchmark_path, root), root)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    validate_benchmark(benchmark, context, kernel_sha256)
    # Validate before copying, so partial/legacy reports never produce a receipt.
    test_receipt(context, kernel_sha256, reports, root, {"stage": "optimize.finalize"})
    copied_reports = []
    artifacts = []
    for index, source in enumerate(reports):
        source = checked_record(file_record(source, root), root)
        target = destination / "tests" / ("command-{:04d}.json".format(index))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied_reports.append(target)
    strict = context["environment"]["anti_hack"] == "single-fused"
    for index, workload in enumerate(benchmark["workloads"]):
        source = Path(workload["performance"]["candidate"]["trace_dir"])
        source = Path(root) / source if not source.is_absolute() else source
        target = destination / "profile" / ("case-{:04d}".format(index))
        names = ["kernel_details.csv", "step_trace_time.csv"]
        if strict:
            names.append("anti_hack_manifest.json")
            names.extend(_anti_hack_iterations(source, context))
        found = 0
        for name in names:
            path = source / name
            if not path.exists() and name in (
                "kernel_details.csv",
                "step_trace_time.csv",
            ):
                matches = sorted(source.glob("**/" + name))
                path = matches[0] if matches else path
            if path.exists():
                path = checked_record(file_record(path, root), root)
                output = target / name
                if not output.resolve().is_relative_to(target.resolve()):
                    raise ValueError("profiler 审计路径越界")
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, output)
                artifacts.append(file_record(output, root))
                found += name in ("kernel_details.csv", "step_trace_time.csv")
        if not found:
            raise ValueError("最终 workload 缺少 profiling CSV")
        workload["performance"]["candidate"]["trace_dir"] = str(target.resolve())
    benchmark["profiling"]["candidate"] = {
        "artifact": str((destination / "profile").resolve())
    }
    target = destination / "benchmark.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(benchmark, ensure_ascii=False), encoding="utf-8")
    receipt = test_receipt(
        context, kernel_sha256, copied_reports, root, {"stage": "optimize.finalize"}
    )
    return {
        "schema": "catlass.dsl.acceptance.v1",
        "tests": receipt,
        "benchmark": file_record(target, root),
        "artifacts": artifacts,
    }


def check_acceptance(acceptance, context, kernel_sha256, root):
    if acceptance.get("schema") != "catlass.dsl.acceptance.v1" or not acceptance.get(
        "artifacts"
    ):
        raise ValueError("缺少最终验收记录")
    validate_receipt(acceptance["tests"], context, kernel_sha256, root)
    for record in acceptance["artifacts"]:
        checked_record(record, root)
    path = checked_record(acceptance["benchmark"], root)
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    validate_benchmark(benchmark, context, kernel_sha256)
    return benchmark
