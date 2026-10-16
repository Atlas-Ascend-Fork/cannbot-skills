#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Eventized workflow state and live-report renderer for ops-perf-optimize."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import html
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from summarize_performance_effects import summarize
from validate_execution_identity import dependency_paths, validate_execution_identity


SCHEMA_VERSION = 3
WORKFLOW_VERSION = "unified-workflow-v12-execution-identity"
PHASES = ("baseline", "preparation", "analysis", "implementation", "integration")
TUNING_MODES = ("standard", "generalization")
STAGE_STATUSES = ("in_progress", "complete", "invalid", "skipped")
TERMINAL_STATUSES = ("success", "failed", "blocked", "awaiting_operator")
TERMINAL_REASON_CODES = (
    "completed",
    "external_blocker",
    "input_invalid",
    "workflow_error",
    "known_baseline_failure",
    "candidate_precision_regression",
    "no_material_improvement",
    "ceiling_bound",
    "structural_headroom_unrealized",
)
# Structural-direction ledger (pursue_optimal). A structural direction proposed by analysis is a
# cross-round obligation that must be discharged before a clean success finish under pursue_optimal.
STRUCTURAL_DIRECTION_STATUSES = (
    "not_attempted",          # proposed, not yet implemented — UNRESOLVED
    "blocked_by_scope",       # frozen scope too narrow — UNRESOLVED (re-freeze + re-attempt)
    "blocked_by_precision",   # attempted, evidenced it breaks the precision contract — resolved
    "infeasible_evidenced",   # attempted, source/model evidence proves infeasibility — resolved
    "realized",               # implemented and selected — resolved
)
UNRESOLVED_STRUCTURAL_STATUSES = ("not_attempted", "blocked_by_scope")
PRECISION_CLASSIFICATIONS = (
    "known_baseline_failure",
    "improved_baseline_failure",
    "new_or_worsened_failure",
    "under_diagnosis",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha(item).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def artifact_identity(raw_path: str | Path) -> dict[str, Any]:
    path = Path(raw_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "sha256": tree_sha(path) if path.is_dir() else file_sha(path),
    }


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_manifest(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if "configuration" not in payload or "collections" not in payload or "inputs" not in payload:
        raise ValueError(f"invalid run manifest: {path}")
    policy = payload["configuration"].get("precision_diagnosis_policy")
    if policy != "persistent":
        raise ValueError(f"invalid precision diagnosis policy: {policy}")
    tuning_mode = payload["configuration"].get("tuning_mode", "standard")
    if tuning_mode not in TUNING_MODES:
        raise ValueError(f"invalid tuning mode: {tuning_mode}")
    target_socs = payload["configuration"].get("target_socs")
    if not isinstance(target_socs, list) or not target_socs:
        raise ValueError("run manifest requires non-empty target_socs")
    if tuning_mode == "generalization":
        if payload["configuration"].get("functional_mode") != "reuse_performance":
            raise ValueError("generalization mode fixes functional mode to reuse_performance")
        generalization = payload["configuration"].get("generalization")
        if not isinstance(generalization, dict):
            raise ValueError("generalization configuration is missing")
        competitor = generalization.get("competitor", {})
        if competitor.get("source") != "file":
            raise ValueError("invalid generalization competitor source")
    return payload


@contextmanager
def state_lock(state_path: Path) -> Iterator[None]:
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_event(state: dict[str, Any], kind: str, **payload: Any) -> None:
    state.setdefault("events", []).append({"at": utc_now(), "kind": kind, **payload})


def stage_first_event_time(state: dict[str, Any], stage: str) -> str | None:
    for event in state.get("events", []):
        if event.get("kind") == "stage" and event.get("stage") == stage and event.get("at"):
            return event["at"]
    return None


def resolve_stage_started_at(state: dict[str, Any], stage: str, record: dict[str, Any]) -> str | None:
    return record.get("started_at") or stage_first_event_time(state, stage)


def manifest_summary(manifest: dict[str, Any]) -> list[str]:
    collections = manifest.get("collections", {})
    tuning_mode = manifest.get("configuration", {}).get("tuning_mode", "standard")
    if tuning_mode == "generalization":
        labels = {
            "generalization_full": "泛化全集",
            "competitor_performance": "竞品性能",
            "baseline_performance": "NPU 基线性能",
            "generalization_tuning": "泛化调优子集",
            "functional_acceptance": "功能验收子集",
            "optimized_involved": "涉及优化用例集",
        }
    else:
        labels = {
            "performance_full": "性能全集",
            "performance_tuning": "性能调优子集",
            "functional_full": "功能全集",
            "functional_acceptance": "功能验收子集",
        }
    rows: list[str] = []
    for key, label in labels.items():
        item = collections.get(key)
        if not item:
            rows.append(f"| {label} | pending | - | - |")
            continue
        selection = item.get("selection", {})
        status = selection.get("confirmation", "frozen")
        rows.append(
            f"| {label} | {item.get('count', '-')} | `{str(item.get('sha256', '-'))[:12]}` | {status} |"
        )
    return rows


def performance_checkpoint(stage: str) -> bool:
    return stage == "final.performance" or bool(
        re.fullmatch(r"(?:gen\.)?round[1-9]\d*\.complete|gen\.outer[1-9]\d*\.exit", stage)
    )


def prepare_performance(args: argparse.Namespace) -> tuple[dict[str, str], list[str]]:
    """Validate existing effect artifacts without collecting performance again."""
    required = (args.performance_summary, args.cumulative_performance_summary, args.selected_candidate)
    if not all(required):
        raise ValueError(
            "performance checkpoint requires --performance-summary, "
            "--cumulative-performance-summary and --selected-candidate"
        )
    references = {
        "round_summary": str(args.performance_summary.resolve()),
        "cumulative_summary": str(args.cumulative_performance_summary.resolve()),
        "selected_candidate": args.selected_candidate,
        "scope": args.performance_scope,
    }
    artifacts: list[str] = []
    summaries = []
    for key in ("round_summary", "cumulative_summary"):
        path = Path(references[key])
        summary = read_json(path)
        if not isinstance(summary, dict) or any(
            not isinstance(summary.get(field), str) for field in ("baseline_path", "candidate_path")
        ):
            raise ValueError(f"{path}: expected an effect summary with performance source paths")
        baseline = Path(summary["baseline_path"])
        candidate = Path(summary["candidate_path"])
        if not baseline.is_absolute() or not candidate.is_absolute():
            raise ValueError(f"{path}: performance source paths must be absolute")
        expected = summarize(baseline, candidate)
        if any(field not in summary or summary[field] != value for field, value in expected.items()):
            raise ValueError(f"{path}: effect summary differs from source performance or jitter policy")
        artifacts.extend([str(path), str(baseline), str(candidate)])
        summaries.append(summary)
    current, cumulative = summaries
    if current["candidate_path"] != cumulative["candidate_path"]:
        raise ValueError("round and cumulative summaries must reference the same selected performance")
    if [case["case_id"] for case in current["cases"]] != [case["case_id"] for case in cumulative["cases"]]:
        raise ValueError("round and cumulative summaries must cover the same ordered cases")
    return references, artifacts


def recorded_performance(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Only display evidence that still matches the completed checkpoint."""
    artifacts = {item["path"]: item for item in record.get("artifacts", [])}
    summaries = []
    for key in ("round_summary", "cumulative_summary"):
        path = record["performance"][key]
        if artifact_identity(path) != artifacts.get(path):
            raise ValueError(f"performance artifact changed or not registered: {path}")
        summary = read_json(Path(path))
        for source in (summary["baseline_path"], summary["candidate_path"]):
            if artifact_identity(source) != artifacts.get(source):
                raise ValueError(f"performance source changed or not registered: {source}")
        summaries.append(summary)
    return summaries[0], summaries[1]


def report_cell(value: Any) -> str:
    return html.escape(str(value)).replace("|", "&#124;").replace("`", "&#96;").replace("\n", "<br>")


def speedup_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}x"


def performance_report(state: dict[str, Any]) -> list[str]:
    lines = ["", "## 当前性能结果", ""]
    checkpoints = [
        (name, record)
        for name, record in state.get("stages", {}).items()
        if performance_checkpoint(name) or record.get("performance")
    ]
    # JSON object keys are sorted on disk; event order resolves equal timestamps.
    completion_order = {
        event.get("stage"): index
        for index, event in enumerate(state.get("events", []))
        if event.get("kind") == "stage" and event.get("status") == "complete"
    }
    checkpoints.sort(key=lambda item: (
        item[1].get("completed_at", item[1].get("updated_at", "")),
        completion_order.get(item[0], -1),
    ))
    completed = [(name, record) for name, record in checkpoints if record.get("completed_at")]
    if not completed:
        return lines + ["- 尚无已完成轮次的性能结果。"]

    available = {}
    errors = {}
    for name, record in completed:
        if record.get("status") != "complete":
            errors[name] = f"保存点状态为 {record.get('status')}"
        elif not record.get("performance"):
            errors[name] = "尚未登记性能摘要；需用已有采集补齐"
        else:
            try:
                available[name] = recorded_performance(record)
            except (OSError, KeyError, ValueError) as error:
                errors[name] = str(error)

    latest_name, latest = completed[-1]
    lines.append(f"- 最近性能保存点：`{report_cell(latest_name)}`（{latest.get('completed_at')}）")
    if latest_name in errors:
        lines.append(f"- 当前性能结果不可用：{report_cell(errors[latest_name])}。")
    else:
        current, cumulative = available[latest_name]
        performance = latest["performance"]
        scope = "性能全集" if performance["scope"] == "full" else "本轮调优子集"
        lines.extend([
            f"- 当前保留实现：`{report_cell(performance['selected_candidate'])}`",
            f"- 统计范围：{scope}，{current['total_case_count']} 个 case；子集结果不代表未测全集。",
            "- 本轮变化相对轮初基线；累计变化相对原始基线，均只统计同一组已实测 case。",
            "- 有效几何加速比只统计超出抖动阈值的 case。",
            "",
            "| 比较 | 总 case | 抖动 | 有效 | 有效提升 | 有效回退 | 有效几何加速比 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for label, summary in (("本轮", current), ("累计", cumulative)):
            lines.append(
                f"| {label} | {summary['total_case_count']} | {summary['jitter_case_count']} | "
                f"{summary['effective_case_count']} | {summary['effective_improvement_count']} | "
                f"{summary['effective_regression_count']} | {speedup_text(summary['effective_geomean_speedup'])} |"
            )
        lines.extend([
            "", "### 逐 case 当前性能", "",
            "差值 = 当前耗时 − 对应基线耗时，负值表示提升；抖动 case 保留展示但不计入有效指标。",
            "",
            "| case_id | 原始基线(us) | 轮初基线(us) | 当前(us) | 本轮差值(us) | 本轮变化(%) | 本轮加速比 | 本轮分类 | "
            "累计差值(us) | 累计变化(%) | 累计加速比 | 累计分类 |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
        ])
        for case, total in zip(current["cases"], cumulative["cases"]):
            lines.append(
                f"| {report_cell(case['case_id'])} | {total['baseline_total_us']:.6f} | "
                f"{case['baseline_total_us']:.6f} | {case['candidate_total_us']:.6f} | "
                f"{case['delta_us']:+.6f} | {case['delta_percent']:+.6f} | "
                f"{speedup_text(case['speedup'])} | {case['classification']} | "
                f"{total['delta_us']:+.6f} | {total['delta_percent']:+.6f} | "
                f"{speedup_text(total['speedup'])} | {total['classification']} |"
            )
        lines.extend([
            "",
            f"- 本轮证据：`{report_cell(performance['round_summary'])}`",
            f"- 累计证据：`{report_cell(performance['cumulative_summary'])}`",
        ])

    lines.extend([
        "", "### 各轮性能汇总", "",
        "各行使用该保存点的实测集合，不跨轮拼接 case 或累计乘算加速比。",
        "",
        "| 保存点 | 保留实现 | 范围 | 总 case | 本轮有效/提升/回退/抖动 | 本轮有效几何加速比 | 累计有效/提升/回退/抖动 | 累计有效几何加速比 | 证据 |",
        "|---|---|---|---:|---|---:|---|---:|---|",
    ])
    for name, record in completed:
        if name in errors:
            continue
        current, cumulative = available[name]
        counts = [
            "/".join(str(summary[field]) for field in (
                "effective_case_count", "effective_improvement_count", "effective_regression_count", "jitter_case_count"
            )) for summary in (current, cumulative)
        ]
        performance = record["performance"]
        scope = "全集" if performance["scope"] == "full" else "调优子集"
        lines.append(
            f"| `{report_cell(name)}` | {report_cell(performance['selected_candidate'])} | "
            f"{scope} | {current['total_case_count']} | {counts[0]} | "
            f"{speedup_text(current['effective_geomean_speedup'])} | {counts[1]} | "
            f"{speedup_text(cumulative['effective_geomean_speedup'])} | "
            f"[本轮]({quote(performance['round_summary'])}) / "
            f"[累计]({quote(performance['cumulative_summary'])}) |"
        )
    if errors:
        lines.append("")
        lines.extend(
            f"- `{report_cell(name)}`：性能结果不可用，{report_cell(error)}。"
            for name, error in errors.items()
        )
    return lines


def structural_directions_report(state: dict[str, Any]) -> list[str]:
    directions = state.get("structural_directions", [])
    intent = state.get("optimization_intent", "safe_incremental")
    if not directions and intent != "pursue_optimal":
        return []
    lines = ["", "## 结构方向台账", ""]
    if intent == "pursue_optimal":
        lines.append(
            "> pursue_optimal：结构方向是跨轮义务。`not_attempted`/`blocked_by_scope` 未了结时，"
            "禁止以 success/completed 收尾（blocked_by_scope 须重冻结更宽 scope 后重试）。"
        )
    if not directions:
        lines.append("- 尚未登记结构方向。")
        return lines
    lines.extend(["", "| 方向 | 状态 | 轮次 | 预估收益 | 说明 | 证据 |", "|---|---|---:|---|---|---|"])
    for d in directions:
        lines.append(
            f"| `{report_cell(d.get('id'))}` | `{report_cell(d.get('status'))}` | "
            f"{report_cell(d.get('round'))} | {report_cell(d.get('estimated_gain'))} | "
            f"{report_cell(d.get('description'))} | {report_cell(d.get('evidence'))} |"
        )
    unresolved = [d.get("id") for d in directions if d.get("status") in UNRESOLVED_STRUCTURAL_STATUSES]
    if unresolved:
        lines.append("")
        lines.append(f"- ⚠ 未了结结构方向：{', '.join(f'`{i}`' for i in unresolved)}（阻止 success/completed 收尾）。")
    return lines


def report_next_action(state: dict[str, Any]) -> str:
    """A lifecycle boundary supersedes the previous stage's action."""
    for event in reversed(state.get("events", [])):
        if event.get("kind") == "finish":
            return "任务已结束" if event.get("status") == "success" else "处理终态记录中的未决事项"
        if event.get("next_action"):
            return report_cell(event["next_action"])
        if event.get("kind") == "stage":
            if event.get("status") == "in_progress":
                return f"等待 `{report_cell(event['stage'])}` 产出并复核"
            return "复核保存点，确定后续阶段"
    return report_cell(state.get("precision", {}).get("next_action") or state.get("next_action") or "等待当前阶段产出")


def report_stage_conclusion(state: dict[str, Any]) -> str:
    for event in reversed(state.get("events", [])):
        if event.get("kind") != "stage" or event.get("status") != "complete":
            continue
        stage = event["stage"]
        record = state.get("stages", {}).get(stage, {})
        if record.get("status") != "complete":
            return f"`{report_cell(stage)}` 当前为 `{report_cell(record.get('status'))}`，原结论待复核"
        message = event.get("message") or "已完成，详见保存点证据"
        return f"`{report_cell(stage)}`：{report_cell(message)}"
    return "尚无已验收阶段结论"


def report_text(state: dict[str, Any]) -> str:
    manifest_path = Path(state["manifest"]["path"])
    manifest = (
        read_manifest(manifest_path)
        if manifest_path.exists()
        else {"inputs": {}, "collections": {}, "configuration": {}}
    )
    configuration = manifest.get("configuration", {})
    current = state.get("current_stage") or ("阶段间歇" if state.get("stages") else "未开始")
    terminal = state.get("terminal", {})
    operator = state.get("operator", {})
    precision = state.get("precision", {})
    objective = configuration.get("optimization_objective", {})
    tuning_mode = configuration.get("tuning_mode", "standard")
    if objective.get("mode") == "fixed_rounds":
        objective_text = f"固定 {objective.get('rounds', '-')} 轮"
    elif objective.get("mode") == "target":
        comparison = {"ge": ">=", "le": "<=", "eq": "="}.get(objective.get("comparison"), "?")
        objective_text = (
            f"{objective.get('metric', '-')} {comparison} {objective.get('value', '-')} "
            f"{objective.get('unit') or ''}"
        ).strip()
        if objective.get("max_rounds") is not None:
            objective_text += f"（最多 {objective['max_rounds']} 轮）"
    else:
        objective_text = "未配置"
    environments = configuration.get("environments", {})
    python_environment = environments.get("python", {})
    cann_environment = environments.get("cann", {})
    request = manifest.get("request", {})

    lines = [
        "# 性能调优报告",
        "",
        "> 本报告由 `run-manifest.json` 与 `workflow-state.json` 持续生成；既是执行现场，也是最终交付。",
        "",
        "## 当前结论",
        "",
        f"- 任务状态：`{terminal.get('status', 'running')}`",
        f"- 当前阶段：`{current}`",
        f"- 调优模式：`{tuning_mode}`",
        f"- 优化意图：`{state.get('optimization_intent', configuration.get('optimization_intent', 'safe_incremental'))}`",
        f"- 性能参考锚点：`{state.get('reference_perf') or configuration.get('reference_perf') or '未设置'}`",
        f"- 已验证目标 SoC：`{', '.join(configuration.get('target_socs', []))}`",
        f"- 优化目标：{objective_text}",
        f"- Python：`{python_environment.get('executable', '-')}` ({python_environment.get('version', '-')})",
        f"- Python 激活命令：`{python_environment.get('activation_command') or '无需'}`",
        f"- CANN 环境：`{cann_environment.get('setup_command', '-')}`",
        f"- 最近验收结论：{report_stage_conclusion(state)}",
        f"- 下一动作：{report_next_action(state)}",
        f"- 安全恢复点：{operator.get('safe_restore_point') or state.get('safe_restore_point') or '输入已冻结'}",
        f"- 最后更新：{state.get('updated_at', '-')}",
    ]
    if tuning_mode == "generalization":
        generalization = configuration.get("generalization", {})
        thresholds = generalization.get("ratio_thresholds", {})
        lines.extend(
            [
                f"- 竞品来源：`{generalization.get('competitor', {}).get('source', '-')}`",
                f"- 泛化目标用例数：{generalization.get('target_case_count', '-')}",
                f"- Ratio 门限：geomean ≥ {thresholds.get('geomean', 1.0)}，min ≥ {thresholds.get('minimum', 0.6)}",
                f"- 循环上限：内层 {generalization.get('inner_max_rounds', 3)} 轮，"
                f"外层 {generalization.get('outer_max_iterations', 2)} 次",
            ]
        )
    reference = configuration.get("performance_reference")
    if reference:
        lines.append(f"- 性能对标：`{report_cell(reference['kind'])}`；目标比率：{reference['target_ratio']}")
    if configuration.get("workflow_layout") == "five-stage-v1":
        lines.extend(["", "## 五阶段交接", "", "| 阶段 | 状态 |", "|---|---|"])
        for phase, label in zip(PHASES, ("基线评估", "前处理", "诊断分析", "实施验证", "后处理")):
            record = state.get("stages", {}).get(f"phase.{phase}", {})
            lines.append(f"| {label} | {record.get('status', 'pending')} |")
        for phase in ("baseline", "integration"):
            record = state.get("stages", {}).get(f"phase.{phase}", {})
            for artifact in record.get("artifacts", []):
                name = Path(artifact["path"]).name
                if name in ("环境检查报告.md", "性能目标.md", "VERIFY_REPORT.md"):
                    lines.append(f"- [{name}]({quote(artifact['path'])})（{record.get('status', 'pending')}）")
    lines.extend(performance_report(state))
    lines.extend(structural_directions_report(state))
    functional_status = (
        manifest.get('collections', {}).get('functional_full', {})
        .get('validation', {}).get('status', 'pending')
    )
    lines.extend(
        [
            "",
            "## 输入与冻结集合",
            "",
            "| 集合 | 数量 | SHA-256 | 状态 |",
            "|---|---:|---|---|",
            *manifest_summary(manifest),
            "",
            f"- 优化请求：`{request.get('path', '-')}` (`{str(request.get('sha256', '-'))[:12]}`)",
            f"- 代码目录：`{manifest.get('inputs', {}).get('code', {}).get('path', '-')}`",
            f"- 功能模式：`{configuration.get('functional_mode', '-')}`",
            f"- 功能库存静态检查：`{functional_status}`",
            "",
            "## 阶段",
            "",
            "| 阶段 | 状态 | 开始 | 完成 | 耗时(s) | 证据 |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    stages = state.get("stages", {})
    if stages:
        for name, record in stages.items():
            elapsed = record.get('elapsed_seconds')
            lines.append(
                f"| `{name}` | {record.get('status', '-') or '-'} | {report_cell(record.get('started_at') or '-')} | "
                f"{report_cell(record.get('completed_at') or '-')} | "
                f"{report_cell(elapsed if elapsed is not None else '-')} | "
                f"{len(record.get('artifacts', []))} |"
            )
    else:
        lines.append("| - | 未开始 | - | - | - | 0 |")

    lines.extend(["", "## 精度状态", ""])
    baseline = precision.get("known_baseline_failures", [])
    regressions = precision.get("new_or_worsened_failures", [])
    lines.append(f"- 已证实基线失败：{len(baseline)}")
    for item in baseline:
        lines.append(f"  - `{item.get('case_id', '-')}`：{item.get('message', '-')} ({item.get('evidence', '-')})")
    lines.append(f"- 候选新增/恶化失败：{len(regressions)}")
    for item in regressions:
        lines.append(f"  - `{item.get('case_id', '-')}`：{item.get('message', '-')} ({item.get('evidence', '-')})")
    lines.append(f"- 当前假设：{precision.get('current_hypothesis') or '无'}")
    excluded = precision.get("excluded_hypotheses", [])
    lines.append(f"- 已排除假设：{'; '.join(excluded) if excluded else '无'}")

    lines.extend(["", "## 最近进展", ""])
    progress_events = [
        event
        for event in state.get("events", [])
        if event.get("kind") in ("progress", "diagnosis", "decision", "stage")
    ]
    if progress_events:
        for event in progress_events[-12:]:
            message = event.get("message") or f"{event.get('stage', '-')} → {event.get('status', '-')}"
            lines.append(f"- {event.get('at', '-')} `{event.get('kind')}`：{message}")
    else:
        lines.append("- 输入已冻结，等待首个阶段。")

    lines.extend(["", "## 候选与逐轮选择", ""])
    candidate_events = [event for event in state.get("events", []) if event.get("candidate")]
    latest_candidates: dict[str, dict[str, Any]] = {}
    for event in candidate_events:
        latest_candidates[event["candidate"]] = event
    if latest_candidates:
        lines.extend(["| 候选 | 阶段 | 最新结论 | 证据 |", "|---|---|---|---|"])
        for candidate, event in sorted(latest_candidates.items()):
            details = event.get("details") or {}
            conclusion = event.get("message", "-")
            if details:
                conclusion += "；" + "，".join(f"{key}={value}" for key, value in sorted(details.items()))
            lines.append(
                f"| `{candidate}` | `{event.get('stage', '-')}` | {conclusion} | {event.get('evidence', '-')} |"
            )
    else:
        lines.append("- 尚无候选结论。")
    selection_stages = [
        (name, record)
        for name, record in stages.items()
        if name.endswith(".selection") or name == "final.selection"
    ]
    for name, record in selection_stages:
        context = "；".join(record.get("context", [])) or "见登记 artifact"
        lines.append(f"- `{name}`：{record.get('status', '-')}；{context}")

    lines.extend(["", "## 人工介入", ""])
    if operator.get("awaiting"):
        lines.append("- 状态：等待人工")
        lines.append(f"- 问题：{operator.get('question', '-')}")
    else:
        lines.append("- 状态：无需人工输入")
    if operator.get("last_decision"):
        lines.append(f"- 最近决策：{operator['last_decision']}")

    lines.extend(["", "## 最终验收", ""])
    if terminal:
        lines.append(f"- 终态：`{terminal.get('status')}`")
        lines.append(f"- 原因码：`{terminal.get('reason_code', '-')}`")
        lines.append(f"- 结论：{terminal.get('message', '-')} ")
        lines.append(f"- 完成时间：{terminal.get('at', '-')} ")
    else:
        lines.append("- 尚未进入终态；已完成证据见阶段表，候选与逐轮选择见对应 artifact。")
    lines.append("")
    return "\n".join(lines)


def commit(state_path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    report = report_text(state)
    atomic_json(state_path, state)
    atomic_text(state_path.parent / "性能调优报告.md", report)


def workflow_version(args: argparse.Namespace) -> str:
    return getattr(args, "workflow_version", None) or WORKFLOW_VERSION


def command_start(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    manifest_path = args.manifest.resolve()
    manifest = read_manifest(manifest_path)
    manifest_id = artifact_identity(manifest_path)
    version = workflow_version(args)
    with state_lock(state_path):
        if state_path.exists():
            state = read_json(state_path)
            manifest_entry = state.get("manifest", {})
            if (
                manifest_entry.get("path") != str(manifest_path)
                or manifest_entry.get("sha256") != manifest_id["sha256"]
            ):
                print(json.dumps({"status": "stale_inputs", "reason": "run_manifest_changed"}, ensure_ascii=False))
                return 3
            if state.get("workflow_version") != version:
                print(
                    json.dumps(
                        {"status": "resume_with_workflow_change", "current": state.get("workflow_version"),
                         "requested": version},
                        ensure_ascii=False,
                    )
                )
                return 5
            invalid = [name for name, record in state.get("stages", {}).items() if record.get("status") == "invalid"]
            status = "resume_with_invalid_checkpoints" if invalid else "resume"
            print(
                json.dumps(
                    {"status": status, "invalid_stages": invalid, "state": str(state_path)},
                    ensure_ascii=False,
                )
            )
            return 4 if invalid else 0

        state = {
            "schema_version": SCHEMA_VERSION,
            "workflow_version": version,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "manifest": manifest_id,
            "tuning_mode": manifest["configuration"].get("tuning_mode", "standard"),
            "optimization_intent": manifest["configuration"].get("optimization_intent", "safe_incremental"),
            "reference_perf": manifest["configuration"].get("reference_perf"),
            "current_stage": None,
            "stages": {},
            "structural_directions": [],
            "events": [],
            "precision": {
                "policy": manifest["configuration"]["precision_diagnosis_policy"],
                "known_baseline_failures": [],
                "new_or_worsened_failures": [],
                "excluded_hypotheses": [],
                "current_hypothesis": None,
                "next_action": None,
            },
            "operator": {"awaiting": False, "safe_restore_point": "inputs_frozen"},
            "safe_restore_point": "inputs_frozen",
        }
        append_event(state, "start", message="输入与配置已冻结")
        commit(state_path, state)
    print(
        json.dumps(
            {"status": "new", "state": str(state_path), "report": str(state_path.parent / '性能调优报告.md')},
            ensure_ascii=False,
        )
    )
    return 0


def command_sync_manifest(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    with state_lock(state_path):
        state = read_json(state_path)
        manifest_path = Path(state["manifest"]["path"])
        manifest = read_manifest(manifest_path)
        state["manifest"] = artifact_identity(manifest_path)
        state["tuning_mode"] = manifest["configuration"].get("tuning_mode", "standard")
        state["optimization_intent"] = manifest["configuration"].get(
            "optimization_intent", state.get("optimization_intent", "safe_incremental")
        )
        state["reference_perf"] = manifest["configuration"].get("reference_perf", state.get("reference_perf"))
        state["precision"]["policy"] = manifest["configuration"]["precision_diagnosis_policy"]
        append_event(state, "manifest", message=args.message or "run manifest synchronized")
        commit(state_path, state)
    print(json.dumps({"status": "manifest_synchronized"}, ensure_ascii=False))
    return 0


def check_phase_artifacts(phase: str, paths: list[str], manifest_path: Path) -> list[str]:
    """Check new phase outputs; source/route semantics remain separate gates."""
    from build_performance_targets import build
    from validate_postprocess_performance import validate

    files = {Path(path).name: Path(path).resolve() for path in paths}
    required = {
        "baseline": ("环境检查报告.md", "性能目标.md", "performance-targets.json", "preflight-report.json"),
        "preparation": ("extraction-manifest.json", "source-compatibility.md", "functional-acceptance.md"),
        "analysis": ("analysis.md",),
        "implementation": ("final-functional-regression.md", "final-performance-comparison.md"),
        "integration": ("VERIFY_REPORT.md",),
    }
    missing = set(required[phase]) - set(files)
    if missing:
        raise ValueError(f"{phase} phase missing artifacts: {sorted(missing)}")
    extra = []
    if phase == "baseline":
        preflight = read_json(files["preflight-report.json"])
        if preflight.get("status") not in ("pass", "warn") or preflight.get("fail_count") != 0:
            raise ValueError("baseline phase requires a passing preflight")
        manifest = read_manifest(manifest_path)
        if preflight.get("request_sha256") != manifest["request"]["sha256"]:
            raise ValueError("preflight request identity mismatch")
        saved = read_json(files["performance-targets.json"])
        evidence = Path(saved["evidence"]["path"])
        if build(manifest_path, evidence) != saved:
            raise ValueError("target evidence or input identity changed")
        extra.append(str(evidence))
        for row in saved["cases"]:
            extra.extend(item["path"] for item in row["source_artifacts"])
        if manifest["configuration"]["performance_reference"]["kind"] == "baseline":
            identity_path = files.get("execution-identity.json")
            if identity_path is None:
                raise ValueError("baseline reference requires execution-identity.json")
            execution_identity = validate_execution_identity(manifest_path, identity_path)
            extra.extend(dependency_paths(execution_identity))
    if phase == "integration":
        manifest = read_manifest(manifest_path)
        gates = {}
        for path in files.values():
            if path.suffix != ".json":
                continue
            gate = read_json(path)
            if gate.get("schema_version") != 1 or "target_soc" not in gate or "collections" not in gate:
                continue
            if gate["target_soc"] in gates:
                raise ValueError("duplicate postprocess gate for target SoC")
            paths_by_kind = {kind: (Path(item["baseline"]["path"]), Path(item["candidate"]["path"]))
                             for kind, item in gate["collections"].items()}
            expected_error_gate = gate["collections"].get("functional", {}).get("expected_errors")
            functional_errors = None
            if expected_error_gate:
                functional_errors = (
                    Path(expected_error_gate["baseline"]["evidence"]["path"]),
                    Path(expected_error_gate["candidate"]["evidence"]["path"]),
                )
            if validate(manifest_path, gate["target_soc"], paths_by_kind,
                        functional_errors=functional_errors) != gate or gate['decision'] != 'pass':
                raise ValueError("postprocess gate is stale or has regressions")
            gates[gate['target_soc']] = gate
            extra.extend(str(p) for pair in paths_by_kind.values() for p in pair)
            if functional_errors:
                extra.extend(str(p) for p in functional_errors)
        if set(gates) != set(manifest['configuration']['target_socs']):
            raise ValueError("postprocess requires both full inventories on every target SoC")
    return list(dict.fromkeys(paths + extra))


def command_mark(args: argparse.Namespace) -> int:
    if args.reason and args.status not in ("skipped", "invalid"):
        raise ValueError("--reason is valid only with --status skipped or invalid")
    has_performance = any((args.performance_summary, args.cumulative_performance_summary, args.selected_candidate))
    if has_performance and args.status != "complete":
        raise ValueError("performance summaries are valid only with --status complete")
    state_path = args.state.resolve()
    with state_lock(state_path):
        state = read_json(state_path)
        manifest_path = Path(state["manifest"]["path"])
        manifest = read_manifest(manifest_path)
        if manifest["configuration"].get("workflow_layout") == "five-stage-v1" and args.stage.startswith("phase."):
            phase = args.stage.removeprefix("phase.")
            if phase not in PHASES:
                raise ValueError("unknown workflow phase")
            if args.status in ("in_progress", "complete"):
                for predecessor in PHASES[:PHASES.index(phase)]:
                    if state.get("stages", {}).get(f"phase.{predecessor}", {}).get("status") != "complete":
                        raise ValueError(f"phase prerequisite incomplete: {predecessor}")
            if args.status == "complete":
                args.artifact = check_phase_artifacts(phase, args.artifact or [], manifest_path)
        previous = state.setdefault("stages", {}).get(args.stage, {})
        timestamp = utc_now()
        record = dict(previous)
        record["status"] = args.status
        record["updated_at"] = timestamp
        if args.status == "in_progress":
            record.setdefault("started_at", timestamp)
            state["current_stage"] = args.stage
        elif args.status == "complete":
            started_at = resolve_stage_started_at(state, args.stage, record)
            if started_at:
                record["started_at"] = started_at
                record["elapsed_seconds"] = round((parse_time(timestamp) - parse_time(started_at)).total_seconds(), 3)
            else:
                record.pop("started_at", None)
                record["elapsed_seconds"] = None
            record["completed_at"] = timestamp
            record["artifacts"] = [artifact_identity(path) for path in (args.artifact or [])]
            if has_performance or performance_checkpoint(args.stage):
                performance, paths = prepare_performance(args)
                record["performance"] = performance
                paths = list(dict.fromkeys([item["path"] for item in record["artifacts"]] + paths))
                record["artifacts"] = [artifact_identity(path) for path in paths]
            else:
                record.pop("performance", None)
            state["safe_restore_point"] = args.stage
            state.setdefault("operator", {})["safe_restore_point"] = args.stage
            current_stage = state.get("current_stage") or ""
            round_prefix = args.stage.removesuffix(".complete") + "."
            if current_stage == args.stage or (
                args.stage.endswith(".complete") and performance_checkpoint(args.stage)
                and current_stage.startswith(round_prefix)
            ):
                state["current_stage"] = None
        elif args.status == "skipped":
            started_at = resolve_stage_started_at(state, args.stage, record)
            if started_at:
                record["started_at"] = started_at
                record["elapsed_seconds"] = round((parse_time(timestamp) - parse_time(started_at)).total_seconds(), 3)
            else:
                record.pop("started_at", None)
                record["elapsed_seconds"] = None
            record["completed_at"] = timestamp
            record["reason"] = args.reason
            record["artifacts"] = [artifact_identity(path) for path in (args.artifact or [])]
            state["safe_restore_point"] = args.stage
        elif args.status == "invalid":
            record["invalidated_at"] = timestamp
            record["reason"] = args.reason or "artifact_or_workflow_changed"
        if args.context:
            record["context"] = args.context
        state["stages"][args.stage] = record
        append_event(
            state, "stage", stage=args.stage, status=args.status,
            message=args.message, next_action=args.next_action,
        )
        commit(state_path, state)
    print(json.dumps({"status": "marked", "stage": args.stage, "stage_status": args.status}, ensure_ascii=False))
    return 0


def unique_append(items: list[Any], value: Any, key: str | None = None) -> None:
    if key is None:
        if value not in items:
            items.append(value)
        return
    if not any(item.get(key) == value.get(key) and item.get("message") == value.get("message") for item in items):
        items.append(value)


def command_progress(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    with state_lock(state_path):
        state = read_json(state_path)
        precision = state.setdefault("precision", {})
        kind = "diagnosis" if args.kind == "diagnosis" else "progress"
        payload: dict[str, Any] = {"message": args.message}
        if args.next_action:
            payload["next_action"] = args.next_action
        for key in ("stage", "evidence", "case_id", "candidate", "classification"):
            value = getattr(args, key, None)
            if value:
                payload[key] = value
        if args.detail:
            details: dict[str, str] = {}
            for item in args.detail:
                if "=" not in item:
                    raise ValueError("--detail must use key=value")
                key, value = item.split("=", 1)
                if not key:
                    raise ValueError("--detail key cannot be empty")
                details[key] = value
            payload["details"] = details
        if args.hypothesis:
            precision["current_hypothesis"] = args.hypothesis
        if args.excluded_hypothesis:
            excluded = precision.setdefault("excluded_hypotheses", [])
            for hypothesis in args.excluded_hypothesis:
                unique_append(excluded, hypothesis)
        if args.next_action:
            precision["next_action"] = args.next_action
            state["next_action"] = args.next_action
        if args.classification:
            if not args.case_id:
                raise ValueError("precision classification requires --case-id")
            item = {
                "case_id": args.case_id,
                "candidate": args.candidate,
                "message": args.message,
                "evidence": args.evidence,
                "at": utc_now(),
            }
            target = (
                "known_baseline_failures"
                if args.classification == "known_baseline_failure"
                else "new_or_worsened_failures"
                if args.classification == "new_or_worsened_failure"
                else "diagnosis_results"
            )
            unique_append(precision.setdefault(target, []), item, "case_id")
        append_event(state, kind, **payload)
        commit(state_path, state)
    print(json.dumps({"status": "recorded", "kind": kind}, ensure_ascii=False))
    return 0


def command_decision(args: argparse.Namespace) -> int:
    if args.awaiting and not args.question:
        raise ValueError("--awaiting requires --question")
    state_path = args.state.resolve()
    with state_lock(state_path):
        state = read_json(state_path)
        operator = state.setdefault("operator", {})
        if args.awaiting:
            operator.update({"awaiting": True, "question": args.question, "requested_at": utc_now()})
            message = args.question
        else:
            operator.update({"awaiting": False, "last_decision": args.decision, "resolved_at": utc_now()})
            operator.pop("question", None)
            if state.get("terminal", {}).get("status") == "awaiting_operator":
                state.pop("terminal")
            message = args.decision
        if args.safe_restore_point:
            operator["safe_restore_point"] = args.safe_restore_point
            state["safe_restore_point"] = args.safe_restore_point
        append_event(state, "decision", message=message, awaiting=args.awaiting)
        commit(state_path, state)
    print(json.dumps({"status": "awaiting" if args.awaiting else "resolved"}, ensure_ascii=False))
    return 0


def command_structural_direction(args: argparse.Namespace) -> int:
    """Upsert a structural-direction ledger entry (pursue_optimal cross-round obligation)."""
    state_path = args.state.resolve()
    with state_lock(state_path):
        state = read_json(state_path)
        ledger = state.setdefault("structural_directions", [])
        entry = next((d for d in ledger if d.get("id") == args.id), None)
        if entry is None:
            entry = {"id": args.id, "created_at": utc_now()}
            ledger.append(entry)
        entry["status"] = args.status
        entry["updated_at"] = utc_now()
        if args.description:
            entry["description"] = args.description
        if args.round is not None:
            entry["round"] = args.round
        if args.evidence:
            entry["evidence"] = args.evidence
        if args.estimated_gain:
            entry["estimated_gain"] = args.estimated_gain
        append_event(
            state,
            "structural_direction",
            message=f"{args.id} -> {args.status}",
            direction_id=args.id,
            status=args.status,
        )
        commit(state_path, state)
    print(json.dumps({"status": "recorded", "direction": args.id, "direction_status": args.status}, ensure_ascii=False))
    return 0


def command_migrate(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    audit = artifact_identity(args.audit)
    with state_lock(state_path):
        state = read_json(state_path)
        for stage in args.invalidate_stage or []:
            record = state.setdefault("stages", {}).setdefault(stage, {})
            record.update({"status": "invalid", "invalidated_at": utc_now(), "reason": "workflow_migration"})
        state["workflow_version"] = args.workflow_version
        append_event(state, "workflow_migration", audit=audit, invalidated=args.invalidate_stage or [])
        commit(state_path, state)
    print(json.dumps({"status": "workflow_migrated", "invalidated": args.invalidate_stage or []}, ensure_ascii=False))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    invalid: list[dict[str, str]] = []
    with state_lock(state_path):
        state = read_json(state_path)
        current_manifest = artifact_identity(state["manifest"]["path"])
        if current_manifest["sha256"] != state["manifest"]["sha256"]:
            invalid.append({"kind": "manifest", "path": current_manifest["path"]})
        for name, record in state.get("stages", {}).items():
            if record.get("status") != "complete":
                continue
            for artifact in record.get("artifacts", []):
                try:
                    current = artifact_identity(artifact["path"])
                except FileNotFoundError:
                    invalid.append({"kind": "artifact", "stage": name, "path": artifact["path"]})
                    record.update({"status": "invalid", "reason": "artifact_missing", "invalidated_at": utc_now()})
                    continue
                if current["sha256"] != artifact["sha256"]:
                    invalid.append({"kind": "artifact", "stage": name, "path": artifact["path"]})
                    record.update({"status": "invalid", "reason": "artifact_changed", "invalidated_at": utc_now()})
        if invalid:
            append_event(state, "verification", message="state verification failed", invalid=invalid)
        commit(state_path, state)
    print(json.dumps({"status": "invalid" if invalid else "valid", "invalid": invalid}, ensure_ascii=False))
    return 2 if invalid else 0


def command_inspect(args: argparse.Namespace) -> int:
    """Read a bounded stage view; no state writes or completion inference."""
    if args.recent < 0:
        raise ValueError("--recent must be non-negative")
    state = read_json(args.state.resolve())
    manifest_id = state["manifest"]
    if file_sha(Path(manifest_id["path"])) != manifest_id["sha256"]:
        raise ValueError("stale_inputs: run_manifest_changed")
    manifest = read_manifest(Path(manifest_id["path"]))
    stages = state.get("stages", {})
    selected = args.stage or ([state["current_stage"]] if state.get("current_stage") else [])
    events = [event for event in state.get("events", []) if event.get("stage") in selected]
    print(json.dumps({
        "manifest": manifest_id,
        "configuration": manifest["configuration"],
        "inputs": manifest["inputs"],
        "collections": {
            name: {key: value[key] for key in ("path", "sha256", "count") if key in value}
            for name, value in manifest["collections"].items() if isinstance(value, dict)
        },
        "current_stage": state.get("current_stage"),
        "stage_statuses": {name: record.get("status") for name, record in stages.items()},
        "stages": {name: stages.get(name) for name in selected},
        "recent_events": events[-args.recent:] if args.recent else [],
        "operator": state.get("operator"),
        "safe_restore_point": state.get("safe_restore_point"),
        "verification": "manifest_sha_only; use verify for saved artifacts and stage gates for evidence ownership",
    }, ensure_ascii=False))
    return 0


def command_render(args: argparse.Namespace) -> int:
    state_path = args.state.resolve()
    with state_lock(state_path):
        state = read_json(state_path)
        atomic_text(args.output.resolve() if args.output else state_path.parent / "性能调优报告.md", report_text(state))
    print(json.dumps({"status": "rendered"}, ensure_ascii=False))
    return 0


def command_finish(args: argparse.Namespace) -> int:
    if args.status in ("failed", "blocked") and args.reason_code in (
        "known_baseline_failure",
        "candidate_precision_regression",
    ):
        raise ValueError(
            "a known baseline failure or one candidate regression cannot terminate the workflow; "
            "record conditional_pass or eliminate that candidate"
        )
    state_path = args.state.resolve()
    with state_lock(state_path):
        state = read_json(state_path)
        # pursue_optimal hard gate: a clean success finish requires every proposed structural
        # direction to be discharged (realized / evidenced-infeasible / precision-blocked).
        # A direction still not_attempted or blocked_by_scope is an unrealized obligation —
        # blocked_by_scope must be re-attempted with a wider re-frozen scope, not abandoned.
        if (
            state.get("optimization_intent") == "pursue_optimal"
            and args.status == "success"
            and args.reason_code == "completed"
        ):
            unresolved = [
                d
                for d in state.get("structural_directions", [])
                if d.get("status") in UNRESOLVED_STRUCTURAL_STATUSES
            ]
            if unresolved:
                ids = ", ".join(f"{d.get('id')}[{d.get('status')}]" for d in unresolved)
                raise ValueError(
                    "pursue_optimal: cannot finish success/completed with unrealized structural directions: "
                    f"{ids}. Re-freeze a wider scope and re-attempt each blocked_by_scope/not_attempted "
                    "direction, or record it as realized / infeasible_evidenced / blocked_by_precision. "
                    "If parking for a human decision, finish with "
                    "--status awaiting_operator --reason-code structural_headroom_unrealized."
                )
        manifest = read_manifest(Path(state["manifest"]["path"]))
        if args.status == "success" and manifest["configuration"].get("workflow_layout") == "five-stage-v1":
            if artifact_identity(state["manifest"]["path"]) != state["manifest"]:
                raise ValueError("run manifest changed; verify before finishing")
            for phase in PHASES:
                record = state.get("stages", {}).get(f"phase.{phase}", {})
                if record.get("status") != "complete":
                    raise ValueError(f"required phase is incomplete: {phase}")
                for artifact in record.get("artifacts", []):
                    if artifact_identity(artifact["path"]) != artifact:
                        raise ValueError(f"phase artifact changed: {artifact['path']}")
        state["terminal"] = {
            "status": args.status,
            "reason_code": args.reason_code,
            "message": args.message,
            "at": utc_now(),
        }
        if args.status == "awaiting_operator":
            state.setdefault("operator", {})["awaiting"] = True
        append_event(state, "finish", message=args.message, status=args.status)
        commit(state_path, state)
    print(json.dumps({"status": args.status, "report": str(state_path.parent / '性能调优报告.md')}, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="start or resume a run")
    start.add_argument("--state", type=Path, required=True)
    start.add_argument("--manifest", type=Path, required=True)
    start.add_argument("--workflow-version")
    start.set_defaults(func=command_start)

    sync = subparsers.add_parser("sync-manifest", help="accept an intentional manifest update")
    sync.add_argument("--state", type=Path, required=True)
    sync.add_argument("--message")
    sync.set_defaults(func=command_sync_manifest)

    mark = subparsers.add_parser("mark", help="change a stage lifecycle state")
    mark.add_argument("--state", type=Path, required=True)
    mark.add_argument("--stage", required=True)
    mark.add_argument("--status", choices=STAGE_STATUSES, required=True)
    mark.add_argument("--reason")
    mark.add_argument("--message")
    mark.add_argument("--next-action", help="next action at this lifecycle checkpoint")
    mark.add_argument("--artifact", action="append")
    mark.add_argument("--context", action="append")
    mark.add_argument(
        "--performance-summary", type=Path,
        help="selected implementation vs round baseline effect-summary.json",
    )
    mark.add_argument(
        "--cumulative-performance-summary", type=Path,
        help="same selected performance vs original baseline effect-summary.json",
    )
    mark.add_argument("--selected-candidate", help="retained implementation identity, including baseline when retained")
    mark.add_argument("--performance-scope", choices=("tuning_subset", "full"), default="tuning_subset")
    mark.set_defaults(func=command_mark)

    progress = subparsers.add_parser("progress", help="record meaningful work or a diagnosis checkpoint")
    progress.add_argument("--state", type=Path, required=True)
    progress.add_argument("--kind", choices=("progress", "diagnosis"), default="progress")
    progress.add_argument("--message", required=True)
    progress.add_argument("--stage")
    progress.add_argument("--evidence")
    progress.add_argument("--hypothesis")
    progress.add_argument("--excluded-hypothesis", action="append")
    progress.add_argument("--next-action")
    progress.add_argument("--case-id")
    progress.add_argument("--candidate")
    progress.add_argument("--classification", choices=PRECISION_CLASSIFICATIONS)
    progress.add_argument("--detail", action="append", help="structured report field as key=value")
    progress.set_defaults(func=command_progress)

    decision = subparsers.add_parser("decision", help="record an operator wait or decision")
    decision.add_argument("--state", type=Path, required=True)
    group = decision.add_mutually_exclusive_group(required=True)
    group.add_argument("--awaiting", action="store_true")
    group.add_argument("--decision")
    decision.add_argument("--question")
    decision.add_argument("--safe-restore-point")
    decision.set_defaults(func=command_decision)

    sdir = subparsers.add_parser(
        "structural-direction",
        help="upsert a structural-direction ledger entry (pursue_optimal cross-round obligation)",
    )
    sdir.add_argument("--state", type=Path, required=True)
    sdir.add_argument("--id", required=True, help="stable direction id (e.g. whole-op-fusion)")
    sdir.add_argument("--status", choices=STRUCTURAL_DIRECTION_STATUSES, required=True)
    sdir.add_argument("--description")
    sdir.add_argument("--round", type=int)
    sdir.add_argument("--evidence")
    sdir.add_argument("--estimated-gain")
    sdir.set_defaults(func=command_structural_direction)

    migrate = subparsers.add_parser("migrate-workflow", help="record workflow impact audit")
    migrate.add_argument("--state", type=Path, required=True)
    migrate.add_argument("--audit", type=Path, required=True)
    migrate.add_argument("--workflow-version", required=True)
    migrate.add_argument("--invalidate-stage", action="append")
    migrate.set_defaults(func=command_migrate)

    verify = subparsers.add_parser("verify", help="verify manifest and completed artifacts")
    verify.add_argument("--state", type=Path, required=True)
    verify.set_defaults(func=command_verify)

    inspect = subparsers.add_parser(
        "inspect", help="read manifest identity and selected stage context without full history"
    )
    inspect.add_argument("--state", type=Path, required=True)
    inspect.add_argument("--stage", action="append", help="exact stage key; repeat for prerequisite stages")
    inspect.add_argument("--recent", type=int, default=3)
    inspect.set_defaults(func=command_inspect)

    render = subparsers.add_parser("render", help="render the live report from current state")
    render.add_argument("--state", type=Path, required=True)
    render.add_argument("--output", type=Path)
    render.set_defaults(func=command_render)

    finish = subparsers.add_parser("finish", help="record terminal status and finalize the live report")
    finish.add_argument("--state", type=Path, required=True)
    finish.add_argument("--status", choices=TERMINAL_STATUSES, required=True)
    finish.add_argument("--reason-code", choices=TERMINAL_REASON_CODES, required=True)
    finish.add_argument("--message", required=True)
    finish.set_defaults(func=command_finish)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
