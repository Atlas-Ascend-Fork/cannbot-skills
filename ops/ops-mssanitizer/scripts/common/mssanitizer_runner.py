#!/usr/bin/env python3
# --------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and
# conditions of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with it.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
# PARTICULAR PURPOSE. See LICENSE in the repository root for the full text of the License.
# --------------------------------------------------------------------------------------------------
"""Shared execution engine for the ops-mssanitizer shell entry points."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence


CHECKS = {
    "ascendc": (
        ("memcheck_device", "memcheck", "--leak-check=yes", "--check-device-heap=yes"),
        ("memcheck_cann", "memcheck", "--leak-check=yes", "--check-cann-heap=yes"),
        ("racecheck", "racecheck"),
        ("initcheck", "initcheck"),
        ("synccheck", "synccheck"),
    ),
    "geir": (
        ("memcheck_device", "memcheck", "--leak-check=yes", "--check-device-heap=yes"),
        ("memcheck_cann", "memcheck", "--leak-check=yes", "--check-cann-heap=yes"),
        ("racecheck", "racecheck"),
        ("initcheck", "initcheck"),
        ("synccheck", "synccheck"),
    ),
    "triton": (
        ("memcheck_device", "memcheck", "--leak-check=yes", "--check-device-heap=yes"),
        ("racecheck", "racecheck"),
        ("initcheck", "initcheck"),
        ("synccheck", "synccheck"),
    ),
}

ERROR_RECORD = re.compile(r"^(?:=+\s+)?(?:\[E\].*)?ERROR:")
LEAK_SUMMARY = re.compile(r"^(?:=+\s+)?SUMMARY:.*leaked", re.IGNORECASE)


class RunnerError(RuntimeError):
    """A preflight or build error that prevents sanitizer execution."""


@dataclass(frozen=True)
class Target:
    """Resolved workload and its execution metadata."""
    mode: str
    path: Path
    project_dir: Path
    command: tuple[str, ...]
    display_mode: str


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one mssanitizer tool invocation."""
    name: str
    log_path: Path
    exit_code: int
    error_count: int
    json_path: Optional[Path] = None

    @property
    def status(self) -> str:
        """Return the user-facing status derived from evidence."""
        if self.exit_code != 0:
            return "⚠️ 未完成"
        if self.error_count == 0:
            return "🟢 通过"
        return "🟡 中等" if self.error_count < 3 else "🔴 严重"


@dataclass(frozen=True)
class ExecutionContext:
    """Stable inputs shared by every check in one run."""

    sanitizer: Path
    log_dir: Path
    timestamp: str
    environment: dict[str, str]
    kernel_name: Optional[str]


def absolute_existing_path(raw_path: str, expected: str) -> Path:
    """Resolve and validate an input file or directory."""
    path = Path(raw_path).expanduser().resolve()
    predicate = path.is_file if expected == "file" else path.is_dir
    if not predicate():
        raise RunnerError(f"路径不是有效的{expected}: {path}")
    return path


def resolve_cann_root(raw_path: Optional[str]) -> Path:
    """Resolve CANN from an argument or the current environment."""
    value = raw_path or os.environ.get("ASCEND_HOME_PATH")
    if not value:
        raise RunnerError("请通过参数或 ASCEND_HOME_PATH 指定 CANN 安装根路径")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise RunnerError(f"CANN 安装根路径不存在: {root}")
    return root


def configure_environment(cann_root: Path, mode: str) -> dict[str, str]:
    """Build a child-process environment pinned to one CANN install."""
    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(cann_root)
    env["ASCEND_AICPU_PATH"] = str(cann_root)
    library_dirs = (
        cann_root / "lib64",
        cann_root / "lib64/plugin/opskernel",
        cann_root / "lib64/plugin/nnengine",
    )
    old_library_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [*(str(item) for item in library_dirs), old_library_path]
    ).rstrip(":")
    if mode == "triton":
        env.update(
            PYTORCH_NO_NPU_MEMORY_CACHING="1",
            TRITON_ALWAYS_COMPILE="1",
            TRITON_ENABLE_SANITIZER="1",
        )
    return env


def locate_mssanitizer(cann_root: Path) -> Path:
    """Locate an executable mssanitizer under the selected CANN root."""
    preferred = cann_root / "tools/mssanitizer/bin/mssanitizer"
    if preferred.is_file() and os.access(preferred, os.X_OK):
        return preferred
    for candidate in cann_root.rglob("mssanitizer"):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise RunnerError(f"在 CANN 路径中未找到可执行的 mssanitizer: {cann_root}")


def error_record_count(log_path: Path) -> int:
    """Count structured error records instead of arbitrary ERROR text."""
    if not log_path.is_file() or log_path.stat().st_size == 0:
        return 0
    count = 0
    saw_leak_summary = False
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            count += bool(ERROR_RECORD.search(line))
            saw_leak_summary = saw_leak_summary or bool(LEAK_SUMMARY.search(line))
    return count or int(saw_leak_summary)


def run_process(
    command: Sequence[str], env: dict[str, str], cwd: Optional[Path] = None
) -> int:
    """Run a command and convert launch failures to a stable exit code."""
    try:
        return subprocess.run(command, cwd=cwd, env=env, check=False).returncode
    except OSError as error:
        print(f"无法启动命令 {command[0]}: {error}", file=sys.stderr)
        return 126


def find_existing_cpp_test(
    project_dir: Path, operator_name: str
) -> Optional[tuple[Path, str]]:
    """Find a previously built GE IR or aclnn test executable."""
    build_roots = (project_dir / "mssanitizer_test/build", project_dir / "build")
    for prefix, display_mode in (("test_geir_", "GE IR"), ("test_aclnn_", "aclnn")):
        for build_root in build_roots:
            executable = build_root / f"{prefix}{operator_name}"
            if executable.is_file():
                return executable.resolve(), display_mode
    return None


def discover_cpp_source(project_dir: Path) -> tuple[Path, str]:
    """Select a GE IR test source before falling back to aclnn."""
    examples = project_dir / "examples"
    for pattern, display_mode in (("test_geir_*.cpp", "GE IR"), ("test_aclnn_*.cpp", "aclnn")):
        matches = sorted(examples.rglob(pattern)) if examples.is_dir() else []
        if matches:
            return matches[0].resolve(), display_mode
    raise RunnerError(
        f"未在 {examples} 中找到 test_geir_*.cpp 或 test_aclnn_*.cpp；"
        "请提供可运行的 C++ 测试"
    )


def cmake_source(source: Path, project_dir: Path, cann_root: Path, display_mode: str) -> str:
    """Generate a minimal CMake project for the selected C++ test."""
    target_name = source.stem
    common = [
        "cmake_minimum_required(VERSION 3.16)",
        f"project({target_name} LANGUAGES CXX)",
        "set(CMAKE_CXX_STANDARD 17)",
        f'add_executable({target_name} "{source}")',
    ]
    if display_mode == "GE IR":
        common.extend(
            [
                "find_package(ASCEND REQUIRED)",
                (
                    f'target_include_directories({target_name} PRIVATE '
                    f'"${{ASCEND_INC_DIR}}" "{project_dir / "op_graph"}")'
                ),
                f'target_link_directories({target_name} PRIVATE "${{ASCEND_LIB_DIR}}")',
                f"target_link_libraries({target_name} PRIVATE ascendcl ge_runner graph register)",
            ]
        )
    else:
        architecture_lib = cann_root / f"{os.uname().machine}-linux/lib64"
        common.extend(
            [
                f'target_include_directories({target_name} PRIVATE "{cann_root / "include"}")',
                (
                    f'target_link_directories({target_name} PRIVATE '
                    f'"{cann_root / "lib64"}" "{architecture_lib}")'
                ),
                f"target_link_libraries({target_name} PRIVATE ascendcl opapi nnopbase)",
            ]
        )
    return "\n".join(common) + "\n"


def build_cpp_test(project_dir: Path, cann_root: Path, env: dict[str, str]) -> tuple[Path, str]:
    """Configure and build a discovered C++ test source."""
    source, display_mode = discover_cpp_source(project_dir)
    if display_mode == "GE IR" and not (project_dir / "op_graph").is_dir():
        raise RunnerError(f"GE IR 测试缺少 op_graph 目录: {project_dir / 'op_graph'}")

    build_dir = project_dir / "mssanitizer_test/build"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "CMakeLists.txt").write_text(
        cmake_source(source, project_dir, cann_root, display_mode), encoding="utf-8"
    )
    configure = ["cmake", "-S", str(build_dir), "-B", str(build_dir)]
    compile_command = [
        "cmake",
        "--build",
        str(build_dir),
        "--parallel",
        str(os.cpu_count() or 1),
    ]
    if run_process(configure, env) != 0 or run_process(compile_command, env) != 0:
        raise RunnerError(f"测试程序构建失败: {source}")
    executable = build_dir / source.stem
    if not executable.is_file():
        raise RunnerError(f"构建完成但未生成预期程序: {executable}")
    return executable.resolve(), display_mode


def prepare_target(mode: str, raw_target: str, cann_root: Path, env: dict[str, str]) -> Target:
    """Resolve a Python or C++ workload for the requested mode."""
    if mode in ("ascendc", "triton"):
        script = absolute_existing_path(raw_target, "file")
        project_dir = script.parent.parent if mode == "ascendc" else script.parent
        return Target(mode, script, project_dir, (sys.executable, str(script)), mode)

    project_dir = absolute_existing_path(raw_target, "directory")
    operator_name = project_dir.name
    located = find_existing_cpp_test(project_dir, operator_name)
    executable, display_mode = located or build_cpp_test(project_dir, cann_root, env)
    if not os.access(executable, os.X_OK):
        raise RunnerError(f"测试程序不可执行: {executable}")
    suffix = ("float",) if display_mode == "GE IR" else ()
    return Target(mode, executable, project_dir, (str(executable), *suffix), display_mode)


def check_arguments(
    specification: Sequence[str], kernel_name: Optional[str]
) -> list[str]:
    """Translate one check specification to mssanitizer arguments."""
    _name, tool_name, *options = specification
    arguments = ["-t", tool_name, *options]
    if kernel_name:
        arguments.extend(("--kernel-name", kernel_name))
    return arguments


def execute_checks(
    target: Target,
    context: ExecutionContext,
) -> list[CheckResult]:
    """Execute the configured checks and retain evidence for reporting."""
    results = []
    prefix = "geir_" if target.mode == "geir" else ""
    for specification in CHECKS[target.mode]:
        check_name = specification[0]
        log_path = context.log_dir / f"{prefix}{check_name}_{context.timestamp}.log"
        json_path = None
        target_command = list(target.command)
        if target.mode == "ascendc":
            json_path = context.log_dir / (
                f"{check_name}_report_{context.timestamp}.json"
            )
            target_command.extend(("--output", str(json_path)))
        command = [
            str(context.sanitizer),
            *check_arguments(specification, context.kernel_name),
            f"--log-file={log_path}",
            "--",
            *target_command,
        ]
        print(f"\n>>> {check_name}")
        exit_code = run_process(command, context.environment)
        count = error_record_count(log_path)
        if exit_code:
            print(f"{check_name} 未正常完成: exit={exit_code}, log={log_path}")
        else:
            print(f"{check_name} 完成: errors={count}, log={log_path}")
        results.append(CheckResult(check_name, log_path, exit_code, count, json_path))
    return results


def overall_rating(results: Sequence[CheckResult]) -> str:
    """Summarize completeness and finding count."""
    if any(item.exit_code for item in results):
        return "⚠️ **检测未完成** — 至少一个检测项执行失败；不得根据空日志判定通过"
    total = sum(item.error_count for item in results)
    if total == 0:
        return "🟢 **良好 (GOOD)** — 未检测到内存错误"
    if total < 5:
        return f"🟡 **中等 (MODERATE)** — 检测到 {total} 个内存问题"
    return f"🔴 **严重 (CRITICAL)** — 检测到 {total} 个内存问题"


def parsed_log(log_path: Path, parser_path: Path) -> str:
    """Return an embedded diagnosis for a non-empty log."""
    if not log_path.is_file() or log_path.stat().st_size == 0 or not parser_path.is_file():
        return ""
    process = subprocess.run(
        [sys.executable, str(parser_path), str(log_path), "--no-title", "--heading-offset", "2"],
        check=False,
        text=True,
        capture_output=True,
    )
    return process.stdout if process.returncode == 0 else ""


def report_metadata(
    target: Target, cann_root: Path, kernel_name: Optional[str]
) -> list[str]:
    """Render mode-specific report metadata."""
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    metadata = [f"**生成时间**: {generated_at}", f"**CANN 路径**: {cann_root}"]
    if target.mode == "geir":
        metadata.extend(
            (
                f"**算子工程**: {target.project_dir}",
                f"**测试可执行文件**: {target.path}",
                f"**检测模式**: {target.display_mode} (C++ 可执行文件)",
            )
        )
    else:
        metadata.append(f"**测试脚本**: {target.path}")
    if kernel_name:
        metadata.append(f"**指定 kernel**: {kernel_name}")
    return metadata


def render_report(
    target: Target,
    cann_root: Path,
    results: Sequence[CheckResult],
    parser_path: Path,
    kernel_name: Optional[str],
) -> str:
    """Render a complete run report from collected evidence."""
    title_suffix = ""
    if target.mode == "triton":
        title_suffix = " (Triton 算子)"
    elif target.mode == "geir":
        title_suffix = f" ({target.display_mode} 模式)"
    lines = [
        f"# mssanitizer 内存检测汇总报告{title_suffix}",
        "",
        *report_metadata(target, cann_root, kernel_name),
        "",
    ]
    lines.extend(
        (
            "## 检测工具执行结果总览",
            "",
            "| 检测工具 | 退出码 | 错误条目数 | 状态 |",
            "|---|---:|---:|---|",
        )
    )
    for result in results:
        lines.append(
            f"| **{result.name}** | {result.exit_code} | "
            f"{result.error_count} | {result.status} |"
        )
    lines.extend(("", f"**整体评级**: {overall_rating(results)}", ""))

    for result in results:
        lines.extend(
            (
                "---",
                "",
                f"## {result.name}",
                "",
                f"- 工具退出码: {result.exit_code}",
                f"- 检测到错误条目数: {result.error_count}",
                "",
            )
        )
        analysis = parsed_log(result.log_path, parser_path)
        if analysis:
            lines.extend(("### 解析报告", "", analysis.rstrip(), ""))
        if result.log_path.is_file() and result.log_path.stat().st_size:
            raw_log = result.log_path.read_text(encoding="utf-8", errors="replace")
            lines.extend(("### 原始日志", "", "```text", raw_log.rstrip(), "```", ""))
        elif result.exit_code == 0:
            lines.extend(("工具正常完成且日志为空，未检测到问题。", ""))
        else:
            lines.extend(("日志为空且工具执行失败，当前结果不能用于通过判定。", ""))

    lines.extend(("---", "", "## 检测输出文件清单", "", "| 文件 | 说明 |", "|---|---|"))
    for result in results:
        lines.append(f"| `{result.log_path.name}` | 原始日志 |")
        if result.json_path and result.json_path.exists():
            lines.append(f"| `{result.json_path.name}` | 测试结果 JSON |")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Create the shared runner command-line interface."""
    parser = argparse.ArgumentParser(description="运行 mssanitizer 并生成 Markdown 汇总报告")
    parser.add_argument("mode", choices=tuple(CHECKS), help=argparse.SUPPRESS)
    parser.add_argument("target", help="Python 测试脚本或算子工程目录")
    parser.add_argument("cann_root", nargs="?", help="CANN 安装根路径，默认读取 ASCEND_HOME_PATH")
    parser.add_argument("--kernel-name", help="Triton 场景下限定目标 kernel")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Resolve inputs, run checks, and save a summary report."""
    arguments = build_parser().parse_args(argv)
    try:
        cann_root = resolve_cann_root(arguments.cann_root)
        env = configure_environment(cann_root, arguments.mode)
        sanitizer = locate_mssanitizer(cann_root)
        target = prepare_target(arguments.mode, arguments.target, cann_root, env)
        if arguments.kernel_name and arguments.mode != "triton":
            raise RunnerError("--kernel-name 仅适用于 Triton 检测")

        log_dir = target.project_dir / "mssanitizer_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"模式: {target.display_mode}")
        print(f"CANN 根路径: {cann_root}")
        print(f"检测目标: {target.path}")
        print(f"日志目录: {log_dir}")
        print(f"使用 mssanitizer: {sanitizer}")
        context = ExecutionContext(
            sanitizer=sanitizer,
            log_dir=log_dir,
            timestamp=timestamp,
            environment=env,
            kernel_name=arguments.kernel_name,
        )
        results = execute_checks(target, context)
        parser_path = Path(__file__).with_name("parse_mssanitizer_log.py")
        report_name = (
            f"geir_mssanitizer_summary_{timestamp}.md"
            if arguments.mode == "geir"
            else f"mssanitizer_summary_{timestamp}.md"
        )
        report_path = log_dir / report_name
        report_path.write_text(
            render_report(target, cann_root, results, parser_path, arguments.kernel_name),
            encoding="utf-8",
        )
        print(f"汇总报告: {report_path}")
        return 2 if any(item.exit_code for item in results) else 0
    except RunnerError as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
