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
"""Turn an mssanitizer text log into a compact Markdown diagnosis."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class FindingRule:
    """Pattern and presentation metadata for one finding category."""
    category: str
    expression: str
    severity: str
    title: str


@dataclass
class Finding:
    """One structured finding extracted from the text log."""
    category: str
    raw_line: str
    size: Optional[int] = None
    address: Optional[str] = None

    def as_mapping(self) -> dict[str, object]:
        """Expose the legacy dictionary representation."""
        value: dict[str, object] = {"type": self.category, "line": self.raw_line}
        if self.size is not None:
            value["size"] = str(self.size)
        if self.address:
            value["address"] = self.address
        return value


RULES = (
    FindingRule(
        "illegal_free",
        r"illegal\s+free",
        "🔴 严重",
        "非法释放",
    ),
    FindingRule(
        "illegal_read",
        r"illegal\s+read\s+of\s+size\s+(\d+)",
        "🔴 严重",
        "非法读取",
    ),
    FindingRule(
        "illegal_write",
        r"illegal\s+write\s+of\s+size\s+(\d+)",
        "🔴 严重",
        "非法写入",
    ),
    FindingRule(
        "memory_leak",
        r"LeakCheck|memory\s+leak|SUMMARY:.*leaked",
        "🟡 中等",
        "内存泄漏",
    ),
    FindingRule(
        "ub_out_of_bounds",
        r"ub\s+address\s+out\s+of\s+bounds|VEC\s+instruction\s+error",
        "🔴 严重",
        "UB 地址越界",
    ),
)
ADDRESS = re.compile(r"\bat\s+(0x[0-9a-f]+)\b", re.IGNORECASE)
HEAP_ADD = re.compile(r"addr:\s*(0x[0-9a-f]+),\s*size:\s*(\d+)", re.IGNORECASE)
HEAP_FREE = re.compile(r"addr:\s*(0x[0-9a-f]+)", re.IGNORECASE)
MEMORY_RECORD = re.compile(r"\btype:([A-Za-z_]+)")


REMEDIATION = {
    "illegal_read": (
        "核对读取偏移和输入 Tensor 的有效元素数；重点检查尾块。",
        "为向量化读取增加边界 mask；block pointer 场景显式声明 boundary_check。",
    ),
    "illegal_write": (
        "核对输出偏移、实际输出 shape 和每个 program 写入范围。",
        "在尾块写回时使用 mask，且不要用向上取整后的长度作为有效长度。",
    ),
    "illegal_free": (
        "先检查同一次运行中是否存在越界写；越界可能破坏内存管理元数据。",
        "PyTorch 场景设置 PYTORCH_NO_NPU_MEMORY_CACHING=1 后复测。",
    ),
    "memory_leak": (
        "确认被测进程退出前已经执行 torch.npu.synchronize()。",
        "关闭 PyTorch NPU 内存缓存后复测，以排除缓存造成的假阳性。",
    ),
    "ub_out_of_bounds": (
        "检查 LocalTensor 起始地址、切片偏移和临时 Buffer 容量。",
        "结合 tiling 计算逐项验证单核处理量、尾块和双缓冲空间。",
    ),
}


class LogScanner:
    """Single-pass state machine for mssanitizer records."""

    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.operations: Counter[str] = Counter()
        self.live_blocks: dict[str, int] = {}

    def consume(self, raw_line: str) -> None:
        """Update scanner state from one log line."""
        line = raw_line.strip()
        if not line:
            return
        self._consume_operation(line)
        self._consume_heap_event(line)
        finding = self._match_finding(line)
        if finding:
            self.findings.append(finding)
            return
        if self.findings and line.startswith("="):
            address = ADDRESS.search(line)
            if address and not self.findings[-1].address:
                self.findings[-1].address = address.group(1)

    def _consume_operation(self, line: str) -> None:
        """Count one MEMORY_RECORD operation if present."""
        if not line.startswith("RecordVersion:MEMORY_RECORD"):
            return
        operation = MEMORY_RECORD.search(line)
        if operation:
            self.operations[operation.group(1).lower()] += 1

    def _consume_heap_event(self, line: str) -> None:
        """Track allocator add/free messages."""
        if "[I] add heap block" in line:
            added = HEAP_ADD.search(line)
            if added:
                self.live_blocks[added.group(1)] = int(added.group(2))
        elif "[I] free heap block" in line:
            freed = HEAP_FREE.search(line)
            if freed:
                self.live_blocks.pop(freed.group(1), None)

    @staticmethod
    def _match_finding(line: str) -> Optional[Finding]:
        """Return the first matching finding for a line."""
        leak_summary = "SUMMARY:" in line and "leaked" in line.lower()
        if "ERROR:" not in line and not leak_summary:
            return None
        for rule in RULES:
            match = re.search(rule.expression, line, re.IGNORECASE)
            if not match:
                continue
            size = (
                int(match.group(1))
                if rule.category in {"illegal_read", "illegal_write"}
                else None
            )
            address_match = ADDRESS.search(line)
            return Finding(
                category=rule.category,
                raw_line=line,
                size=size,
                address=address_match.group(1) if address_match else None,
            )
        return None

    def grouped_findings(self) -> dict[str, list[Finding]]:
        """Group findings by category while retaining their input order."""
        grouped: dict[str, list[Finding]] = defaultdict(list)
        for finding in self.findings:
            grouped[finding.category].append(finding)
        return grouped


class MarkdownReport:
    """Render scanner state while keeping report policy separate from parsing."""

    def __init__(
        self,
        source: Path,
        scanner: LogScanner,
        heading_offset: int = 0,
        include_title: bool = True,
    ) -> None:
        self.source = source
        self.scanner = scanner
        self.heading_offset = heading_offset
        self.include_title = include_title

    def render(self) -> str:
        """Render the complete Markdown report."""
        sections: list[str] = []
        if self.include_title:
            generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            sections.extend(
                (
                    "# mssanitizer 内存检测问题报告",
                    "",
                    f"**日志文件**: {self.source}",
                    f"**生成时间**: {generated}",
                    "",
                )
            )
        sections.extend(self._summary())
        sections.extend(self._operation_table())
        if self.scanner.findings:
            sections.extend(self._finding_details())
            sections.extend(self._recommendations())
        if self.scanner.live_blocks:
            sections.extend(self._live_block_table())
        shifted = (self._shift_heading(line) for line in sections)
        return "\n".join(shifted)

    def _summary(self) -> list[str]:
        """Render counts and overall severity."""
        counts = Counter(item.category for item in self.scanner.findings)
        lines = [
            "## 问题摘要",
            "",
            "| 问题类型 | 数量 | 严重程度 |",
            "|---|---:|---|",
        ]
        for rule in RULES:
            if counts[rule.category]:
                lines.append(
                    f"| **{rule.category}** | **{counts[rule.category]}** | {rule.severity} |"
                )
        if not counts:
            lines.append("| 无错误 | 0 | 🟢 无 |")
        total = sum(counts.values())
        if total == 0:
            rating = "🟢 **良好 (GOOD)** — 未检测到内存错误。"
        elif total < 5:
            rating = f"🟡 **中等 (MODERATE)** — 检测到 {total} 个内存问题。"
        else:
            rating = f"🔴 **严重 (CRITICAL)** — 检测到 {total} 个内存问题。"
        lines.extend(("", "### 问题严重性评估", "", f"**整体评级**: {rating}", ""))
        return lines

    def _operation_table(self) -> list[str]:
        """Render memory-operation counters."""
        lines = ["## 内存操作统计", "", "| 操作类型 | 数量 |", "|---|---:|"]
        for operation in ("malloc", "free", "load", "store"):
            lines.append(f"| {operation.upper()} | {self.scanner.operations[operation]} |")
        malloc_count = self.scanner.operations["malloc"]
        free_count = self.scanner.operations["free"]
        if malloc_count and free_count:
            ratio = 100 * free_count / malloc_count
            lines.extend(("", f"**内存分配/释放比例**: {free_count}/{malloc_count} ({ratio:.1f}%)"))
            if malloc_count != free_count:
                lines.append("⚠️ MALLOC 与 FREE 数量不一致，需要核对资源生命周期。")
        lines.append("")
        return lines

    def _finding_details(self) -> list[str]:
        """Render bounded samples for every finding category."""
        grouped = self.scanner.grouped_findings()
        lines = ["## 详细错误分析", ""]
        for rule in RULES:
            findings = grouped.get(rule.category, [])
            if not findings:
                continue
            lines.extend((f"### {rule.category}", ""))
            for index, finding in enumerate(findings[:10], start=1):
                lines.extend((f"**错误 {index}**", "", "```text", finding.raw_line, "```"))
                if finding.address:
                    lines.append(f"- 地址: `{finding.address}`")
                if finding.size is not None:
                    lines.append(f"- 访问大小: {finding.size} 字节")
                lines.append("")
            if len(findings) > 10:
                lines.extend((f"其余 {len(findings) - 10} 条同类记录未展开。", ""))
        return lines

    def _recommendations(self) -> list[str]:
        """Render category-specific remediation guidance."""
        categories = {item.category for item in self.scanner.findings}
        lines = ["## 修复建议", ""]
        for rule in RULES:
            if rule.category not in categories:
                continue
            lines.extend((f"### {rule.title} (`{rule.category}`)", ""))
            lines.extend(f"- {action}" for action in REMEDIATION[rule.category])
            lines.append("")
        return lines

    def _live_block_table(self) -> list[str]:
        """Render allocations not paired with a free message."""
        lines = ["## 未释放的内存块", "", "| 地址 | 大小（字节） |", "|---|---:|"]
        visible = list(self.scanner.live_blocks.items())[:20]
        lines.extend(f"| `{address}` | {size} |" for address, size in visible)
        hidden = len(self.scanner.live_blocks) - len(visible)
        if hidden:
            lines.append(f"| ... | 另有 {hidden} 个未显示 |")
        lines.append("")
        return lines

    def _shift_heading(self, line: str) -> str:
        """Increase a Markdown heading level for embedded reports."""
        if self.heading_offset <= 0 or not line.startswith("#"):
            return line
        hashes, content = line.split(" ", maxsplit=1)
        return f"{'#' * self.heading_offset}{hashes} {content}"

    def write(self, destination: Path) -> None:
        """Save the rendered report to a UTF-8 text file."""
        destination.write_text(self.render(), encoding="utf-8")


class MssanitizerLogParser:
    """Compatibility facade for callers of the original parser API."""

    def __init__(self, log_file: str, heading_offset: int = 0, no_title: bool = False):
        self.log_file = log_file
        self.heading_offset = heading_offset
        self.no_title = no_title
        self.errors: dict[str, list[dict[str, object]]] = defaultdict(list)
        self.heap_blocks: dict[str, int] = {}
        self.stats: Counter[str] = Counter()
        self._scanner = LogScanner()

    def parse(self) -> None:
        """Parse the configured file and refresh compatibility fields."""
        self._scanner = LogScanner()
        with Path(self.log_file).open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                self._scanner.consume(line)
        self.errors = defaultdict(list)
        for finding in self._scanner.findings:
            self.errors[finding.category].append(finding.as_mapping())
        self.heap_blocks = dict(self._scanner.live_blocks)
        self.stats = Counter(
            {f"total_{name}": count for name, count in self._scanner.operations.items()}
        )
        self.stats.update(item.category for item in self._scanner.findings)

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Return the report and optionally save it."""
        report = MarkdownReport(
            Path(self.log_file),
            self._scanner,
            heading_offset=self.heading_offset,
            include_title=not self.no_title,
        ).render()
        if output_file:
            Path(output_file).write_text(report, encoding="utf-8")
            print(f"报告已保存到: {output_file}", file=sys.stderr)
        return report


def build_parser() -> argparse.ArgumentParser:
    """Create the log-parser command-line interface."""
    parser = argparse.ArgumentParser(description="解析 mssanitizer 日志并生成 Markdown 报告")
    parser.add_argument("log_file", help="mssanitizer 日志文件")
    parser.add_argument("-o", "--output", help="报告输出路径；缺省时写入标准输出")
    parser.add_argument(
        "--heading-offset",
        type=int,
        default=0,
        help="嵌入其他文档时增加标题层级",
    )
    parser.add_argument("--no-title", action="store_true", help="省略报告标题和生成元数据")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the parser command-line interface."""
    arguments = build_parser().parse_args(argv)
    parser = MssanitizerLogParser(
        arguments.log_file,
        heading_offset=max(arguments.heading_offset, 0),
        no_title=arguments.no_title,
    )
    try:
        parser.parse()
    except FileNotFoundError:
        print(f"错误: 文件不存在: {arguments.log_file}", file=sys.stderr)
        return 1
    report = parser.generate_report(arguments.output)
    if not arguments.output:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
