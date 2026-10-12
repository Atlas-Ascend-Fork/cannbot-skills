#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------
"""Audit a completed CANN operator's error reporting, include chain, and returns."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from error_code_triggers import (
    CALL_RE,
    CODE_RE,
    DEFAULT_EXCLUDES,
    DEPRECATED,
    DEPRECATED_MACROS,
    E9_RE,
    ERRNO_REPORT_NAMES,
    FORMAL_REPORT_NAMES,
    INTERNAL_REPORT_NAMES,
    LOG_ONLY_NAMES,
    MIXED_FORMAL_REPORT_NAMES,
    RETURN_CHECK_ENTRIES,
    SOURCE_SUFFIXES,
    bare_name,
    errno_code,
    first_argument,
)
from scan_error_code_candidates import Finding, scan as scan_candidates

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
DEFINE_RE = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s*(?:\((.*?)\))?\s*(.*)$", re.DOTALL)
RETURN_RE = re.compile(r"\breturn\s+([^;]+);", re.DOTALL)
ERROR_RETURN_RE = re.compile(
    r"\b(ERR|ERROR|FAIL|FAILED|INVALID|UNSUPPORTED|NOT_SUPPORT|NULLPTR|PARAM|GRAPH_FAILED|ACLNN_|ACL_ERROR)|"
    r"^(?:false|nullptr|-\d+|[1-9]\d*)$",
    re.IGNORECASE,
)
FUNCTION_RE = re.compile(
    r"(?:^|[;{}]\s*|\n\s*)"
    r"(?:[A-Za-z_]\w*(?:::\w+)*(?:\s*[*&])?\s+)+"
    r"([~A-Za-z_]\w*(?:::\w+)*)\s*\([^;{}]*\)\s*(?:const\s*)?(?:noexcept\s*)?\{",
    re.MULTILINE,
)


@dataclass(frozen=True)
class MacroDefinition:
    name: str
    path: str
    line: int
    body: str


@dataclass(frozen=True)
class MacroResolution:
    status: str
    code: str
    leaf: str
    definition_path: str
    definition_line: int
    trace: tuple[str, ...]


@dataclass
class AuditRecord:
    path: str
    line: int
    function: str
    stage: str
    entry: str
    final_code: str
    code_status: str
    report_status: str
    return_expr: str
    return_marker: str
    macro_leaf: str
    definition_path: str
    definition_line: int
    include_chain: str
    source_text: str
    recommended_action: str


@dataclass
class IncludeStats:
    cmake_files: int = 0
    include_dirs: int = 0
    resolved_includes: int = 0
    unresolved_includes: int = 0


def parse_args() -> argparse.Namespace:
    ops_root = Path(__file__).resolve().parents[2]
    default_catalog = ops_root / "cann-error-code-standard" / "references" / "documented-error-codes.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Completed operator directory to audit")
    parser.add_argument(
        "--cmake-root",
        type=Path,
        help="Repository/build root containing CMakeLists.txt; defaults to --root",
    )
    parser.add_argument("--cann-root", type=Path, help="CANN package root used to expand include variables")
    parser.add_argument(
        "--include-dir",
        action="append",
        default=[],
        type=Path,
        help="Additional include directory; repeatable",
    )
    parser.add_argument("--exclude-dir", action="append", default=[], help="Additional excluded directory name")
    parser.add_argument("--suffix", action="append", default=[], help="Additional source suffix")
    parser.add_argument(
        "--documented-codes",
        type=Path,
        default=default_catalog,
        help=(
            "Machine-readable baseline; defaults to "
            "ops/cann-error-code-standard/references/documented-error-codes.json"
        ),
    )
    parser.add_argument("--output", type=Path, help="Markdown output path; stdout when omitted")
    return parser.parse_args()


def iter_sources(root: Path, excludes: set[str], suffixes: set[str]) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        try:
            relative_parts = path.relative_to(root).parts[:-1]
        except ValueError:
            continue
        if any(part in excludes for part in relative_parts):
            continue
        yield path.resolve()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def strip_cmake_comments(text: str) -> str:
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def cmake_blocks(text: str, command: str) -> Iterable[str]:
    pattern = re.compile(rf"\b{re.escape(command)}\s*\(", re.IGNORECASE)
    for match in pattern.finditer(text):
        depth = 1
        pos = match.end()
        quoted = False
        escaped = False
        while pos < len(text) and depth:
            char = text[pos]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = not quoted
            elif not quoted and char == "(":
                depth += 1
            elif not quoted and char == ")":
                depth -= 1
            pos += 1
        if depth == 0:
            start = match.end()
            end = pos - 1
            yield text[start:end]


def tokenize_cmake(block: str) -> list[str]:
    normalized = block.replace("\n", " ").replace("\r", " ").replace(";", " ")
    try:
        return shlex.split(normalized, posix=True)
    except ValueError:
        return normalized.split()


def cmake_variables(text: str, cmake_file: Path, cmake_root: Path, cann_root: Path | None) -> dict[str, str]:
    values = {
        "CMAKE_CURRENT_SOURCE_DIR": str(cmake_file.parent),
        "CMAKE_CURRENT_LIST_DIR": str(cmake_file.parent),
        "PROJECT_SOURCE_DIR": str(cmake_root),
        "CMAKE_SOURCE_DIR": str(cmake_root),
    }
    if cann_root:
        values["ASCEND_CANN_PACKAGE_PATH"] = str(cann_root)
        values["CANN_ROOT"] = str(cann_root)
    for block in cmake_blocks(text, "set"):
        tokens = tokenize_cmake(block)
        if len(tokens) >= 2 and re.fullmatch(r"[A-Za-z_]\w*", tokens[0]):
            value = tokens[1]
            if not re.search(r"\$[{<]", value):
                candidate = Path(value)
                values[tokens[0]] = str(candidate if candidate.is_absolute() else cmake_file.parent / candidate)
    return values


def expand_cmake_token(token: str, values: dict[str, str], cmake_file: Path) -> tuple[Path | None, str]:
    visibility = {"PUBLIC", "PRIVATE", "INTERFACE", "SYSTEM", "BEFORE", "AFTER"}
    token = token.strip().strip('"')
    if not token or token.upper() in visibility or token.startswith("$<INSTALL_INTERFACE:"):
        return None, ""
    if token.startswith("$<BUILD_INTERFACE:") and token.endswith(">"):
        prefix_length = len("$<BUILD_INTERFACE:")
        token = token[prefix_length:-1]
    token = re.sub(
        r"\$ENV\{([A-Za-z_]\w*)\}",
        lambda match: values.get(match.group(1), os.environ.get(match.group(1), match.group(0))),
        token,
    )
    token = re.sub(r"\$\{([A-Za-z_]\w*)\}", lambda m: values.get(m.group(1), m.group(0)), token)
    if "$<" in token or re.search(r"\$\{?|\$ENV\{", token):
        return None, token
    candidate = Path(token)
    if not candidate.is_absolute():
        candidate = cmake_file.parent / candidate
    return candidate.resolve(), ""


def discover_include_dirs(
    cmake_root: Path,
    operator_root: Path,
    cann_root: Path | None,
    explicit: list[Path],
) -> tuple[list[Path], list[str], int]:
    dirs: list[Path] = []
    unresolved: list[str] = []
    cmake_candidates = set(operator_root.rglob("CMakeLists.txt"))
    current = operator_root
    while True:
        candidate = current / "CMakeLists.txt"
        if candidate.is_file():
            cmake_candidates.add(candidate)
        if current == cmake_root or current.parent == current:
            break
        try:
            current.relative_to(cmake_root)
        except ValueError:
            break
        current = current.parent
    cmake_files = sorted(cmake_candidates)
    for cmake_file in cmake_files:
        text = strip_cmake_comments(read_text(cmake_file))
        values = cmake_variables(text, cmake_file, cmake_root, cann_root)
        for command in ("target_include_directories", "include_directories"):
            for block in cmake_blocks(text, command):
                tokens = tokenize_cmake(block)
                if command == "target_include_directories" and tokens:
                    tokens = tokens[1:]
                for token in tokens:
                    path, pending = expand_cmake_token(token, values, cmake_file)
                    if path and path.is_dir():
                        dirs.append(path)
                    elif pending:
                        unresolved.append(f"{cmake_file}:{pending}")

    defaults = [operator_root, operator_root / "include", cmake_root, cmake_root / "include"]
    for base in [cmake_root, *cmake_root.parents[:2]]:
        defaults.extend([base / "opbase" / "include", base / "include"])
    if cann_root:
        defaults.extend(
            [
                cann_root / "include",
                cann_root / "include" / "op_common",
                cann_root / "include" / "nnopbase",
                cann_root / "pkg_inc",
                cann_root / "aarch64-linux" / "include",
                cann_root / "x86_64-linux" / "include",
            ]
        )
    defaults.extend(path.resolve() for path in explicit)
    dirs.extend(path.resolve() for path in defaults if path.is_dir())

    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique, sorted(set(unresolved)), len(cmake_files)


def resolve_include(name: str, quoted: bool, source: Path, include_dirs: list[Path]) -> Path | None:
    candidates: list[Path] = []
    if quoted:
        candidates.append(source.parent / name)
    candidates.extend(base / name for base in include_dirs)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def include_closure(
    source: Path,
    include_dirs: list[Path],
    max_depth: int = 32,
) -> tuple[list[Path], dict[Path, Path], list[str]]:
    ordered: list[Path] = []
    parent: dict[Path, Path] = {}
    unresolved: list[str] = []
    seen: set[Path] = {source.resolve()}

    def visit(path: Path, depth: int) -> None:
        if depth > max_depth:
            unresolved.append(f"{path}:include-depth>{max_depth}")
            return
        try:
            text = read_text(path)
        except OSError:
            return
        for match in INCLUDE_RE.finditer(text):
            name = match.group(2).strip()
            resolved = resolve_include(name, match.group(1) == '"', path, include_dirs)
            if not resolved:
                is_error_header = re.search(
                    r"(?:^|/)(?:log|error|op_log|opdev|op_common|nnopbase)",
                    name,
                    re.IGNORECASE,
                )
                if match.group(1) == '"' or is_error_header:
                    unresolved.append(f"{path}:{name}")
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            parent[resolved] = path
            visit(resolved, depth + 1)
            ordered.append(resolved)

    visit(source.resolve(), 0)
    ordered.append(source.resolve())
    return ordered, parent, unresolved


def logical_preprocessor_lines(text: str) -> Iterable[tuple[int, str]]:
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        start = index + 1
        chunk = lines[index]
        while chunk.rstrip().endswith("\\") and index + 1 < len(lines):
            index += 1
            chunk = chunk.rstrip()[:-1] + " " + lines[index].lstrip()
        if chunk.lstrip().startswith("#define"):
            yield start, chunk
        index += 1


def preprocessor_line_numbers(text: str) -> set[int]:
    """Return physical lines that belong to preprocessor directives."""
    result: set[int] = set()
    continued = False
    for line_no, raw in enumerate(text.splitlines(), start=1):
        is_directive = continued or raw.lstrip().startswith("#")
        if is_directive:
            result.add(line_no)
        continued = is_directive and raw.rstrip().endswith("\\")
    return result


def macro_definitions(paths: list[Path]) -> dict[str, MacroDefinition]:
    macros: dict[str, MacroDefinition] = {}
    for path in paths:
        try:
            text = read_text(path)
        except OSError:
            continue
        for line, logical in logical_preprocessor_lines(text):
            match = DEFINE_RE.match(logical)
            if not match:
                continue
            macros[match.group(1)] = MacroDefinition(match.group(1), str(path), line, match.group(3).strip())
    return macros


def macro_calls(body: str) -> list[str]:
    names = [match.group(1).split("::")[-1] for match in CALL_RE.finditer(body)]
    names.extend(re.findall(r"\b([A-Z][A-Z0-9_]+)\s*\(", body))
    return list(dict.fromkeys(name for name in names if name not in {"IF", "FOR", "WHILE", "SWITCH"}))


def reporting_aliases(macros: dict[str, MacroDefinition]) -> set[str]:
    """Find arbitrary macro aliases that eventually call a known reporting entry."""
    aliases: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, definition in macros.items():
            if name in aliases:
                continue
            calls = macro_calls(definition.body)
            if CALL_RE.search(definition.body) or any(child in aliases for child in calls):
                aliases.add(name)
                changed = True
    return aliases


def alias_call_regex(names: set[str]) -> re.Pattern[str] | None:
    if not names:
        return None
    pattern = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(rf"\b((?:[A-Za-z_]\w*::)*(?:{pattern}))\s*\(")


def resolve_macro(
    name: str,
    macros: dict[str, MacroDefinition],
    documented: set[str],
    visited: tuple[str, ...] = (),
) -> MacroResolution:
    bare = name.split("::")[-1]
    if bare in visited:
        return MacroResolution("UNRESOLVED", "", bare, "", 0, visited + (bare,))
    definition = macros.get(bare)
    if bare in LOG_ONLY_NAMES:
        return MacroResolution(
            "LOG_ONLY",
            "",
            bare,
            definition.path if definition else "",
            definition.line if definition else 0,
            visited + (bare,),
        )
    if not definition:
        return MacroResolution("UNRESOLVED", "", bare, "", 0, visited + (bare,))

    trace = visited + (bare,)
    body = definition.body
    codes = CODE_RE.findall(body)
    code = codes[0] if codes else ""
    called = macro_calls(body)
    if any(item in INTERNAL_REPORT_NAMES for item in called) or (code and E9_RE.fullmatch(code)):
        return MacroResolution("INTERNAL_REPORTED", code, bare, definition.path, definition.line, trace)
    if any(item in FORMAL_REPORT_NAMES for item in called):
        if code in documented:
            status = "FORMAL_REPORTED"
        elif code:
            status = "REPORTED_UNDOCUMENTED"
        else:
            status = "UNRESOLVED"
        return MacroResolution(status, code, bare, definition.path, definition.line, trace)
    if any(item in ERRNO_REPORT_NAMES for item in called):
        report_call = next(item for item in called if item in ERRNO_REPORT_NAMES)
        mapped = errno_code(first_argument(body, report_call))
        if mapped:
            status = "INTERNAL_REPORTED" if E9_RE.fullmatch(mapped) else "FORMAL_REPORTED"
            return MacroResolution(status, mapped, bare, definition.path, definition.line, trace)
    if bare in DEPRECATED_MACROS:
        return MacroResolution(
            "FORMAL_REPORTED",
            DEPRECATED_MACROS[bare],
            bare,
            definition.path,
            definition.line,
            trace,
        )
    if "return" in body and not called:
        return MacroResolution("RETURN_ONLY", code, bare, definition.path, definition.line, trace)

    best = MacroResolution("UNRESOLVED", code, bare, definition.path, definition.line, trace)
    priority = {
        "INTERNAL_REPORTED": 5,
        "FORMAL_REPORTED": 4,
        "REPORTED_UNDOCUMENTED": 3,
        "LOG_ONLY": 2,
        "RETURN_ONLY": 1,
        "UNRESOLVED": 0,
    }
    for child in called:
        if child == bare:
            continue
        result = resolve_macro(child, macros, documented, trace)
        if priority.get(result.status, 0) > priority.get(best.status, 0):
            best = MacroResolution(
                result.status,
                result.code or code,
                result.leaf,
                result.definition_path or definition.path,
                result.definition_line or definition.line,
                result.trace,
            )
    log_tokens = ("DLOG", "GELOG", "OPS_LOG", "printf", "fprintf")
    if best.status == "UNRESOLVED" and any(token in body for token in log_tokens):
        return MacroResolution("LOG_ONLY", code, bare, definition.path, definition.line, trace)
    return best


def logical_statements(text: str, alias_re: re.Pattern[str] | None = None) -> Iterable[tuple[int, str]]:
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        raw = lines[index]
        has_call = CALL_RE.search(raw) or (alias_re.search(raw) if alias_re else None)
        if raw.lstrip().startswith("#") or not (has_call or RETURN_RE.search(raw)):
            index += 1
            continue
        start = index + 1
        chunk = raw.strip()
        balance = chunk.count("(") - chunk.count(")")
        while index + 1 < len(lines) and (balance > 0 or (";" not in chunk and len(chunk) < 8000)):
            index += 1
            addition = lines[index].strip()
            chunk += " " + addition
            balance += addition.count("(") - addition.count(")")
            if balance <= 0 and ";" in addition:
                break
            if index + 1 - start > 30:
                break
        yield start, re.sub(r"\s+", " ", chunk).strip()
        index += 1


def function_at(text: str, line: int) -> str:
    offset = sum(len(value) + 1 for value in text.splitlines()[: max(0, line - 1)])
    prefix = text[:offset]
    matches = list(FUNCTION_RE.finditer(prefix[-20000:]))
    return matches[-1].group(1) if matches else "<unknown>"


def stage_for(path: str) -> str:
    lowered = path.lower().replace("\\", "/")
    if "/op_api/" in lowered or "/aclnn" in lowered or lowered.endswith("_api.cpp"):
        return "op_api"
    if "/op_kernel/" in lowered or "/kernel/" in lowered or "_kernel" in lowered:
        return "op_kernel"
    if "/op_host/" in lowered or "tiling" in lowered or "infer_shape" in lowered or "infershape" in lowered:
        return "op_host"
    return "shared"


def nearby_return(lines: list[str], line: int, statement: str) -> str:
    direct = RETURN_RE.search(statement)
    if direct:
        return re.sub(r"\s+", " ", direct.group(1)).strip()
    start = max(0, line - 1)
    end = min(len(lines), start + 12)
    window = "\n".join(lines[start:end])
    match = RETURN_RE.search(window)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def macro_return(resolution: MacroResolution, macros: dict[str, MacroDefinition]) -> str:
    for name in resolution.trace:
        definition = macros.get(name)
        if not definition:
            continue
        match = RETURN_RE.search(definition.body)
        if match:
            return re.sub(r"\s+", " ", match.group(1)).strip()
    return ""


def include_chain_text(source: Path, target: str, parents: dict[Path, Path], root: Path) -> str:
    if not target:
        return ""
    target_path = Path(target).resolve()
    if target_path == source.resolve():
        return source.relative_to(root).as_posix() if source.is_relative_to(root) else str(source)
    chain = [target_path]
    while chain[-1] in parents:
        chain.append(parents[chain[-1]])
        if chain[-1] == source.resolve():
            break
    chain.reverse()
    rendered = []
    for item in chain:
        rendered.append(item.relative_to(root).as_posix() if item.is_relative_to(root) else str(item))
    return " -> ".join(rendered)


def code_status(code: str, documented: set[str]) -> str:
    if not code:
        return "NO_CODE"
    if code in DEPRECATED:
        return "GOVERNANCE_DEPRECATED"
    if E9_RE.fullmatch(code):
        return "INTERNAL_E9"
    if code in documented:
        return "DOCUMENTED_FORMAL"
    return "NOT_IN_DOCUMENT_REFERENCE"


def marker_for(report_status: str, status: str, return_expr: str) -> str:
    if status == "GOVERNANCE_DEPRECATED":
        return "DEPRECATED_CODE_RETURN" if return_expr else "DEPRECATED_CODE_NO_RETURN"
    if report_status == "INTERNAL_REPORTED" or status == "INTERNAL_E9":
        return "INTERNAL_E9_RETURN" if return_expr else "INTERNAL_E9_NO_RETURN"
    if report_status == "FORMAL_REPORTED" and status == "DOCUMENTED_FORMAL":
        return "FORMAL_REPORTED_RETURN" if return_expr else "FORMAL_REPORTED_NO_RETURN_OBSERVED"
    if report_status == "LOG_ONLY":
        if return_expr and re.search(r"\b(rc|ret|status|result|error|errno)\b", return_expr, re.IGNORECASE):
            return "LOWER_LAYER_RETURN"
        return "LOG_ONLY_RETURN" if return_expr else "LOG_ONLY_NO_RETURN_OBSERVED"
    if report_status == "RETURN_ONLY":
        return "RETURN_WITHOUT_REPORT"
    if report_status == "REPORTED_UNDOCUMENTED":
        return "UNDOCUMENTED_CODE_RETURN" if return_expr else "UNDOCUMENTED_CODE_NO_RETURN"
    return "UNRESOLVED_RETURN" if return_expr else "UNRESOLVED_NO_RETURN"


def action_for(marker: str, code: str) -> str:
    if marker.startswith("FORMAL_REPORTED"):
        return "核对注册Arglist、actual/expected/reason和首报唯一性；通过后保留"
    if marker.startswith("DEPRECATED_CODE"):
        return "按参数根因改用文档收录的EZ00xx语义宏"
    if marker.startswith("INTERNAL_E9"):
        return "展开调用链；参数/能力/执行/下层故障分类后改用正式码或仅日志透传"
    if marker == "LOWER_LAYER_RETURN":
        return "确认下层已首报正式码并保留关联字段；当前层不得重复上报"
    if marker.startswith("LOG_ONLY"):
        return "确认是否存在下层首错；不存在则在根因层补正式码"
    if marker == "RETURN_WITHOUT_REPORT":
        return "确认该失败路径的首报责任；无下层首错时在根因层补正式码"
    if marker.startswith("UNDOCUMENTED_CODE"):
        return f"核对目标版本注册表和官方资料；未注册时不得继续使用{code or '该码'}"
    return "补齐CMake/include路径并解析实际宏叶子后再决定整改"


def load_documented_codes(path: Path) -> tuple[set[str], dict[str, dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = {str(row["code"]): row for row in payload.get("codes", [])}
    return set(items), items


def audit_source(
    source: Path,
    root: Path,
    include_dirs: list[Path],
    documented: set[str],
) -> tuple[list[AuditRecord], int, list[str]]:
    paths, parents, unresolved = include_closure(source, include_dirs)
    macros = macro_definitions(paths)
    alias_re = alias_call_regex(reporting_aliases(macros))
    text = read_text(source)
    lines = text.splitlines()
    preprocessor_lines = preprocessor_line_numbers(text)
    relative = source.relative_to(root).as_posix()
    records: list[AuditRecord] = []
    occupied_returns: set[int] = set()

    for line, statement in logical_statements(text, alias_re):
        if line in preprocessor_lines:
            continue
        call = CALL_RE.search(statement) or (alias_re.search(statement) if alias_re else None)
        if not call:
            continue
        entry = call.group(1)
        bare_entry = bare_name(entry)
        codes = CODE_RE.findall(statement)
        explicit_code = codes[0] if codes else DEPRECATED_MACROS.get(bare_entry, "")
        mapped_code = ""
        if bare_entry == "OP_LOGE" or bare_entry in ERRNO_REPORT_NAMES:
            mapped_code = errno_code(first_argument(statement, entry))
        elif bare_entry in RETURN_CHECK_ENTRIES:
            errno_match = re.search(r"\b(ACLNN_ERR_[A-Z0-9_]+|ACL_ERROR_[A-Z0-9_]+|[135]\d{5})\b", statement)
            mapped_code = errno_code(errno_match.group(1)) if errno_match else ""
        if bare_entry in INTERNAL_REPORT_NAMES:
            resolution = MacroResolution(
                "INTERNAL_REPORTED",
                explicit_code,
                bare_entry,
                str(source),
                line,
                (bare_entry,),
            )
        elif bare_entry in ERRNO_REPORT_NAMES:
            if mapped_code:
                report_status = "INTERNAL_REPORTED" if E9_RE.fullmatch(mapped_code) else "FORMAL_REPORTED"
            else:
                report_status = "UNRESOLVED"
            resolution = MacroResolution(report_status, mapped_code, bare_entry, str(source), line, (bare_entry,))
        elif bare_entry in FORMAL_REPORT_NAMES:
            if explicit_code and E9_RE.fullmatch(explicit_code):
                status = "INTERNAL_REPORTED"
            elif explicit_code in documented:
                status = "FORMAL_REPORTED"
            elif explicit_code:
                status = "REPORTED_UNDOCUMENTED"
            else:
                status = "UNRESOLVED" if bare_entry in MIXED_FORMAL_REPORT_NAMES else "REPORTED_UNDOCUMENTED"
            resolution = MacroResolution(status, explicit_code, bare_entry, str(source), line, (bare_entry,))
        elif bare_entry in LOG_ONLY_NAMES:
            resolution = resolve_macro(entry, macros, documented)
        else:
            resolution = resolve_macro(entry, macros, documented)
        if mapped_code:
            mapped_status = "INTERNAL_REPORTED" if E9_RE.fullmatch(mapped_code) else "FORMAL_REPORTED"
            if bare_entry == "OP_LOGE" or bare_entry in ERRNO_REPORT_NAMES:
                resolution = MacroResolution(
                    mapped_status,
                    mapped_code,
                    resolution.leaf or bare_entry,
                    resolution.definition_path,
                    resolution.definition_line,
                    resolution.trace or (bare_entry,),
                )
            elif bare_entry in RETURN_CHECK_ENTRIES and resolution.status == "UNRESOLVED":
                resolution = MacroResolution("RETURN_ONLY", mapped_code, bare_entry, "", 0, (bare_entry,))
        final_code = mapped_code or explicit_code or resolution.code
        ret = nearby_return(lines, line, statement) or macro_return(resolution, macros)
        if ret:
            for ret_line in range(line, min(len(lines) + 1, line + 12)):
                if "return" in lines[ret_line - 1]:
                    occupied_returns.add(ret_line)
                    break
        status = code_status(final_code, documented)
        marker = marker_for(resolution.status, status, ret)
        records.append(
            AuditRecord(
                path=relative,
                line=line,
                function=function_at(text, line),
                stage=stage_for(relative),
                entry=entry,
                final_code=final_code,
                code_status=status,
                report_status=resolution.status,
                return_expr=ret,
                return_marker=marker,
                macro_leaf=resolution.leaf,
                definition_path=resolution.definition_path,
                definition_line=resolution.definition_line,
                include_chain=include_chain_text(source, resolution.definition_path, parents, root),
                source_text=statement[:500],
                recommended_action=action_for(marker, final_code),
            )
        )

    for line, statement in logical_statements(text, alias_re):
        has_call = CALL_RE.search(statement) or (alias_re.search(statement) if alias_re else None)
        if line in preprocessor_lines or line in occupied_returns or has_call:
            continue
        match = RETURN_RE.search(statement)
        if not match:
            continue
        expr = re.sub(r"\s+", " ", match.group(1)).strip()
        if not ERROR_RETURN_RE.search(expr):
            continue
        marker = "RETURN_WITHOUT_REPORT"
        records.append(
            AuditRecord(
                path=relative,
                line=line,
                function=function_at(text, line),
                stage=stage_for(relative),
                entry="return",
                final_code="",
                code_status="NO_CODE",
                report_status="RETURN_ONLY",
                return_expr=expr,
                return_marker=marker,
                macro_leaf="",
                definition_path="",
                definition_line=0,
                include_chain="",
                source_text=statement[:500],
                recommended_action=action_for(marker, ""),
            )
        )
    return records, len(paths) - 1, unresolved


def audit(
    root: Path,
    cmake_root: Path,
    cann_root: Path | None,
    explicit_includes: list[Path],
    excludes: set[str],
    suffixes: set[str],
    documented: set[str],
) -> tuple[list[AuditRecord], IncludeStats, list[Path], list[str], list[str]]:
    include_dirs, unresolved_cmake, cmake_count = discover_include_dirs(cmake_root, root, cann_root, explicit_includes)
    records: list[AuditRecord] = []
    unresolved_includes: list[str] = []
    resolved_total = 0
    sources = list(sorted(iter_sources(root, excludes, suffixes)))
    for source in sources:
        try:
            found, resolved, unresolved = audit_source(source, root, include_dirs, documented)
        except OSError as exc:
            print(f"warning: cannot audit {source}: {exc}", file=sys.stderr)
            continue
        records.extend(found)
        resolved_total += resolved
        unresolved_includes.extend(unresolved)
    records.sort(key=lambda item: (item.path, item.line, item.entry))
    stats = IncludeStats(cmake_count, len(include_dirs), resolved_total, len(set(unresolved_includes)))
    return records, stats, include_dirs, unresolved_cmake, sorted(set(unresolved_includes))


def static_decision(
    records: list[AuditRecord],
    unresolved_cmake: list[str],
    unresolved_includes: list[str],
    unmatched_candidates: list[Finding],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not records:
        reasons.append("未识别到错误上报或失败返回路径，需确认扫描边界是否完整")
    blocking = [row for row in records if row.return_marker != "FORMAL_REPORTED_RETURN"]
    if blocking:
        reasons.append(f"{len(blocking)}条路径尚未达到正式码上报并返回的静态完成条件")
    if unresolved_cmake:
        reasons.append(f"{len(unresolved_cmake)}个CMake路径表达式未展开")
    if unresolved_includes:
        reasons.append(f"{len(unresolved_includes)}个错误码相关include未解析")
    if unmatched_candidates:
        reasons.append(f"{len(unmatched_candidates)}条候选未进入调用与返回审计明细")
    return not reasons, reasons


def unmatched_candidates(candidates: list[Finding], records: list[AuditRecord]) -> list[Finding]:
    """Return scanner findings not represented by any audit row on the same source line."""
    audited_locations = {(row.path, row.line) for row in records}
    return [item for item in candidates if (item.path, item.line) not in audited_locations]


def md_cell(value: object) -> str:
    return str(value or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def markdown_text(
    root: Path,
    cmake_root: Path,
    records: list[AuditRecord],
    stats: IncludeStats,
    include_dirs: list[Path],
    unresolved_cmake: list[str],
    unresolved_includes: list[str],
    candidates: list[Finding],
    unmatched: list[Finding],
) -> str:
    static_passed, static_reasons = static_decision(records, unresolved_cmake, unresolved_includes, unmatched)
    marker_counts = Counter(row.return_marker for row in records)
    code_counts = Counter(row.code_status for row in records)
    stage_counts = Counter(row.stage for row in records)
    lines = [
        "# 算子错误码返回状态报告",
        "",
        f"- 扫描目录：`{root}`",
        f"- CMake目录：`{cmake_root}`",
        f"- 静态审计是否通过：`{'是' if static_passed else '否'}`",
        f"- 治理候选：`{len(candidates)}`",
        f"- 未进入审计明细的候选：`{len(unmatched)}`",
        f"- 调用/返回路径：`{len(records)}`",
        "",
        "## 状态汇总",
        "",
        "| 返回标记 | 数量 |",
        "|---|---:|",
    ]
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(marker_counts.items()))
    lines.extend(["", "| 错误码状态 | 数量 |", "|---|---:|"])
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(code_counts.items()))
    lines.extend(["", "| 阶段 | 数量 |", "|---|---:|"])
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(stage_counts.items()))

    lines.extend(
        [
            "",
            "## CMake与include链",
            "",
            f"- CMake文件：`{stats.cmake_files}`",
            f"- 有效include目录：`{stats.include_dirs}`",
            f"- 递归解析include：`{stats.resolved_includes}`",
            f"- 未解析include：`{stats.unresolved_includes}`",
            "",
            "已采用的include目录：",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in include_dirs)
    if unresolved_cmake:
        lines.extend(["", "未展开的CMake路径表达式：", ""])
        lines.extend(f"- `{item}`" for item in unresolved_cmake[:100])
    if unresolved_includes:
        lines.extend(["", "未解析include（最多列出100项）：", ""])
        lines.extend(f"- `{item}`" for item in unresolved_includes[:100])

    lines.extend(
        [
            "",
            "## 候选—审计对账",
            "",
            f"- 候选扫描器发现：`{len(candidates)}`",
            f"- 未进入审计明细：`{len(unmatched)}`",
        ]
    )
    if unmatched:
        lines.extend(
            [
                "",
                "| 文件 | 行 | 类型 | 入口 | 错误码/参数 |",
                "|---|---:|---|---|---|",
            ]
        )
        for item in unmatched:
            lines.append(
                f"| `{md_cell(item.path)}` | {item.line} | `{item.kind}` | "
                f"`{md_cell(item.entry)}` | `{md_cell(item.code)}` |"
            )

    lines.extend(
        [
            "",
            "## 调用与返回明细",
            "",
            "| 文件 | 行 | 阶段 | 函数 | 入口 | 最终码 | 码状态 | 上报状态 | 返回标记 | 宏定义来源 | include链 | 建议 |",
            "|---|---:|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in records:
        definition = f"{row.definition_path}:{row.definition_line}" if row.definition_path else ""
        lines.append(
            f"| `{md_cell(row.path)}` | {row.line} | {row.stage} | `{md_cell(row.function)}` | "
            f"`{md_cell(row.entry)}` | `{md_cell(row.final_code)}` | `{row.code_status}` | `{row.report_status}` | "
            f"`{row.return_marker}` | `{md_cell(definition)}` | "
            f"`{md_cell(row.include_chain)}` | {md_cell(row.recommended_action)} |"
        )

    lines.extend(
        [
            "",
            "## 静态审计结论",
            "",
        ]
    )
    if static_passed:
        lines.append("- 未发现待整改返回路径或include缺口。此结论不包含编译和测试结果，不能单独作为最终关闭依据。")
    else:
        lines.extend(f"- {reason}" for reason in static_reasons)
    lines.append("- `LOWER_LAYER_RETURN`只有在下层正式首错已通过负向日志验证后，才可由评审者判定该路径完成。")
    lines.extend(
        [
            "",
            "## 编译验证",
            "",
            "- 状态：`待补充`",
            "- 受影响target：",
            "- 构建命令与环境：",
            "- 结果与首个有效错误：",
            "",
            "## 测试用例",
            "",
            "- 状态：`待补充`",
            "- 新增或修改的测试：",
            "- 负向场景与预期错误码：",
            "- 执行命令、结果与首错唯一性：",
            "- 未覆盖项：",
            "",
            "## 最终结论",
            "",
            "- 是否可关闭：`否`",
            "- 原因：编译验证与测试用例结果尚未补充。完成验证后由执行者更新本节。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    cmake_root = (args.cmake_root or root).resolve()
    cann_root = args.cann_root.resolve() if args.cann_root else None
    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return 2
    if not cmake_root.is_dir():
        print(f"error: --cmake-root is not a directory: {cmake_root}", file=sys.stderr)
        return 2
    if not args.documented_codes.is_file():
        print(f"error: documented-code reference not found: {args.documented_codes}", file=sys.stderr)
        return 2
    documented, _ = load_documented_codes(args.documented_codes)
    excludes = DEFAULT_EXCLUDES | set(args.exclude_dir)
    suffixes = SOURCE_SUFFIXES | {value if value.startswith(".") else f".{value}" for value in args.suffix}
    records, stats, include_dirs, unresolved_cmake, unresolved_includes = audit(
        root,
        cmake_root,
        cann_root,
        args.include_dir,
        excludes,
        suffixes,
        documented,
    )
    candidates = scan_candidates(root, excludes, suffixes, False)
    unmatched = unmatched_candidates(candidates, records)
    rendered = markdown_text(
        root,
        cmake_root,
        records,
        stats,
        include_dirs,
        unresolved_cmake,
        unresolved_includes,
        candidates,
        unmatched,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
