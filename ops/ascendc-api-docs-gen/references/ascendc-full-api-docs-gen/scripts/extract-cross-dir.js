#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

'use strict';

const fs = require('fs');
const path = require('path');
const {
    trackAscendCReferences,
    detectProjectRoot,
    normPath,
} = require('./lib/ascendc-cross-dir-tracker');

function relPath(filePath, base) {
    if (!filePath) return '—';
    if (base) {
        const rel = path.relative(base, filePath).replace(/\\/g, '/');
        if (!rel.startsWith('..')) return rel;
    }
    return normPath(filePath);
}

function kindLabel(kind) {
    switch (kind) {
        case 'input': return '输入范围内代码文件';
        case 'include-internal': return '#include 引入的内部文件';
        case 'include-external': return '#include 引入的外部文件';
        case 'impl-pair': return '头文件对应实现文件';
        case 'api-check': return 'api_check/kernel_check 文件';
        case 'api-check-impl': return 'api_check/kernel_check 对应实现文件';
        default: return kind || '代码文件';
    }
}

function reasonText(meta, base) {
    if (!meta) return '代码文件';
    if (meta.kind === 'input') return '输入范围内代码文件';
    if (meta.kind === 'impl-pair') {
        return `**由 \`${relPath(meta.includedBy, base)}\` 定位对应实现文件**`;
    }
    if (meta.kind === 'api-check') {
        return `**由 \`${relPath(meta.includedBy, base)}\` 定位对应 api_check/kernel_check 文件**`;
    }
    if (meta.kind === 'api-check-impl') {
        return `**由 \`${relPath(meta.includedBy, base)}\` 定位对应的 .cpp 实现文件**`;
    }
    if (meta.kind === 'include-external' || meta.kind === 'include-internal') {
        const depth = meta.depth > 1 ? `（第 ${meta.depth} 层）` : '';
        return `**由 \`${relPath(meta.includedBy, base)}\` #include "${meta.includeText}" 引入**${depth}`;
    }
    return meta.reason || '代码文件';
}

function reasonPlain(meta, base) {
    if (!meta) return '代码文件';
    if (meta.kind === 'input') return '输入范围内代码文件';
    if (meta.kind === 'impl-pair') return `由 ${relPath(meta.includedBy, base)} 定位对应实现文件`;
    if (meta.kind === 'api-check') return `由 ${relPath(meta.includedBy, base)} 定位对应 api_check/kernel_check 文件`;
    if (meta.kind === 'api-check-impl') return `由 ${relPath(meta.includedBy, base)} 定位对应的 .cpp 实现文件`;
    if (meta.kind === 'include-external' || meta.kind === 'include-internal') {
        const depth = meta.depth > 1 ? `（第 ${meta.depth} 层）` : '';
        return `由 ${relPath(meta.includedBy, base)} #include "${meta.includeText}" 引入${depth}`;
    }
    return meta.reason || '代码文件';
}

function buildStructuredClosure(tracking) {
    const base = tracking.projectRoot || path.dirname(tracking.inputPath);
    const candidateFiles = tracking.files
        .slice()
        .sort((a, b) => a.localeCompare(b))
        .map(filePath => {
            const meta = tracking.metadata.get(filePath) || {};
            return {
                path: relPath(filePath, base),
                kind: meta.kind || 'code',
                reason: reasonPlain(meta, base),
                includedBy: meta.includedBy ? relPath(meta.includedBy, base) : null,
                includeText: meta.includeText || null,
                depth: Number.isInteger(meta.depth) ? meta.depth : 0,
            };
        });
    return {
        schemaVersion: 1,
        kind: 'ascendc_code_closure',
        inputPath: relPath(tracking.inputPath, base),
        projectRoot: tracking.projectRoot ? normPath(tracking.projectRoot) : null,
        candidateFiles,
        unresolved: [...tracking.unresolved],
    };
}
function generateReport(tracking) {
    const base = tracking.projectRoot || path.dirname(tracking.inputPath);
    const files = tracking.files.slice().sort((a, b) => a.localeCompare(b));
    const groups = {
        input: files.filter(f => tracking.metadata.get(f)?.kind === 'input'),
        include: files.filter(f => ['include-internal', 'include-external'].includes(tracking.metadata.get(f)?.kind)),
        impl: files.filter(f => tracking.metadata.get(f)?.kind === 'impl-pair'),
        check: files.filter(f => tracking.metadata.get(f)?.kind === 'api-check'),
        checkImpl: files.filter(f => tracking.metadata.get(f)?.kind === 'api-check-impl'),
    };

    const out = [];
    out.push('# 跨目录引用追踪报告');
    out.push('');
    out.push(`- 输入路径：\`${relPath(tracking.inputPath, base)}\``);
    if (tracking.projectRoot) out.push(`- 项目根目录：\`${tracking.projectRoot}\``);
    out.push(`- 输入范围代码文件数：${tracking.inputFiles.length}`);
    out.push(`- 纳入分析文件总数：${files.length}`);
    out.push(`- #include 引用文件数：${groups.include.length}`);
    out.push(`- 实现文件数：${groups.impl.length}`);
    out.push(`- api_check/kernel_check 文件数：${groups.check.length + groups.checkImpl.length}`);
    out.push('');

    out.push('## 优先核验文件');
    out.push('');
    out.push('以下是目标 API 代码闭包中的候选文件，通常包含函数声明、实现分发、kernel 侧校验或参数硬约束。它们是后续核验范围，不代表已形成代码事实。');
    out.push('');
    out.push('| 文件路径 | 类型 |');
    out.push('|---------|------|');
    for (const f of [...groups.input, ...groups.impl, ...groups.check, ...groups.checkImpl].slice(0, 80)) {
        const meta = tracking.metadata.get(f);
        out.push(`| \`${relPath(f, base)}\` | ${kindLabel(meta?.kind)} |`);
    }
    if ([...groups.input, ...groups.impl, ...groups.check, ...groups.checkImpl].length > 80) {
        out.push(`| *(另有 ${[...groups.input, ...groups.impl, ...groups.check, ...groups.checkImpl].length - 80} 个文件省略)* | 见完整清单 |`);
    }
    out.push('');

    if (tracking.archGuards && tracking.archGuards.size > 0) {
        out.push('## 产品支持证据线索');
        out.push('');
        out.push('以下文件包含 `__NPU_ARCH__`、`ASCEND_IS_AIC` 或相关产品/执行单元条件。该列表只用于辅助判断当前接口支持哪些产品型号；不能直接作为最终结论，仍需结合 API 主头文件、include 链路和实现文件中的有效条件综合判定。');
        out.push('');
        out.push('| 文件路径 | 命中位置 |');
        out.push('|---------|---------|');
        for (const [filePath, guards] of [...tracking.archGuards.entries()].slice(0, 80)) {
            const snippets = guards.slice(0, 4).map(g => `L${g.line}: \`${g.text.replace(/\|/g, '\\|')}\``).join('<br>');
            const suffix = guards.length > 4 ? `<br>另有 ${guards.length - 4} 处` : '';
            out.push(`| \`${relPath(filePath, base)}\` | ${snippets}${suffix} |`);
        }
        if (tracking.archGuards.size > 80) {
            out.push(`| *(另有 ${tracking.archGuards.size - 80} 个文件省略)* | 见完整清单后按需搜索 |`);
        }
        out.push('');
    }

    out.push('## 代码分析清单（完整引用列表）');
    out.push('');
    out.push('| 文件路径 | 分析原因 |');
    out.push('|---------|---------|');
    for (const f of files) {
        out.push(`| \`${relPath(f, base)}\` | ${reasonText(tracking.metadata.get(f), base)} |`);
    }
    out.push('');

    if (tracking.unresolved.length > 0) {
        out.push('## 未解析项');
        out.push('');
        out.push('以下引用未能自动解析，模型分析时可跳过或按需人工核验：');
        out.push('');
        for (const item of tracking.unresolved.slice(0, 50)) out.push(`- ${item}`);
        if (tracking.unresolved.length > 50) {
            out.push(`- *(另有 ${tracking.unresolved.length - 50} 条，已省略)*`);
        }
        out.push('');
    }

    out.push('## 验证清单');
    out.push('');
    out.push('- [x] 已自动索引输入范围内所有 `.h`/`.hpp`/`.cpp`/`.cc`/`.cxx` 文件的 `#include "..."` 指令；这不代表已完成语义核验');
    out.push('- [x] 已解析同目录、相对路径、项目根目录、`include/`、`impl/`、`include/{api_type}/`、`impl/{api_type}/` 下的引用路径');
    out.push('- [x] 已将内部和外部 `#include` 文件纳入完整引用列表');
    out.push('- [x] 已尝试定位 `include/` 到 `impl/` 的对应实现文件，以及同目录同名 `.cpp` 实现文件');
    out.push('- [x] 已尝试定位 `impl/{api_type}/detail/api_check/` 和 `impl/{api_type}/detail/api_check/kernel_check/` 下的校验文件及其对应实现文件');
    out.push('- [x] 已递归追踪头文件引用（最多 3 层）和 `.cpp` 实现文件引用（最多 2 层）');
    out.push('- [x] 已输出候选代码闭包并注明引用链、实现定位链或 api_check 定位链；后续必须逐项记录核验结果');
    out.push('- [ ] 阶段分析时必须结合"产品支持证据线索"判断当前接口支持哪些产品型号，并在 `code-evidence.md` 的产品支持与差异结论中体现');
    out.push('');

    return out.join('\n');
}

function main() {
    const argv = process.argv.slice(2);
    const outIdx = argv.indexOf('--output');
    if (outIdx !== -1 && (!argv[outIdx + 1] || argv[outIdx + 1].startsWith('--'))) {
        throw new Error('--output 缺少文件路径');
    }
    const outputFile = outIdx !== -1 ? path.resolve(argv.splice(outIdx, 2)[1]) : null;

    if (argv.length < 1) {
        console.error('用法: node scripts/extract-cross-dir.js <AscendC API文件或目录> [项目根目录] [--output <文件>]');
        console.error('示例: node scripts/extract-cross-dir.js <repo-root>/include/basic_api/kernel_operator_mm_intf.h <repo-root> --output outputs/BroadCastVecToMM/0_cross_dir.md');
        process.exit(1);
    }

    const inputPath = path.resolve(argv[0]);
    if (!fs.existsSync(inputPath)) {
        console.error(`错误: 输入路径不存在: ${inputPath}`);
        process.exit(1);
    }
    const projectRoot = argv[1] ? path.resolve(argv[1]) : detectProjectRoot(inputPath);
    const tracking = trackAscendCReferences(inputPath, projectRoot);
    const report = generateReport(tracking);

    if (outputFile) {
        fs.mkdirSync(path.dirname(outputFile), { recursive: true });
        fs.writeFileSync(outputFile, report + '\n', 'utf8');
        console.error(`已写入: ${outputFile}`);
    } else {
        process.stdout.write(report + '\n', 'utf8');
    }
}

if (require.main === module) main();

module.exports = {
    buildStructuredClosure,
    generateReport,
    relPath,
};
