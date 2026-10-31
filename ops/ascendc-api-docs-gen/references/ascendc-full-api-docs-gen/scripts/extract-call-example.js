#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

/**
 * AscendC 调用示例提取脚本。
 * 仅从 examples/ 中提取当前 API 的直接调用：先 Compute()，再 Process()。
 * Helper 调用仅输出受限模型回查所需的候选定位，不自动拼接调用链。
 * 默认输出 Markdown；加 --json 输出候选元数据。
 */
const fs = require('fs');
const path = require('path');
const { hasApiSignature } = require('./lib/ascendc-signatures');

const API_TYPE_TO_EXAMPLE_DIR = {
    adv_api: '01_simd_cpp_api',
    basic_api: '01_simd_cpp_api',
    tensor_api: '01_simd_cpp_api',
    c_api: '02_simd_c_api',
    simt_api: '03_simt_api'
};
const MAX_HELPER_CANDIDATES = 8;
const NON_FUNCTION_NAMES = new Set([
    'alignof', 'catch', 'decltype', 'for', 'if', 'new', 'noexcept',
    'operator', 'return', 'sizeof', 'static_assert', 'switch', 'typeid', 'while'
]);


function ensureTrailingNewline(content) {
    return content.endsWith('\n') ? content : content + '\n';
}

function cleanExampleCode(content) {
    return content
        .replace(/\/\*[\s\S]*?[Cc]opyright[\s\S]*?\*\//g, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

function escapeRegExp(value) {
    return value.replace(/[.*+?^$()|[\]{}\\]/g, '\\$&');
}

function lineNumberAt(content, index) {
    return content.slice(0, index).split(/\r?\n/).length;
}

function maskCommentsKeepLines(content) {
    return content
        .replace(/\/\*[\s\S]*?\*\//g, match => match.replace(/[^\n]/g, ' '))
        .replace(/\/\/[^\n]*/g, match => ' '.repeat(match.length));
}

function findClosing(content, openIndex, openChar, closeChar) {
    let depth = 1;
    let quote = '';
    let escaped = false;
    for (let index = openIndex + 1; index < content.length; index += 1) {
        const char = content[index];
        if (quote) {
            if (escaped) escaped = false;
            else if (char === '\\') escaped = true;
            else if (char === quote) quote = '';
            continue;
        }
        if (char === '"' || char === "'") {
            quote = char;
            continue;
        }
        if (char === openChar) depth += 1;
        else if (char === closeChar && --depth === 0) return index;
    }
    return -1;
}

function directApiCallExpression(apiName, flags = '') {
    const escapedName = escapeRegExp(apiName);
    return new RegExp(
        '(^|[^A-Za-z0-9_])(?:AscendC::)?' + escapedName +
        '(?:\\s*<[^;{}()]*>)?\\s*\\(',
        flags
    );
}

function findDirectApiCalls(content, apiName) {
    const code = maskCommentsKeepLines(content);
    const expression = directApiCallExpression(apiName, 'g');
    const calls = [];
    let match;
    while ((match = expression.exec(code)) !== null) {
        const prefix = match[1] || '';
        const index = match.index + prefix.length;
        calls.push({ index, line: lineNumberAt(content, index) });
    }
    return calls;
}

function hasDirectApiCall(content, apiName) {
    return findDirectApiCalls(content, apiName).length > 0;
}

function findFunctionOpeningBrace(content, startIndex) {
    let parenDepth = 0;
    let bracketDepth = 0;
    for (let index = startIndex; index < content.length; index += 1) {
        const char = content[index];
        if (char === '(') parenDepth += 1;
        else if (char === ')' && parenDepth > 0) parenDepth -= 1;
        else if (char === '[') bracketDepth += 1;
        else if (char === ']' && bracketDepth > 0) bracketDepth -= 1;
        else if (parenDepth === 0 && bracketDepth === 0 && char === '{') return index;
        else if (parenDepth === 0 && bracketDepth === 0 && (char === ';' || char === '}')) return -1;
    }
    return -1;
}

function findFunctionBlocks(content, functionName = null) {
    const masked = maskCommentsKeepLines(content);
    const expression = functionName
        ? new RegExp('\\b' + escapeRegExp(functionName) + '\\s*\\(', 'g')
        : /\b([A-Za-z_][A-Za-z0-9_]*)(?:\s*<[^;{}()]*>)?\s*\(/g;
    const blocks = [];
    let match;
    while ((match = expression.exec(masked)) !== null) {
        const name = functionName || match[1];
        if (NON_FUNCTION_NAMES.has(name)) continue;
        const openParen = masked.indexOf('(', match.index + name.length);
        const closeParen = findClosing(masked, openParen, '(', ')');
        if (closeParen < 0) continue;

        const openingBrace = findFunctionOpeningBrace(masked, closeParen + 1);
        if (openingBrace < 0) {
            expression.lastIndex = closeParen + 1;
            continue;
        }

        const closingBrace = findClosing(masked, openingBrace, '{', '}');
        if (closingBrace < 0) {
            expression.lastIndex = openingBrace + 1;
            continue;
        }

        blocks.push({
            name,
            line: lineNumberAt(content, match.index),
            endLine: lineNumberAt(content, closingBrace),
            bodyStart: openingBrace + 1,
            bodyEnd: closingBrace,
            body: content.slice(openingBrace + 1, closingBrace).trim()
        });
        expression.lastIndex = closingBrace + 1;
    }
    return blocks;
}

function findDirectCallFunctions(content, apiName) {
    const calls = findDirectApiCalls(content, apiName);
    if (calls.length === 0) return [];
    return findFunctionBlocks(content)
        .map(block => ({
            ...block,
            apiCallLines: calls
                .filter(call => call.index >= block.bodyStart && call.index < block.bodyEnd)
                .map(call => call.line)
        }))
        .filter(block => block.apiCallLines.length > 0);
}

function selectExampleFunction(content, apiName) {
    for (const functionName of ['Compute', 'Process']) {
        const matching = findFunctionBlocks(content, functionName)
            .find(block => hasDirectApiCall(block.body, apiName));
        if (matching) return matching;
    }
    return null;
}

function getSearchDirs(rootDir, apiType) {
    const subDirs = [];
    if (apiType && API_TYPE_TO_EXAMPLE_DIR[apiType]) {
        subDirs.push(API_TYPE_TO_EXAMPLE_DIR[apiType]);
    }
    // 新 API 家族可复用已有 examples 分支；类型目录未命中时回退整个 examples。
    subDirs.push('');
    return [...new Set(subDirs.map(subDir => path.join(rootDir, 'examples', subDir)))];
}

function collectAscFiles(rootDir) {
    const result = [];
    if (!fs.existsSync(rootDir)) return result;
    function walk(directory) {
        let entries;
        try {
            entries = fs.readdirSync(directory, { withFileTypes: true });
        } catch {
            return;
        }
        for (const entry of entries) {
            const filePath = path.join(directory, entry.name);
            if (entry.isDirectory()) walk(filePath);
            else if (entry.isFile() && entry.name.toLowerCase().endsWith('.asc')) result.push(filePath);
        }
    }
    walk(rootDir);
    return result;
}

function scoreExampleFile(filePath, content, rootDir, apiName) {
    const stem = path.basename(filePath, path.extname(filePath)).toLowerCase();
    const lowerName = apiName.toLowerCase();
    const relative = path.relative(rootDir, filePath).replace(/\\/g, '/').toLowerCase();
    let score = 0;
    if (hasDirectApiCall(content, apiName)) score += 1000;
    if (stem === lowerName) score += 200;
    else if (stem.includes(lowerName)) score += 80;
    if (relative.includes('/' + lowerName + '/')) score += 100;
    return score;
}

function collectHelperCandidates(ranked, apiName, rootDir) {
    const candidates = [];
    for (const item of ranked) {
        if (!hasDirectApiCall(item.content, apiName)) continue;
        for (const block of findDirectCallFunctions(item.content, apiName)) {
            if (block.name === 'Compute' || block.name === 'Process') continue;
            candidates.push({
                source: {
                    kind: 'repository-example-helper',
                    path: path.resolve(item.filePath),
                    relativePath: path.relative(rootDir, item.filePath).replace(/\\/g, '/')
                },
                functionName: block.name,
                functionLine: block.line,
                functionEndLine: block.endLine,
                apiCallLines: block.apiCallLines
            });
        }
    }
    return {
        candidates: candidates.slice(0, MAX_HELPER_CANDIDATES),
        truncated: candidates.length > MAX_HELPER_CANDIDATES
    };
}

function findExampleForMember(rootDir, apiName, apiType) {
    const files = [...new Set(getSearchDirs(rootDir, apiType).flatMap(collectAscFiles))];
    const ranked = files
        .map(filePath => {
            const content = fs.readFileSync(filePath, 'utf8');
            return { filePath, content, score: scoreExampleFile(filePath, content, rootDir, apiName) };
        })
        .sort((left, right) => right.score - left.score || left.filePath.localeCompare(right.filePath));

    let sawDirectCall = false;
    for (const item of ranked) {
        if (!hasDirectApiCall(item.content, apiName)) continue;
        sawDirectCall = true;
        const block = selectExampleFunction(item.content, apiName);
        if (!block) continue;
        return {
            status: 'available',
            member: apiName,
            source: {
                kind: 'repository-example',
                path: path.resolve(item.filePath),
                relativePath: path.relative(rootDir, item.filePath).replace(/\\/g, '/'),
                entryFunction: block.name,
                line: block.line
            },
            code: cleanExampleCode(block.body)
        };
    }

    const helperCandidates = sawDirectCall
        ? collectHelperCandidates(ranked, apiName, rootDir)
        : { candidates: [], truncated: false };
    const result = {
        status: 'unavailable',
        member: apiName,
        reason: sawDirectCall
            ? '找到当前 API 的直接调用，但没有包含该调用的 Compute() 或 Process()。'
            : 'examples/ 中未找到当前 API 的直接调用。'
    };
    if (helperCandidates.candidates.length > 0) {
        result.helperCandidates = helperCandidates.candidates;
        result.helperCandidatesTruncated = helperCandidates.truncated;
    }
    return result;

}

function buildCallExampleResult(rootDir, documentApiName, apiType, memberApis) {
    const members = memberApis.length > 0 ? [...new Set(memberApis)] : [documentApiName];
    const entries = members.map(member => findExampleForMember(rootDir, member, apiType));
    const availableCount = entries.filter(entry => entry.status === 'available').length;
    return {
        kind: 'ascendc_call_example',
        documentApiName,
        apiType: apiType || 'unknown',
        status: availableCount === entries.length
            ? 'available'
            : availableCount > 0 ? 'partial' : 'unavailable',
        members: entries
    };
}

function renderSingleExample(example) {
    if (example.status !== 'available') {
        const hasHelperCandidates = Array.isArray(example.helperCandidates) &&
            example.helperCandidates.length > 0;
        const lines = [
            hasHelperCandidates
                ? '- 固定规则未直接提取到 Compute() 或 Process()；可按受限调用链规则继续核验。'
                : '- <span style="color:#dc2626;">【待开发补齐】' + example.reason + '</span>'
        ];
        if (hasHelperCandidates) {
            lines.push('', '- 可供受限调用链核验的 Helper：');
            for (const candidate of example.helperCandidates) {
                lines.push(
                    '  - ' + candidate.source.relativePath + '：' +
                    candidate.functionName + '()（第 ' + candidate.functionLine +
                    '–' + candidate.functionEndLine + ' 行；目标调用第 ' +
                    candidate.apiCallLines.join('、') + ' 行）'
                );
            }
            if (example.helperCandidatesTruncated) {
                lines.push('  - 候选列表已截断；不能据此断言仓内不存在调用示例。');
            }
        }
        return lines.join('\n');
    }
    const fence = String.fromCharCode(96).repeat(3);
    return [
        fence + 'cpp',
        ensureTrailingNewline(example.code),
        fence
    ].join('\n');
}

function renderMarkdown(result) {
    if (result.members.length === 1) return renderSingleExample(result.members[0]);
    const sections = [];
    for (const example of result.members) {
        sections.push('### ' + example.member, '', renderSingleExample(example), '');
    }
    return sections.join('\n').trim();
}

function detectApiType(rootDir, apiName) {
    for (const apiType of Object.keys(API_TYPE_TO_EXAMPLE_DIR)) {
        const includeDir = path.join(rootDir, 'include', apiType);
        if (!fs.existsSync(includeDir)) continue;
        const stack = [includeDir];
        while (stack.length > 0) {
            const directory = stack.pop();
            let entries;
            try {
                entries = fs.readdirSync(directory, { withFileTypes: true });
            } catch {
                continue;
            }
            for (const entry of entries) {
                const filePath = path.join(directory, entry.name);
                if (entry.isDirectory()) stack.push(filePath);
                else if (/\.(h|hpp|hh)$/i.test(entry.name) &&
                    hasApiSignature(fs.readFileSync(filePath, 'utf8'), apiName)) {
                    return apiType;
                }
            }
        }
    }
    return null;
}

function parseArgs(argv) {
    if (argv.length < 2) {
        throw new Error('用法: node extract-call-example.js <代码仓根目录> <文档API名> [API类型] [--member <API名>]... [--json]');
    }
    const rootDir = argv[0];
    const documentApiName = argv[1];
    let apiType = null;
    const memberApis = [];
    let json = false;
    for (let index = 2; index < argv.length; index += 1) {
        const arg = argv[index];
        if (arg === '--member') {
            const member = argv[++index];
            if (!member) throw new Error('--member 缺少 API 名称');
            memberApis.push(member);
        } else if (arg === '--json') {
            json = true;
        } else if (!apiType) {
            apiType = arg;
        } else {
            throw new Error('未知参数: ' + arg);
        }
    }
    return { rootDir, documentApiName, apiType, memberApis, json };
}

function main() {
    const args = parseArgs(process.argv.slice(2));
    let apiType = args.apiType;
    if (!apiType) {
        apiType = detectApiType(args.rootDir, args.documentApiName);
        if (apiType) console.error('自动检测到 API 类型: ' + apiType);
    }
    const result = buildCallExampleResult(args.rootDir, args.documentApiName, apiType, args.memberApis);
    if (args.json) console.log(JSON.stringify(result, null, 2));
    else console.log(renderMarkdown(result));
}

if (require.main === module) main();

module.exports = {
    buildCallExampleResult,
    findDirectApiCalls,
    findDirectCallFunctions,
    findFunctionBlocks
};
