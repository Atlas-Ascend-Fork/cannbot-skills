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
    normalizeSpaces,
    parseAscendcSignaturesFromContent,
    parseSignatureSuffix,
} = require('./lib/ascendc-signatures');

function escapeRegex(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function flexibleText(value) {
    return escapeRegex(value).replace(/\s+/g, '\\s+');
}

function maskCommentsKeepLines(source) {
    return source
        .replace(/\/\*[\s\S]*?\*\//g, match => match.replace(/[^\n]/g, ' '))
        .replace(/\/\/[^\n]*/g, match => ' '.repeat(match.length));
}

function findClosingParen(source, openIndex) {
    let depth = 1;
    let quote = '';
    let escaped = false;
    for (let index = openIndex + 1; index < source.length; index++) {
        const char = source[index];
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
        if (char === '(') depth++;
        else if (char === ')' && --depth === 0) return index;
    }
    return -1;
}

function lineNumberAt(source, index) {
    return source.slice(0, index).split(/\r?\n/).length;
}

function relativePath(repoPath, filePath) {
    const relative = path.relative(repoPath, filePath).replace(/\\/g, '/');
    return relative && !relative.startsWith('../') && relative !== '..' && !path.isAbsolute(relative)
        ? relative
        : path.resolve(filePath).replace(/\\/g, '/');
}

function isPublicDeclarationFile(relativeFile) {
    return relativeFile.startsWith('include/');
}

function isDeclarationPrefix(line) {
    const text = line.trim();
    return text.startsWith('template') || text.startsWith('__') || text.startsWith('[[') ||
        /^(?:static|inline|constexpr)\b/.test(text);
}

function isCommentLine(line) {
    const text = line.trim();
    return text.startsWith('//') || text.startsWith('/*') || text.startsWith('*') || text.endsWith('*/');
}

function extractLeadingComment(source, declarationLine) {
    const lines = source.split(/\r?\n/);
    let cursor = declarationLine - 2;
    while (cursor >= 0 && isDeclarationPrefix(lines[cursor])) cursor--;
    if (cursor < 0 || !isCommentLine(lines[cursor])) return null;

    const endLine = cursor + 1;
    while (cursor >= 0 && isCommentLine(lines[cursor])) cursor--;
    const startLine = cursor + 2;
    const raw = lines.slice(startLine - 1, endLine).join('\n').trim();
    if (!raw) return null;
    const text = raw
        .replace(/\/\*+!?/g, '')
        .replace(/\*\//g, '')
        .replace(/^\s*\/\/\/?\s?/gm, '')
        .replace(/^\s*\*\s?/gm, '')
        .trim();
    if (!text) return null;
    return {
        startLine,
        endLine,
        kind: /\/\*!?|\/\/\//.test(raw) ? 'doxygen' : 'comment',
        text,
        tags: [...new Set([...text.matchAll(/@(brief|param|note|warning|attention|return)\b/gi)]
            .map(match => '@' + match[1].toLowerCase()))],
    };
}

function conditionalContext(source, lineNumber) {
    const lines = source.split(/\r?\n/);
    const stack = [];
    for (let index = 0; index < lineNumber - 1; index++) {
        const text = lines[index].trim();
        if (/^#\s*(?:if|ifdef|ifndef)\b/.test(text)) {
            stack.push(text);
        } else if (/^#\s*(?:elif|else)\b/.test(text) && stack.length > 0) {
            stack[stack.length - 1] = text;
        } else if (/^#\s*endif\b/.test(text)) {
            stack.pop();
        }
    }
    return stack;
}

function findSignatureLocations(maskedSource, signature) {
    const pattern = new RegExp(
        `${flexibleText(signature.returnType)}\\s+${escapeRegex(signature.funcName)}\\s*\\(`,
        'g'
    );
    const results = [];
    let match;
    while ((match = pattern.exec(maskedSource)) !== null) {
        const openParen = maskedSource.indexOf('(', match.index + match[0].length - 1);
        const closeParen = findClosingParen(maskedSource, openParen);
        if (closeParen === -1) continue;
        const suffixInfo = parseSignatureSuffix(maskedSource, closeParen + 1);
        if (suffixInfo.suffixError || suffixInfo.suffix !== signature.suffix) continue;
        const rawParams = normalizeSpaces(maskedSource.slice(openParen + 1, closeParen));
        if (rawParams !== signature.rawParams) continue;
        results.push({
            declarationLine: lineNumberAt(maskedSource, match.index),
        });
        pattern.lastIndex = closeParen + 1;
    }
    return results;
}

function displaySignature(signature) {
    const template = signature.templateParams ? `template <${signature.templateParams}>` : '';
    const parts = [template, signature.qualifiers, signature.returnType]
        .filter(Boolean)
        .join(' ');
    return `${parts} ${signature.funcName}(${signature.rawParams})${signature.suffix ? ' ' + signature.suffix : ''}`.trim();
}

function collectPublicContracts({ repoPath, sourceFiles, members, callingConvention }) {
    const memberNames = new Set((members || []).map(member =>
        typeof member === 'string' ? member : member?.name).filter(Boolean));
    const files = [...new Set((sourceFiles || []).map(filePath => path.resolve(filePath)))].sort();
    const contracts = [];

    for (const filePath of files) {
        const relativeFile = relativePath(repoPath, filePath);
        if (!isPublicDeclarationFile(relativeFile)) continue;
        const source = fs.readFileSync(filePath, 'utf8');
        const maskedSource = maskCommentsKeepLines(source);
        const signatures = parseAscendcSignaturesFromContent(source)
            .filter(signature => memberNames.has(signature.funcName) &&
                (!callingConvention || signature.callingConvention === callingConvention));
        for (const signature of signatures) {
            for (const location of findSignatureLocations(maskedSource, signature)) {
                const comment = extractLeadingComment(source, location.declarationLine);
                if (!comment) continue;
                contracts.push({
                    member: signature.funcName,
                    callingConvention: signature.callingConvention,
                    declaration: displaySignature(signature),
                    source: {
                        file: relativeFile,
                        declarationLine: location.declarationLine,
                        commentStartLine: comment.startLine,
                        commentEndLine: comment.endLine,
                    },
                    conditionalContext: conditionalContext(source, location.declarationLine),
                    comment,
                });
            }
        }
    }

    contracts.sort((left, right) =>
        left.source.file.localeCompare(right.source.file) ||
        left.source.declarationLine - right.source.declarationLine ||
        left.member.localeCompare(right.member));
    return {
        schemaVersion: 1,
        kind: 'ascendc_public_contract_candidates',
        sourceFiles: files.map(filePath => relativePath(repoPath, filePath)),
        contracts: contracts.map((contract, index) => ({
            id: `PC-${String(index + 1).padStart(3, '0')}`,
            ...contract,
        })),
    };
}

function parseArgs(argv) {
    const result = { sourceFiles: [], members: [] };
    for (let index = 0; index < argv.length; index++) {
        const arg = argv[index];
        if (arg === '--repo') result.repoPath = argv[++index];
        else if (arg === '--file') result.sourceFiles.push(argv[++index]);
        else if (arg === '--member') result.members.push(argv[++index]);
        else if (arg === '--calling-convention') result.callingConvention = argv[++index];
        else if (arg === '--help' || arg === '-h') {
            console.log('用法: node scripts/extract-public-contracts.js --repo <代码仓> --file <声明文件> --member <API成员> [--file ...] [--member ...] [--calling-convention <调用约定>]');
            process.exit(0);
        } else {
            throw new Error('未知参数: ' + arg);
        }
    }
    if (!result.repoPath || result.sourceFiles.length === 0 || result.members.length === 0) {
        throw new Error('缺少 --repo、--file 或 --member');
    }
    return result;
}

function main() {
    const options = parseArgs(process.argv.slice(2));
    const repoPath = path.resolve(options.repoPath);
    const sourceFiles = options.sourceFiles.map(filePath =>
        path.isAbsolute(filePath) ? path.resolve(filePath) : path.resolve(repoPath, filePath));
    for (const filePath of sourceFiles) {
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            throw new Error('声明文件不可读取: ' + filePath);
        }
    }
    process.stdout.write(JSON.stringify(collectPublicContracts({
        repoPath,
        sourceFiles,
        members: options.members,
        callingConvention: options.callingConvention,
    }), null, 2) + '\n');
}

if (require.main === module) {
    try {
        main();
    } catch (error) {
        console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
        process.exit(1);
    }
}

module.exports = {
    collectPublicContracts,
    extractLeadingComment,
    findSignatureLocations,
    maskCommentsKeepLines,
};
