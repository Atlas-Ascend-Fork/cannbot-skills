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
    CALLING_CONVENTIONS,
    hasApiSignature,
    normalizeSpaces,
    parseAscendcSignaturesFromContent,
    splitTopLevel
} = require('./lib/ascendc-signatures');

function isApiHeaderFile(filePath) {
    return /\.(h|hpp|hh)$/i.test(filePath);
}

function memberRole(name) {
    return name.endsWith('_sync') ? 'sync' : 'regular';
}

function formatSignature(signature) {
    if (signature.suffixError) throw new Error(`${signature.funcName}: ${signature.suffixError}，不能生成精确原型`);
    const lines = [];
    if (signature.templateParams) lines.push(`template <${signature.templateParams}>`);
    const prefix = signature.qualifiers ? `${signature.qualifiers} ` : '';
    const params = !signature.rawParams || signature.rawParams === 'void'
        ? ''
        : splitTopLevel(signature.rawParams).map(normalizeSpaces).join(', ');
    const suffix = signature.suffix ? ` ${signature.suffix}` : '';
    lines.push(`${prefix}${signature.returnType} ${signature.funcName}(${params})${suffix};`);
    return lines.join('\n');
}

function collectHeaderFiles(dir, results) {
    if (!fs.existsSync(dir)) return;
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) collectHeaderFiles(fullPath, results);
        else if (isApiHeaderFile(fullPath)) results.push(fullPath);
    }
}

function findApiHeaderFiles(rootDir, memberNames, apiType, inputPath) {
    const candidates = [];
    if (inputPath) {
        const resolvedInput = path.resolve(inputPath);
        if (!fs.existsSync(resolvedInput)) throw new Error(`Input file or directory does not exist: ${resolvedInput}`);
        const stat = fs.statSync(resolvedInput);
        if (stat.isFile()) {
            if (isApiHeaderFile(resolvedInput)) candidates.push(resolvedInput);
        } else {
            collectHeaderFiles(resolvedInput, candidates);
        }
    } else {
        collectHeaderFiles(apiType ? path.join(rootDir, 'include', apiType) : path.join(rootDir, 'include'), candidates);
    }
    return [...new Set(candidates.filter(filePath => {
        const content = fs.readFileSync(filePath, 'utf8');
        return memberNames.some(name => hasApiSignature(content, name));
    }))];
}

function buildPrototypeSection(rootDir, documentApiName, apiType, inputPath, callingConvention, memberApis) {
    const members = memberApis.length > 0 ? memberApis : [{ name: documentApiName, role: memberRole(documentApiName) }];
    const memberNames = members.map(member => member.name);
    const memberSet = new Set(memberNames);
    const headerFiles = findApiHeaderFiles(rootDir, memberNames, apiType, inputPath);
    if (headerFiles.length === 0) return null;

    const allSignatures = [];
    const seen = new Set();
    for (const filePath of headerFiles) {
        const content = fs.readFileSync(filePath, 'utf8');
        const signatures = parseAscendcSignaturesFromContent(content).filter(signature =>
            memberSet.has(signature.funcName) && (!callingConvention || signature.callingConvention === callingConvention));
        for (const signature of signatures) {
            if (seen.has(signature.key)) continue;
            seen.add(signature.key);
            allSignatures.push({ ...signature, sourceFile: filePath });
        }
    }
    if (allSignatures.length === 0) return null;

    const conventions = [...new Set(allSignatures.map(signature => signature.callingConvention))];
    if (!callingConvention && conventions.length > 1) {
        throw new Error(`Document ${documentApiName} matched multiple calling conventions: ${conventions.join(', ')}`);
    }

    const sections = [];
    for (const member of members) {
        const signatures = allSignatures.filter(signature => signature.funcName === member.name);
        if (signatures.length === 0) throw new Error(`No signatures found for document member ${member.name}`);
        sections.push('## ' + member.name + (member.role === 'sync' ? '（同步接口）' : '（常规接口）'), '');
        sections.push('```cpp', signatures.map(formatSignature).join('\n'), '```', '');
    }
    return sections.join('\n').trim();
}

function parseArgs(argv) {
    const result = { rootDir: argv[0], documentApiName: argv[1], members: [] };
    for (let index = 2; index < argv.length; index++) {
        const arg = argv[index];
        if (arg === '--input') result.inputPath = argv[++index];
        else if (arg === '--calling-convention') result.callingConvention = argv[++index];
        else if (arg === '--member') {
            const name = argv[++index];
            result.members.push({ name, role: memberRole(name) });
        } else if (!result.apiType) result.apiType = arg;
        else throw new Error(`Unknown argument: ${arg}`);
    }
    if (!result.members.some(member => member.name === result.documentApiName)) {
        result.members.unshift({ name: result.documentApiName, role: memberRole(result.documentApiName) });
    }
    return result;
}

function usage() {
    console.error('Usage: node extract-prototype.js <projectRoot> <documentApiName> [apiType] [--input <fileOrDir>] [--calling-convention <value>] [--member <apiName>]...');
    console.error(`Calling conventions: ${CALLING_CONVENTIONS.join(', ')}`);
}

function main() {
    const argv = process.argv.slice(2);
    if (argv.length < 2) {
        usage();
        process.exit(1);
    }
    const options = parseArgs(argv);
    if (options.callingConvention && !CALLING_CONVENTIONS.includes(options.callingConvention)) {
        throw new Error(`Unsupported calling convention: ${options.callingConvention}`);
    }
    const result = buildPrototypeSection(options.rootDir, options.documentApiName, options.apiType || null,
        options.inputPath, options.callingConvention, options.members);
    if (!result) throw new Error('No signatures found for the requested document identity');
    console.log(result);
}

module.exports = { buildPrototypeSection, formatSignature, memberRole, parseArgs };

if (require.main === module) {
    try {
        main();
    } catch (error) {
        console.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
        process.exit(1);
    }
}
