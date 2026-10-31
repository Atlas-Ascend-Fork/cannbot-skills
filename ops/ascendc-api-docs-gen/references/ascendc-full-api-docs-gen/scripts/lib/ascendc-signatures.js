// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const CALLING_CONVENTIONS = [
    '__aicore__',
    '__simd_callee__',
    '__SIMT_DEVICE_FUNCTIONS_DECL__',
    'unqualified'
];

function removeComments(source) {
    return source
        .replace(/\/\/.*$/gm, '')
        .replace(/\/\*[\s\S]*?\*\//g, '');
}

function normalizeSpaces(value) {
    return value.replace(/\s+/g, ' ').trim();
}

// Split only at the outermost level so template arguments remain intact.
function splitTopLevel(value, delimiter = ',') {
    const parts = [];
    let start = 0;
    let quote = '';
    let escaped = false;
    const depth = { '(': 0, '[': 0, '{': 0, '<': 0 };
    const closing = { ')': '(', ']': '[', '}': '{', '>': '<' };
    for (let index = 0; index < value.length; index++) {
        const char = value[index];
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
        if (Object.prototype.hasOwnProperty.call(depth, char)) depth[char]++;
        else if (Object.prototype.hasOwnProperty.call(closing, char)) {
            const open = closing[char];
            if (depth[open] > 0) depth[open]--;
        } else if (char === delimiter && Object.values(depth).every(item => item === 0)) {
            parts.push(value.slice(start, index).trim());
            start = index + 1;
        }
    }
    const tail = value.slice(start).trim();
    if (tail || parts.length > 0) parts.push(tail);
    return parts;
}

function stripTopLevelDefaultValue(param) {
    let quote = '';
    let escaped = false;
    const depth = { '(': 0, '[': 0, '{': 0, '<': 0 };
    const closing = { ')': '(', ']': '[', '}': '{', '>': '<' };
    for (let index = 0; index < param.length; index++) {
        const char = param[index];
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
        if (Object.prototype.hasOwnProperty.call(depth, char)) {
            depth[char]++;
            continue;
        }
        if (Object.prototype.hasOwnProperty.call(closing, char)) {
            const open = closing[char];
            if (depth[open] > 0) depth[open]--;
            continue;
        }
        if (char === '=' && Object.values(depth).every(item => item === 0)) {
            return param.slice(0, index).trim();
        }
    }
    return param.trim();
}

function extractParameterName(param) {
    const declaration = stripTopLevelDefaultValue(param)
        .replace(/\[\s*[^\]]*\s*\]\s*$/g, '')
        .trim();
    if (!declaration || declaration === 'void') return '';

    const functionPointer = declaration.match(/\(\s*[*&]\s*([A-Za-z_]\w*)\s*\)/);
    if (functionPointer) return functionPointer[1];
    const match = declaration.match(/([A-Za-z_]\w*)\s*$/);
    return match ? match[1] : '';
}

function getParameterNames(rawParams) {
    if (!rawParams || rawParams === 'void') return [];
    return splitTopLevel(rawParams).map(extractParameterName);
}

function signatureShapeKey(signature) {
    return [
        normalizeSpaces(signature.templateParams || ''),
        normalizeSpaces(signature.qualifiers || ''),
        normalizeSpaces(signature.returnType || ''),
        normalizeSpaces(signature.rawParams || ''),
        normalizeSpaces(signature.suffix || '')
    ].join('|');
}

function findClosingParen(source, openParenIdx) {
    let depth = 1;
    let quote = '';
    let closeParenIdx = openParenIdx + 1;
    while (closeParenIdx < source.length && depth > 0) {
        const char = source[closeParenIdx];
        if (quote) {
            if (char === '\\') closeParenIdx++;
            else if (char === quote) quote = '';
        } else if (char === '"' || char === "'") quote = char;
        else if (char === '(') depth++;
        else if (char === ')') depth--;
        closeParenIdx++;
    }
    return depth === 0 ? closeParenIdx - 1 : -1;
}

function parseSignatureSuffix(source, start) {
    let cursor = start;
    let quote = '';
    const stack = [];
    const pairs = { ')': '(', ']': '[' };
    // Preserve the bounded declarator instead of maintaining a C++ suffix whitelist.
    while (cursor < source.length) {
        const char = source[cursor];
        if (quote) {
            if (char === '\\') cursor++;
            else if (char === quote) quote = '';
        } else if (char === '"' || char === "'") quote = char;
        else if (char === '(' || char === '[') stack.push(char);
        else if (Object.hasOwn(pairs, char)) {
            if (stack.pop() !== pairs[char]) return { suffix: '', suffixError: '声明后缀括号不匹配' };
        } else if ((char === ';' || char === '{') && stack.length === 0) break;
        cursor++;
    }
    const suffix = normalizeSpaces(source.slice(start, cursor));
    if (quote || stack.length) return { suffix, suffixError: '声明后缀未闭合，无法可靠定位' };
    if (source[cursor] === '{' && /\brequires\b/.test(suffix)) {
        return { suffix, suffixError: 'requires 表达式与函数体边界需进一步解析' };
    }
    return { suffix, isDefinition: source[cursor] === '{' };
}

function getCallingConvention(qualifiers) {
    for (const convention of CALLING_CONVENTIONS.slice(0, -1)) {
        if (new RegExp(`(?:^|\\s)${convention}(?:\\s|$)`).test(qualifiers)) {
            return convention;
        }
    }
    return 'unqualified';
}

function parseAscendcSignaturesFromContent(content) {
    const source = removeComments(content);
    const results = [];

    function addSignature({ templateParams = '', qualifiers = '', returnType, funcName, openParenIdx }) {
        const normalizedFuncName = normalizeSpaces(funcName).replace(/\s+/g, '');
        const closeParenIdx = findClosingParen(source, openParenIdx);
        if (closeParenIdx === -1) return;
        const rawParams = source.slice(openParenIdx + 1, closeParenIdx).trim();
        const suffixInfo = parseSignatureSuffix(source, closeParenIdx + 1);
        const normalizedQualifiers = normalizeSpaces(qualifiers);
        const key = [
            normalizeSpaces(templateParams),
            normalizedQualifiers,
            normalizeSpaces(returnType),
            normalizedFuncName,
            normalizeSpaces(rawParams),
            suffixInfo.suffix,
            suffixInfo.suffixError || ''
        ].join('|');

        if (!results.some(result => result.key === key)) {
            results.push({
                funcName: normalizedFuncName,
                sourceFuncName: funcName,
                returnType: normalizeSpaces(returnType),
                qualifiers: normalizedQualifiers,
                callingConvention: getCallingConvention(normalizedQualifiers),
                templateParams: normalizeSpaces(templateParams),
                rawParams: normalizeSpaces(rawParams),
                ...suffixInfo,
                key
            });
        }
    }

    const templateHeaders = [];
    const templateStart = /\btemplate\s*</g;
    let header;
    while ((header = templateStart.exec(source)) !== null) {
        const opening = templateStart.lastIndex - 1;
        let depth = 1;
        let cursor = opening + 1;
        let quote = '';
        for (; cursor < source.length && depth; cursor++) {
            const char = source[cursor];
            if (quote) {
                if (char === '\\') cursor++;
                else if (char === quote) quote = '';
            } else if (char === '"' || char === "'") quote = char;
            else if (char === '<') depth++;
            else if (char === '>') depth--;
        }
        if (depth) continue;
        templateHeaders.push({ start: header.index, end: cursor,
            params: source.slice(opening + 1, cursor - 1) });
        templateStart.lastIndex = cursor;
    }
    let declarations = source;
    for (const range of [...templateHeaders].reverse()) {
        declarations = declarations.slice(0, range.start) +
            declarations.slice(range.start, range.end).replace(/[^\n\r]/g, ' ') +
            declarations.slice(range.end);
    }

    const plainFuncRegex = /(?:^|[;{}]\s*|\n\s*)((?:(?:__aicore__|__simd_callee__|__SIMT_DEVICE_FUNCTIONS_DECL__|static|inline|__in_pipe__\([^)]*\)|__out_pipe__\([^)]*\)|__inout_pipe__\([^)]*\))\s+)*)\s*((?:decltype\s*\([^)]*\)|[A-Za-z_][A-Za-z0-9_:<>*&]*|void|bool|auto|int\d+_t|uint\d+_t|float|double))\s+(operator\s*(?:\[\]|\(\)|<<|>>|<=|>=|==|!=|&&|\|\||[!~+\-*\/%<>&|^=,])|\w+)\s*\(/gm;
    let plainMatch;
    while ((plainMatch = plainFuncRegex.exec(declarations)) !== null) {
        const declarationStart = plainMatch.index + plainMatch[0].search(/[^\s;{}]/);
        const template = templateHeaders.find(range => range.end <= declarationStart &&
            !source.slice(range.end, declarationStart).trim());
        addSignature({
            templateParams: template?.params || '',
            qualifiers: plainMatch[1],
            returnType: plainMatch[2],
            funcName: plainMatch[3],
            openParenIdx: source.indexOf('(', plainMatch.index + plainMatch[0].length - 1)
        });
    }

    return results;
}

function hasApiSignature(content, apiName, callingConvention) {
    return parseAscendcSignaturesFromContent(content).some(signature =>
        signature.funcName === apiName.replace(/\s+/g, '') &&
        (!callingConvention || signature.callingConvention === callingConvention)
    );
}

module.exports = {
    CALLING_CONVENTIONS,
    extractParameterName,
    getParameterNames,
    getCallingConvention,
    hasApiSignature,
    normalizeSpaces,
    parseAscendcSignaturesFromContent,
    parseSignatureSuffix,
    removeComments,
    signatureShapeKey,
    splitTopLevel,
    stripTopLevelDefaultValue
};
