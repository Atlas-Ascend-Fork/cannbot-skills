// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

// Inspect expression syntax only. Name lookup, overload resolution, macro expansion
// and execution still require source review or a compiler with the target SDK.
function scanExample(source) {
    const tokens = [];
    const comments = [];
    const lexeme = /(?:[A-Za-z_]\w*|(?:\d|\.\d)(?:[eEpP][+-]|[\w.'])*|::|->\*|->|\.\*|\.\.\.|<<=?|>>=?|[+\-*/%&|^!=<>]=|&&|\|\||\+\+|--|[^\s])/y;
    let cursor = 0;
    while (cursor < source.length) {
        if (/\s/.test(source[cursor])) { cursor++; continue; }
        const start = cursor;
        if (source.startsWith('//', cursor)) {
            do {
                const newline = source.indexOf('\n', cursor);
                cursor = newline < 0 ? source.length : newline + 1;
            } while (cursor < source.length && /\\\r?\n$/.test(source.slice(start, cursor)));
            comments.push([start, cursor]);
            continue;
        }
        if (source.startsWith('/*', cursor)) {
            const end = source.indexOf('*/', cursor + 2);
            if (end < 0) throw new Error('调用示例的块注释未闭合');
            cursor = end + 2;
            comments.push([start, cursor]);
            continue;
        }
        const raw = /^(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(/.exec(source.slice(cursor));
        const quoted = /^(?:u8|u|U|L)?(["'])/.exec(source.slice(cursor));
        if (raw) {
            const closing = ')' + raw[1] + '"';
            const end = source.indexOf(closing, cursor + raw[0].length);
            if (end < 0) throw new Error('调用示例的原始字符串未闭合');
            cursor = end + closing.length;
        } else if (quoted) {
            cursor += quoted[0].length;
            while (cursor < source.length && source[cursor] !== quoted[1]) {
                if (source[cursor] === '\\') cursor++;
                cursor++;
            }
            if (cursor >= source.length) throw new Error('调用示例的字符串或字符字面量未闭合');
            cursor++;
        }
        if (raw || quoted) {
            tokens.push({ value: source.slice(start, cursor), kind: 'literal' });
            continue;
        }
        // A macro definition is not a demonstrated invocation. Do not expand it.
        if (source[cursor] === '#' && !source.slice(source.lastIndexOf('\n', cursor - 1) + 1, cursor).trim()) {
            do {
                const newline = source.indexOf('\n', cursor);
                cursor = newline < 0 ? source.length : newline + 1;
            } while (cursor < source.length && /\\\r?\n$/.test(source.slice(start, cursor)));
            continue;
        }
        lexeme.lastIndex = cursor;
        const value = lexeme.exec(source)[0];
        tokens.push({ value, kind: /^[A-Za-z_]/.test(value) ? 'identifier' : /^\d|^\.\d/.test(value) ? 'literal' : 'punctuation' });
        cursor += value.length;
    }
    let withoutComments = source;
    for (const [start, end] of comments.reverse()) {
        withoutComments = withoutComments.slice(0, start) +
            withoutComments.slice(start, end).replace(/[^\r\n]/g, ' ') + withoutComments.slice(end);
    }
    return { tokens, withoutComments };
}

function analyzeExample(source) {
    const { tokens, withoutComments } = scanExample(source);
    const pairs = new Map();
    const stack = [];
    const openings = { ')': '(', ']': '[', '}': '{' };
    const unevaluated = new Set(['decltype', 'sizeof', 'alignof', 'noexcept', 'requires']);
    const excluded = new Set();
    for (let index = 0; index < tokens.length; index++) {
        const token = tokens[index];
        if (token.kind === 'literal') continue;
        if (['(', '[', '{'].includes(token.value)) stack.push(index);
        else if (openings[token.value]) {
            const opening = stack.pop();
            if (tokens[opening]?.value !== openings[token.value]) throw new Error('调用示例的代码括号不匹配');
            pairs.set(opening, index);
            pairs.set(index, opening);
            if (unevaluated.has(tokens[opening - 1]?.value)) {
                for (let item = opening; item <= index; item++) excluded.add(item);
            }
        }
    }
    if (stack.length) throw new Error('调用示例的代码括号未闭合');

    function templateEnd(opening) {
        let depth = 0;
        for (let index = opening; index < tokens.length; index++) {
            const value = tokens[index].value;
            if (value === '<') depth++;
            else if (value === '>' || value === '>>') {
                depth -= value.length;
                if (depth === 0) return index + 1;
                if (depth < 0) return -1;
            } else if (value === '(' || value === '[') index = pairs.get(index);
            else if (value === ';' || value === '{' || value === '}') return -1;
        }
        return -1;
    }

    function templateBeginning(ending) {
        let depth = 0;
        for (let cursor = ending; cursor >= 0; cursor--) {
            const value = tokens[cursor].value;
            if (value === '>' || value === '>>') depth += value.length;
            else if (value === '<' && --depth === 0) return cursor;
            else if (value === ')' || value === ']') cursor = pairs.get(cursor);
            else if (value === ';' || value === '{' || value === '}') break;
        }
        return -1;
    }

    function expressionStart(index) {
        let start = index;
        if (tokens[start - 1]?.value === 'template') start--;
        while (['::', '.', '->'].includes(tokens[start - 1]?.value)) {
            start--;
            if (['>', '>>'].includes(tokens[start - 1]?.value)) {
                const opening = templateBeginning(start - 1);
                if (opening >= 0) start = opening;
            }
            if (tokens[start - 1]?.kind === 'identifier') start--;
            else if (tokens[start - 1]?.value === ')' || tokens[start - 1]?.value === ']') {
                start = pairs.get(start - 1);
                if (tokens[start - 1]?.kind === 'identifier') start--;
            } else break;
        }
        return start;
    }

    function expressionContext(index) {
        const start = expressionStart(index);
        const previous = tokens[start - 1];
        if (!previous) return true;
        if (['return', 'co_return', 'co_await', 'throw', 'else'].includes(previous.value)) return true;
        if (previous.kind !== 'punctuation') return false;
        if (previous.value === ',') {
            // A comma continues the enclosing expression or declaration list.
            let beginning = start - 2;
            for (; beginning >= 0; beginning--) {
                const value = tokens[beginning].value;
                if ([')', ']', '}'].includes(value)) beginning = pairs.get(beginning);
                else if ([';', '{', '('].includes(value)) break;
            }
            for (let cursor = beginning + 1; cursor < start - 1; cursor++) {
                const token = tokens[cursor];
                if (token.value === '=' || token.value === ',') break;
                if (['(', '[', '{'].includes(token.value)) cursor = pairs.get(cursor);
                else if (token.value === '<' && templateEnd(cursor) > cursor) cursor = templateEnd(cursor) - 1;
                else if (token.kind === 'identifier' && !expressionContext(cursor)) return false;
            }
            return true;
        }
        if (['*', '&', '&&', '>', '>>'].includes(previous.value)) {
            // A return type/pointer before a declarator is not an expression.
            // In an initializer or return expression the same tokens are operators.
            for (let cursor = start - 2; cursor >= 0; cursor--) {
                const value = tokens[cursor].value;
                if (value === ')' || value === ']') cursor = pairs.get(cursor);
                else if (['>', '>>'].includes(value) && templateBeginning(cursor) >= 0) cursor = templateBeginning(cursor);
                else if (['=', 'return', 'co_return', '(', '?', ':'].includes(value)) return true;
                else if ([';', '{', '}'].includes(value)) break;
            }
            return start === 1;
        }
        return !['::', '.', '->'].includes(previous.value);
    }

    function isCall(start, end) {
        if (excluded.has(start) || unevaluated.has(tokens[start]?.value) ||
            ['if', 'for', 'while', 'switch', 'catch'].includes(tokens[start]?.value)) return false;
        let opening = end;
        if (tokens[opening]?.value === '<') opening = templateEnd(opening);
        let beginning = expressionStart(start);
        while (tokens[opening]?.value === ')' && pairs.get(opening) === beginning - 1) {
            beginning--;
            opening++;
        }
        return expressionContext(beginning) && opening >= 0 && tokens[opening]?.value === '(' && pairs.has(opening);
    }

    function hasInvocation(memberName) {
        const name = memberName.replace(/\s+/g, '');
        const nameTokens = scanExample(name).tokens.map(token => token.value);
        for (let index = 0; index < tokens.length; index++) {
            if (nameTokens.every((value, offset) => tokens[index + offset]?.kind !== 'literal' && tokens[index + offset]?.value === value) &&
                isCall(index, index + nameTokens.length)) return true;
        }
        if (!name.startsWith('operator')) return false;
        const symbol = name.slice('operator'.length);
        // Expression syntax covers operators; binding to the target type is reviewed
        // against evidence. String contents and declarators cannot supply operands.
        for (let index = 1; index < tokens.length; index++) {
            if (excluded.has(index) || tokens[index - 1].value === 'operator') continue;
            const left = tokens[index - 1];
            if (symbol === '()' && left.kind === 'identifier' && isCall(index - 1, index)) return true;
            if (symbol === '[]' && tokens[index].value === '[' && left.kind === 'identifier' && expressionContext(index - 1)) return true;
            if (tokens[index].value !== symbol || tokens[index].kind !== 'punctuation') continue;
            const right = tokens[index + 1];
            if (!right) continue;
            const leftOperand = left.kind === 'identifier' || left.kind === 'literal' || [')', ']'].includes(left.value);
            const rightOperand = right.kind === 'identifier' || right.kind === 'literal' || right.value === '(';
            const start = [')', ']'].includes(left.value) ? pairs.get(index - 1) : index - 1;
            if (leftOperand && rightOperand && expressionContext(start)) return true;
        }
        return false;
    }
    return { hasInvocation, withoutComments };
}

function hasMemberInvocation(code, memberName) {
    return analyzeExample(code).hasInvocation(memberName);
}

module.exports = { analyzeExample, hasMemberInvocation };
