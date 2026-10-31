// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const { validatePublicDocumentBoundary: validateBoundary } = require('./public-document-boundary.js');
const { validateProductDocument } = require('./product-document-policy.js');
const { codeBlocks: parseCodeBlocks, mapOutsideCodeBlocks } = require('./markdown-resource-links.js');
const { analyzeExample, hasMemberInvocation } = require('./cpp-example-calls.js');

const { parseAscendcSignaturesFromContent } = require('./ascendc-signatures');

function escapeRegex(value) {
    return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function extractSection(content, heading) {
    const match = content.match(new RegExp(`(?:^|\\n)## ${escapeRegex(heading)}\\s*\\n([\\s\\S]*?)(?=\\n## |$)`, 'u'));
    return match ? match[1].trim() : '';
}

function normalizeMemberNames(apiName, memberApis) {
    const names = (memberApis || []).map(member => typeof member === 'string' ? member : member.name).filter(Boolean);
    if (names.length === 0) names.push(apiName);
    return [...new Set(names.map(name => name.replace(/\s+/g, '')))];
}

function collectApiSignatures(content, apiNames) {
    const names = new Set(Array.isArray(apiNames) ? apiNames : [apiNames]);
    return parseAscendcSignaturesFromContent(signatureSource(content))
        .filter(signature => names.has(signature.funcName));
}

function signatureSource(content) {
    const prototypeSection = extractSection(content, '函数原型');
    const source = prototypeSection || content;
    const codeBlocks = [...source.matchAll(/\x60{3}(?:cpp|c\+\+|c)?\s*\n([\s\S]*?)\x60{3}/gi)]
        .map(match => match[1]);
    return codeBlocks.length > 0 ? codeBlocks.join('\n') : source;
}

function normalizePrototypeSemicolons({ apiName, memberApis, prototypeDoc, finalDoc, callingConvention }) {
    const memberNames = normalizeMemberNames(apiName, memberApis);
    const memberSet = new Set(memberNames);
    const expectedKeys = new Set(collectApiSignatures(prototypeDoc, memberNames)
        .filter(signature => !callingConvention || signature.callingConvention === callingConvention)
        .map(signature => signature.key));
    const sectionPattern = /((?:^|\n)## 函数原型\s*\n)([\s\S]*?)(?=\n## |$)/u;

    return finalDoc.replace(sectionPattern, (section, heading, body) => {
        const normalizedBody = body.replace(/(```(?:cpp|c\+\+|c)?\s*\n)([\s\S]*?)(```)/gi,
            (block, opening, code, closing) => {
                const lines = code.split(/\r?\n/);
                const normalizedCode = lines.map((line, index) => {
                    if (/;\s*$/.test(line) || !line.trim()) return line;
                    const candidateLine = line.replace(/\s*$/, ';');
                    const previousLine = index > 0 ? lines[index - 1] : '';
                    const candidate = /^\s*template\s*</.test(previousLine)
                        ? `${previousLine}\n${candidateLine}`
                        : candidateLine;
                    const signatures = parseAscendcSignaturesFromContent(candidate)
                        .filter(signature => memberSet.has(signature.funcName) &&
                            (!callingConvention || signature.callingConvention === callingConvention));
                    return signatures.length === 1 && !signatures[0].suffixError &&
                        expectedKeys.has(signatures[0].key) ? candidateLine : line;
                }).join('\n');
                return `${opening}${normalizedCode}${closing}`;
            });
        return `${heading}${normalizedBody}`;
    });
}

function validatePrototypeCompleteness({ apiName, memberApis, prototypeDoc, finalDoc, callingConvention }) {
    const prototypeSection = extractSection(finalDoc, '函数原型');
    if (!prototypeSection) throw new Error('最终文档缺少“## 函数原型”章节');
    if (/^#{3,6}\s+/m.test(prototypeSection)) {
        throw new Error('函数原型章节禁止包含子标题，请使用普通列表项组织原型');
    }
    if (/参数集合(?:为)?\s*(?:<code>|[A-Za-z_])/.test(prototypeSection)) {
        throw new Error('函数原型禁止使用参数集合进行机械分组，请使用有代码证据的功能模式名称或真实函数名');
    }

    const memberNames = normalizeMemberNames(apiName, memberApis);
    const memberSet = new Set(memberNames);
    const expected = collectApiSignatures(prototypeDoc, memberNames);
    const allActual = parseAscendcSignaturesFromContent(signatureSource(prototypeSection));
    const actual = allActual.filter(signature => memberSet.has(signature.funcName));
    const unexpectedMembers = allActual.filter(signature => !memberSet.has(signature.funcName));
    for (const signature of [...expected, ...allActual]) {
        if (signature.suffixError) throw new Error(`${signature.funcName}: ${signature.suffixError}，无法校验精确原型`);
    }
    if (expected.length === 0) throw new Error(`0_prototype.md中未解析到文档成员 ${memberNames.join(', ')} 的原型`);

    for (const memberName of memberNames) {
        if (!expected.some(signature => signature.funcName === memberName)) {
            throw new Error(`0_prototype.md缺少文档成员 ${memberName} 的原型`);
        }
    }

    const expectedKeys = new Set(expected.map(signature => signature.key));
    const actualKeys = new Set(actual.map(signature => signature.key));
    const missing = expected.filter(signature => !actualKeys.has(signature.key));
    const extra = actual.filter(signature => !expectedKeys.has(signature.key));
    if (actual.length !== expected.length || missing.length > 0 || extra.length > 0 || unexpectedMembers.length > 0) {
        const details = [`函数原型不完整：0_prototype.md为${expected.length}个，最终文档为${actual.length}个`];
        if (missing.length > 0) details.push(`缺少${missing.length}个原型`);
        if (extra.length > 0) details.push(`多出或被改写${extra.length}个原型`);
        if (unexpectedMembers.length > 0) {
            details.push(`混入非文档成员原型：${[...new Set(unexpectedMembers.map(item => item.funcName))].join(', ')}`);
        }
        throw new Error(details.join('；'));
    }

    const memberPattern = new RegExp(`\\b(?:${memberNames.map(escapeRegex).join('|')})\\s*\\(`);
    const declarationLines = signatureSource(prototypeSection).split(/\r?\n/).filter(line => memberPattern.test(line));
    if (declarationLines.length !== expected.length || declarationLines.some(line => !/;\s*$/.test(line))) {
        throw new Error('函数原型格式错误：每个函数声明必须独占一行并保留末尾分号');
    }

    if (callingConvention) {
        const mismatched = actual.filter(signature => signature.callingConvention !== callingConvention);
        if (mismatched.length > 0) {
            throw new Error(`最终文档混入${mismatched.length}个非${callingConvention}调用约定原型`);
        }
    }
    if (/以[^\n]{0,40}为例|其余类型同|同样包含/.test(prototypeSection)) {
        throw new Error('函数原型章节使用了代表性原型压缩表述，必须逐个展示0_prototype.md中的全部原型');
    }
}
function requiredExampleMemberNames(apiName, memberApis) {
    const requiredMembers = (memberApis || []).filter(member =>
        typeof member === 'string' || member.relation !== 'source-document-group');
    return normalizeMemberNames(apiName, requiredMembers);
}

function exampleCodeBlocks(section) {
    return parseCodeBlocks(section).filter(block =>
        ['', 'c', 'cpp', 'c++', 'cc', 'cxx', 'h', 'hpp'].includes(block.language));
}

function validateCallExample(finalDoc, apiName, memberApis, exampleRestrictions = []) {
    const section = extractSection(finalDoc, '调用示例');
    if (!section) throw new Error('最终文档缺少“## 调用示例”章节');

    const codeBlocks = exampleCodeBlocks(section);
    if (/【待开发补齐】缺少可验证调用示例/.test(section)) {
        throw new Error('调用示例不能使用缺失占位，必须根据代码生成模型示例');
    }
    if (codeBlocks.length === 0) {
        throw new Error('调用示例没有代码块，必须根据代码生成模型示例');
    }

    // Keep block boundaries: declarations or literals in one block must not
    // combine with another block to manufacture a call.
    const examples = codeBlocks.map(block => analyzeExample(block.content));
    const codeForChecks = examples.map(example => example.withoutComments).join('\n');
    if (/\.\.\.|=\s*\.\.\.|伪地址/.test(codeForChecks)) throw new Error('调用示例包含省略号或伪地址占位内容');
    if (/(?:__cbuf__|__gm__|__ubuf__|__ca__|__cb__|__cc__)\s+[^;\n=]+\b[A-Za-z_]\w*\s*;/.test(codeForChecks)) {
        throw new Error('调用示例包含未初始化的地址空间指针');
    }
    const memberNames = requiredExampleMemberNames(apiName, memberApis);
    const missingMembers = memberNames.filter(memberName =>
        !examples.some(example => example.hasInvocation(memberName)));
    if (missingMembers.length > 0) {
        throw new Error('调用示例未识别到文档成员的调用表达式（声明、定义、注释和字符串不计入）：' + missingMembers.join(', '));
    }
    if (!/结果示例如下：\s*\n\s*```text[\s\S]*?```/u.test(section)) {
        throw new Error('存在调用示例代码时，必须提供text代码块格式的结果示例');
    }
    validateExampleSafety(section, exampleRestrictions);
}


function validateExampleSafety(section, exampleRestrictions = []) {
    if (!Array.isArray(exampleRestrictions) || exampleRestrictions.length === 0) return;

    const resultBlocks = parseCodeBlocks(section).filter(block => block.language === 'text').map(block => block.content);
    // Restrictions may refer to literal values: hide comments, retain literals.
    const codePayload = exampleCodeBlocks(section).map(block => analyzeExample(block.content).withoutComments).join('\n');
    const examplePayload = [codePayload, ...resultBlocks].join('\n');

    for (const restriction of exampleRestrictions) {
        for (const expression of restriction.staticPatterns || []) {
            if (!expression) continue;
            if (new RegExp(escapeRegex(expression), 'iu').test(examplePayload)) {
                throw new Error(
                    `调用示例使用了受限制的写法“${expression}”（${restriction.member}：${restriction.subject}）`
                );
            }
        }
    }
}

function mapOutsideFencedCodeBlocks(content, transform) {
    return mapOutsideCodeBlocks(content, transform);
}

function validateTableSyntax(finalDoc) {
    const groupedProse = finalDoc.replace(/\x60\x60\x60[\s\S]*?\x60\x60\x60/g, '');
    if (/参数集合(?:为)?\s*(?:<code>|[A-Za-z_])/.test(groupedProse)) {
        throw new Error('最终文档禁止使用参数集合进行机械分组，请使用有代码证据的功能模式名称或真实函数名');
    }
    const parameterSection = extractSection(finalDoc, '参数说明');
    if (/\$[^\n$]+\$|\\le\b|\\ge\b|\\_[A-Za-z0-9]/.test(parameterSection)) {
        throw new Error('参数说明表格包含不允许的LaTeX表达');
    }

    const prose = finalDoc.replace(/```[\s\S]*?```/g, '');
    if (/`[^`\n]+`/.test(prose)) {
        throw new Error('最终正文包含Markdown行内代码，请改用<code>...</code>');
    }

    const htmlTables = [...finalDoc.matchAll(/<table\b[\s\S]*?<\/table>/gi)].map(match => match[0]);
    for (const table of htmlTables) {
        if (/\*\*[^*\n]+\*\*/.test(table) || /\[[^\]\n]+\]\([^\)\n]+\)/.test(table)) {
            throw new Error('HTML表格内包含Markdown强调或链接语法，请使用对应HTML标签');
        }
    }
}

function normalizeInlineCodeOutsideFences(content) {
    return mapOutsideFencedCodeBlocks(content, prose =>
        prose.replace(/`([^`\n]+)`/g, '<code>$1</code>'));
}

function normalizeInlineCodeInHtmlTables(content) {
    return content.replace(/<table\b[\s\S]*?<\/table>/gi, table =>
        table.replace(/`([^`\n]+)`/g, '<code>$1</code>'));
}
function validatePublicFactMappings(finalDoc, publicFactMappings = []) {
    if (!Array.isArray(publicFactMappings)) {
        throw new Error('对外事实映射必须是数组');
    }
    for (const mapping of publicFactMappings) {
        const factId = mapping && mapping.factId;
        if (!factId) throw new Error('对外事实映射缺少事实 ID');
        if (mapping.disposition === '用户明确暂缓') {
            if (!finalDoc.includes('【待开发补齐】')) {
                throw new Error('已暂缓的对外事实必须在最终文档中保留【待开发补齐】标记：' + factId);
            }
            continue;
        }
        if (mapping.disposition !== '写入正文') {
            throw new Error('对外事实映射处理方式无效：' + factId);
        }
        const sections = Array.isArray(mapping.sections) ? mapping.sections : [];
        const identifiers = Array.isArray(mapping.requiredIdentifiers) ? mapping.requiredIdentifiers : [];
        if (sections.length === 0 || identifiers.length === 0) {
            throw new Error('写入正文的对外事实映射必须记录正文位置和必含标识符：' + factId);
        }
        const sectionText = sections.map(section => extractSection(finalDoc, section)).join('\n');
        if (!sectionText) throw new Error('最终文档缺少对外事实映射指定的章节：' + factId);
        for (const identifier of identifiers) {
            if (!sectionText.includes(identifier)) {
                throw new Error('最终文档遗漏对外事实的必含标识符“' + identifier + '”：' + factId);
            }
        }
    }
}

function validatePublicDocumentBoundary(finalDoc, options) {
    const warnings = validateBoundary(finalDoc, options);
    const prose = finalDoc.replace(/[\x60]{3}[\s\S]*?[\x60]{3}/g, '');
    const forbidden = [
        /\b(?:op_host|op_kernel|op_tiling)\//u,
        /工程运行结果未由本接口代码证实/u,
        /未由(?:本接口)?代码证实/u,
        /(?:测试覆盖|覆盖状态|构建状态|编译状态|运行状态|验证状态)\s*[：:][^\n]*/u
    ];
    if (forbidden.some(pattern => pattern.test(prose))) {
        throw new Error('最终 API 文档不得包含测试或工程验证状态，请保留在证据或测试产物中');
    }
    return warnings;
}

function validateFinalDocument(options) {
    validateDocumentStructure(options.finalDoc, options.apiName);
    validateProductDocument(options.finalDoc, 'ascendc');
    validatePrototypeCompleteness(options);
    validateCallExample(options.finalDoc, options.apiName, options.memberApis, options.exampleRestrictions);
    validateTableSyntax(options.finalDoc);
    validatePublicFactMappings(options.finalDoc, options.publicFactMappings || []);
    return validatePublicDocumentBoundary(options.finalDoc, options);
}

function validateDocumentStructure(finalDoc, apiName) {
    const prose = finalDoc.replace(/^[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$/gm,
        match => match.replace(/[^\r\n]/g, ' '));
    const titles = [...prose.matchAll(/^# (.+?)[ \t]*\r?$/gm)];
    if (titles.length !== 1 || titles[0][1].trim() !== apiName) throw new Error('最终文档必须包含唯一且匹配目标 API 的一级标题');
    const expected = ['产品支持情况', '功能说明', '函数原型', '参数说明', '返回值说明', '约束说明', '调用示例'];
    const matches = [...prose.matchAll(/^## (.+?)[ \t]*\r?$/gm)];
    const headings = matches.map(match => match[1].trim());
    if (JSON.stringify(headings) !== JSON.stringify(expected)) {
        throw new Error('最终文档的必需章节缺失、重复、顺序错误或包含未定义章节；应依次为：' + expected.join('、'));
    }
    if (titles[0].index > matches[0].index) throw new Error('最终文档一级标题必须位于必需章节之前');
    for (const [index, match] of matches.entries()) {
        const body = finalDoc.slice(match.index + match[0].length, matches[index + 1]?.index ?? finalDoc.length)
            .replace(/<!--[\s\S]*?-->/g, '').trim();
        if (!body) throw new Error('最终文档章节不能为空：' + headings[index]);
    }
}

module.exports = {
    collectApiSignatures,
    extractSection,
    normalizeInlineCodeInHtmlTables,
    normalizeInlineCodeOutsideFences,
    normalizeMemberNames,
    normalizePrototypeSemicolons,
    hasMemberInvocation,
    validateCallExample,
    validateExampleSafety,
    validateFinalDocument,
    validateDocumentStructure,
    validatePublicDocumentBoundary,
    validatePublicFactMappings,
    validatePrototypeCompleteness,
    validateTableSyntax,
};
