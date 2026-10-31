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
const { trackAscendCReferences } = require('./ascendc-cross-dir-tracker');
const { buildStructuredClosure, generateReport } = require('../extract-cross-dir');
const {
    getParameterNames,
    parseAscendcSignaturesFromContent,
    splitTopLevel
} = require('./ascendc-signatures');
const {
    buildSemanticLinks,
    collectCodeFiles,
    extractDefinitions
} = require('./ascendc-semantic-tracker');

function normPath(filePath) {
    return path.resolve(filePath).replace(/\\/g, '/');
}

function unique(values) {
    return [...new Set(values.filter(Boolean))];
}

function apiFamilyFromPath(filePath) {
    const match = normPath(filePath).match(/\/(?:include|impl)\/([^/]+)(?:\/|$)/);
    return match ? match[1] : 'unknown';
}

function commonInputPath(files) {
    if (files.length === 1) return files[0];
    let candidate = path.dirname(files[0]);
    while (!files.every(filePath => isInside(candidate, filePath))) {
        const parent = path.dirname(candidate);
        if (parent === candidate) break;
        candidate = parent;
    }
    return candidate;
}

function isInside(parent, child) {
    const relative = path.relative(parent, child);
    return relative === '' || (
        !relative.startsWith('..' + path.sep) &&
        relative !== '..' &&
        !path.isAbsolute(relative)
    );
}

function readText(filePath) {
    try {
        return fs.readFileSync(filePath, 'utf8');
    } catch {
        return '';
    }
}

function lineNumberAt(source, offset) {
    return source.slice(0, offset).split(/\r?\n/).length;
}

function architectureHintsFromExpressions(expressions) {
    const hints = new Set();
    const patterns = [
        /\b(?:ArchVersion::V|NPU_ARCH_|npu_arch_|ARCH_)(\d{3,5})\b/gi,
        /\b(?:CURRENT_ARCH_VERSION|__NPU_ARCH__|NPU_ARCH|ARCH_VERSION)\b[^0-9]{0,48}(\d{3,5})\b/gi
    ];
    for (const expression of expressions) {
        for (const pattern of patterns) {
            let match;
            while ((match = pattern.exec(expression)) !== null) hints.add(match[1]);
        }
    }
    return [...hints].sort();
}

function preprocessorConditionsAt(source, offset) {
    const stack = [];
    const lines = source.slice(0, offset).split(/\r?\n/);
    for (const line of lines) {
        const directive = /^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$/u.exec(line);
        if (!directive) continue;
        const kind = directive[1];
        const value = directive[2].trim();
        if (kind === "if" || kind === "ifdef" || kind === "ifndef") {
            stack.push(kind === "ifndef" ? "!defined(" + value + ")" : value || "#" + kind);
        } else if (kind === "elif" && stack.length > 0) {
            stack[stack.length - 1] = value || "#elif";
        } else if (kind === "else" && stack.length > 0) {
            stack[stack.length - 1] = "#else";
        } else if (kind === "endif") {
            stack.pop();
        }
    }
    const expressions = unique(stack);
    return {
        expressions,
        architectures: architectureHintsFromExpressions(expressions)
    };
}

function escapeRegExp(value) {
    return value.replace(/[.*+?^$()|[\]{}\\]/g, '\\$&');
}

function findClosing(source, openIndex, openChar, closeChar) {
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
        if (char === openChar) depth++;
        else if (char === closeChar && --depth === 0) return index;
    }
    return -1;
}

function sourceFilesForTarget(repoPath, targetApi) {
    const files = Array.isArray(targetApi.files) ? targetApi.files : [];
    const absolute = files.map(file => path.resolve(repoPath, file));
    if (absolute.length === 0) throw new Error('targetApi.files 为空');
    for (const filePath of absolute) {
        if (!isInside(repoPath, filePath)) throw new Error('目标 API 源文件越出代码仓: ' + filePath);
        if (!fs.existsSync(filePath)) throw new Error('目标 API 源文件不存在: ' + filePath);
    }
    return absolute;
}

function collectTargetSignatures(sourceFiles, targetApi) {
    const members = new Set((targetApi.members || [])
        .map(member => typeof member === 'string' ? member : member && member.name)
        .filter(Boolean));
    const signatures = [];
    for (const filePath of sourceFiles) {
        const content = readText(filePath);
        for (const signature of parseAscendcSignaturesFromContent(content)) {
            if (!members.has(signature.funcName)) continue;
            if (targetApi.callingConvention && signature.callingConvention !== targetApi.callingConvention) continue;
            signatures.push({ ...signature, sourceFile: filePath });
        }
    }
    if (signatures.length === 0) {
        throw new Error('未在目标 API 源文件中解析到当前成员的公开签名');
    }
    return signatures;
}

function publicDescriptors(repoPath, names) {
    const wanted = new Set(names.filter(Boolean));
    const result = new Map();
    if (wanted.size === 0) return result;
    const includeRoot = path.join(repoPath, 'include');
    for (const filePath of collectCodeFiles(includeRoot)) {
        const content = readText(filePath);
        for (const signature of parseAscendcSignaturesFromContent(content)) {
            if (!wanted.has(signature.funcName)) continue;
            const descriptor = {
                name: signature.funcName,
                apiFamily: apiFamilyFromPath(filePath),
                callingConvention: signature.callingConvention,
                file: normPath(filePath),
                parameterCount: getParameterNames(signature.rawParams).length
            };
            if (!result.has(descriptor.name)) result.set(descriptor.name, []);
            const key = [
                descriptor.apiFamily,
                descriptor.callingConvention,
                descriptor.file,
                descriptor.parameterCount
            ].join('|');
            if (!result.get(descriptor.name).some(item =>
                [item.apiFamily, item.callingConvention, item.file, item.parameterCount].join('|') === key)) {
                result.get(descriptor.name).push(descriptor);
            }
        }
    }
    return result;
}

function selectDescriptor(descriptorMap, name, candidate) {
    const descriptors = descriptorMap.get(name) || [];
    let compatible = descriptors.filter(descriptor =>
        !candidate ||
        candidate.callingConvention === 'unqualified' ||
        descriptor.callingConvention === 'unqualified' ||
        descriptor.callingConvention === candidate.callingConvention
    );

    const parameterCounts = [
        ...new Set(
            candidate && Array.isArray(candidate.parameterCounts)
                ? candidate.parameterCounts
                : candidate && Number.isInteger(candidate.parameterCount)
                    ? [candidate.parameterCount]
                    : []
        )
    ];
    if (parameterCounts.length > 0) {
        compatible = compatible.filter(descriptor => parameterCounts.includes(descriptor.parameterCount));
    }

    // A known calling convention or parameter count is evidence, not a
    // preference. Do not fall back to a same-name declaration that violates it.
    const selected = candidate ? compatible : descriptors;
    if (selected.length === 0) {
        return {
            status: descriptors.length === 0 ? 'declaration-not-found' : 'no-compatible-declaration',
            descriptor: null
        };
    }
    const families = unique(selected.map(descriptor => descriptor.apiFamily + '|' + descriptor.callingConvention));
    const files = unique(selected.map(descriptor => descriptor.file));
    if (families.length !== 1 || files.length !== 1) {
        return { status: 'ambiguous-declaration', descriptor: null, descriptors: [] };
    }
    return { status: 'resolved', descriptor: selected[0], descriptors: selected };
}

function directRelations(semanticLinks, descriptorMap, targetNames) {
    const result = [];
    const seen = new Set();
    for (const reachable of semanticLinks.reachableCalls || []) {
        if (!['resolved', 'resolved_overload_family'].includes(reachable.resolutionStatus)) continue;
        for (const candidate of reachable.selectedCandidates || []) {
            const name = candidate.name || reachable.call.symbol;
            if (!name || targetNames.has(name)) continue;
            const declaration = selectDescriptor(descriptorMap, name, {
                ...candidate,
                parameterCount: Array.isArray(candidate.parameters) ? candidate.parameters.length : undefined
            });
            if (!declaration.descriptor) continue;
            const key = [
                'direct',
                reachable.memberApi,
                name,
                reachable.call.file,
                reachable.call.line,
                declaration.descriptor.apiFamily,
                declaration.descriptor.callingConvention
            ].join('|');
            if (seen.has(key)) continue;
            seen.add(key);
            result.push({
                id: 'direct-' + result.length.toString().padStart(3, '0'),
                targetMember: reachable.memberApi,
                kind: 'direct-call',
                lowerApi: {
                    name,
                    apiFamily: declaration.descriptor.apiFamily,
                    callingConvention: declaration.descriptor.callingConvention,
                    declarationFile: declaration.descriptor.file,
                    parameterCounts: unique(declaration.descriptors.map(descriptor => descriptor.parameterCount))
                },
                conditions: {
                    expressions: reachable.call.conditions || [],
                    architectures: reachable.call.architectures || []
                },
                source: {
                    file: reachable.call.file,
                    line: reachable.call.line
                },
                route: [
                    ...(reachable.route || []),
                    reachable.caller.name + ' -> ' + name
                ],
                resolution: reachable.resolutionStatus === 'resolved'
                    ? '已唯一定位'
                    : '同一重载族，需按类型继续核验',
                documentCandidate: true
            });
        }
    }
    return result;
}

function callableParameterArities(body, parameter) {
    const arities = [];
    const expression = new RegExp('\\b' + escapeRegExp(parameter) + '\\s*\\(', 'g');
    let match;
    while ((match = expression.exec(body)) !== null) {
        const open = body.indexOf('(', match.index + parameter.length);
        const close = findClosing(body, open, '(', ')');
        if (close < 0) continue;
        const argumentText = body.slice(open + 1, close).trim();
        arities.push(argumentText ? splitTopLevel(argumentText).length : 0);
        expression.lastIndex = close + 1;
    }
    return [...new Set(arities)];
}

function collectFunctionMacros(content, filePath) {
    const macros = [];
    const lines = content.split(/\r?\n/);
    for (let index = 0; index < lines.length; index++) {
        const first = /^\s*#\s*define\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(.*)$/.exec(lines[index]);
        if (!first) continue;
        let body = first[3].replace(/\\\s*$/, '');
        const startLine = index + 1;
        while (/\\\s*$/.test(lines[index]) && index + 1 < lines.length) {
            index += 1;
            body += '\n' + lines[index].replace(/\\\s*$/, '');
        }
        const params = splitTopLevel(first[2]).map(value => value.trim()).filter(Boolean);
        const callableParameters = params
            .map((parameter, parameterIndex) => ({
                parameter,
                parameterIndex
            }))
            .filter(item => new RegExp('\\b' + escapeRegExp(item.parameter) + '\\s*\\(').test(body))
            .map(item => ({
                ...item,
                callArities: callableParameterArities(body, item.parameter)
            }));
        if (callableParameters.length === 0) continue;
        const generatedSymbols = [...body.matchAll(/\b(?:struct|class)\s+([A-Za-z_]\w*)/g)]
            .map(match => match[1]);
        macros.push({
            name: first[1],
            params,
            callableParameters,
            generatedSymbols,
            file: normPath(filePath),
            line: startLine
        });
    }
    return macros;
}

function macroInvocations(content, macro) {
    const result = [];
    const expression = new RegExp('\\b' + escapeRegExp(macro.name) + '\\s*\\(', 'g');
    let match;
    while ((match = expression.exec(content)) !== null) {
        const lineStart = content.lastIndexOf('\n', match.index) + 1;
        if (/^\s*#\s*define\b/.test(content.slice(lineStart, match.index))) continue;
        const open = content.indexOf('(', match.index + macro.name.length);
        const close = findClosing(content, open, '(', ')');
        if (close < 0) continue;
        const args = splitTopLevel(content.slice(open + 1, close));
        if (args.length !== macro.params.length) continue;
        const functions = macro.callableParameters
            .map(item => ({
                parameter: item.parameter,
                name: args[item.parameterIndex].trim(),
                callArities: item.callArities
            }))
            .filter(item => /^[A-Za-z_]\w*$/.test(item.name));
        if (functions.length === 0) continue;
        const preprocessor = preprocessorConditionsAt(content, match.index);
        const conditions = {};
        for (let index = 0; index < macro.params.length; index++) {
            if (macro.callableParameters.some(item => item.parameterIndex === index)) continue;
            conditions[macro.params[index]] = args[index].trim();
        }
        result.push({
            line: lineNumberAt(content, match.index),
            functions,
            conditions,
            preprocessor
        });
        expression.lastIndex = close + 1;
    }
    return result;
}

function closureDefinitions(files) {
    const map = new Map();
    for (const filePath of files) {
        const content = readText(filePath);
        for (const definition of extractDefinitions(content, filePath)) {
            map.set(definition.id, definition);
        }
    }
    return map;
}

function macroRelations({ files, semanticLinks, descriptorMap, targetNames }) {
    const contents = new Map(files.map(filePath => [normPath(filePath), readText(filePath)]));
    const macros = [];
    for (const [filePath, content] of contents) {
        macros.push(...collectFunctionMacros(content, filePath));
    }
    const definitions = closureDefinitions(files);
    const reachable = semanticLinks.reachableDefinitions || [];
    const result = [];
    const diagnostics = [];
    const seen = new Set();

    for (const macro of macros) {
        const implementationFamily = macro.generatedSymbols[0] || macro.name;
        const boundMembers = new Map();
        for (const entry of reachable) {
            const definition = definitions.get(entry.definition && entry.definition.id);
            if (!definition) continue;
            const bound = macro.generatedSymbols.some(symbol =>
                new RegExp('\\b' + escapeRegExp(symbol) + '\\b').test(definition.body)
            ) || new RegExp('\\b' + escapeRegExp(macro.name) + '\\s*\\(').test(definition.body);
            if (!bound) continue;
            if (!boundMembers.has(entry.memberApi)) boundMembers.set(entry.memberApi, []);
            boundMembers.get(entry.memberApi).push(entry);
        }
        if (boundMembers.size === 0) {
            diagnostics.push('宏 ' + macro.name + ' 的函数映射未能绑定到当前公开成员。');
            continue;
        }
        for (const [filePath, content] of contents) {
            for (const invocation of macroInvocations(content, macro)) {
                for (const functionValue of invocation.functions) {
                    if (targetNames.has(functionValue.name)) continue;
                    for (const [memberApi, entries] of boundMembers) {
                        const callingConvention = entries
                            .map(entry => entry.definition && entry.definition.callingConvention)
                            .find(Boolean) || 'unqualified';
                        const declaration = selectDescriptor(descriptorMap, functionValue.name, {
                            callingConvention,
                            parameterCounts: functionValue.callArities
                        });
                        if (!declaration.descriptor) {
                            diagnostics.push('宏表中的 ' + functionValue.name +
                                ' 未能按调用方式唯一定位到公开 API 声明。');
                            continue;
                        }
                        const key = [
                            memberApi,
                            macro.name,
                            functionValue.name,
                            filePath,
                            invocation.line,
                            JSON.stringify(invocation.conditions)
                        ].join('|');
                        if (seen.has(key)) continue;
                        seen.add(key);
                        result.push({
                            id: 'macro-' + result.length.toString().padStart(3, '0'),
                            targetMember: memberApi,
                            kind: 'macro-function-mapping',
                            implementationFamily,
                            lowerApi: {
                                name: functionValue.name,
                                apiFamily: declaration.descriptor.apiFamily,
                                callingConvention: declaration.descriptor.callingConvention,
                                declarationFile: declaration.descriptor.file,
                                parameterCounts: unique(declaration.descriptors.map(descriptor => descriptor.parameterCount))
                            },
                            conditions: {
                                macro: macro.name,
                                arguments: invocation.conditions,
                                callArities: functionValue.callArities,
                                expressions: invocation.preprocessor.expressions,
                                architectures: invocation.preprocessor.architectures
                            },
                            source: {
                                file: filePath,
                                line: invocation.line,
                                macroDefinition: {
                                    file: macro.file,
                                    line: macro.line,
                                    parameter: functionValue.parameter,
                                    callArities: functionValue.callArities,
                                    generatedSymbols: macro.generatedSymbols
                                }
                            },
                            route: unique(entries.flatMap(entry => entry.route || [])),
                            resolution: '宏表已映射；仍需按条件核验',
                            documentCandidate: false
                        });
                    }
                }
            }
        }
    }
    return { relations: result, diagnostics };
}

function groupMacroRelations(relations) {
    const groups = [];
    const byKey = new Map();
    for (const relation of relations) {
        const implementationFamily = relation.implementationFamily || relation.conditions.macro;
        const key = [relation.targetMember, implementationFamily].join("|");
        let group = byKey.get(key);
        if (!group) {
            group = {
                id: "macro-group-" + groups.length.toString().padStart(3, "0"),
                targetMembers: [relation.targetMember],
                implementationFamily,
                macroNames: [],
                relationIds: [],
                sourceLocations: [],
                expressions: [],
                architectures: [],
                conditionVariants: new Map(),
                lowerApiMap: new Map()
            };
            byKey.set(key, group);
            groups.push(group);
        }
        relation.mappingGroupId = group.id;
        group.relationIds.push(relation.id);
        group.macroNames.push(relation.conditions.macro);
        group.expressions.push(...(relation.conditions.expressions || []));
        group.architectures.push(...(relation.conditions.architectures || []));
        const conditionVariantKey = JSON.stringify({
            expressions: relation.conditions.expressions || [],
            architectures: relation.conditions.architectures || []
        });
        const conditionVariant = group.conditionVariants.get(conditionVariantKey) || {
            expressions: relation.conditions.expressions || [],
            architectures: relation.conditions.architectures || [],
            relationCount: 0
        };
        conditionVariant.relationCount += 1;
        group.conditionVariants.set(conditionVariantKey, conditionVariant);
        group.sourceLocations.push({
            file: relation.source.file,
            line: relation.source.line,
            macro: relation.conditions.macro
        });
        const lowerKey = [
            relation.lowerApi.name,
            relation.lowerApi.apiFamily,
            relation.lowerApi.callingConvention,
            relation.lowerApi.declarationFile
        ].join("|");
        if (!group.lowerApiMap.has(lowerKey)) {
            group.lowerApiMap.set(lowerKey, {
                ...relation.lowerApi,
                parameterCounts: [...(relation.lowerApi.parameterCounts || [])]
            });
        }
    }
    return groups.map((group) => {
        const lowerApis = [...group.lowerApiMap.values()]
            .sort((left, right) => left.name.localeCompare(right.name));
        return {
            id: group.id,
            targetMembers: group.targetMembers,
            implementationFamily: group.implementationFamily,
            macroNames: unique(group.macroNames),
            relationIds: group.relationIds,
            relationCount: group.relationIds.length,
            lowerApiCount: lowerApis.length,
            lowerApis,
            conditions: {
                expressions: unique(group.expressions),
                architectures: unique(group.architectures),
                hasMappingsWithoutArchitecture: [...group.conditionVariants.values()]
                    .some((variant) => variant.architectures.length === 0),
                variants: [...group.conditionVariants.values()]
            },
            sourceLocations: group.sourceLocations,
            documentReview: {
                required: true,
                reason: "宏表列出的是条件映射，须先按当前 API 的实际可达路径和用户目标判断后才可选择参考文档",
                maxRecommendedDocuments: 3
            }
        };
    });
}

function relativeRepoPath(repoPath, filePath) {
    const relative = path.relative(repoPath, filePath).replace(/\\/g, '/');
    return relative && !relative.startsWith('../') && relative !== '..' ? relative : normPath(filePath);
}

function addEvidenceFile(entries, repoPath, filePath, reason) {
    if (!filePath || !fs.existsSync(filePath)) return;
    const absolute = path.resolve(filePath);
    const key = normPath(absolute);
    let entry = entries.get(key);
    if (!entry) {
        entry = {
            path: relativeRepoPath(repoPath, absolute),
            kind: 'direct-evidence',
            reasons: []
        };
        entries.set(key, entry);
    }
    if (!entry.reasons.includes(reason)) entry.reasons.push(reason);
}

function typeTokensForSignatures(signatures) {
    const ignored = new Set(['const', 'volatile', 'typename', 'template', 'inline', 'void', 'auto', 'return', 'class', 'struct']);
    return unique(signatures.flatMap(signature =>
        ((signature.returnType || '') + ' ' + (signature.rawParams || '') + ' ' + (signature.templateParams || ''))
            .match(/[A-Za-z_]\w*/g) || []
    ).filter(token => token.length >= 3 && !ignored.has(token))).slice(0, 48);
}

function addDirectTypeDependencies({ entries, repoPath, tracking, sourceFiles, targetSignatures }) {
    const sourceSet = new Set(sourceFiles.map(normPath));
    const tokens = typeTokensForSignatures(targetSignatures);
    if (tokens.length === 0) return;
    for (const filePath of tracking.files || []) {
        const meta = tracking.metadata.get(filePath) || {};
        if (meta.depth !== 1 || !sourceSet.has(normPath(meta.includedBy || ''))) continue;
        const content = readText(filePath);
        const matched = tokens.filter(token => new RegExp('\\b(?:class|struct|enum|using|typedef)\\b[\\s\\S]{0,240}\\b' + escapeRegExp(token) + '\\b').test(content));
        if (matched.length > 0) {
            addEvidenceFile(entries, repoPath, filePath, '公开声明直接依赖的类型或枚举：' + matched.slice(0, 4).join(', '));
        }
    }
}

function addDirectMemberChecks({ entries, repoPath, tracking, targetNames }) {
    for (const filePath of tracking.files || []) {
        const meta = tracking.metadata.get(filePath) || {};
        if (!['api-check', 'api-check-impl'].includes(meta.kind)) continue;
        const content = readText(filePath);
        const matched = [...targetNames].filter(name => content.includes(name));
        if (matched.length > 0) {
            addEvidenceFile(entries, repoPath, filePath,
                '直接提及当前成员的检查代码：' + matched.join(', '));
        }
    }
}

function buildEvidenceClosure({ repoPath, sourceFiles, targetSignatures, semanticLinks, directRelations, macroRelations, tracking }) {
    const entries = new Map();
    for (const filePath of sourceFiles) addEvidenceFile(entries, repoPath, filePath, '目标 API 声明或定义文件');

    for (const reachable of semanticLinks.reachableDefinitions || []) {
        addEvidenceFile(entries, repoPath, reachable.definition && reachable.definition.file,
            '当前成员 ' + reachable.memberApi + ' 的可达实现：' + (reachable.definition && reachable.definition.name || '未知符号'));
    }
    for (const reachable of semanticLinks.reachableCalls || []) {
        addEvidenceFile(entries, repoPath, reachable.call && reachable.call.file,
            '当前成员 ' + reachable.memberApi + ' 的直接调用位置：' + (reachable.call && reachable.call.symbol || '未知符号'));
    }
    for (const relation of directRelations || []) {
        addEvidenceFile(entries, repoPath, relation.source && relation.source.file,
            '已定位的直接下层调用：' + relation.lowerApi.name);
        addEvidenceFile(entries, repoPath, relation.lowerApi && relation.lowerApi.declarationFile,
            '直接下层 API 声明：' + relation.lowerApi.name);
    }
    for (const relation of macroRelations || []) {
        addEvidenceFile(entries, repoPath, relation.source && relation.source.file,
            '当前可达宏表映射位置：' + (relation.conditions && relation.conditions.macro || relation.id));
        addEvidenceFile(entries, repoPath, relation.source && relation.source.macroDefinition && relation.source.macroDefinition.file,
            '当前可达宏定义：' + (relation.conditions && relation.conditions.macro || relation.id));
        addEvidenceFile(entries, repoPath, relation.lowerApi && relation.lowerApi.declarationFile,
            '宏映射候选下层 API 声明：' + relation.lowerApi.name);
    }
    addDirectTypeDependencies({ entries, repoPath, tracking, sourceFiles, targetSignatures });
    addDirectMemberChecks({ entries, repoPath, tracking, targetNames: new Set((targetSignatures || []).map(signature => signature.funcName)) });

    const candidateFiles = [...entries.values()]
        .map(entry => ({ ...entry, reason: entry.reasons.join('；') }))
        .sort((left, right) => left.path.localeCompare(right.path));
    return {
        schemaVersion: 2,
        kind: 'ascendc_code_closure',
        selectionRule: '仅包含当前 API 的声明、可达实现、已定位调用、实际关联宏映射、直接类型或约束，以及直接提及当前成员的代码。',
        candidateFiles,
        unresolved: [...(tracking.unresolved || [])]
    };
}

function buildApiRelationPlan({ repoPath, documentApiName, targetApi }) {
    const absoluteRepo = path.resolve(repoPath);
    const sourceFiles = sourceFilesForTarget(absoluteRepo, targetApi);
    const inputPath = commonInputPath(sourceFiles);
    const targetSignatures = collectTargetSignatures(sourceFiles, targetApi);
    const members = unique((targetApi.members || [])
        .map(member => typeof member === 'string' ? member : member && member.name)
        .filter(Boolean));
    const targetNames = new Set(members);

    const tracking = trackAscendCReferences(inputPath, absoluteRepo);
    const referenceClosure = {
        ...buildStructuredClosure(tracking),
        schemaVersion: 2,
        usage: '仅用于定位 include、宏、类型和未解析引用；不要求逐项形成人工证据。'
    };
    const crossDirectory = generateReport(tracking);
    const semanticLinks = buildSemanticLinks({
        projectRoot: absoluteRepo,
        memberApis: members.map(name => ({ name, role: name.endsWith('_sync') ? 'sync' : 'regular' })),
        publicSignatures: targetSignatures
    });

    const closureFiles = referenceClosure.candidateFiles
        .map(item => path.resolve(absoluteRepo, item.path))
        .filter(filePath => fs.existsSync(filePath));
    const directNames = (semanticLinks.reachableCalls || [])
        .flatMap(item => (item.selectedCandidates || []).map(candidate => candidate.name || item.call.symbol));
    // Macro definitions and their table entries are often in different files.
    // Discover callable values across the resolved include closure.
    const macroNames = [];
    const closureContents = new Map(
        closureFiles.map(filePath => [normPath(filePath), readText(filePath)])
    );
    const closureMacros = [];
    for (const [filePath, content] of closureContents) {
        closureMacros.push(...collectFunctionMacros(content, filePath));
    }
    for (const macro of closureMacros) {
        for (const content of closureContents.values()) {
            for (const invocation of macroInvocations(content, macro)) {
                for (const functionValue of invocation.functions) macroNames.push(functionValue.name);
            }
        }
    }
    const descriptorMap = publicDescriptors(absoluteRepo, unique([...directNames, ...macroNames]));
    const direct = directRelations(semanticLinks, descriptorMap, targetNames);
    const macro = macroRelations({
        files: closureFiles,
        semanticLinks,
        descriptorMap,
        targetNames
    });
    const macroMappingGroups = groupMacroRelations(macro.relations);
    const codeClosure = buildEvidenceClosure({
        repoPath: absoluteRepo,
        sourceFiles,
        targetSignatures,
        semanticLinks,
        directRelations: direct,
        macroRelations: macro.relations,
        tracking
    });
    const downstreamApis = [...direct, ...macro.relations];
    const uniqueRelations = [];
    const relationKeys = new Set();
    for (const relation of downstreamApis) {
        const key = [
            relation.targetMember,
            relation.kind,
            relation.lowerApi.name,
            relation.lowerApi.apiFamily,
            relation.source.file,
            relation.source.line,
            JSON.stringify(relation.conditions)
        ].join('|');
        if (relationKeys.has(key)) continue;
        relationKeys.add(key);
        uniqueRelations.push(relation);
    }

    return {
        schemaVersion: 2,
        kind: 'ascendc_api_relation_plan',
        generatedAt: new Date().toISOString(),
        target: {
            documentApiName,
            apiType: targetApi.apiType,
            callingConvention: targetApi.callingConvention,
            members,
            sourceFiles: sourceFiles.map(normPath)
        },
        codeClosure,
        referenceClosure,
        crossDirectory,
        semanticLinks,
        downstreamApis: uniqueRelations,
        macroMappingGroups,
        diagnostics: macro.diagnostics
    };
}

function matchesTargetSelection(plan, { repoPath, documentApiName, targetApi }) {
    if (!plan || plan.kind !== 'ascendc_api_relation_plan') return false;
    const expectedFiles = sourceFilesForTarget(path.resolve(repoPath), targetApi).map(normPath).sort();
    const actualFiles = Array.isArray(plan.target && plan.target.sourceFiles)
        ? plan.target.sourceFiles.map(normPath).sort()
        : [];
    const expectedMembers = unique((targetApi.members || [])
        .map(member => typeof member === 'string' ? member : member && member.name)
        .filter(Boolean)).sort();
    const actualMembers = Array.isArray(plan.target && plan.target.members)
        ? [...plan.target.members].sort()
        : [];
    return plan.target.documentApiName === documentApiName &&
        plan.target.callingConvention === targetApi.callingConvention &&
        JSON.stringify(actualFiles) === JSON.stringify(expectedFiles) &&
        JSON.stringify(actualMembers) === JSON.stringify(expectedMembers);
}

module.exports = {
    apiFamilyFromPath,
    buildApiRelationPlan,
    matchesTargetSelection
};
