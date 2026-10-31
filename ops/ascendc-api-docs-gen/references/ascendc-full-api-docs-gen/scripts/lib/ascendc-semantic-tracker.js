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
    getParameterNames,
    parseAscendcSignaturesFromContent,
    splitTopLevel,
    extractParameterName,
    stripTopLevelDefaultValue
} = require('./ascendc-signatures');

const CODE_EXTENSIONS = new Set(['.h', '.hpp', '.hh', '.cpp', '.cc', '.cxx']);
const SKIP_DIRS = new Set(['.git', 'build', 'out', 'output', 'outputs', 'test', 'tests', 'docs']);
const CALL_KEYWORDS = new Set(['if', 'for', 'while', 'switch', 'sizeof', 'alignof', 'decltype', 'return', 'static_assert', 'constexpr', 'defined', 'static_cast',
    'reinterpret_cast', 'const_cast', 'dynamic_cast']);
const MAX_TRACE_DEPTH = 7;
const MAX_LINKS_PER_PARAMETER = 30;

function uniqueStrings(values) {
    return [...new Set(values.flat().filter(Boolean))].sort();
}

function architectureHintsFromText(value) {
    const hints = new Set();
    const patterns = [
        /\b(?:ArchVersion::V|NPU_ARCH_|npu_arch_|ARCH_)(\d{3,5})\b/gi,
        /\b(?:CURRENT_ARCH_VERSION|__NPU_ARCH__|NPU_ARCH|ARCH_VERSION)\b[^0-9]{0,48}(\d{3,5})\b/gi
    ];
    for (const pattern of patterns) {
        let match;
        while ((match = pattern.exec(value)) !== null) hints.add(match[1]);
    }
    return [...hints].sort();
}

function architectureHintsFromPath(filePath) {
    const hints = new Set();
    const pattern = /(?:^|\/)(?:npu_arch_|arch_)(\d{3,5})(?:\/|$)/gi;
    const value = normPath(filePath);
    let match;
    while ((match = pattern.exec(value)) !== null) hints.add(match[1]);
    return [...hints].sort();
}

function collectConditionalBlocks(source) {
    const blocks = [];
    const pattern = /\bif\s*(?:constexpr\s*)?\(/g;
    let match;
    while ((match = pattern.exec(source)) !== null) {
        const openParen = source.indexOf('(', match.index);
        const closeParen = findClosing(source, openParen, '(', ')');
        if (closeParen === -1) continue;
        let cursor = closeParen + 1;
        while (/\s/.test(source[cursor] || '')) cursor++;
        if (source[cursor] !== '{') continue;
        const closeBrace = findClosing(source, cursor, '{', '}');
        if (closeBrace === -1) continue;
        const expression = source.slice(openParen + 1, closeParen).trim();
        blocks.push({ start: cursor + 1, end: closeBrace, expression });
    }
    return blocks;
}

function preprocessorConditionsAt(source, offset) {
    const stack = [];
    const lines = source.slice(0, offset).split(/\r?\n/);
    for (const line of lines) {
        const directive = /^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)$/u.exec(line);
        if (!directive) continue;
        const kind = directive[1];
        const value = directive[2].trim();
        if (kind === 'if' || kind === 'ifdef' || kind === 'ifndef') {
            stack.push(kind === 'ifndef' ? '!defined(' + value + ')' : value || '#' + kind);
        } else if (kind === 'elif' && stack.length > 0) {
            stack[stack.length - 1] = value || '#elif';
        } else if (kind === 'else' && stack.length > 0) {
            stack[stack.length - 1] = '#else';
        } else if (kind === 'endif') {
            stack.pop();
        }
    }
    return stack;
}

function conditionContextAt(source, offset, conditionalBlocks) {
    const expressions = uniqueStrings([
        preprocessorConditionsAt(source, offset),
        conditionalBlocks
            .filter(block => block.start <= offset && offset < block.end)
            .map(block => block.expression)
    ]);
    return {
        expressions,
        architectures: uniqueStrings(expressions.map(architectureHintsFromText))
    };
}

function parameterDescriptor(rawParameter) {
    const declaration = stripTopLevelDefaultValue(rawParameter).trim();
    const name = extractParameterName(declaration);
    const escapedName = name.replace(/[.*+?^\x24()|[\]\\]/g, '\\$&');
    const type = name
        ? declaration.replace(new RegExp('\\b' + escapedName + '\\b\\s*(?:\\[[^\\]]*\\]\\s*)?$'), '').trim()
        : declaration;
    const lower = type.toLowerCase();
    const pointer = /\*/.test(type) || /\b__ubuf__\b/i.test(type);
    const reference = /&/.test(type);
    let category = 'unknown';
    let typeFamily = '';
    if (/\b(?:vector_bool|vector_mask|mask_t)\b/i.test(type)) {
        category = 'mask';
        typeFamily = 'vector-mask';
    } else if (pointer) {
        category = 'pointer';
        typeFamily = 'pointer';
    } else if (/\b(?:vector_[A-Za-z0-9_]*|reg_[A-Za-z0-9_]*)\b/.test(type)) {
        category = 'vector';
        typeFamily = 'vector-register';
    } else if (/\b(?:T|U|V|DType|DataType)\b/.test(type)) {
        category = 'generic';
        typeFamily = 'generic';
    } else if (/\b(?:bool|char|short|int|long|float|double|half|bfloat16|u?int\d+_t|size_t)\b/.test(lower)) {
        category = 'scalar';
        typeFamily = 'scalar';
    }
    return {
        name,
        raw: rawParameter.trim(),
        type: type.replace(/\s+/g, ' ').trim(),
        category,
        typeFamily,
        passing: pointer ? 'pointer' : reference ? 'reference' : 'value'
    };
}

function parameterDescriptors(rawParams) {
    if (!rawParams || rawParams.trim() === 'void') return [];
    return splitTopLevel(rawParams).map(parameterDescriptor);
}

function extractCallingConvention(prefix) {
    const conventions = prefix.match(/__(?:aicore|simd_callee|SIMT_DEVICE_FUNCTIONS_DECL)__/g);
    return conventions?.[conventions.length - 1] || 'unqualified';
}

function argumentDescriptor(definition, expression) {
    const value = expression.trim();
    const exactIndex = definition.params.indexOf(value);
    if (exactIndex >= 0) return { ...definition.parameterDescriptors[exactIndex], source: 'parameter' };
    // Tensor wrappers commonly forward reg_tensor fields. Preserve the field
    // category instead of treating src.mask as the enclosing vector object.
    const field = /^([A-Za-z_]\w*)\s*(?:\.|->)\s*(mask|reg)\b/u.exec(value);
    if (field) {
        const ownerIndex = definition.params.indexOf(field[1]);
        if (ownerIndex >= 0) {
            const owner = definition.parameterDescriptors[ownerIndex];
            if (field[2] === 'mask') {
                return { ...owner, category: 'mask', typeFamily: 'vector-mask', passing: 'value', source: 'parameter-mask-field' };
            }
            return { ...owner, category: 'vector', typeFamily: 'vector-register', passing: 'value', source: 'parameter-reg-field' };
        }
    }
    const referenced = definition.params
        .map((name, index) => ({ name, index }))
        .filter(item => item.name && new RegExp('\\b' + item.name.replace(/[.*+?^\x24()|[\]\\]/g, '\\$&') + '\\b').test(value));
    if (referenced.length === 1) {
        return { ...definition.parameterDescriptors[referenced[0].index], source: 'parameter-expression' };
    }
    if (/^(?:nullptr|NULL)$/u.test(value) || /^&\s*[A-Za-z_]\w*/u.test(value)) {
        return { category: 'pointer', typeFamily: 'pointer', passing: 'value', type: '', source: 'expression' };
    }
    if (/^(?:true|false|[-+]?\d+(?:\.\d+)?(?:[fFuUlL]*)?)$/u.test(value)) {
        return { category: 'scalar', typeFamily: 'scalar', passing: 'value', type: '', source: 'literal' };
    }
    return { category: 'unknown', typeFamily: '', passing: 'value', type: '', source: 'expression' };
}

function parameterShapeKey(definition) {
    return definition.parameterDescriptors
        .map(parameter => parameter.category + ':' + parameter.passing)
        .join(',');
}

function summarizeDefinition(definition) {
    return {
        id: definition.id,
        name: definition.name,
        file: definition.file,
        line: definition.line,
        callingConvention: definition.callingConvention,
        architectures: definition.architectures,
        parameters: definition.parameterDescriptors.map(parameter => ({
            type: parameter.type,
            category: parameter.category,
            passing: parameter.passing
        }))
    };
}

function normPath(filePath) {
    return path.resolve(filePath).replace(/\\/g, '/');
}

function collectCodeFiles(root) {
    const files = [];
    if (!fs.existsSync(root)) return files;
    const stat = fs.statSync(root);
    if (stat.isFile()) return CODE_EXTENSIONS.has(path.extname(root).toLowerCase()) ? [path.resolve(root)] : [];
    const walk = (dir) => {
        let entries = [];
        try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
        for (const entry of entries) {
            const fullPath = path.join(dir, entry.name);
            if (entry.isDirectory()) {
                if (!SKIP_DIRS.has(entry.name.toLowerCase())) walk(fullPath);
            } else if (CODE_EXTENSIONS.has(path.extname(fullPath).toLowerCase())) {
                files.push(path.resolve(fullPath));
            }
        }
    };
    walk(root);
    return files;
}

function removeCommentsKeepLines(source) {
    return source
        .replace(/\/\*[\s\S]*?\*\//g, match => match.replace(/[^\n]/g, ' '))
        .replace(/\/\/[^\n]*/g, match => ' '.repeat(match.length));
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

function lineNumberAt(source, index) {
    return source.slice(0, index).split('\n').length;
}

function extractDefinitions(content, filePath) {
    const source = removeCommentsKeepLines(content);
    const definitions = [];
    const conditionalBlocks = collectConditionalBlocks(source);
    const regex = /\b([A-Za-z_]\w*)\s*\(/g;
    let match;
    while ((match = regex.exec(source)) !== null) {
        const name = match[1];
        if (CALL_KEYWORDS.has(name)) continue;
        const openParen = source.indexOf('(', match.index + name.length);
        const closeParen = findClosing(source, openParen, '(', ')');
        if (closeParen === -1) continue;
        let cursor = closeParen + 1;
        while (/\s/.test(source[cursor] || '')) cursor++;
        if (source[cursor] !== '{') continue;
        const closeBrace = findClosing(source, cursor, '{', '}');
        if (closeBrace === -1) continue;
        const prefix = source.slice(Math.max(0, match.index - 160), match.index);
        if (!/(?:\b(?:void|bool|auto|int\d+_t|uint\d+_t|float|double)\b|[>*&])\s*$/.test(prefix)) continue;
        const rawParams = source.slice(openParen + 1, closeParen).trim();
        const descriptors = parameterDescriptors(rawParams);
        const conditionContext = conditionContextAt(source, match.index, conditionalBlocks);
        definitions.push({
            id: `${normPath(filePath)}:${match.index}`,
            name,
            params: getParameterNames(rawParams),
            parameterDescriptors: descriptors,
            rawParams,
            callingConvention: extractCallingConvention(prefix),
            conditions: conditionContext.expressions,
            architectures: uniqueStrings([
                architectureHintsFromPath(filePath),
                conditionContext.architectures
            ]),
            conditionalBlocks,
            body: source.slice(cursor + 1, closeBrace),
            bodyOffset: cursor + 1,
            source,
            file: normPath(filePath),
            line: lineNumberAt(source, match.index)
        });
        regex.lastIndex = closeBrace + 1;
    }
    return definitions;
}

function scopedCallInfo(definition, callIndex) {
    const prefix = definition.body.slice(0, callIndex);
    if (/(?:\.|->)\s*$/.test(prefix)) {
        return { qualifier: null, kind: "instance-member" };
    }
    const scopeMatch = /([A-Za-z_]\w*)\s*::\s*$/.exec(prefix);
    if (!scopeMatch) return { qualifier: null, kind: "unqualified" };
    const qualifier = scopeMatch[1];
    const isLocalAlias = new RegExp("\\busing\\s+" + qualifier + "\\s*=").test(definition.body) ||
        new RegExp("\\btypedef\\b[^;]*\\b" + qualifier + "\\s*;").test(definition.body);
    return {
        qualifier,
        kind: isLocalAlias ? "local-type-member" : "namespace-qualified"
    };
}

function extractCalls(definition) {
    if (definition.calls) return definition.calls;
    const calls = [];
    const regex = /\b([A-Za-z_]\w*)\b/g;
    let match;
    while ((match = regex.exec(definition.body)) !== null) {
        const name = match[1];
        if (CALL_KEYWORDS.has(name)) continue;
        let cursor = match.index + name.length;
        while (/\s/.test(definition.body[cursor] || '')) cursor++;
        if (definition.body[cursor] === '<') {
            const closeTemplate = findClosing(definition.body, cursor, '<', '>');
            if (closeTemplate === -1) continue;
            cursor = closeTemplate + 1;
            while (/\s/.test(definition.body[cursor] || '')) cursor++;
        }
        if (definition.body[cursor] !== '(') continue;
        const closeParen = findClosing(definition.body, cursor, '(', ')');
        if (closeParen === -1) continue;
        const rawArgs = definition.body.slice(cursor + 1, closeParen);
        const args = rawArgs.trim() ? splitTopLevel(rawArgs) : [];
        const callOffset = definition.bodyOffset + match.index;
        const conditionContext = conditionContextAt(
            definition.source,
            callOffset,
            definition.conditionalBlocks || collectConditionalBlocks(definition.source)
        );
        const callScope = scopedCallInfo(definition, match.index);
        calls.push({
            name,
            qualifier: callScope.qualifier,
            invocationKind: callScope.kind,
            args,
            rawArgs,
            argumentDescriptors: args.map(argument => argumentDescriptor(definition, argument)),
            callingConvention: definition.callingConvention,
            conditions: conditionContext.expressions,
            architectures: uniqueStrings([definition.architectures, conditionContext.architectures]),
            line: lineNumberAt(definition.source, callOffset),
            file: definition.file
        });
        regex.lastIndex = closeParen + 1;
    }
    definition.calls = calls;
    return calls;
}

function identifiers(expression) {
    return [...new Set((expression.match(/\b[A-Za-z_]\w*\b/g) || [])
        .filter(identifier => !CALL_KEYWORDS.has(identifier) && !/^(?:true|false|nullptr)$/.test(identifier)))];
}

function assignmentDependencies(definition, variable) {
    const escaped = variable.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const dependencies = new Set();
    const assignment = new RegExp(`\\b${escaped}\\s*=\\s*([^;]+)`, 'g');
    let match;
    while ((match = assignment.exec(definition.body)) !== null) {
        for (const identifier of identifiers(match[1])) dependencies.add(identifier);
        const before = definition.body.slice(Math.max(0, match.index - 500), match.index);
        const conditions = [...before.matchAll(/\bif\s*\(([^()]*)\)\s*\{?/g)];
        const nearest = conditions[conditions.length - 1];
        if (nearest) for (const identifier of identifiers(nearest[1])) dependencies.add(identifier);
    }
    return dependencies;
}

function dependencyClosure(definition, initialIdentifiers) {
    const result = new Set(initialIdentifiers);
    const queue = [...initialIdentifiers];
    while (queue.length > 0 && result.size < 80) {
        const variable = queue.shift();
        for (const dependency of assignmentDependencies(definition, variable)) {
            if (result.has(dependency)) continue;
            result.add(dependency);
            queue.push(dependency);
        }
    }
    return result;
}

function originsForExpression(definition, expression, provenance) {
    const closure = dependencyClosure(definition, identifiers(expression));
    const origins = new Set();
    for (const identifier of closure) {
        for (const origin of provenance.get(identifier) || []) origins.add(origin);
    }
    return origins;
}

function indexDefinitions(codeFiles) {
    const definitions = [];
    const fileContents = new Map();
    for (const file of codeFiles) {
        let content;
        try { content = fs.readFileSync(file, 'utf8'); } catch { continue; }
        fileContents.set(normPath(file), content);
        definitions.push(...extractDefinitions(content, file));
    }
    const byName = new Map();
    for (const definition of definitions) {
        if (!byName.has(definition.name)) byName.set(definition.name, []);
        byName.get(definition.name).push(definition);
    }
    return { definitions, byName, fileContents };
}

function architectureSetsCompatible(left, right) {
    if (left.length === 0 || right.length === 0) return true;
    return left.some(value => right.includes(value));
}

function architectureText(hints) {
    return hints.length > 0 ? hints.map(value => 'V' + value).join(', ') : '未标出芯片条件';
}

function parameterMismatchReasons(call, candidate) {
    const reasons = [];
    for (let index = 0; index < call.argumentDescriptors.length; index++) {
        const argument = call.argumentDescriptors[index];
        const parameter = candidate.parameterDescriptors[index];
        if (!argument || !parameter) continue;
        const candidateCategory = parameter.category;
        if (argument.category === 'mask' && !['mask', 'generic', 'unknown'].includes(candidateCategory)) {
            reasons.push('第 ' + (index + 1) + ' 个实参是向量掩码，候选参数是' + candidateCategory);
        } else if (argument.category === 'vector' && ['pointer', 'scalar', 'mask'].includes(candidateCategory)) {
            reasons.push('第 ' + (index + 1) + ' 个实参是寄存器向量，候选参数是' + candidateCategory);
        } else if (argument.category === 'pointer' && ['vector', 'mask'].includes(candidateCategory)) {
            reasons.push('第 ' + (index + 1) + ' 个实参是指针，候选参数是' + candidateCategory);
        } else if (argument.category === 'scalar' && ['vector', 'mask'].includes(candidateCategory)) {
            reasons.push('第 ' + (index + 1) + ' 个实参是标量，候选参数是' + candidateCategory);
        }
    }
    return reasons;
}

function resolveCalleeDefinitions({ call, callerDefinition, activeArchitectures, index }) {
    const effectiveArchitectures = uniqueStrings([
        activeArchitectures || [],
        callerDefinition.architectures || [],
        call.architectures || []
    ]);
    if (call.invocationKind === "instance-member" || call.invocationKind === "local-type-member") {
        return {
            status: "unresolved_scoped_member",
            candidates: [],
            selected: [],
            rejected: [],
            effectiveArchitectures,
            reason: call.invocationKind === "local-type-member"
                ? "调用的是当前函数内类型别名的成员，不能按裸函数名连接到全局同名实现"
                : "调用的是对象或指针成员，不能按裸函数名连接到全局同名实现"
        };
    }
    const candidates = (index.byName.get(call.name) || [])
        .filter(definition => definition.params.length === call.args.length);
    if (candidates.length === 0) {
        return {
            status: 'unresolved_external',
            candidates,
            selected: [],
            rejected: [],
            effectiveArchitectures
        };
    }

    const rejected = [];
    const compatible = [];
    for (const candidate of candidates) {
        const reasons = [];
        if (!architectureSetsCompatible(effectiveArchitectures, candidate.architectures || [])) {
            reasons.push(
                '调用适用 ' + architectureText(effectiveArchitectures) +
                '，候选仅适用 ' + architectureText(candidate.architectures || [])
            );
        }
        if (call.callingConvention === '__simd_callee__' &&
            candidate.callingConvention !== 'unqualified' &&
            candidate.callingConvention !== '__simd_callee__') {
            reasons.push('调用位于 __simd_callee__ 函数，候选调用方式为 ' + candidate.callingConvention);
        }
        reasons.push(...parameterMismatchReasons(call, candidate));
        if (reasons.length > 0) rejected.push({ candidate, reasons });
        else compatible.push(candidate);
    }

    if (compatible.length === 0) {
        return {
            status: 'no_compatible_candidate',
            candidates,
            selected: [],
            rejected,
            effectiveArchitectures
        };
    }
    if (compatible.length === 1) {
        return {
            status: 'resolved',
            candidates,
            selected: compatible,
            rejected,
            effectiveArchitectures
        };
    }

    const families = new Map();
    for (const candidate of compatible) {
        const key = [
            candidate.file,
            candidate.callingConvention,
            (candidate.architectures || []).join(','),
            parameterShapeKey(candidate)
        ].join('|');
        if (!families.has(key)) families.set(key, []);
        families.get(key).push(candidate);
    }
    if (families.size === 1) {
        return {
            status: 'resolved_overload_family',
            candidates,
            selected: compatible,
            rejected,
            effectiveArchitectures
        };
    }
    return {
        status: 'ambiguous',
        candidates,
        selected: [],
        compatible,
        rejected,
        effectiveArchitectures
    };
}

function callCandidateDiagnostic(memberApi, callerDefinition, call, resolution) {
    return {
        memberApi,
        caller: {
            symbol: callerDefinition.name,
            file: callerDefinition.file,
            line: callerDefinition.line,
            callingConvention: callerDefinition.callingConvention
        },
        call: {
            symbol: call.name,
            qualifier: call.qualifier,
            invocationKind: call.invocationKind,
            file: call.file,
            line: call.line,
            argumentCount: call.args.length,
            architectures: resolution.effectiveArchitectures,
            conditions: call.conditions
        },
        status: resolution.status,
        retainedCandidates: resolution.selected.map(summarizeDefinition),
        ambiguousCandidates: (resolution.compatible || []).map(summarizeDefinition),
        rejectedCandidates: resolution.rejected.map(item => ({
            ...summarizeDefinition(item.candidate),
            reasons: item.reasons
        }))
    };
}

function traceTargetMembers(memberApis, publicSignatures, index) {
    const terminalEdges = [];
    const allEdges = [];
    const callCandidateDiagnostics = [];
    const reachableCalls = [];
    const reachableDefinitions = [];
    const diagnosticKeys = new Set();
    const reachableCallKeys = new Set();
    const reachableDefinitionKeys = new Set();
    for (const member of memberApis) {
        const signatures = publicSignatures.filter(signature => signature.funcName === member.name);
        const signatureParamsByArity = signatures.map(signature => getParameterNames(signature.rawParams));
        const starts = index.byName.get(member.name) || [];
        const queue = starts.map(definition => {
            const matchingParameters = signatureParamsByArity
                .filter(parameters => parameters.length === definition.params.length);
            const provenance = new Map(definition.params.map((parameter, parameterIndex) => [
                parameter,
                new Set(matchingParameters.map(parameters => parameters[parameterIndex]).filter(Boolean))
            ]));
            return {
                definition,
                depth: 0,
                provenance,
                activeArchitectures: definition.architectures || [],
                path: []
            };
        });
        const visited = new Set();
        while (queue.length > 0) {
            const state = queue.shift();
            const provenanceKey = [...state.provenance.entries()]
                .map(([key, value]) => key + ':' + [...value].sort().join(',')).sort().join('|');
            const visitKey = member.name + '|' + state.definition.id + '|' + provenanceKey + '|' +
                (state.activeArchitectures || []).join(',');
            if (visited.has(visitKey) || state.depth > MAX_TRACE_DEPTH) continue;
            visited.add(visitKey);

            const definitionKey = member.name + '|' + state.definition.id + '|' + state.path.join('>');
            if (!reachableDefinitionKeys.has(definitionKey)) {
                reachableDefinitionKeys.add(definitionKey);
                reachableDefinitions.push({
                    memberApi: member.name,
                    definition: summarizeDefinition(state.definition),
                    route: [...state.path],
                    conditions: [...(state.activeArchitectures || [])]
                });
            }

            for (const call of extractCalls(state.definition)) {
                const argumentOrigins = call.args.map(argument => originsForExpression(state.definition, argument, state.provenance));
                const resolution = resolveCalleeDefinitions({
                    call,
                    callerDefinition: state.definition,
                    activeArchitectures: state.activeArchitectures,
                    index
                });
                const isRelevant = argumentOrigins.some(origins => origins.size > 0);
                const reachableCallKey = [
                    member.name,
                    state.definition.id,
                    call.line,
                    call.name,
                    resolution.status,
                    (resolution.effectiveArchitectures || []).join(',')
                ].join('|');
                if (!reachableCallKeys.has(reachableCallKey)) {
                    reachableCallKeys.add(reachableCallKey);
                    reachableCalls.push({
                        memberApi: member.name,
                        route: [...state.path],
                        caller: summarizeDefinition(state.definition),
                        call: {
                            symbol: call.name,
                            qualifier: call.qualifier,
                            invocationKind: call.invocationKind,
                            file: call.file,
                            line: call.line,
                            argumentCount: call.args.length,
                            callingConvention: call.callingConvention,
                            conditions: call.conditions,
                            architectures: resolution.effectiveArchitectures
                        },
                        resolutionStatus: resolution.status,
                        selectedCandidates: resolution.selected.map(summarizeDefinition),
                        ambiguousCandidates: (resolution.compatible || []).map(summarizeDefinition),
                        rejectedCandidates: resolution.rejected.map(item => ({
                            ...summarizeDefinition(item.candidate),
                            reasons: item.reasons
                        }))
                    });
                }
                if (isRelevant && (resolution.rejected.length > 0 ||
                    resolution.status === "ambiguous" || resolution.status === "no_compatible_candidate" || resolution.status === "unresolved_scoped_member")) {
                    const key = [
                        member.name,
                        state.definition.id,
                        call.line,
                        call.name,
                        resolution.status,
                        resolution.effectiveArchitectures.join(',')
                    ].join('|');
                    if (!diagnosticKeys.has(key)) {
                        diagnosticKeys.add(key);
                        callCandidateDiagnostics.push(callCandidateDiagnostic(member.name, state.definition, call, resolution));
                    }
                }
                for (let argumentIndex = 0; argumentIndex < argumentOrigins.length; argumentIndex++) {
                    for (const targetParameter of argumentOrigins[argumentIndex]) {
                        const edge = {
                            memberApi: member.name,
                            targetParameter,
                            caller: state.definition.name,
                            callee: call.name,
                            argumentIndex,
                            argumentCount: call.args.length,
                            expression: call.args[argumentIndex],
                            file: call.file,
                            line: call.line,
                            architectures: resolution.effectiveArchitectures,
                            callingConvention: call.callingConvention,
                            resolutionStatus: resolution.status
                        };
                        allEdges.push(edge);
                        if (resolution.status === 'unresolved_external') terminalEdges.push(edge);
                    }
                }
                if (!['resolved', 'resolved_overload_family'].includes(resolution.status)) continue;
                for (const calleeDefinition of resolution.selected) {
                    const nextProvenance = new Map();
                    for (let indexArg = 0; indexArg < calleeDefinition.params.length; indexArg++) {
                        nextProvenance.set(calleeDefinition.params[indexArg], argumentOrigins[indexArg] || new Set());
                    }
                    queue.push({
                        definition: calleeDefinition,
                        depth: state.depth + 1,
                        provenance: nextProvenance,
                        activeArchitectures: uniqueStrings([
                            resolution.effectiveArchitectures,
                            calleeDefinition.architectures || []
                        ]),
                        path: [...state.path, call.name]
                    });
                }
            }
        }
    }
    return {
        terminalEdges,
        allEdges,
        callCandidateDiagnostics,
        reachableCalls,
        reachableDefinitions
    };
}

function buildCallers(index) {
    const externalBySymbol = new Map();
    const byDefinitionId = new Map();
    for (const definition of index.definitions) {
        for (const call of extractCalls(definition)) {
            const resolution = resolveCalleeDefinitions({
                call,
                callerDefinition: definition,
                activeArchitectures: definition.architectures || [],
                index
            });
            const item = { definition, call, resolution };
            if (resolution.status === 'unresolved_external') {
                if (!externalBySymbol.has(call.name)) externalBySymbol.set(call.name, []);
                externalBySymbol.get(call.name).push(item);
                continue;
            }
            if (!['resolved', 'resolved_overload_family'].includes(resolution.status)) continue;
            for (const calleeDefinition of resolution.selected) {
                if (!byDefinitionId.has(calleeDefinition.id)) byDefinitionId.set(calleeDefinition.id, []);
                byDefinitionId.get(calleeDefinition.id).push(item);
            }
        }
    }
    return { externalBySymbol, byDefinitionId };
}

function reverseTraceSink(edge, index, publicNames, callers = buildCallers(index)) {
    const links = [];
    const sinkCallers = (callers.externalBySymbol.get(edge.callee) || [])
        .filter(item => item.call.args.length === edge.argumentCount)
        .filter(item => architectureSetsCompatible(
            edge.architectures || [],
            item.resolution.effectiveArchitectures || []
        ));
    const queue = [];
    for (const item of sinkCallers) {
        const provenance = new Map(item.definition.params.map(param => [param, new Set([param])]));
        const origins = originsForExpression(item.definition, item.call.args[edge.argumentIndex] || '', provenance);
        if (origins.size === 0) continue;
        queue.push({ definition: item.definition, trackedParams: origins, path: [{
            function: item.definition.name,
            callee: edge.callee,
            argumentIndex: edge.argumentIndex,
            file: item.call.file,
            line: item.call.line
        }], depth: 0 });
    }

    const visited = new Set();
    while (queue.length > 0 && links.length < MAX_LINKS_PER_PARAMETER) {
        const state = queue.shift();
        const key = state.definition.id + '|' + [...state.trackedParams].sort().join(',');
        if (visited.has(key) || state.depth > MAX_TRACE_DEPTH) continue;
        visited.add(key);
        if (publicNames.has(state.definition.name)) {
            links.push({
                relatedPublicApi: state.definition.name,
                relatedParameters: [...state.trackedParams].filter(param => state.definition.params.includes(param)),
                evidenceLevel: 'mapping_only',
                path: state.path
            });
        }

        for (const caller of callers.byDefinitionId.get(state.definition.id) || []) {
            if (caller.call.args.length !== state.definition.params.length) continue;
            const provenance = new Map(caller.definition.params.map(param => [param, new Set([param])]));
            const nextTracked = new Set();
            for (const trackedParam of state.trackedParams) {
                const parameterIndex = state.definition.params.indexOf(trackedParam);
                if (parameterIndex === -1) continue;
                const expression = caller.call.args[parameterIndex] || '';
                for (const origin of originsForExpression(caller.definition, expression, provenance)) nextTracked.add(origin);
            }
            if (nextTracked.size === 0) continue;
            queue.push({
                definition: caller.definition,
                trackedParams: nextTracked,
                path: [{
                    function: caller.definition.name,
                    callee: state.definition.name,
                    argumentIndex: [...state.trackedParams]
                        .map(param => state.definition.params.indexOf(param)).filter(value => value >= 0),
                    file: caller.call.file,
                    line: caller.call.line
                }, ...state.path],
                depth: state.depth + 1
            });
        }
    }
    return links;
}

function normalizedIdentifier(identifier) {
    return identifier.replace(/_/g, '').toLowerCase();
}

function collectNameHints(targetParameters, fileContents) {
    const hints = Object.fromEntries(targetParameters.map(parameter => [parameter, []]));
    const parametersByNormalizedName = new Map();
    for (const parameter of targetParameters) {
        const normalized = normalizedIdentifier(parameter);
        if (!parametersByNormalizedName.has(normalized)) parametersByNormalizedName.set(normalized, []);
        parametersByNormalizedName.get(normalized).push(parameter);
    }

    for (const [file, content] of fileContents) {
        const lines = content.split(/\r?\n/);
        for (let index = 0; index < lines.length; index++) {
            const matchedParameters = new Set();
            for (const identifier of identifiers(lines[index])) {
                for (const parameter of parametersByNormalizedName.get(normalizedIdentifier(identifier)) || []) {
                    matchedParameters.add(parameter);
                }
            }
            for (const parameter of matchedParameters) {
                if (hints[parameter].length >= 20) continue;
                hints[parameter].push({
                    file,
                    line: index + 1,
                    text: lines[index].trim(),
                    evidenceLevel: 'name_hint_only'
                });
            }
        }
    }
    return hints;
}


function buildSemanticLinks({ projectRoot, memberApis, publicSignatures }) {
    const roots = [path.join(projectRoot, 'include'), path.join(projectRoot, 'impl')];
    const codeFiles = [...new Set(roots.flatMap(collectCodeFiles))];
    const index = indexDefinitions(codeFiles);
    const publicNames = new Set();
    for (const file of codeFiles.filter(file => normPath(file).includes('/include/'))) {
        const content = index.fileContents.get(normPath(file)) || '';
        for (const signature of parseAscendcSignaturesFromContent(content)) publicNames.add(signature.funcName);
    }

    const trace = traceTargetMembers(memberApis, publicSignatures, index);
    const callers = buildCallers(index);
    const memberNames = new Set(memberApis.map(member => member.name));
    const links = [];
    const seen = new Set();
    for (const edge of trace.terminalEdges) {
        for (const related of reverseTraceSink(edge, index, publicNames, callers)) {
            if (memberNames.has(related.relatedPublicApi)) continue;
            const key = `${edge.memberApi}|${edge.targetParameter}|${edge.callee}|${edge.argumentIndex}|` +
                `${related.relatedPublicApi}|${related.relatedParameters.join(',')}`;
            if (seen.has(key)) continue;
            seen.add(key);
            links.push({
                targetMember: edge.memberApi,
                targetParameter: edge.targetParameter,
                sink: {
                    symbol: edge.callee,
                    argumentIndex: edge.argumentIndex,
                    argumentCount: edge.argumentCount,
                    file: edge.file,
                    line: edge.line
                },
                ...related
            });
        }
    }

    const targetParameters = [...new Set(publicSignatures
        .filter(signature => memberApis.some(member => member.name === signature.funcName))
        .flatMap(signature => getParameterNames(signature.rawParams)))];
    const diagnostics = links.length === 0
        ? ['No exact same-sink public-wrapper link was resolved; name hints remain weak evidence only.']
        : [];
    if (trace.callCandidateDiagnostics.some(item =>
        item.status === 'ambiguous' || item.status === 'no_compatible_candidate')) {
        diagnostics.push('Some same-name call candidates could not be resolved safely; they are listed in callCandidateDiagnostics and are not semantic evidence.');
    }
    return {
        scannedFileCount: codeFiles.length,
        definitionCount: index.definitions.length,
        targetEdges: trace.allEdges,
        links,
        callCandidateDiagnostics: trace.callCandidateDiagnostics,
        reachableCalls: trace.reachableCalls,
        reachableDefinitions: trace.reachableDefinitions,
        nameHints: collectNameHints(targetParameters, index.fileContents),
        diagnostics
    };
}

module.exports = {
    buildSemanticLinks,
    collectCodeFiles,
    dependencyClosure,
    extractCalls,
    extractDefinitions,
    normalizedIdentifier,
    resolveCalleeDefinitions,
    reverseTraceSink
};
