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

const CODE_EXTS = new Set(['.h', '.hpp', '.cpp', '.cc', '.cxx']);
const SKIP_DIRS = new Set(['.git', 'build', 'cmake-build', 'out', 'output', 'outputs', 'tests', 'test']);
const IGNORED_UNRESOLVED_INCLUDES = new Set(['stub_def.h', 'stub_fun.h']);
const MAX_INCLUDE_DEPTH = 3;
const MAX_CPP_INCLUDE_DEPTH = 2;

function normPath(p) {
    return path.resolve(p).replace(/\\/g, '/');
}

function readFile(p) {
    try {
        return fs.readFileSync(p, 'utf8');
    } catch {
        return null;
    }
}

function isCodeFile(p) {
    return CODE_EXTS.has(path.extname(p).toLowerCase());
}

function collectFiles(inputPath) {
    const result = [];
    if (!fs.existsSync(inputPath)) return result;
    const stat = fs.statSync(inputPath);
    if (stat.isFile()) return isCodeFile(inputPath) ? [path.resolve(inputPath)] : [];
    for (const entry of fs.readdirSync(inputPath, { withFileTypes: true })) {
        const full = path.join(inputPath, entry.name);
        if (entry.isDirectory()) {
            if (!SKIP_DIRS.has(entry.name.toLowerCase())) result.push(...collectFiles(full));
        } else if (entry.isFile() && isCodeFile(full)) {
            result.push(path.resolve(full));
        }
    }
    return result;
}

function isDirectory(candidate) {
    try {
        return fs.statSync(candidate).isDirectory();
    } catch {
        return false;
    }
}

function isAscendCProjectRoot(candidate) {
    return isDirectory(path.join(candidate, 'include')) && isDirectory(path.join(candidate, 'impl'));
}

function detectProjectRoot(inputPath) {
    let current = fs.existsSync(inputPath) && fs.statSync(inputPath).isFile()
        ? path.dirname(path.resolve(inputPath))
        : path.resolve(inputPath);
    while (true) {
        if (isAscendCProjectRoot(current)) {
            return current;
        }
        const parent = path.dirname(current);
        if (parent === current) break;
        current = parent;
    }
    return null;
}
function apiTypeFromPath(filePath) {
    const n = normPath(filePath);
    const m = n.match(/\/(?:include|impl)\/([^/]+)\//);
    return m ? m[1] : null;
}

function projectRootsFor(sourceFile, projectRoot) {
    const roots = [];
    const add = root => {
        if (!root) return;
        const absolute = path.resolve(root);
        if (!isAscendCProjectRoot(absolute)) return;
        if (!roots.some(item => normPath(item) === normPath(absolute))) roots.push(absolute);
    };
    // Resolve from the target source first. A nested path supplied by a caller
    // must not make a public header in <repo>/include look missing.
    add(detectProjectRoot(sourceFile));
    add(projectRoot);
    return roots;
}

function resolveInclude(includePath, sourceFile, projectRoot) {
    const apiType = apiTypeFromPath(sourceFile);
    const candidates = [
        path.resolve(path.dirname(sourceFile), includePath),
    ];
    for (const root of projectRootsFor(sourceFile, projectRoot)) {
        candidates.push(path.resolve(root, includePath));
        candidates.push(path.resolve(root, 'include', includePath));
        candidates.push(path.resolve(root, 'impl', includePath));
        if (apiType) {
            candidates.push(path.resolve(root, 'include', apiType, includePath));
            candidates.push(path.resolve(root, 'impl', apiType, includePath));
        }
    }
    for (const candidate of [...new Set(candidates.map(item => path.resolve(item)))]) {
        if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
            return path.resolve(candidate);
        }
    }
    return null;
}
function replaceIncludeWithImpl(filePath) {
    const n = normPath(filePath);
    const idx = n.indexOf('/include/');
    if (idx === -1) return [];
    const implBase = n.slice(0, idx) + '/impl/' + n.slice(idx + '/include/'.length);
    const parsed = path.parse(implBase);
    const names = [
        parsed.name,
        `${parsed.name}_impl`,
        parsed.name.replace(/_intf$/, '_intf_impl'),
        parsed.name.replace(/_intf$/, '_base_impl'),
    ];
    const exts = ['.cpp', '.cc', '.cxx', '.h', '.hpp'];
    return [...new Set(names.flatMap(name => exts.map(ext => path.join(parsed.dir, name + ext))))];
}

function sameDirImplCandidates(filePath) {
    const parsed = path.parse(filePath);
    const names = [parsed.name, `${parsed.name}_impl`];
    const exts = ['.cpp', '.cc', '.cxx'];
    return names.flatMap(name => exts.map(ext => path.join(parsed.dir, name + ext)));
}

function findImplementationFiles(headerPath) {
    return [...replaceIncludeWithImpl(headerPath), ...sameDirImplCandidates(headerPath)]
        .filter((p, i, arr) => arr.indexOf(p) === i)
        .filter(p => fs.existsSync(p) && fs.statSync(p).isFile());
}

function findApiCheckFiles(filePath, projectRoot) {
    if (!projectRoot) return [];
    const apiType = apiTypeFromPath(filePath);
    if (!apiType) return [];
    const checkRoots = [
        path.join(projectRoot, 'impl', apiType, 'detail', 'api_check'),
        path.join(projectRoot, 'impl', apiType, 'detail', 'api_check', 'kernel_check'),
    ];
    const base = path.basename(filePath, path.extname(filePath))
        .replace(/_intf$/, '')
        .replace(/^kernel_operator_/, '')
        .replace(/^kernel_/, '');
    const tokens = [...new Set(base.split(/[_-]+/).filter(Boolean))];
    const files = [];
    for (const root of checkRoots) {
        if (!fs.existsSync(root)) continue;
        for (const f of collectFiles(root)) {
            const lower = path.basename(f).toLowerCase();
            if (tokens.some(t => t.length >= 2 && lower.includes(t.toLowerCase()))) files.push(f);
        }
    }
    return [...new Set(files)];
}

function parseIncludes(content) {
    const out = [];
    const re = /^#\s*include\s+"([^"]+)"/gm;
    let m;
    while ((m = re.exec(content)) !== null) out.push(m[1]);
    return out;
}

function detectArchGuards(filePath) {
    const content = readFile(filePath);
    if (!content) return [];
    const guards = [];
    const patterns = [
        /__NPU_ARCH__/,
        /ASCEND_IS_AIC/,
        /__CCE_AICORE__/,
        /#\s*(?:if|elif)\b[^\n]*(?:__DAV_|__CCE_|AIC|VECTOR|CUBE)/,
    ];
    const lines = content.split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
        if (patterns.some(re => re.test(lines[i]))) {
            guards.push({ line: i + 1, text: lines[i].trim() });
        }
    }
    return guards;
}

function addTracked(state, filePath, meta) {
    const n = normPath(filePath);
    if (!state.files.has(n)) {
        state.files.add(n);
        state.metadata.set(n, meta);
    }
}

function trackAscendCReferences(inputPath, projectRootArg) {
    // The target source determines the usable repository root. Keep the
    // explicit value only as a fallback for intentionally external inputs.
    const projectRoot = detectProjectRoot(inputPath) || projectRootArg;
    const rootNorm = normPath(projectRoot || inputPath);
    const inputFiles = collectFiles(inputPath);
    const inputRoot = fs.existsSync(inputPath) && fs.statSync(inputPath).isFile()
        ? normPath(path.dirname(path.resolve(inputPath))) + '/'
        : normPath(inputPath) + '/';

    const state = {
        files: new Set(),
        metadata: new Map(),
        unresolved: [],
        unresolvedSet: new Set(),
    };

    for (const f of inputFiles) {
        addTracked(state, f, { reason: '输入范围内代码文件', includedBy: null, depth: 0, kind: 'input' });
    }

    function trackFrom(files, depth, maxDepth) {
        if (depth > maxDepth) return;
        const next = [];
        for (const file of files) {
            const content = readFile(file);
            if (!content) continue;
            for (const inc of parseIncludes(content)) {
                const resolved = resolveInclude(inc, file, projectRoot);
                if (!resolved) {
                    if (!IGNORED_UNRESOLVED_INCLUDES.has(inc)) {
                        const item = `#include "${inc}" (from ${normPath(file)}) — file not found`;
                        if (!state.unresolvedSet.has(item)) {
                            state.unresolvedSet.add(item);
                            state.unresolved.push(item);
                        }
                    }
                    continue;
                }
                const resolvedNorm = normPath(resolved);
                const isExternal = !resolvedNorm.startsWith(inputRoot);
                addTracked(state, resolved, {
                    reason: isExternal ? '由 #include 引入的外部文件' : '由 #include 引入的内部文件',
                    includedBy: normPath(file),
                    includeText: inc,
                    depth,
                    kind: isExternal ? 'include-external' : 'include-internal',
                });
                next.push(resolved);
                for (const impl of findImplementationFiles(resolved)) {
                    addTracked(state, impl, {
                        reason: '由头文件定位到的实现文件',
                        includedBy: resolvedNorm,
                        includeText: null,
                        depth,
                        kind: 'impl-pair',
                    });
                    next.push(impl);
                }
            }
        }
        if (next.length > 0) trackFrom(next, depth + 1, maxDepth);
    }

    trackFrom(inputFiles, 1, MAX_INCLUDE_DEPTH);

    const known = [...state.files];
    for (const f of known) {
        for (const check of findApiCheckFiles(f, projectRoot)) {
            addTracked(state, check, {
                reason: '由 API 路径定位到的 api_check/kernel_check 文件',
                includedBy: f,
                includeText: null,
                depth: 0,
                kind: 'api-check',
            });
            for (const impl of findImplementationFiles(check)) {
                addTracked(state, impl, {
                    reason: '由 api_check/kernel_check 头文件定位到的实现文件',
                    includedBy: normPath(check),
                    includeText: null,
                    depth: 0,
                    kind: 'api-check-impl',
                });
            }
        }
    }

    const cppFiles = [...state.files].filter(f => ['.cpp', '.cc', '.cxx'].includes(path.extname(f).toLowerCase()));
    trackFrom(cppFiles, 1, MAX_CPP_INCLUDE_DEPTH);

    const archGuards = new Map();
    for (const f of state.files) {
        const guards = detectArchGuards(f);
        if (guards.length > 0) archGuards.set(f, guards);
    }

    return {
        projectRoot: projectRoot ? normPath(projectRoot) : null,
        inputPath: normPath(inputPath),
        inputFiles: inputFiles.map(normPath),
        files: [...state.files],
        metadata: state.metadata,
        unresolved: state.unresolved,
        archGuards,
        rootNorm,
    };
}

module.exports = {
    trackAscendCReferences,
    detectProjectRoot,
    collectFiles,
    normPath,
};
