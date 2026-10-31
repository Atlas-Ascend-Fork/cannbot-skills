#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { assertReferenceSources, assertV2Manifest, providedReferences } from './lib/reference-materials.mjs';

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.manifestPath) throw new Error('缺少必需参数: --manifest');
  if (!args.pattern) throw new Error('缺少必需参数: --pattern');
  if (!args.searchPath) throw new Error('缺少必需参数: --path');

  const manifestPath = path.resolve(args.manifestPath);
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const repoRoot = requiredRepoRoot(manifest);
  const searchPath = resolveSearchPath(repoRoot, args.searchPath);
  const excludedGlobs = buildExcludedGlobs(manifest, repoRoot);
  const relativeSearchPath = path.relative(repoRoot, searchPath);

  const result = spawnSync('rg', [
    '--line-number',
    '--no-heading',
    '--fixed-strings',
    ...excludedGlobs.flatMap(glob => ['--glob', glob]),
    '--',
    args.pattern,
    relativeSearchPath
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
    windowsHide: true,
    maxBuffer: 16 * 1024 * 1024
  });

  let backend = 'rg';
  let output = result.stdout || '';
  if (result.error?.code === 'ENOENT') {
    backend = 'node';
    output = await searchWithNode({
      repoRoot,
      searchPath,
      pattern: args.pattern,
      excludedPaths: buildExcludedPaths(manifest, repoRoot)
    });
  } else if (result.error) {
    throw new Error('无法执行 rg: ' + result.error.message);
  } else if (result.status !== 0 && result.status !== 1) {
    throw new Error((result.stderr || result.stdout || 'rg 执行失败').trim());
  }

  process.stderr.write(
    '受限代码搜索（' + backend + '）：仅搜索指定的目标代码路径，已排除 docs/、outputs/、测试目录、Markdown 与清单登记的参考文件。\n'
  );
  process.stdout.write(output);
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--manifest') result.manifestPath = argv[++index];
    else if (arg === '--pattern') result.pattern = argv[++index];
    else if (arg === '--path') result.searchPath = argv[++index];
    else if (arg === '--help' || arg === '-h') {
      console.log(
        '用法: node tools/ascendc-code-search.mjs --manifest <manifest.json> --pattern <精确符号> --path <阅读清单或直接调用链给出的仓内代码文件或目录>'
      );
      process.exit(0);
    } else {
      throw new Error('未知参数: ' + arg);
    }
  }
  return result;
}

function requiredRepoRoot(manifest) {
  assertV2Manifest(manifest);
  assertReferenceSources(manifest.sources, { requireReady: true });
  const repoRoot = manifest.sources?.repo?.path;
  if (typeof repoRoot !== 'string' || !repoRoot.trim()) {
    throw new Error('manifest.json 缺少 sources.repo.path');
  }
  return path.resolve(repoRoot);
}

function resolveSearchPath(repoRoot, requestedPath) {
  const candidate = path.resolve(repoRoot, requestedPath);
  if (!isInside(repoRoot, candidate)) {
    throw new Error('搜索路径必须位于代码仓内: ' + candidate);
  }
  const relative = path.relative(repoRoot, candidate).replace(/\\/g, '/');
  if (!relative || relative === '.' || isBlockedRelativePath(relative)) {
    throw new Error('代码证据阶段不允许搜索该路径: ' + (relative || '.'));
  }
  return candidate;
}

function buildExcludedGlobs(manifest, repoRoot) {
  const globs = ['!docs/**', '!outputs/**', '!tests/**', '!.git/**', '!**/*.md'];
  for (const source of providedReferences(manifest.sources)) {
    const sourcePath = path.resolve(source.path);
    if (!isInside(repoRoot, sourcePath)) continue;
    const relative = path.relative(repoRoot, sourcePath).replace(/\\/g, '/');
    if (relative) globs.push('!' + relative);
  }
  return [...new Set(globs)];
}


function buildExcludedPaths(manifest, repoRoot) {
  const paths = new Set();
  for (const source of providedReferences(manifest.sources)) {
    const sourcePath = path.resolve(source.path);
    if (isInside(repoRoot, sourcePath)) paths.add(sourcePath);
  }
  return paths;
}

async function searchWithNode({ repoRoot, searchPath, pattern, excludedPaths }) {
  const files = await collectSearchFiles(searchPath, repoRoot, excludedPaths);
  const matches = [];
  for (const filePath of files) {
    const content = await fs.readFile(filePath);
    if (content.includes(0)) continue;
    const relativePath = path.relative(repoRoot, filePath).replace(/\\/g, '/');
    const lines = content.toString('utf8').split(/\n/);
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index].replace(/\r$/u, '');
      if (line.includes(pattern)) matches.push(relativePath + ':' + (index + 1) + ':' + line);
    }
  }
  return matches.length > 0 ? matches.join('\n') + '\n' : '';
}

async function collectSearchFiles(searchPath, repoRoot, excludedPaths) {
  const stat = await fs.lstat(searchPath);
  if (stat.isSymbolicLink()) return [];
  if (stat.isFile()) return isExcludedSearchPath(searchPath, repoRoot, excludedPaths) ? [] : [searchPath];
  if (!stat.isDirectory()) return [];

  const files = [];
  const entries = await fs.readdir(searchPath, { withFileTypes: true });
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const entryPath = path.join(searchPath, entry.name);
    if (isExcludedSearchPath(entryPath, repoRoot, excludedPaths) || entry.isSymbolicLink()) continue;
    if (entry.isDirectory()) {
      files.push(...await collectSearchFiles(entryPath, repoRoot, excludedPaths));
    } else if (entry.isFile()) {
      files.push(entryPath);
    }
  }
  return files;
}

function isExcludedSearchPath(candidate, repoRoot, excludedPaths) {
  if (excludedPaths.has(path.resolve(candidate))) return true;
  const relative = path.relative(repoRoot, candidate).replace(/\\/g, '/');
  return !relative || relative.startsWith('../') || isBlockedRelativePath(relative);
}

function isBlockedRelativePath(relativePath) {
  const normalized = relativePath.toLowerCase();
  return normalized === 'docs' || normalized.startsWith('docs/') ||
    normalized === 'outputs' || normalized.startsWith('outputs/') ||
    normalized === 'tests' || normalized.startsWith('tests/') ||
    normalized === '.git' || normalized.startsWith('.git/') ||
    normalized.endsWith('.md');
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (
    !relative.startsWith('..' + path.sep) &&
    relative !== '..' &&
    !path.isAbsolute(relative)
  );
}

main().catch(error => {
  console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
  process.exit(1);
});
