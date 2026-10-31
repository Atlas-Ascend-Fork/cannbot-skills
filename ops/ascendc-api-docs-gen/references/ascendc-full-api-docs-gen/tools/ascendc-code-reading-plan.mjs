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
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { assertFinalApiConfirmed, assertReferenceSources, assertV2Manifest, providedReferences, refreshManifestFingerprint } from './lib/reference-materials.mjs';

const toolDir = path.dirname(fileURLToPath(import.meta.url));
const skillRoot = path.resolve(toolDir, '..');
const scriptsDir = path.join(skillRoot, 'scripts');
const require = createRequire(import.meta.url);
const { collectPublicContracts } = require('../scripts/extract-public-contracts.js');
const { buildApiRelationPlan, matchesTargetSelection } = require('../scripts/lib/ascendc-api-relations.js');
const { captureSourceSnapshot } = require('../scripts/lib/source-snapshot.js');

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.manifestPath) throw new Error('缺少必需参数: --manifest');
  const manifestPath = path.resolve(args.manifestPath);
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  const context = await resolveContext(manifest);
  const relationPlan = await readOrBuildRelationPlan(args.relationPlanPath, context);
  const sourceFiles = [...(relationPlan.codeClosure?.candidateFiles || []).map(item => path.resolve(context.repoPath, item.path)),
    ...args.sourceFiles.map(file => path.resolve(context.repoPath, file))];
  if (sourceFiles.length === 0) throw new Error('没有可登记的目标源码');
  const snapshot = captureSourceSnapshot(sourceFiles, manifest.sources.repo.evidenceFiles || []);

  const memberArgs = context.members.flatMap(name => ['--member', name]);
  const prototype = runRequired('精确原型提取', 'extract-prototype.js', [
    context.repoPath,
    context.documentApiName,
    context.apiType,
    '--input',
    context.inputPath,
    '--calling-convention',
    context.callingConvention,
    ...memberArgs
  ]);
  const publicContracts = collectPublicContracts({
    repoPath: context.repoPath,
    sourceFiles: context.sourceFiles,
    members: context.members,
    callingConvention: context.callingConvention
  });

  const callExampleRun = runOptional('调用示例提取', 'extract-call-example.js', [
    context.repoPath,
    context.documentApiName,
    context.apiType,
    ...memberArgs,
    '--json'
  ]);
  const callExample = parseCallExample(callExampleRun);

  // Record inputs before evidence is authored; reruns cannot bless changed inputs.
  manifest.sources.repo.evidenceFiles = snapshot;
  refreshManifestFingerprint(manifest);
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n', 'utf8');

  const readingPlan = {
    schemaVersion: 2,
    kind: 'ascendc_code_reading_plan',
    generatedAt: new Date().toISOString(),
    manifestFingerprint: manifest.fingerprint || '',
    documentApiName: context.documentApiName,
    targetApi: manifest.targetApi,
    sourceInput: context.inputPath,
    searchBoundary: await buildSearchBoundary(manifest, context),
    prototype: prototype.stdout.trim(),
    // 完整 include 索引只为定位；只有 codeClosure 是逐项核验范围。
    referenceClosure: relationPlan.referenceClosure || null,
    crossDirectory: relationPlan.crossDirectory,
    codeClosure: relationPlan.codeClosure,
    publicContracts,
    semanticLinks: relationPlan.semanticLinks,
    apiRelations: relationPlan.downstreamApis || [],
    macroMappingGroups: relationPlan.macroMappingGroups || [],
    relationPlanDiagnostics: relationPlan.diagnostics || [],
    callExample
  };
  const text = JSON.stringify(readingPlan, null, 2) + '\n';
  if (args.outputPath) {
    const outputPath = path.resolve(args.outputPath);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, text, 'utf8');
    process.stderr.write('已写入临时代码阅读清单: ' + outputPath + '\n');
  } else {
    process.stdout.write(text);
  }
}

async function buildSearchBoundary(manifest, context) {
  const preferredRoots = ['include', 'impl', 'src', 'kernel', 'examples'];
  const startingSearchRoots = [];
  for (const relativePath of preferredRoots) {
    const absolutePath = path.join(context.repoPath, relativePath);
    const stat = await fs.stat(absolutePath).catch(() => null);
    if (stat?.isDirectory()) startingSearchRoots.push(relativePath + '/');
  }

  const excludedRepoPaths = ['docs/**', 'outputs/**', 'tests/**', '.git/**', '**/*.md'];
  for (const source of providedReferences(manifest.sources)) {
    const sourcePath = path.resolve(source.path);
    if (!isInside(context.repoPath, sourcePath)) continue;
    const relativePath = path.relative(context.repoPath, sourcePath).replace(/\\/g, '/');
    if (relativePath) excludedRepoPaths.push(relativePath);
  }

  return {
    helper: 'tools/ascendc-code-search.mjs',
    startingSearchRoots,
    excludedRepoPaths: [...new Set(excludedRepoPaths)],
    rule: '基础编译只能在目标代码链内搜索；不得搜索 docs/、outputs/、测试目录、Markdown 或清单登记的参考文件。'
  };
}

function parseArgs(argv) {
  const result = { sourceFiles: [] };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--manifest') result.manifestPath = argv[++index];
    else if (arg === '--output') result.outputPath = argv[++index];
    else if (arg === '--relation-plan') result.relationPlanPath = argv[++index];
    else if (arg === '--source-file') {
      const value = argv[++index];
      if (!value || value.startsWith('--')) throw new Error('--source-file 缺少参数值');
      result.sourceFiles.push(value);
    }
    else if (arg === '--help' || arg === '-h') {
      console.log('用法: node tools/ascendc-code-reading-plan.mjs --manifest <manifest.json> [--relation-plan <临时调用关系结果.json>] [--source-file <额外直接证据文件>] [--output <temporary-code-reading-plan.json>]');
      process.exit(0);
    } else {
      throw new Error('未知参数: ' + arg);
    }
  }
  return result;
}

async function readOrBuildRelationPlan(relationPlanPath, context) {
  if (!relationPlanPath) {
    return buildApiRelationPlan({
      repoPath: context.repoPath,
      documentApiName: context.documentApiName,
      targetApi: context.targetApi
    });
  }
  const absolutePath = path.resolve(relationPlanPath);
  let plan;
  try {
    plan = JSON.parse(await fs.readFile(absolutePath, 'utf8'));
  } catch (error) {
    throw new Error('临时调用关系定位结果不可读: ' + absolutePath + '；' + error.message);
  }
  if (!matchesTargetSelection(plan, context)) {
    throw new Error('临时调用关系定位结果与当前 manifest.json 的 API、成员或调用方式不一致');
  }
  if (!plan.codeClosure || !plan.crossDirectory || !plan.semanticLinks) {
    throw new Error('临时调用关系定位结果不完整，缺少代码闭包或调用链信息');
  }
  return plan;
}

async function resolveContext(manifest) {
  assertV2Manifest(manifest);
  assertFinalApiConfirmed(manifest);
  assertReferenceSources(manifest.sources, { requireReady: true });
  const targetApi = manifest.targetApi;
  const repoPath = manifest.sources?.repo?.path;
  if (!targetApi || !repoPath) throw new Error('manifest 缺少 targetApi 或 sources.repo.path');
  const documentApiName = requiredString(manifest.documentApiName, 'documentApiName');
  const apiType = requiredString(targetApi.apiType, 'targetApi.apiType');
  const callingConvention = requiredString(targetApi.callingConvention, 'targetApi.callingConvention');
  const members = (targetApi.members || [])
    .map(member => typeof member === 'string' ? member : member?.name)
    .filter(Boolean);
  if (members.length === 0) throw new Error('targetApi.members 为空');
  const sourceFiles = targetApi.files || [];
  if (!Array.isArray(sourceFiles) || sourceFiles.length === 0) {
    throw new Error('targetApi.files 为空');
  }
  const absoluteFiles = [];
  for (const relativePath of sourceFiles) {
    if (typeof relativePath !== 'string') throw new Error('targetApi.files 包含无效值');
    const absolutePath = path.resolve(repoPath, relativePath);
    const stat = await fs.stat(absolutePath).catch(() => null);
    if (!stat?.isFile()) throw new Error('targetApi 源文件不存在: ' + absolutePath);
    absoluteFiles.push(absolutePath);
  }
  return {
    repoPath: path.resolve(repoPath),
    documentApiName,
    targetApi,
    apiType,
    callingConvention,
    members: [...new Set(members)],
    inputPath: commonInputPath(absoluteFiles),
    sourceFiles: absoluteFiles
  };
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

function parseCallExample(result) {
  if (!result.ok) return { status: 'unavailable', reason: result.detail, members: [] };
  try {
    const parsed = JSON.parse(result.stdout);
    if (parsed?.kind !== 'ascendc_call_example' || !Array.isArray(parsed.members)) {
      throw new Error('返回格式不正确');
    }
    return parsed;
  } catch (error) {
    return {
      status: 'unavailable',
      reason: '调用示例提取结果无法解析：' + (error instanceof Error ? error.message : String(error)),
      members: []
    };
  }
}

function runRequired(label, scriptName, args) {
  const result = runScript(scriptName, args);
  if (!result.ok) throw new Error(label + '失败: ' + result.detail);
  return result;
}

function runOptional(label, scriptName, args) {
  const result = runScript(scriptName, args);
  if (!result.ok) result.detail = label + '失败: ' + result.detail;
  return result;
}

function runScript(scriptName, args) {
  const scriptPath = path.join(scriptsDir, scriptName);
  const result = spawnSync(process.execPath, [scriptPath, ...args], {
    cwd: scriptsDir,
    encoding: 'utf8',
    windowsHide: true,
    maxBuffer: 64 * 1024 * 1024
  });
  const detail = [result.error?.message, result.stderr, result.stdout]
    .filter(Boolean)
    .join('\n')
    .trim();
  return {
    ok: !result.error && result.status === 0,
    stdout: result.stdout || '',
    detail: detail || '无错误输出'
  };
}

function requiredString(value, name) {
  if (typeof value !== 'string' || !value.trim()) {
    throw new Error(name + ' 必须是非空字符串');
  }
  return value.trim();
}

main().catch(error => {
  console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
  process.exit(1);
});
