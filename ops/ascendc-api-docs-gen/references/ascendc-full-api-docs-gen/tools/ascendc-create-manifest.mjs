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
import { execFile } from 'node:child_process';
import { createRequire } from 'node:module';
import { promisify } from 'node:util';
import {
  createPendingFinalApiConfirmation,
  createReferenceSources,
  refreshManifestFingerprint,
  MANIFEST_SCHEMA_VERSION,
  sha256File,
} from './lib/reference-materials.mjs';
import { documentFileStem, legacyOutputDirectoryName, outputDirectoryName } from './lib/output-directory-name.mjs';

const execFileAsync = promisify(execFile);
const require = createRequire(import.meta.url);
const { detectInterfaces } = require('../scripts/detect-interfaces.js');

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputMode = resolveInputMode(args);
  const rawApi = inputMode === 'selected-api'
    ? await readJson(args.apiPath, '已选 API 信息')
    : null;
  const repoInput = inputMode === 'selected-api'
    ? args.repoPath || inputProjectRoot(rawApi)
    : args.repoPath;
  requireValue(repoInput, inputMode === 'selected-api'
    ? '--repo 或已选 API 信息中的 projectRoot'
    : '--repo');
  const repoPath = path.resolve(repoInput);
  await assertDirectory(repoPath, '代码仓根目录');

  const selectedApi = inputMode === 'selected-api'
    ? resolveSelectedApi(rawApi, args.selectedInterfaceId)
    : await resolveDirectApi(args, repoPath);
  const manifestInput = rawApi ?? { projectRoot: repoPath, targetApi: selectedApi };
  const targetApi = await normalizeTargetApi(selectedApi, repoPath);
  const documentApiName = stringValue(manifestInput.documentApiName) || targetApi.name;
  const outputDir = await resolveOutputDir(args.outputDir, repoPath, documentApiName);
  const referenceSources = await createReferenceSources({
    repoPath,
    referenceMaterials: inputReferenceMaterials(manifestInput, selectedApi),
    referenceCollectionComplete: inputReferenceCollectionComplete(manifestInput, selectedApi),
    internalReferencePath: args.internalReferencePath,
    existingDocumentPath: args.existingDocumentPath,
  });
  const sources = {
    repo: {
      path: repoPath,
      revision: await readGitRevision(repoPath),
      targetFiles: await Promise.all(targetApi.files.map(async relativePath => {
        const absolutePath = path.join(repoPath, relativePath);
        return { path: relativePath, sha256: await sha256File(absolutePath) };
      }))
    },
    ...referenceSources
  };
  const manifest = {
    schemaVersion: MANIFEST_SCHEMA_VERSION,
    kind: 'ascendc_doc_manifest',
    generatedAt: new Date().toISOString(),
    documentApiName,
    output: {
      documentFileStem: documentFileStem(documentApiName)
    },
    targetApi,
    finalApiConfirmation: createPendingFinalApiConfirmation(targetApi),
    sources
  };
  const fingerprint = refreshManifestFingerprint(manifest);

  const manifestPath = path.join(outputDir, 'manifest.json');
  const existing = await readOptionalJson(manifestPath);
  if (existing) {
    if (existing.fingerprint === fingerprint) {
      process.stdout.write(JSON.stringify({
        manifest: manifestPath,
        fingerprint,
        status: 'unchanged'
      }, null, 2) + '\n');
      return;
    }
    if (!args.force) {
      const message = inputMode === 'direct-api'
        ? '默认输出目录已有内容不同的 manifest。为避免覆盖已有确认决议，请提供该清单继续任务，或另行指定输出目录：'
        : '已有 manifest 的输入内容不同。为避免覆盖已有确认决议，请先归档输出目录，或显式传入 --force：';
      throw new Error(message + manifestPath);
    }
  }

  await fs.mkdir(outputDir, { recursive: true });
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
  process.stdout.write(JSON.stringify({
    manifest: manifestPath,
    fingerprint,
    status: existing ? 'replaced' : 'created'
  }, null, 2) + '\n');
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--api') result.apiPath = argv[++index];
    else if (arg === '--repo') result.repoPath = argv[++index];
    else if (arg === '--interface-file') result.interfaceFile = argv[++index];
    else if (arg === '--api-name') result.apiName = argv[++index];
    else if (arg === '--selected-interface') result.selectedInterfaceId = argv[++index];
    else if (arg === '--internal-reference') result.internalReferencePath = argv[++index];
    else if (arg === '--existing-document') result.existingDocumentPath = argv[++index];
    else if (arg === '--output-dir') result.outputDir = argv[++index];
    else if (arg === '--force') result.force = true;
    else if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    } else {
      throw new Error('未知参数: ' + arg);
    }
  }
  return result;
}

function usage() {
  console.log(
    '用法: node tools/ascendc-create-manifest.mjs --api <已选 API 信息或接口列表.json> ' +
    '[--selected-interface <接口 ID>] [--repo <代码仓>] [--internal-reference <内部资料.md>] ' +
    '[--existing-document <已有文档.md>] [--output-dir <目录>] [--force]\n' +
    '或: node tools/ascendc-create-manifest.mjs --repo <代码仓> --interface-file <接口文件> ' +
    '--api-name <API名称> [--selected-interface <候选 ID>] [--output-dir <目录>]'
  );
}

function resolveInputMode(args) {
  const hasSelectedApiInput = Boolean(args.apiPath);
  const hasDirectInput = Boolean(args.interfaceFile || args.apiName);
  if (hasSelectedApiInput && hasDirectInput) {
    throw new Error('--api 与 --interface-file/--api-name 不能同时使用');
  }
  if (hasSelectedApiInput) return 'selected-api';
  if (!args.interfaceFile || !args.apiName) {
    throw new Error('未传 --api 时，必须同时提供 --repo、--interface-file 和 --api-name');
  }
  return 'direct-api';
}

function requireValue(value, flag) {
  if (!value) throw new Error('缺少必需参数: ' + flag);
}

async function readJson(filePath, label) {
  const absolutePath = path.resolve(filePath);
  await assertFile(absolutePath, label);
  try {
    return JSON.parse(await fs.readFile(absolutePath, 'utf8'));
  } catch (error) {
    throw new Error(label + ' 无法解析为 JSON: ' + absolutePath + '；' + error.message);
  }
}

async function readOptionalJson(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    if (error && error.code === 'ENOENT') return null;
    throw error;
  }
}

function inputProjectRoot(raw) {
  return stringValue(raw?.projectRoot) || stringValue(raw?.detection?.projectRoot);
}

function inputReferenceMaterials(raw, selectedApi) {
  const candidates = [
    raw?.referenceMaterials,
    raw?.selection?.referenceMaterials,
    selectedApi?.referenceMaterials,
  ];
  return candidates.find(Array.isArray) ?? [];
}

function inputReferenceCollectionComplete(raw, selectedApi) {
  const candidates = [
    raw?.referenceCollectionComplete,
    raw?.selection?.referenceCollectionComplete,
    selectedApi?.referenceCollectionComplete,
  ];
  return candidates.find(value => typeof value === 'boolean') ?? false;
}

function resolveSelectedApi(raw, selectedInterfaceId) {
  const explicitApi = raw?.targetApi ?? raw?.selectedInterface ?? raw?.selectedApi;
  if (explicitApi) return explicitApi;

  const detection = Array.isArray(raw?.interfaces) ? raw : raw?.detection;
  if (!Array.isArray(detection?.interfaces)) return raw;

  const selectionId = stringValue(selectedInterfaceId) ||
    stringValue(raw?.selectedInterfaceId) ||
    stringValue(raw?.selectedApiId) ||
    stringValue(raw?.selection?.id) ||
    stringValue(raw?.selection?.interfaceId);
  if (!selectionId) {
    if (detection.interfaces.length === 1) return detection.interfaces[0];
    throw new Error('接口列表包含多个候选；请提供用户选择的接口 ID');
  }

  const selected = detection.interfaces.find(item => item && item.id === selectionId);
  if (!selected) throw new Error('未找到用户选择的接口 ID: ' + selectionId);
  return selected;
}

async function resolveDirectApi(args, repoPath) {
  const interfaceFile = path.isAbsolute(args.interfaceFile)
    ? path.resolve(args.interfaceFile)
    : path.resolve(repoPath, args.interfaceFile);
  if (!isInside(repoPath, interfaceFile)) {
    throw new Error('接口文件必须位于代码仓内: ' + interfaceFile);
  }
  await assertFile(interfaceFile, '接口文件');
  const requestedName = requiredString(args.apiName, '--api-name').replace(/\s+/g, '');
  const detected = detectInterfaces(interfaceFile, repoPath);
  const candidates = detected.interfaces.filter(candidate =>
    candidate?.name === requestedName ||
    (Array.isArray(candidate?.members) && candidate.members.some(member =>
      (typeof member === 'string' ? member : member?.name) === requestedName
    ))
  );
  if (candidates.length === 0) {
    const knownNames = [...new Set(detected.interfaces.flatMap(candidate => [
      candidate.name,
      ...(candidate.members || []).map(member => typeof member === 'string' ? member : member?.name)
    ]).filter(Boolean))].sort((a, b) => a.localeCompare(b));
    const suffix = knownNames.length > 0
      ? '；该文件可识别的接口: ' + knownNames.join(', ')
      : '；该文件中未识别到可用接口';
    throw new Error('在指定接口文件中未找到 API: ' + requestedName + suffix);
  }
  if (args.selectedInterfaceId) {
    const selected = candidates.find(candidate => candidate.id === args.selectedInterfaceId);
    if (!selected) {
      throw new Error('指定的候选接口不属于 API ' + requestedName + '：' + args.selectedInterfaceId + '\n' +
        formatDirectCandidates(candidates));
    }
    return selected;
  }
  if (candidates.length === 1) return candidates[0];
  throw new Error(
    'API ' + requestedName + ' 在指定接口文件中匹配到多个调用方式或 API 类型。' +
    '请确认后使用 --selected-interface <候选 ID> 重试：\n' + formatDirectCandidates(candidates)
  );
}

function formatDirectCandidates(candidates) {
  return candidates.map(candidate => JSON.stringify({
    id: candidate.id,
    name: candidate.name,
    apiType: candidate.apiType,
    callingConvention: candidate.callingConvention,
    members: (candidate.members || []).map(member => typeof member === 'string' ? member : member?.name),
    files: (candidate.files || []).map(file => typeof file === 'string' ? file : file.relativePath || file.path)
  })).join('\n');
}

async function normalizeTargetApi(raw, repoPath) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('targetApi 必须是对象');
  }
  const name = requiredString(raw.name, 'targetApi.name');
  const apiType = requiredString(raw.apiType, 'targetApi.apiType');
  const callingConvention = requiredString(raw.callingConvention, 'targetApi.callingConvention');
  const rawMembers = raw.members ?? raw.memberApis;
  if (!Array.isArray(rawMembers) || rawMembers.length === 0) {
    throw new Error('targetApi.members 必须是非空数组');
  }
  const members = uniqueMembers(rawMembers, name);
  const rawFiles = raw.files ?? raw.sourceFiles;
  if (!Array.isArray(rawFiles) || rawFiles.length === 0) {
    throw new Error('targetApi.files 必须是非空数组');
  }
  const files = [];
  for (const item of rawFiles) {
    const inputPath = typeof item === 'string' ? item : item?.path ?? item?.relativePath;
    if (!inputPath) throw new Error('targetApi.files 中存在无效文件项');
    const absolutePath = path.isAbsolute(inputPath)
      ? path.resolve(inputPath)
      : path.resolve(repoPath, inputPath);
    if (!isInside(repoPath, absolutePath)) {
      throw new Error('targetApi.files 必须位于代码仓内: ' + absolutePath);
    }
    await assertFile(absolutePath, 'targetApi 源文件');
    files.push(path.relative(repoPath, absolutePath).split(path.sep).join('/'));
  }
  return {
    id: stringValue(raw.id) || [apiType, callingConvention, name].join('|'),
    name,
    apiType,
    callingConvention,
    members,
    files: [...new Set(files)].sort((a, b) => a.localeCompare(b))
  };
}

function uniqueMembers(rawMembers, documentApiName) {
  const members = [];
  const seen = new Set();
  for (const item of rawMembers) {
    const name = typeof item === 'string' ? item : item?.name;
    if (!name || typeof name !== 'string') {
      throw new Error('targetApi.members 中存在无效成员');
    }
    if (seen.has(name)) continue;
    seen.add(name);
    const member = typeof item === 'string' ? { name } : { ...item, name };
    members.push(member);
  }
  if (!seen.has(documentApiName)) members.unshift({ name: documentApiName, relation: 'document-root' });
  return members;
}

async function resolveOutputDir(value, repoPath, documentApiName) {
  if (value) {
    const outputDir = path.resolve(value);
    if (!isInside(repoPath, outputDir)) {
      throw new Error('输出目录必须位于代码仓内: ' + outputDir);
    }
    return outputDir;
  }

  const outputRoot = path.join(repoPath, 'outputs');
  const outputDir = path.join(outputRoot, outputDirectoryName(documentApiName));
  const legacyOutputDir = path.join(outputRoot, legacyOutputDirectoryName(documentApiName));
  // Existing sessions keep their original directory so a naming-rule update
  // never silently forks or overwrites a previously confirmed manifest.
  const selectedOutputDir = !await isDirectory(outputDir) && await isDirectory(legacyOutputDir)
    ? legacyOutputDir
    : outputDir;
  if (!isInside(repoPath, selectedOutputDir)) {
    throw new Error('输出目录必须位于代码仓内: ' + selectedOutputDir);
  }
  return selectedOutputDir;
}

async function isDirectory(dirPath) {
  const stat = await fs.stat(dirPath).catch(() => null);
  return Boolean(stat?.isDirectory());
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative));
}

async function assertDirectory(dirPath, label) {
  const stat = await fs.stat(dirPath).catch(() => null);
  if (!stat?.isDirectory()) throw new Error(label + '不存在或不是目录: ' + dirPath);
}

async function assertFile(filePath, label) {
  const stat = await fs.stat(filePath).catch(() => null);
  if (!stat?.isFile()) throw new Error(label + '不存在或不是文件: ' + filePath);
}

async function readGitRevision(repoPath) {
  try {
    const { stdout } = await execFileAsync('git', ['-C', repoPath, 'rev-parse', 'HEAD'], {
      windowsHide: true
    });
    return stdout.trim() || null;
  } catch {
    return null;
  }
}

function requiredString(value, name) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(name + ' 必须是非空字符串');
  }
  return value.trim();
}

function stringValue(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : '';
}

main().catch(error => {
  console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
  process.exit(1);
});
