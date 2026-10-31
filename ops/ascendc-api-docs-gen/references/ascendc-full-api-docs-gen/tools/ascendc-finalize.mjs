#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFile } from 'node:child_process';
import { createRequire } from 'node:module';
import { promisify } from 'node:util';
import {
  activeProvidedReferences,
  assertFinalApiConfirmed,
  assertReferenceSources,
  assertV2Manifest,
} from './lib/reference-materials.mjs';
import { outputDirectoryNameCandidates } from './lib/output-directory-name.mjs';

const execFileAsync = promisify(execFile);

const require = createRequire(import.meta.url);
const { normalizeProductNames } = require('../scripts/lib/product-document-policy.js');
const { loadBoundaryContext } = require('../scripts/lib/public-document-boundary.js');
const { validateSourceSnapshot } = require('../scripts/lib/source-snapshot.js');
const { mapOutsideCodeBlocks } = require('../scripts/lib/markdown-resource-links.js');
const {
  normalizeInlineCodeOutsideFences,
  normalizePrototypeSemicolons,
  validateFinalDocument
} = require('../scripts/lib/final-document-validator.js');
const { buildApiRelationPlan } = require('../scripts/lib/ascendc-api-relations.js');
const { collectPublicContracts } = require('../scripts/extract-public-contracts.js');

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.documentPath) throw new Error('缺少必需参数: --document');
  const outputDir = await resolveOutputDir(args);
  const manifest = await readJson(path.join(outputDir, 'manifest.json'), 'manifest.json');
  assertManifest(manifest);
  await assertManifestFresh(manifest);

  const resolution = await readRequired(path.join(outputDir, 'resolution.md'));
  const codeEvidence = await readRequired(path.join(outputDir, 'code-evidence.md'));
  assertCodeEvidenceAudit(codeEvidence, manifest);
  const { exampleSafetyRestrictions, publicFactMappings } =
    assertFrozenResolution(resolution, manifest.fingerprint, codeEvidence);

  const documentPath = path.resolve(args.documentPath);
  if (!isInside(outputDir, documentPath)) {
    throw new Error('最终文档必须位于当前输出目录内: ' + documentPath);
  }
  const original = await readRequired(documentPath);
  const normalized = normalizeProductNames(normalizeDoc(normalizeInlineCodeOutsideFences(original)), 'ascendc');
  const finalDoc = normalizePrototypeSemicolons({
    apiName: manifest.documentApiName,
    memberApis: manifest.targetApi.members,
    prototypeDoc: codeEvidence,
    finalDoc: normalized,
    callingConvention: manifest.targetApi.callingConvention
  });
  const warnings = validateFinalDocument({
    apiName: manifest.documentApiName,
    memberApis: manifest.targetApi.members,
    prototypeDoc: codeEvidence,
    finalDoc,
    callingConvention: manifest.targetApi.callingConvention,
    exampleRestrictions: exampleSafetyRestrictions,
    publicFactMappings,
    localRoots: [manifest.sources.repo.path],
    ...loadBoundaryContext('ascendc', outputDir, documentPath)
  });
  for (const warning of warnings) console.warn('审阅提示：' + warning);

  if (args.checkOnly) {
    console.log('最终文档预检通过（未写入文件' + (warnings.length ? `；另有 ${warnings.length} 项审阅提示` : '') + '）: ' + documentPath);
    return;
  }
  await fs.writeFile(documentPath, finalDoc.trimEnd() + '\n', 'utf8');
  console.log('最终文档确定性校验通过' + (warnings.length ? `，另有 ${warnings.length} 项审阅提示` : '') + ': ' + documentPath);
}

function assertCodeEvidenceAudit(content, manifest) {
  const repoPath = path.resolve(manifest.sources.repo.path);
  const sourceFiles = (manifest.targetApi.files || []).map(relativePath => path.resolve(repoPath, relativePath));
  if (sourceFiles.length === 0) {
    throw new Error('manifest.json 缺少目标 API 源文件，无法核验代码处理记录');
  }
  const relationPlan = buildApiRelationPlan({
    repoPath,
    documentApiName: manifest.documentApiName,
    targetApi: manifest.targetApi
  });
  const closure = relationPlan.codeClosure;
  const fileRows = getRequiredTable(content, '源码处理记录', [
    '候选文件', '进入原因', '实际核验范围', '处理结果', '关联事实或缺口'
  ]);
  assertTerminalAuditRows(fileRows, '源码处理记录');
  const rowsByPath = new Map();
  for (const row of fileRows) {
    const filePath = normalizeAuditPath(row[0]);
    if (rowsByPath.has(filePath)) {
      throw new Error('源码处理记录存在重复候选文件: ' + filePath);
    }
    rowsByPath.set(filePath, row);
  }
  for (const candidate of closure.candidateFiles) {
    if (!rowsByPath.has(normalizeAuditPath(candidate.path))) {
      throw new Error('源码处理记录遗漏候选文件: ' + candidate.path);
    }
  }
  assertEvidenceClosureSummary(content, closure, rowsByPath);
  validateSourceSnapshot(manifest.sources.repo.evidenceFiles, [...rowsByPath.keys()].map(file => path.resolve(repoPath, file)));

  const publicContracts = collectPublicContracts({
    repoPath,
    sourceFiles,
    members: manifest.targetApi.members,
    callingConvention: manifest.targetApi.callingConvention
  });
  if (publicContracts.contracts.length === 0) return;
  const commentRows = getRequiredTable(content, '公开注释核验', [
    '注释定位', '所属成员和条件', '核验结果', '关联事实或原因'
  ]);
  assertTerminalAuditRows(commentRows, '公开注释核验');
  const rowsByLocation = new Map();
  for (const row of commentRows) {
    const location = normalizeAuditPath(row[0]);
    if (rowsByLocation.has(location)) {
      throw new Error('公开注释核验存在重复定位: ' + location);
    }
    rowsByLocation.set(location, row);
  }
  for (const contract of publicContracts.contracts) {
    const location = `${contract.source.file}:${contract.source.commentStartLine}-${contract.source.commentEndLine}`;
    if (!rowsByLocation.has(normalizeAuditPath(location))) {
      throw new Error('公开注释核验遗漏候选注释: ' + location);
    }
  }
}

function assertEvidenceClosureSummary(content, closure, rowsByPath) {
  const section = getSection(content, '自动索引与候选代码闭包');
  if (section == null) throw new Error('code-evidence.md 缺少“自动索引与候选代码闭包”章节');
  const declaredInitial = readEvidenceCount(section, '初始待核验源码文件数');
  const declaredAdded = readEvidenceCount(section, '阅读中新增直接证据链文件数');
  const declaredFinal = readEvidenceCount(section, '最终直接证据链文件数');
  const initialPaths = new Set((closure.candidateFiles || []).map(item => normalizeAuditPath(item.path)));
  const addedPaths = [...rowsByPath.keys()].filter(filePath => !initialPaths.has(filePath));
  if (declaredInitial !== initialPaths.size) {
    throw new Error('初始待核验源码文件数与 codeClosure.candidateFiles 不一致');
  }
  if (declaredAdded !== addedPaths.length) {
    throw new Error('阅读中新增直接证据链文件数与源码处理记录不一致');
  }
  if (declaredFinal !== rowsByPath.size) {
    throw new Error('最终直接证据链文件数与源码处理记录不一致');
  }
  if (addedPaths.length > 0) {
    const addedSection = getSection(content, '新增直接证据链文件');
    if (addedSection == null) {
      throw new Error('存在新增直接证据链文件时，code-evidence.md 必须单列“新增直接证据链文件”章节');
    }
    for (const filePath of addedPaths) {
      if (!containsExactToken(addedSection.replace(/\\/g, '/'), filePath)) {
        throw new Error('新增直接证据链文件章节遗漏: ' + filePath);
      }
    }
  }
}

function readEvidenceCount(section, label) {
  const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp('(?:^|\\n)\\s*-\\s*' + escaped + '：\\s*(\\d+)', 'u').exec(section);
  if (!match) throw new Error('code-evidence.md 缺少“' + label + '”统计项');
  return Number.parseInt(match[1], 10);
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

function getRequiredTable(content, heading, expectedHeader) {
  const section = getSection(content, heading);
  if (section == null) {
    throw new Error(`code-evidence.md 缺少“${heading}”章节`);
  }
  const lines = section.split(/\r?\n/);
  const headerIndex = lines.findIndex(line => {
    const cells = parseMarkdownRow(line);
    return cells.length === expectedHeader.length &&
      cells.every((cell, index) => cell === expectedHeader[index]);
  });
  if (headerIndex < 0 || !isMarkdownSeparator(lines[headerIndex + 1] || '')) {
    throw new Error(`code-evidence.md 的“${heading}”必须包含完整核验表`);
  }
  const rows = [];
  for (const line of lines.slice(headerIndex + 2)) {
    if (!/^\s*\|/u.test(line)) {
      if (rows.length > 0 && line.trim()) break;
      continue;
    }
    if (isMarkdownSeparator(line)) continue;
    rows.push(parseMarkdownRow(line));
  }
  if (rows.length === 0) {
    throw new Error(`code-evidence.md 的“${heading}”核验表不能为空`);
  }
  return rows;
}

function getSection(content, heading) {
  const escapedHeading = heading.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const match = new RegExp(`^## ${escapedHeading}\\s*$`, 'mu').exec(content);
  if (!match) return null;
  const afterHeading = content.slice(match.index + match[0].length);
  const nextHeading = afterHeading.search(/^##\s+/mu);
  return nextHeading < 0 ? afterHeading : afterHeading.slice(0, nextHeading);
}

function assertTerminalAuditRows(rows, label) {
  const terminalStates = new Set([
    '已核验并形成事实',
    '已核验，无当前 API 事实',
    '不适用'
  ]);
  for (const cells of rows) {
    if (cells.some(cell => !cell)) {
      throw new Error(`code-evidence.md 的“${label}”每行必须填写全部列`);
    }
    if (!terminalStates.has(cells[cells.length - 2])) {
      throw new Error(`code-evidence.md 的“${label}”存在未完成项：${cells[0]}（处理结果：${cells[cells.length - 2]}）`);
    }
  }
}

function normalizeAuditPath(value) {
  return value.trim().replace(/^`|`$/g, '').replace(/\\/g, '/');
}
function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--interface') result.interfaceName = argv[++index];
    else if (arg === '--output-dir') result.outputDir = argv[++index];
    else if (arg === '--document') result.documentPath = argv[++index];
    else if (arg === '--check') result.checkOnly = true;
    else if (arg === '--help' || arg === '-h') {
      usage();
      process.exit(0);
    } else if (!result.projectRoot) {
      result.projectRoot = arg;
    } else {
      throw new Error('未知参数: ' + arg);
    }
  }
  return result;
}

function usage() {
  console.log(
    '用法: node tools/ascendc-finalize.mjs <代码仓根目录> --interface <API名> ' +
    '--document <最终文档.md> [--check]\n' +
    '或: node tools/ascendc-finalize.mjs --output-dir <输出目录> --document <最终文档.md> [--check]\n' +
    '--check 只运行全部最终化校验与规范化预览，不写入文档。'
  );
}

async function resolveOutputDir(args) {
  if (args.outputDir) {
    const outputDir = path.resolve(args.outputDir);
    await assertDirectory(outputDir, '输出目录');
    return outputDir;
  }
  if (!args.projectRoot || !args.interfaceName) {
    throw new Error('未传 --output-dir 时，必须提供代码仓根目录和 --interface');
  }
  const outputRoot = path.join(path.resolve(args.projectRoot), 'outputs');
  for (const directoryName of outputDirectoryNameCandidates(args.interfaceName)) {
    const outputDir = path.join(outputRoot, directoryName);
    if (await isDirectory(outputDir)) return outputDir;
  }
  const expectedOutputDir = path.join(outputRoot, outputDirectoryNameCandidates(args.interfaceName)[0]);
  await assertDirectory(expectedOutputDir, '输出目录');
  return expectedOutputDir;
}

async function isDirectory(dirPath) {
  const stat = await fs.stat(dirPath).catch(() => null);
  return Boolean(stat?.isDirectory());
}

function assertManifest(manifest) {
  assertV2Manifest(manifest);
  assertFinalApiConfirmed(manifest);
  if (!manifest.fingerprint) {
    throw new Error('manifest.json 缺少本次任务的来源登记信息');
  }
  const target = manifest.targetApi;
  if (!manifest.documentApiName || !target?.callingConvention || !Array.isArray(target.members)) {
    throw new Error('manifest.json 缺少目标 API 信息');
  }
}

async function assertManifestFresh(manifest) {
  const sources = manifest.sources;
  assertReferenceSources(sources, { requireReady: true, requireClassified: true });
  const repo = sources?.repo;
  if (!repo?.path || !Array.isArray(repo.targetFiles)) {
    throw new Error('manifest.json 缺少可校验的代码来源指纹');
  }

  for (const targetFile of repo.targetFiles) {
    if (!targetFile?.path || !targetFile.sha256) {
      throw new Error('manifest.json 包含无效的目标源文件指纹');
    }
    const filePath = path.resolve(repo.path, targetFile.path);
    if (!isInside(repo.path, filePath)) {
      throw new Error('manifest.json 中的目标源文件越出代码仓: ' + filePath);
    }
    await assertFingerprint(filePath, targetFile.sha256, '目标源文件');
  }
  for (const reference of activeProvidedReferences(sources)) {
    await assertFingerprint(reference.path, reference.sha256, `资料“${reference.name}”`);
  }

  if (repo.revision) {
    const currentRevision = await readGitRevision(repo.path);
    if (currentRevision && currentRevision !== repo.revision) {
      throw new Error('代码仓 Git revision 已变化，必须重新创建会话并确认事实');
    }
  }
}

async function assertFingerprint(filePath, expectedHash, label) {
  if (!filePath || !expectedHash) {
    throw new Error('manifest.json 缺少' + label + '指纹');
  }
  let content;
  try {
    content = await fs.readFile(filePath);
  } catch {
    throw new Error(label + '不可读取，必须重新创建会话: ' + filePath);
  }
  const actualHash = crypto.createHash('sha256').update(content).digest('hex');
  if (actualHash !== expectedHash) {
    throw new Error(label + '已变化，必须重新创建会话并确认事实: ' + filePath);
  }
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

function assertFrozenResolution(content, manifestFingerprint, codeEvidence) {
  const metadata = parseFrontMatter(content);
  if (metadata.status !== 'frozen') {
    throw new Error('resolution.md 尚未冻结，不能生成最终文档');
  }
  if (metadata.unresolvedConflicts !== '0') {
    throw new Error('resolution.md 仍有未解决冲突，不能生成最终文档');
  }
  if (metadata.pendingReferenceApprovals !== '0') {
    throw new Error('resolution.md 仍有待确认的资料补充事实，不能生成最终文档');
  }
  if (metadata.manifestFingerprint !== manifestFingerprint) {
    throw new Error('resolution.md 对应的输入指纹已过期，必须重新确认');
  }
  assertNoPendingActiveDecision(content);
  assertCoverageComplete(content, codeEvidence);
  assertDownstreamApplicability(content);
  assertCallExampleBasis(content);
  const exampleSafetyRestrictions = assertExampleSafetyRestrictions(content);
  const publicFactMappings = assertPublicFactMappings(codeEvidence, content);
  return { exampleSafetyRestrictions, publicFactMappings };
}

function assertNoPendingActiveDecision(content) {
  const historyIndex = content.search(/^## 决策变更记录\s*$/mu);
  const activeContent = historyIndex < 0 ? content : content.slice(0, historyIndex);
  if (/状态\s*[:：]\s*(?:待确认|需二次确认)/u.test(activeContent)) {
    throw new Error('resolution.md 仍包含待确认或需二次确认的事实');
  }
}
function assertCoverageComplete(content, codeEvidence) {
  const section = getCoverageSection(content);
  if (section == null) {
    throw new Error('resolution.md 缺少“覆盖与待补证”章节');
  }

  const lines = section.split(/\r?\n/);
  const expectedHeader = ['槽位', '适用范围', '状态', '依据'];
  const headerIndex = lines.findIndex(line => {
    const cells = parseMarkdownRow(line);
    return cells.length === expectedHeader.length &&
      cells.every((cell, index) => cell === expectedHeader[index]);
  });
  if (headerIndex < 0 || !isMarkdownSeparator(lines[headerIndex + 1] || '')) {
    throw new Error('resolution.md 的“覆盖与待补证”必须包含逐槽位覆盖表');
  }

  const rows = [];
  for (const line of lines.slice(headerIndex + 2)) {
    if (!/^\s*\|/u.test(line)) {
      if (rows.length > 0 && line.trim()) break;
      continue;
    }
    if (isMarkdownSeparator(line)) continue;
    rows.push(parseMarkdownRow(line));
  }
  if (rows.length === 0) {
    throw new Error('resolution.md 的逐槽位覆盖表不能为空');
  }

  const terminalStates = new Set([
    '已由代码证实',
    '经用户确认并允许保留',
    '用户明确暂缓',
    '不适用'
  ]);
  for (const cells of rows) {
    if (cells.length !== 4 || cells.some(cell => !cell)) {
      throw new Error('逐槽位覆盖表每行必须填写槽位、适用范围、状态和依据');
    }
    if (!terminalStates.has(cells[2])) {
      throw new Error('覆盖表仍存在未完成槽位：' + cells[0] + '（状态：' + cells[2] + '）');
    }
    if (cells[2] === '不适用') {
      const ids = cells[3].match(/\bK-\d+\b/g) || [];
      const reason = cells[3].replace(/\bK-\d+\b/g, '').replace(/[\s`，,。；;：:()（）]/g, '').replace(/^(?:原因|依据)/u, '');
      if (!ids.length || /^(?:无|不适用|未知|待确认|待补齐|暂无依据|原因|依据)?$/u.test(reason)) {
        throw new Error('不适用槽位必须引用 K-ID 并说明具体原因：' + cells[0]);
      }
      const knownIds = new Set((codeEvidence || '').match(/\bK-\d+\b/g) || []);
      if (ids.some(id => !knownIds.has(id))) {
        throw new Error('不适用槽位引用了代码证据中不存在的 K-ID：' + cells[0]);
      }
    }
  }
}

function assertDownstreamApplicability(content) {
  const rows = getRequiredResolutionTable(content, '下层资料适用性', [
    '下层资料事实', '当前成员与条件', '代码判断', '对当前用户的处理', '依据'
  ]);
  const terminalActions = new Set([
    '写入当前 API 文档',
    '不写入当前 API 文档',
    '仅作为实现说明',
    '经用户确认写入当前 API 文档',
    '用户明确暂缓',
    '不适用'
  ]);
  for (const cells of rows) {
    if (cells.length !== 5 || cells.some((cell) => !cell)) {
      throw new Error('下层资料适用性表每行必须填写全部列');
    }
    const action = cells[3];
    if (!terminalActions.has(action)) {
      throw new Error('下层资料适用性表存在未完成处理：' + cells[0]);
    }
    if ((action === '写入当前 API 文档' || action === '经用户确认写入当前 API 文档') &&
        !/(?:K|C|S)-\d+/u.test(cells[4])) {
      throw new Error('写入当前 API 的下层资料事实必须记录代码或确认依据：' + cells[0]);
    }
  }
}

function assertCallExampleBasis(content) {
  const rows = getRequiredResolutionTable(content, '调用示例组成依据', [
    '示例部分', '使用依据', '允许使用的资料'
  ]);
  for (const cells of rows) {
    if (cells.length !== 3 || cells.some((cell) => !cell)) {
      throw new Error('调用示例组成依据表每行必须填写示例部分、使用依据和允许使用的资料');
    }
  }

  const sourceRows = rows.filter((cells) => cells[0] === '示例来源');
  if (sourceRows.length !== 1) {
    throw new Error('调用示例组成依据表必须且只能有一行“示例来源”');
  }

  const [_, basis, materials] = sourceRows[0];
  const hasCodeEvidence = /(?:K-\d+|代码证据)/u.test(materials);
  const hasReferenceEvidence = /R-\d+/u.test(materials);
  if (basis.includes('代码仓 examples')) {
    if (!hasCodeEvidence) {
      throw new Error('代码仓 examples 作为示例来源时必须记录代码证据');
    }
    return;
  }
  if (basis.includes('当前 API 资料')) {
    if (!hasReferenceEvidence) {
      throw new Error('当前 API 资料作为示例来源时必须记录 R-ID');
    }
    return;
  }
  if (basis.includes('当前 API 实现与同类 API 结构')) {
    if (!hasCodeEvidence) {
      throw new Error('当前 API 实现作为示例来源时必须记录代码证据');
    }
    return;
  }
  throw new Error('示例来源必须是代码仓 examples、当前 API 资料或当前 API 实现与同类 API 结构');
}

function assertPublicFactMappings(codeEvidence, resolution) {
  const candidateHeader = [
    '事实 ID', '类别', '候选对外事实', '适用范围',
    '来源层与透传关系', '建议正文位置', '必含标识符'
  ];
  const mappingHeader = [
    '事实 ID', '正文位置', '适用范围', '来源与透传关系',
    '必含标识符', '正文处理', '依据'
  ];
  const candidateRows = getRequiredTable(codeEvidence, '对外事实候选', candidateHeader);
  const mappingRows = getRequiredResolutionTable(resolution, '对外事实映射', mappingHeader);
  const candidatesAreNone = isAllNoneTable(candidateRows, candidateHeader.length);
  const mappingsAreNone = isAllNoneTable(mappingRows, mappingHeader.length);
  if (candidatesAreNone) {
    if (!mappingsAreNone) throw new Error('没有对外事实候选时，对外事实映射必须为“无”');
    return [];
  }
  if (mappingsAreNone) throw new Error('存在对外事实候选时，resolution.md 不能将对外事实映射写为“无”');

  const candidates = candidateRows.map(cells => {
    if (cells.length !== candidateHeader.length || cells.some(cell => !cell)) {
      throw new Error('对外事实候选表每行必须填写全部列');
    }
    if (!/^(?:K|C|S)-\d+$/u.test(cells[0])) {
      throw new Error('对外事实候选只能引用 K-ID、C-ID 或 S-ID，不能直接把下层 D-ID 写入正文：' + cells[0]);
    }
    return {
      factId: cells[0],
      scope: cells[3],
      source: cells[4],
      sections: splitPublicFactCells(cells[5]),
      requiredIdentifiers: splitPublicFactCells(cells[6])
    };
  });
  const mapped = new Map();
  for (const cells of mappingRows) {
    if (cells.length !== mappingHeader.length || cells.some(cell => !cell)) {
      throw new Error('对外事实映射表每行必须填写全部列');
    }
    const factId = cells[0];
    if (mapped.has(factId)) throw new Error('对外事实映射重复：' + factId);
    const disposition = cells[5];
    if (!['写入正文', '用户明确暂缓'].includes(disposition)) {
      throw new Error('对外事实映射的正文处理必须是“写入正文”或“用户明确暂缓”：' + factId);
    }
    mapped.set(factId, {
      factId,
      sections: splitPublicFactCells(cells[1]),
      scope: cells[2],
      source: cells[3],
      requiredIdentifiers: splitPublicFactCells(cells[4]),
      disposition,
      basis: cells[6]
    });
  }
  for (const candidate of candidates) {
    const mapping = mapped.get(candidate.factId);
    if (!mapping) throw new Error('对外事实候选缺少正文映射：' + candidate.factId);
    if (mapping.scope !== candidate.scope || mapping.source !== candidate.source) {
      throw new Error('对外事实映射必须保留候选事实的适用范围和透传关系：' + candidate.factId);
    }
    if (!sameStringSet(mapping.requiredIdentifiers, candidate.requiredIdentifiers)) {
      throw new Error('对外事实映射的必含标识符必须与候选一致：' + candidate.factId);
    }
    if (mapping.disposition === '写入正文' && !sameStringSet(mapping.sections, candidate.sections)) {
      throw new Error('写入正文的对外事实映射必须使用候选指定的正文位置：' + candidate.factId);
    }
    if (!(mapping.basis.match(/\b[KCRSD]-\d+\b/g) || []).includes(candidate.factId)) {
      throw new Error('对外事实映射的依据必须引用对应事实 ID：' + candidate.factId);
    }
  }
  for (const factId of mapped.keys()) {
    if (!candidates.some(candidate => candidate.factId === factId)) {
      throw new Error('对外事实映射包含不存在的候选事实：' + factId);
    }
  }
  return candidates.map(candidate => mapped.get(candidate.factId));
}

function containsExactToken(content, token) {
  const escaped = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp('(?<![\\w./\\\\-])' + escaped + '(?![\\w./\\\\-])', 'u').test(content);
}

function isAllNoneTable(rows, width) {
  return rows.length === 1 && rows[0].length === width && rows[0].every(cell => cell === '无');
}

function splitPublicFactCells(value) {
  return [...new Set(String(value).split(/[、,]/u).map(item => item.trim()).filter(Boolean))];
}

function sameStringSet(left, right) {
  const a = [...new Set(left || [])].sort();
  const b = [...new Set(right || [])].sort();
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function getRequiredResolutionTable(content, heading, expectedHeader) {
  const section = getSection(content, heading);
  if (section == null) {
    throw new Error('resolution.md 缺少“' + heading + '”章节');
  }
  const lines = section.split(/\r?\n/);
  const headerIndex = lines.findIndex((line) => {
    const cells = parseMarkdownRow(line);
    return cells.length === expectedHeader.length &&
      cells.every((cell, index) => cell === expectedHeader[index]);
  });
  if (headerIndex < 0 || !isMarkdownSeparator(lines[headerIndex + 1] || '')) {
    throw new Error('resolution.md 的“' + heading + '”必须包含完整表格');
  }
  const rows = [];
  for (const line of lines.slice(headerIndex + 2)) {
    if (!/^\s*\|/u.test(line)) {
      if (rows.length > 0 && line.trim()) break;
      continue;
    }
    if (isMarkdownSeparator(line)) continue;
    rows.push(parseMarkdownRow(line));
  }
  if (rows.length === 0) {
    throw new Error('resolution.md 的“' + heading + '”表不能为空');
  }
  return rows;
}

function assertExampleSafetyRestrictions(content) {
  const section = getSection(content, '示例使用限制');
  if (section == null) {
    throw new Error('resolution.md 缺少“示例使用限制”章节');
  }

  const expectedHeader = ['成员', '受限制对象', '示例只能使用', '不得使用', '校验写法', '依据'];
  const lines = section.split(/\r?\n/);
  const headerIndex = lines.findIndex(line => {
    const cells = parseMarkdownRow(line);
    return cells.length === expectedHeader.length &&
      cells.every((cell, index) => cell === expectedHeader[index]);
  });
  if (headerIndex < 0 || !isMarkdownSeparator(lines[headerIndex + 1] || '')) {
    throw new Error('resolution.md 的“示例使用限制”必须包含完整表格');
  }

  const rows = [];
  for (const line of lines.slice(headerIndex + 2)) {
    if (!/^\s*\|/u.test(line)) {
      if (rows.length > 0 && line.trim()) break;
      continue;
    }
    if (isMarkdownSeparator(line)) continue;
    rows.push(parseMarkdownRow(line));
  }
  if (rows.length === 0) {
    throw new Error('resolution.md 的“示例使用限制”表不能为空');
  }

  const restrictions = [];
  for (const cells of rows) {
    if (cells.length !== expectedHeader.length || cells.some(cell => !cell)) {
      throw new Error('示例使用限制表每行必须填写成员、受限制对象、允许范围、禁止条件、校验写法和依据');
    }
    const [member, subject, allowedRange, forbiddenConditions, checkExpressions, evidence] = cells;
    if (subject === '无') {
      if (checkExpressions !== '无') {
        throw new Error('受限制对象为“无”时，校验写法必须为“无”');
      }
      continue;
    }
    if (/^(?:无|不适用|未知|待确认|待补齐|正常(?:值)?|合适(?:值)?|安全(?:值)?)$/u.test(allowedRange)) {
      throw new Error(`示例使用限制缺少具体允许范围：${member}（${subject}）`);
    }
    if (/^(?:无|不适用)$/u.test(forbiddenConditions)) {
      throw new Error(`示例使用限制缺少不得使用的条件：${member}（${subject}）`);
    }
    restrictions.push({
      member,
      subject,
      allowedRange,
      forbiddenConditions,
      staticPatterns: parseExampleSafetyPatterns(checkExpressions, member, subject),
      evidence
    });
  }
  return restrictions;
}

function parseExampleSafetyPatterns(value, member, subject) {
  if (value === '无' || value === '人工审阅') return [];
  const patterns = [...value.matchAll(/`([^`\r\n]+)`/g)]
    .map(match => match[1].trim())
    .filter(Boolean);
  if (patterns.length === 0) {
    throw new Error(`示例使用限制的校验写法必须使用反引号列出，或写“人工审阅”：${member}（${subject}）`);
  }
  return [...new Set(patterns)];
}

function getCoverageSection(content) {
  const match = /^## 覆盖与待补证\s*$/mu.exec(content);
  if (!match) return null;
  const afterHeading = content.slice(match.index + match[0].length);
  const nextHeading = afterHeading.search(/^##\s+/mu);
  return nextHeading < 0 ? afterHeading : afterHeading.slice(0, nextHeading);
}

function parseMarkdownRow(line) {
  return line
    .trim()
    .replace(/^\|/u, '')
    .replace(/\|$/u, '')
    .split('|')
    .map(cell => cell.trim());
}

function isMarkdownSeparator(line) {
  return /^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$/u.test(line);
}
function parseFrontMatter(content) {
  const match = content.match(/^---\r?\n([\s\S]*?)\r?\n---/u);
  if (!match) throw new Error('resolution.md 缺少 YAML front matter');
  const result = {};
  for (const line of match[1].split(/\r?\n/)) {
    const entry = line.match(/^([A-Za-z][A-Za-z0-9]*):\s*(.*?)\s*$/u);
    if (entry) result[entry[1]] = entry[2];
  }
  return result;
}

async function readJson(filePath, label) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (error) {
    throw new Error('无法读取 ' + label + ': ' + error.message);
  }
}

async function readRequired(filePath) {
  try {
    return await fs.readFile(filePath, 'utf8');
  } catch {
    throw new Error('缺少必需文件: ' + filePath);
  }
}

async function assertDirectory(dirPath, label) {
  const stat = await fs.stat(dirPath).catch(() => null);
  if (!stat?.isDirectory()) throw new Error(label + '不存在或不是目录: ' + dirPath);
}

function isInside(parent, child) {
  const relative = path.relative(parent, child);
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative));
}

function normalizeDoc(content) {
  return mapOutsideCodeBlocks(content, text => text
    .replace(/\r\n/g, '\n')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n{3,}/g, '\n\n'));
}

export { assertFrozenResolution, assertCodeEvidenceAudit, assertNoPendingActiveDecision, containsExactToken, assertPublicFactMappings };

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
    process.exitCode = 1;
  });
}
