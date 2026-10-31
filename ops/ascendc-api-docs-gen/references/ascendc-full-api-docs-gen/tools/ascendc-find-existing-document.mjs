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
import { createRequire } from 'node:module';
import { assertFinalApiConfirmed, assertV2Manifest, providedReferences } from './lib/reference-materials.mjs';

const require = createRequire(import.meta.url);
const { parseAscendcSignaturesFromContent } = require('../scripts/lib/ascendc-signatures.js');
const { apiFamilyFromPath, matchesTargetSelection } = require('../scripts/lib/ascendc-api-relations.js');

const DOCUMENT_EXTENSIONS = new Set(['.md', '.markdown', '.mdx']);
const CODE_EXTENSIONS = new Set(['.h', '.hpp', '.hh', '.cpp', '.cc', '.cxx']);
const MAX_CONTENT_BYTES = 256 * 1024;
const DOCUMENT_TEXT_CACHE = new Map();

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.manifestPath) throw new Error('缺少必需参数: --manifest');

  const manifestPath = path.resolve(args.manifestPath);
  const manifest = await readManifest(manifestPath);
  assertV2Manifest(manifest);
  assertFinalApiConfirmed(manifest);

  const repoRoot = requiredRepoRoot(manifest);
  const docsRoot = path.join(repoRoot, 'docs');
  const documentApiName = manifest.documentApiName;
  const currentTargets = targetNames(manifest);
  const apiFamily = targetApiFamily(manifest);
  const relationPlan = args.relationPlanPath
    ? await readRelationPlan(args.relationPlanPath, manifest)
    : null;
  const downstreamRelations = relationPlan?.downstreamApis || [];
  const directDownstreamRelations = downstreamRelations.filter((relation) =>
    relation?.kind === "direct-call" && relation?.documentCandidate !== false);
  const macroMappingGroups = relationPlan?.macroMappingGroups || [];
  const currentSpecs = targetSpecs(currentTargets, apiFamily);
  const peerNames = (await familyApiNames(repoRoot, apiFamily))
    .filter((name) => !currentTargets.includes(name));
  const peerSpecs = targetSpecs(peerNames, apiFamily);
  const directDownstreamSpecs = targetSpecsForRelations(directDownstreamRelations);
  const topicTokens = targetTopicTokens(manifest);
  const maxPeerResults = args.maxPeerResults ?? args.maxResults ?? 3;
  const maxCurrentResults = args.maxCurrentResults ?? args.maxResults ?? 3;
  const maxDownstreamResults = args.maxDownstreamResults ?? args.maxResults ?? 6;
  const maxMacroCandidateResults = Math.max(3, maxPeerResults * 3);
  const docsStat = await fs.stat(docsRoot).catch(() => null);

  if (!docsStat?.isDirectory()) {
    writeResult(emptyResult({
      manifestPath,
      repoRoot,
      documentApiName,
      apiFamily,
      relationPlanPath: args.relationPlanPath || null
    }));
    return;
  }

  const files = [];
  await collectDocumentFiles(docsRoot, files);
  const excluded = new Set(providedReferences(manifest.sources).map((reference) => normalizePath(reference.path)));
  const currentApiCandidates = [];
  const peerApiCandidates = [];
  const downstreamApiCandidates = [];

  for (const filePath of files) {
    if (excluded.has(normalizePath(filePath)) || isExcludedDocument(filePath, repoRoot)) continue;

    const current = await candidateForFile({
      filePath,
      repoRoot,
      targetSpecs: currentSpecs,
      documentApiName,
      role: "current-api",
      topicTokens
    });
    if (current) currentApiCandidates.push(current);

    const peer = await candidateForFile({
      filePath,
      repoRoot,
      targetSpecs: peerSpecs,
      documentApiName,
      role: "peer-api",
      topicTokens
    });
    if (peer) peerApiCandidates.push(peer);

    const downstream = await candidateForFile({
      filePath,
      repoRoot,
      targetSpecs: directDownstreamSpecs,
      documentApiName,
      role: "downstream-api",
      topicTokens: []
    });
    if (downstream) {
      const matchingRelations = relationSummaries(
        directDownstreamRelations.filter((relation) => downstream.matchedApis.includes(relation?.lowerApi?.name))
      );
      downstreamApiCandidates.push({ ...downstream, relations: matchingRelations });
    }
  }

  const macroMappingGroupCandidates = [];
  for (const group of macroMappingGroups) {
    const groupCandidates = [];
    const groupSpecs = targetSpecsForLowerApis(group.lowerApis || []);
    const groupRelations = downstreamRelations.filter((relation) =>
      (group.relationIds || []).includes(relation.id));
    for (const filePath of files) {
      if (excluded.has(normalizePath(filePath)) || isExcludedDocument(filePath, repoRoot)) continue;
      const candidate = await candidateForFile({
        filePath,
        repoRoot,
        targetSpecs: groupSpecs,
        documentApiName,
        role: "macro-mapping-group",
        topicTokens: []
      });
      if (!candidate) continue;
      const relations = relationSummaries(groupRelations.filter((relation) =>
        candidate.matchedApis.includes(relation?.lowerApi?.name)));
      if (relations.length > 0) groupCandidates.push({ ...candidate, relations });
    }
    sortCandidates(groupCandidates);
    macroMappingGroupCandidates.push(compactMacroMappingGroup(
      group,
      groupRelations,
      groupCandidates.slice(0, maxMacroCandidateResults)
    ));
  }

  sortCandidates(currentApiCandidates);
  sortCandidates(peerApiCandidates);
  sortCandidates(downstreamApiCandidates);
  const groups = {
    currentApiCandidates: currentApiCandidates.slice(0, maxCurrentResults).map(stripScore),
    peerApiCandidates: peerApiCandidates.slice(0, maxPeerResults).map(stripScore),
    downstreamApiCandidates: downstreamApiCandidates.slice(0, maxDownstreamResults).map(stripScore),
    macroMappingGroupCandidates
  };
  const macroCandidates = macroMappingGroupCandidates.flatMap((group) =>
    group.candidateDocuments.map((candidate) => ({ ...candidate, mappingGroupId: group.id })));
  const candidates = [
    ...groups.currentApiCandidates,
    ...groups.peerApiCandidates,
    ...groups.downstreamApiCandidates,
    ...macroCandidates
  ];

  writeResult({
    manifest: manifestPath,
    repoRoot,
    docsDirectory: docsRoot,
    documentApiName,
    apiFamily,
    relationPlan: args.relationPlanPath ? path.resolve(args.relationPlanPath) : null,
    candidateOnly: true,
    checkedFileCount: files.length,
    candidateCount: candidates.length,
    groups,
    candidates
  });
}

function emptyResult({ manifestPath, repoRoot, documentApiName, apiFamily, relationPlanPath }) {
  return {
    manifest: manifestPath,
    repoRoot,
    docsDirectory: null,
    documentApiName,
    apiFamily,
    relationPlan: relationPlanPath ? path.resolve(relationPlanPath) : null,
    candidateOnly: true,
    checkedFileCount: 0,
    candidateCount: 0,
    groups: {
      currentApiCandidates: [],
      peerApiCandidates: [],
      downstreamApiCandidates: [],
      macroMappingGroupCandidates: []
    },
    candidates: []
  };
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--manifest') result.manifestPath = argv[++index];
    else if (arg === '--relation-plan') result.relationPlanPath = argv[++index];
    else if (arg === '--max-peer-results') result.maxPeerResults = parseMaxResults(argv[++index]);
    else if (arg === '--max-results') result.maxResults = parseMaxResults(argv[++index]);
    else if (arg === '--help' || arg === '-h') {
      console.log('用法: node tools/ascendc-find-existing-document.mjs --manifest <manifest.json> [--relation-plan <临时调用关系结果.json>] [--max-peer-results <同类候选数量>]；仅返回精确锚定的当前 API、同类 API 和直接下层 API 文档；宏表映射按实现族分组，供模型审阅后推荐，兼容 --max-results。');
      process.exit(0);
    } else {
      throw new Error('未知参数: ' + arg);
    }
  }
  return result;
}

function parseMaxResults(value) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1 || parsed > 50) {
    throw new Error('候选数量必须是 1 到 50 的整数');
  }
  return parsed;
}

async function readManifest(manifestPath) {
  const stat = await fs.stat(manifestPath).catch(() => null);
  if (!stat?.isFile()) throw new Error('manifest.json 不存在或不是文件: ' + manifestPath);
  try {
    return JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  } catch (error) {
    throw new Error('manifest.json 无法解析: ' + manifestPath + '；' + error.message);
  }
}

async function readRelationPlan(value, manifest) {
  const relationPlanPath = path.resolve(value);
  let plan;
  try {
    plan = JSON.parse(await fs.readFile(relationPlanPath, 'utf8'));
  } catch (error) {
    throw new Error('临时调用关系定位结果无法读取: ' + relationPlanPath + '；' + error.message);
  }
  if (!matchesTargetSelection(plan, { repoPath: requiredRepoRoot(manifest),
    documentApiName: manifest.documentApiName, targetApi: manifest.targetApi })) {
    throw new Error('临时调用关系定位结果与当前 manifest.json 不匹配');
  }
  return plan;
}

function requiredRepoRoot(manifest) {
  const repoPath = manifest.sources?.repo?.path;
  if (typeof repoPath !== 'string' || repoPath.trim() === '') {
    throw new Error('manifest.json 缺少代码仓路径');
  }
  return path.resolve(repoPath);
}

function targetNames(manifest) {
  const names = [manifest.documentApiName, manifest.targetApi?.name];
  for (const member of manifest.targetApi?.members || []) names.push(member?.name);
  const result = unique(names);
  if (result.length === 0) throw new Error('manifest.json 缺少文档 API 或成员函数');
  return result;
}

function targetApiFamily(manifest) {
  const explicit = manifest.targetApi?.apiType;
  if (typeof explicit === 'string' && explicit && explicit !== 'unknown') return explicit;
  for (const file of manifest.targetApi?.files || []) {
    const family = apiFamilyFromPath(file);
    if (family !== 'unknown') return family;
  }
  return 'unknown';
}

async function familyApiNames(repoRoot, family) {
  if (!family || family === 'unknown') return [];
  const files = [];
  await collectCodeFiles(path.join(repoRoot, 'include', family), files);
  const names = [];
  for (const filePath of files) {
    const text = await fs.readFile(filePath, 'utf8').catch(() => '');
    for (const signature of parseAscendcSignaturesFromContent(text)) names.push(signature.funcName);
  }
  return unique(names);
}

async function collectCodeFiles(directory, files) {
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await collectCodeFiles(entryPath, files);
    } else if (entry.isFile() && CODE_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      files.push(entryPath);
    }
  }
}

async function collectDocumentFiles(directory, files) {
  const entries = await fs.readdir(directory, { withFileTypes: true }).catch(() => []);
  for (const entry of entries) {
    if (entry.name === '.git' || entry.name === 'outputs') continue;
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await collectDocumentFiles(entryPath, files);
    } else if (entry.isFile() && DOCUMENT_EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      files.push(entryPath);
    }
  }
}

async function candidateForFile({ filePath, repoRoot, targetSpecs, documentApiName, role, topicTokens }) {
  if (!targetSpecs.length) return null;
  const extension = path.extname(filePath);
  const stem = path.basename(filePath, extension);
  const relativePath = path.relative(repoRoot, filePath).replace(/\\/g, "/");
  const documentFamily = documentApiFamily(relativePath);
  if (documentFamily === "unknown") return null;
  const text = await readCachedTextPrefix(filePath);
  const headings = headingLines(text);
  const reasons = [];
  const matchingHeadings = [];
  const matchedApis = [];
  const anchorKinds = [];
  let score = 0;

  for (const spec of targetSpecs) {
    let targetScore = 0;
    if (!spec.apiFamilies.includes(documentFamily)) continue;
    const anchor = documentAnchor({ text, headings, stem, identifier: spec.name });
    if (!anchor.matched) continue;
    matchedApis.push(spec.name);
    const isDocumentApi = spec.name === documentApiName;
    if (anchor.sameStem) {
      targetScore += isDocumentApi ? 220 : 180;
      reasons.push(isDocumentApi ? "文件名与当前 API 一致" : "文件名与候选 API 一致");
      anchorKinds.push("file-name");
    }
    if (anchor.h1Match) {
      targetScore += isDocumentApi ? 160 : 130;
      reasons.push(isDocumentApi ? "一级标题含当前 API" : "一级标题含候选 API");
      anchorKinds.push("h1");
    }
    if (anchor.sectionMatch) {
      targetScore += isDocumentApi ? 100 : 80;
      reasons.push(isDocumentApi ? "明确章节含当前 API" : "明确章节含候选 API");
      anchorKinds.push("api-section");
    }
    if (anchor.prototypeMatch) {
      targetScore += isDocumentApi ? 75 : 60;
      reasons.push(isDocumentApi ? "文档中有当前 API 的调用原型" : "文档中有候选 API 的调用原型");
      anchorKinds.push("prototype");
    }
    for (const heading of anchor.headings) {
      if (matchingHeadings.length < 3) matchingHeadings.push(trimDisplay(heading));
    }
    score = Math.max(score, targetScore);
  }

  if (score === 0) return null;
  const overlap = topicOverlap(relativePath, topicTokens);
  score += overlap * 12;
  if (overlap > 0) reasons.push("文档路径与当前代码主题相近");
  return {
    role,
    path: filePath,
    relativePath,
    documentApiFamily: documentFamily,
    matchedApis: unique(matchedApis),
    reasons: unique(reasons),
    matchingHeadings: unique(matchingHeadings),
    anchorKinds: unique(anchorKinds),
    topicOverlap: overlap,
    score
  };
}

function documentAnchor({ text, headings, stem, identifier }) {
  const sameStem = normalizeIdentifier(stem) === normalizeIdentifier(identifier);
  const h1 = headings.filter((heading) => heading.level === 1);
  const sections = headings.filter((heading) => heading.level >= 2 && heading.level <= 3);
  const h1Match = h1.some((heading) => containsIdentifier(heading.text, identifier));
  const matchingSections = sections.filter((heading) => containsIdentifier(heading.text, identifier));
  const prototypeMatch = documentedPrototypeMatch(text, identifier);
  return {
    matched: sameStem || h1Match || matchingSections.length > 0 || prototypeMatch,
    sameStem,
    h1Match,
    sectionMatch: matchingSections.length > 0,
    prototypeMatch,
    headings: [
      ...h1.filter((heading) => containsIdentifier(heading.text, identifier)),
      ...matchingSections
    ].map((heading) => heading.raw)
  };
}

function documentedPrototypeMatch(text, identifier) {
  const prototype = new RegExp("\\b" + escapeRegExp(identifier) + "\\s*(?:<[^\\n(){};]*>)?\\s*\\(", "ig");
  let match;
  while ((match = prototype.exec(text)) !== null) {
    const before = text.slice(Math.max(0, match.index - 2400), match.index);
    const precedingHeadings = headingLines(before);
    const nearestHeading = precedingHeadings[precedingHeadings.length - 1];
    if (nearestHeading && /(?:函数原型|接口原型|函数声明|function\s+prototype|prototype)/i.test(nearestHeading.text)) {
      return true;
    }
  }
  return false;
}

function headingLines(text) {
  return text.split(/\r?\n/)
    .map((line) => /^(#{1,6})\s+(.+?)\s*$/.exec(line))
    .filter(Boolean)
    .map((match) => ({ level: match[1].length, text: match[2], raw: match[0] }));
}

function documentApiFamily(relativePath) {
  const segments = relativePath.toLowerCase().split("/");
  const found = segments.find((segment) => /^[a-z0-9]+_api$/.test(segment));
  return found || "unknown";
}

function targetSpecs(names, apiFamily) {
  if (!apiFamily || apiFamily === "unknown") return [];
  return unique(names).map((name) => ({ name, apiFamilies: [apiFamily.toLowerCase()] }));
}

function targetSpecsForRelations(relations) {
  return targetSpecsForLowerApis(relations.map((relation) => relation?.lowerApi));
}

function targetSpecsForLowerApis(lowerApis) {
  const result = [];
  const seen = new Set();
  for (const lowerApi of lowerApis) {
    if (!lowerApi?.name || !lowerApi?.apiFamily || lowerApi.apiFamily === "unknown") continue;
    const key = lowerApi.name + "|" + lowerApi.apiFamily.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push({ name: lowerApi.name, apiFamilies: [lowerApi.apiFamily.toLowerCase()] });
  }
  return result;
}

function targetTopicTokens(manifest) {
  const ignored = new Set([
    "include", "impl", "docs", "doc", "api", "tensor", "basic", "simd", "c", "arch",
    "experimental", "vector", "matrix", "asc", "devkit", "h", "hpp", "md", "zh", "en"
  ]);
  const values = [manifest.documentApiName, ...(manifest.targetApi?.files || [])];
  const tokens = [];
  for (const value of values) {
    for (const token of String(value).toLowerCase().split(/[^a-z0-9]+/)) {
      if (token.length >= 3 && !ignored.has(token)) tokens.push(token);
    }
  }
  return unique(tokens);
}

function topicOverlap(relativePath, topicTokens) {
  if (!topicTokens?.length) return 0;
  const fileTokens = new Set(relativePath.toLowerCase().split(/[^a-z0-9]+/));
  return topicTokens.filter((token) => fileTokens.has(token)).length;
}

function isExcludedDocument(filePath, repoRoot) {
  const relativePath = path.relative(repoRoot, filePath).replace(/\\/g, "/").toLowerCase();
  const segments = relativePath.split("/");
  const stem = path.basename(relativePath, path.extname(relativePath));
  if (["readme", "changelog", "change_log", "release_notes", "release-note", "release-notes"].includes(stem)) return true;
  const family = documentApiFamily(relativePath);
  if (family !== "unknown" && normalizeIdentifier(stem) === normalizeIdentifier(family)) return true;
  return segments.some((segment) => [
    "appendix", "appendices", "deprecated", "deprecated_interface", "release_notes", "release-note", "release-notes"
  ].includes(segment));
}

function relationSummaries(relations) {
  return relations.map((relation) => ({
    relationId: relation.id,
    mappingGroupId: relation.mappingGroupId || null,
    targetMember: relation.targetMember,
    relationKind: relation.kind,
    lowerApi: relation.lowerApi,
    conditions: relation.conditions,
    source: relation.source,
    route: relation.route,
    resolution: relation.resolution
  }));
}

function compactMacroMappingGroup(group, relations, candidates) {
  const variants = new Map();
  for (const relation of relations) {
    const expressions = relation.conditions?.expressions || [];
    const architectures = relation.conditions?.architectures || [];
    const key = JSON.stringify({ expressions, architectures });
    const previous = variants.get(key) || { expressions, architectures, relationCount: 0 };
    previous.relationCount += 1;
    variants.set(key, previous);
  }
  const sourceFiles = unique(relations.flatMap((relation) => [
    relation.source?.file,
    relation.source?.macroDefinition?.file
  ]));
  return {
    id: group.id,
    targetMembers: group.targetMembers,
    implementationFamily: group.implementationFamily,
    macroNames: group.macroNames,
    relationCount: group.relationCount,
    lowerApiCount: group.lowerApiCount,
    conditions: {
      architectures: unique(relations.flatMap((relation) => relation.conditions?.architectures || [])),
      hasMappingsWithoutArchitecture: relations.some((relation) =>
        !(relation.conditions?.architectures || []).length),
      variants: [...variants.values()]
    },
    sourceFiles,
    documentReview: group.documentReview,
    candidateDocumentCount: candidates.length,
    candidateDocuments: candidates.map(compactMacroCandidate)
  };
}

function compactMacroCandidate(candidate) {
  const { relations, ...document } = stripScore(candidate);
  const mappedLowerApis = unique(relations.map((relation) => relation.lowerApi?.name));
  const architectures = unique(relations.flatMap((relation) => relation.conditions?.architectures || []));
  return {
    ...document,
    mappedLowerApis,
    relationCount: relations.length,
    architectures,
    hasMappingsWithoutArchitecture: relations.some((relation) =>
      !(relation.conditions?.architectures || []).length)
  };
}

async function readCachedTextPrefix(filePath) {
  const key = normalizePath(filePath);
  if (!DOCUMENT_TEXT_CACHE.has(key)) {
    DOCUMENT_TEXT_CACHE.set(key, readTextPrefix(filePath).catch(() => ""));
  }
  return DOCUMENT_TEXT_CACHE.get(key);
}

async function readTextPrefix(filePath) {
  const handle = await fs.open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(MAX_CONTENT_BYTES);
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0);
    return buffer.subarray(0, bytesRead).toString('utf8');
  } finally {
    await handle.close();
  }
}

function sortCandidates(candidates) {
  candidates.sort((left, right) => right.score - left.score || left.relativePath.localeCompare(right.relativePath));
}

function stripScore(candidate) {
  const { score, ...rest } = candidate;
  return rest;
}

function containsIdentifier(text, identifier) {
  return new RegExp('(^|[^A-Za-z0-9_])' + escapeRegExp(identifier) + '($|[^A-Za-z0-9_])', 'i').test(text);
}

function escapeRegExp(value) {
  return value.replace(/[|\\{}()[\]^$+*?.-]/g, '\\$&');
}

function normalizeIdentifier(value) {
  return value.toLowerCase().replace(/[^a-z0-9_]/g, '');
}

function normalizePath(value) {
  return path.resolve(value).toLowerCase();
}

function trimDisplay(value) {
  return value.length > 180 ? value.slice(0, 177) + '...' : value;
}

function unique(values) {
  return [...new Set(values.filter((value) => typeof value === 'string' && value.trim()).map((value) => value.trim()))];
}

function writeResult(result) {
  process.stdout.write(JSON.stringify({ kind: 'ascendc_existing_document_candidates', ...result }, null, 2) + '\n');
}

main().catch(error => {
  console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
  process.exit(1);
});
