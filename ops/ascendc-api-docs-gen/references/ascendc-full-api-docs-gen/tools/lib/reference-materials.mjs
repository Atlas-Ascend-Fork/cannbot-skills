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

export const MANIFEST_SCHEMA_VERSION = 2;

const ACCEPTED_MANIFEST_KINDS = new Set(['ascendc_doc_manifest', 'ascendc_doc_session']);

export const DEFAULT_REFERENCE_MATERIALS = Object.freeze([
  Object.freeze({ id: 'internal-reference', name: '内部脱敏资料' }),
  Object.freeze({ id: 'existing-document', name: '仓内已有人工文档' }),
]);

const REFERENCE_STATUSES = new Set(['pending', 'provided', 'absent', 'unreadable']);
const DOCUMENT_TARGET_KINDS = new Set([
  'unclassified',
  'current-api',
  'mixed-api',
  'other-api',
  'peer-api',
  'downstream-api',
  'needs-confirmation',
  'excluded',
]);
const DOCUMENT_TARGET_HANDLING = new Map([
  ['current-api', 'facts'],
  ['mixed-api', 'selected-sections-only'],
  ['other-api', 'comparison-only'],
  ['peer-api', 'example-template'],
  ['downstream-api', 'dependency-evidence'],
  ['excluded', 'excluded'],
]);

export function sha256Text(text) {
  return crypto.createHash('sha256').update(text).digest('hex');
}

export async function sha256File(filePath) {
  return crypto.createHash('sha256').update(await fs.readFile(filePath)).digest('hex');
}

export function isDefaultReferenceId(id) {
  return DEFAULT_REFERENCE_MATERIALS.some((reference) => reference.id === id);
}

export function defaultReferenceName(id) {
  return DEFAULT_REFERENCE_MATERIALS.find((reference) => reference.id === id)?.name;
}

function targetApiId(targetApi) {
  return typeof targetApi?.id === 'string' && targetApi.id.trim() ? targetApi.id.trim() : null;
}

export function createPendingFinalApiConfirmation(targetApi) {
  const apiId = targetApiId(targetApi);
  if (!apiId) throw new Error('targetApi.id 不能为空，无法记录最终文档接口确认');
  return { status: 'pending', apiId };
}

export function assertFinalApiConfirmation(manifest) {
  const confirmation = manifest?.finalApiConfirmation;
  const apiId = targetApiId(manifest?.targetApi);
  if (!apiId) throw new Error('manifest.json 缺少 targetApi.id');
  if (!confirmation || typeof confirmation !== 'object' || Array.isArray(confirmation)) {
    throw new Error('manifest.json 缺少最终文档接口确认记录');
  }
  if (!['pending', 'confirmed'].includes(confirmation.status)) {
    throw new Error('最终文档接口确认状态无效：' + confirmation.status);
  }
  if (confirmation.apiId !== apiId) {
    throw new Error('最终文档接口确认记录与 targetApi.id 不一致');
  }
  if (confirmation.status === 'confirmed' &&
      (typeof confirmation.confirmedAt !== 'string' || !confirmation.confirmedAt.trim())) {
    throw new Error('已确认的最终文档接口必须记录确认时间');
  }
  if (confirmation.status === 'pending' && Object.hasOwn(confirmation, 'confirmedAt')) {
    throw new Error('待确认的最终文档接口不能记录确认时间');
  }
  return confirmation;
}

export function assertFinalApiConfirmed(manifest) {
  const confirmation = assertFinalApiConfirmation(manifest);
  if (confirmation.status !== 'confirmed') {
    throw new Error('尚未确认本次最终要生成文档的接口，不能继续分析代码或资料');
  }
  return confirmation;
}

export function confirmFinalApi(manifest, apiId, confirmedAt = new Date().toISOString()) {
  const expected = targetApiId(manifest?.targetApi);
  if (!expected) throw new Error('manifest.json 缺少 targetApi.id');
  if (apiId !== expected) {
    throw new Error('确认的接口 ID 与 manifest.json 的 targetApi.id 不一致');
  }
  manifest.finalApiConfirmation = { status: 'confirmed', apiId: expected, confirmedAt };
  assertFinalApiConfirmation(manifest);
  return manifest.finalApiConfirmation;
}

export function providedReferences(sources) {
  if (!sources || !Array.isArray(sources.references)) {
    return [];
  }
  return sources.references.filter((reference) => reference.status === 'provided');
}

// 资料路径已提供，但用户明确选择“不使用”时，它不应再影响最终文档的有效性。
export function activeProvidedReferences(sources) {
  return providedReferences(sources).filter((reference) => documentTargetOf(reference).kind !== 'excluded');
}

export function documentTargetOf(reference) {
  return reference?.documentTarget || { kind: 'unclassified' };
}

export function unresolvedDocumentTargets(sources) {
  return activeProvidedReferences(sources).filter((reference) => {
    const kind = documentTargetOf(reference).kind;
    return kind === 'unclassified' || kind === 'needs-confirmation';
  });
}

export function unresolvedReferences(sources) {
  const references = Array.isArray(sources?.references) ? sources.references : [];
  return references.filter((reference) => reference.status !== 'provided' && reference.status !== 'absent');
}

export function assertReferenceSources(sources, { requireReady = false, requireClassified = false } = {}) {
  if (!sources || !Array.isArray(sources.references)) {
    throw new Error('manifest.json 的 sources.references 必须是资料列表');
  }
  if (typeof sources.referenceCollectionComplete !== 'boolean') {
    throw new Error('manifest.json 的 sources.referenceCollectionComplete 必须是布尔值');
  }

  const ids = new Set();
  for (const reference of sources.references) {
    if (!reference || typeof reference !== 'object') {
      throw new Error('manifest.json 的资料记录必须是对象');
    }
    if (typeof reference.id !== 'string' || reference.id.trim() === '') {
      throw new Error('资料记录缺少 id');
    }
    if (ids.has(reference.id)) {
      throw new Error(`资料记录 id 重复：${reference.id}`);
    }
    ids.add(reference.id);
    if (typeof reference.name !== 'string' || reference.name.trim() === '') {
      throw new Error(`资料 ${reference.id} 缺少名称`);
    }
    const expectedDefaultName = defaultReferenceName(reference.id);
    if (expectedDefaultName && reference.name !== expectedDefaultName) {
      throw new Error(`默认资料 ${reference.id} 的名称必须是“${expectedDefaultName}”`);
    }
    if (!REFERENCE_STATUSES.has(reference.status)) {
      throw new Error(`资料 ${reference.name} 的状态无效：${reference.status}`);
    }
    if (reference.status === 'provided') {
      if (typeof reference.inputPath !== 'string' || reference.inputPath.trim() === '' ||
          typeof reference.path !== 'string' || reference.path.trim() === '' ||
          !path.isAbsolute(reference.path) ||
          typeof reference.sha256 !== 'string' || reference.sha256.trim() === '') {
        throw new Error(`资料 ${reference.name} 已提供时必须记录原始输入、解析后的绝对路径和文件摘要`);
      }
      assertDocumentTarget(reference);
      continue;
    }
    if (reference.status === 'unreadable') {
      if (typeof reference.inputPath !== 'string' || reference.inputPath.trim() === '' ||
          typeof reference.requestedPath !== 'string' || reference.requestedPath.trim() === '' ||
          !path.isAbsolute(reference.requestedPath) ||
          typeof reference.error !== 'string' || reference.error.trim() === '') {
        throw new Error(`资料 ${reference.name} 不可读时必须记录原始输入、解析后的绝对路径和原因`);
      }
      continue;
    }
    if (Object.hasOwn(reference, 'inputPath') || Object.hasOwn(reference, 'path') || Object.hasOwn(reference, 'sha256') ||
        Object.hasOwn(reference, 'requestedPath') || Object.hasOwn(reference, 'error') ||
        Object.hasOwn(reference, 'documentTarget')) {
      throw new Error(`资料 ${reference.name} 的状态为 ${reference.status} 时不能保留文件路径、读取错误或资料适用性结论`);
    }
  }

  for (const defaultReference of DEFAULT_REFERENCE_MATERIALS) {
    if (!ids.has(defaultReference.id)) {
      throw new Error(`manifest.json 缺少默认资料：${defaultReference.name}`);
    }
  }

  if (requireReady) {
    if (!sources.referenceCollectionComplete) {
      throw new Error('尚未确认是否还有其他资料，不能继续生成文档');
    }
    const unresolved = unresolvedReferences(sources);
    if (unresolved.length > 0) {
      const details = unresolved.map((reference) => {
        if (reference.status === 'unreadable') {
          return `${reference.name}（原始输入：${reference.inputPath}；解析路径：${reference.requestedPath}；${reference.error}）`;
        }
        return `${reference.name}（尚未提供或确认不存在）`;
      });
      throw new Error(`资料尚未准备完成：${details.join('；')}`);
    }
  }

  if (requireClassified) {
    const unresolvedTargets = unresolvedDocumentTargets(sources);
    if (unresolvedTargets.length > 0) {
      const details = unresolvedTargets.map((reference) => {
        const kind = documentTargetOf(reference).kind;
        return `${reference.name}（${kind === 'needs-confirmation' ? '等待用户确认资料适用对象' : '尚未判断资料对应的 API'}）`;
      });
      throw new Error(`资料适用对象尚未处理完成：${details.join('；')}`);
    }
  }
}

function assertDocumentTarget(reference) {
  const target = reference.documentTarget;
  // 为兼容已有 v2 清单，缺失字段等同于尚未判定；最终化时会拦截。
  if (target == null) return;
  if (!target || typeof target !== 'object' || Array.isArray(target)) {
    throw new Error(`资料 ${reference.name} 的 documentTarget 必须是对象`);
  }
  if (!DOCUMENT_TARGET_KINDS.has(target.kind)) {
    throw new Error(`资料 ${reference.name} 的 documentTarget.kind 无效：${target.kind}`);
  }

  const detectedApis = target.detectedApis;
  if (detectedApis != null) {
    if (!Array.isArray(detectedApis) || detectedApis.length === 0 ||
        detectedApis.some((api) => typeof api !== 'string' || api.trim() === '') ||
        new Set(detectedApis.map((api) => api.trim())).size !== detectedApis.length) {
      throw new Error(`资料 ${reference.name} 的 documentTarget.detectedApis 必须是去重后的 API 名称列表`);
    }
  }

  if (target.kind === 'unclassified') {
    if (Object.hasOwn(target, 'detectedApis') || Object.hasOwn(target, 'handling') || Object.hasOwn(target, 'reason')) {
      throw new Error(`资料 ${reference.name} 尚未判定时不能预填 API、处理方式或原因`);
    }
    return;
  }

  if (typeof target.reason !== 'string' || target.reason.trim() === '') {
    throw new Error(`资料 ${reference.name} 的资料适用性结论必须说明原因`);
  }
  if (['current-api', 'mixed-api', 'other-api', 'peer-api', 'downstream-api'].includes(target.kind) &&
      (!Array.isArray(detectedApis) || detectedApis.length === 0)) {
    throw new Error(`资料 ${reference.name} 的 ${target.kind} 结论必须记录检测到的 API 名称`);
  }

  const expectedHandling = DOCUMENT_TARGET_HANDLING.get(target.kind);
  if (expectedHandling) {
    if (target.handling !== expectedHandling) {
      throw new Error(`资料 ${reference.name} 的 ${target.kind} 结论处理方式必须是 ${expectedHandling}`);
    }
  } else if (Object.hasOwn(target, 'handling')) {
    throw new Error(`资料 ${reference.name} 的 ${target.kind} 结论不能记录 handling`);
  }
}

function userPathInput(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function normalizePath(value, { repoPath } = {}) {
  const inputPath = userPathInput(value);
  if (!inputPath) return null;
  if (path.isAbsolute(inputPath)) return path.resolve(inputPath);
  if (typeof repoPath !== 'string' || !repoPath.trim()) {
    throw new Error('相对资料路径必须提供代码仓根目录作为解析基准');
  }
  if (!path.isAbsolute(repoPath.trim())) {
    throw new Error('相对资料路径的代码仓根目录必须是绝对路径');
  }
  return path.resolve(repoPath.trim(), inputPath);
}

function readErrorMessage(error) {
  if (error?.code === 'ENOENT') {
    return '文件不存在';
  }
  if (error?.code === 'EACCES' || error?.code === 'EPERM') {
    return '没有读取权限';
  }
  return error?.message || '无法读取文件';
}

export async function materialFromPath({ id, name, value, repoPath }) {
  const inputPath = userPathInput(value);
  const requestedPath = normalizePath(value, { repoPath });
  if (!requestedPath) {
    throw new Error(`资料 ${name} 缺少文件路径`);
  }
  try {
    const stat = await fs.stat(requestedPath);
    if (!stat.isFile()) {
      throw new Error('路径不是文件');
    }
    return {
      id,
      name,
      status: 'provided',
      inputPath,
      path: requestedPath,
      sha256: await sha256File(requestedPath),
      documentTarget: { kind: 'unclassified' },
    };
  } catch (error) {
    return {
      id,
      name,
      status: 'unreadable',
      inputPath,
      requestedPath,
      error: readErrorMessage(error),
    };
  }
}

function nextCustomId(references) {
  const existing = new Set(references.map((reference) => reference.id));
  let sequence = 1;
  while (existing.has(`reference-${String(sequence).padStart(3, '0')}`)) {
    sequence += 1;
  }
  return `reference-${String(sequence).padStart(3, '0')}`;
}

function rawReferenceName(raw, id) {
  const defaultName = defaultReferenceName(id);
  const name = typeof raw?.name === 'string' ? raw.name.trim() : '';
  if (defaultName) {
    if (name && name !== defaultName) {
      throw new Error(`默认资料 ${id} 的名称必须是“${defaultName}”`);
    }
    return defaultName;
  }
  if (!name) {
    throw new Error('新增资料必须提供名称');
  }
  return name;
}

function rawReferenceStatus(raw) {
  if (raw?.status == null || raw.status === '') {
    return null;
  }
  if (!REFERENCE_STATUSES.has(raw.status)) {
    throw new Error(`资料 ${raw.name || raw.id || ''} 的状态无效：${raw.status}`);
  }
  return raw.status;
}

async function normalizeRawReference(raw, id, name, repoPath) {
  const status = rawReferenceStatus(raw);
  const suppliedPath = raw?.path ?? raw?.requestedPath;
  if (status === 'absent') {
    if (userPathInput(suppliedPath)) {
      throw new Error(`资料 ${name} 已标记不存在时不能同时提供路径`);
    }
    return { id, name, status: 'absent' };
  }
  if (userPathInput(suppliedPath)) {
    return materialFromPath({ id, name, value: suppliedPath, repoPath });
  }
  if (status === 'provided' || status === 'unreadable') {
    throw new Error(`资料 ${name} 的状态为 ${status} 时必须提供文件路径`);
  }
  return { id, name, status: 'pending' };
}

function applyRawReference(raw, references) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('referenceMaterials 中的每项必须是对象');
  }
  const id = typeof raw.id === 'string' && raw.id.trim() ? raw.id.trim() : nextCustomId(references);
  const name = rawReferenceName(raw, id);
  const existingIndex = references.findIndex((reference) => reference.id === id);
  if (existingIndex >= 0) {
    if (!isDefaultReferenceId(id)) {
      throw new Error(`资料记录 id 重复：${id}`);
    }
    references[existingIndex] = { __raw: raw, id, name };
  } else {
    references.push({ __raw: raw, id, name });
  }
}

export async function createReferenceSources({
  repoPath,
  referenceMaterials = [],
  referenceCollectionComplete = false,
  internalReferencePath,
  existingDocumentPath,
} = {}) {
  if (!Array.isArray(referenceMaterials)) {
    throw new Error('referenceMaterials 必须是数组');
  }
  if (typeof referenceCollectionComplete !== 'boolean') {
    throw new Error('referenceCollectionComplete 必须是布尔值');
  }

  const rawReferences = DEFAULT_REFERENCE_MATERIALS.map((reference) => ({
    __raw: {},
    id: reference.id,
    name: reference.name,
  }));
  for (const raw of referenceMaterials) {
    applyRawReference(raw, rawReferences);
  }
  const convenience = [
    { id: 'internal-reference', name: '内部脱敏资料', path: internalReferencePath },
    { id: 'existing-document', name: '仓内已有人工文档', path: existingDocumentPath },
  ];
  for (const item of convenience) {
    if (userPathInput(item.path)) {
      const index = rawReferences.findIndex((reference) => reference.id === item.id);
      rawReferences[index] = { __raw: { path: item.path }, id: item.id, name: item.name };
    }
  }

  const references = [];
  for (const reference of rawReferences) {
    references.push(await normalizeRawReference(reference.__raw, reference.id, reference.name, repoPath));
  }
  const sources = { references, referenceCollectionComplete };
  assertReferenceSources(sources);
  return sources;
}

export async function replaceReferenceWithPath(reference, value, { repoPath } = {}) {
  return materialFromPath({ id: reference.id, name: reference.name, value, repoPath });
}

export function absentReference(reference) {
  return { id: reference.id, name: reference.name, status: 'absent' };
}

export function classifyReferenceDocumentTarget(reference, { kind, detectedApis, reason }) {
  if (!reference || reference.status !== 'provided') {
    throw new Error('只有路径可读的资料才能判断其对应的 API');
  }
  const documentTarget = { kind };
  if (detectedApis?.length) {
    documentTarget.detectedApis = [...new Set(detectedApis.map((api) => String(api).trim()).filter(Boolean))];
  }
  if (typeof reason === 'string' && reason.trim()) {
    documentTarget.reason = reason.trim();
  }
  const handling = DOCUMENT_TARGET_HANDLING.get(kind);
  if (handling) documentTarget.handling = handling;
  const updated = { ...reference, documentTarget };
  assertDocumentTarget(updated);
  return updated;
}

export async function createCustomReference({ references, name, value, absent = false, repoPath }) {
  if (typeof name !== 'string' || name.trim() === '') {
    throw new Error('新增资料必须提供名称');
  }
  if (absent) {
    return { id: nextCustomId(references), name: name.trim(), status: 'absent' };
  }
  return materialFromPath({ id: nextCustomId(references), name: name.trim(), value, repoPath });
}

export function manifestFingerprintPayload(manifest) {
  return {
    schemaVersion: MANIFEST_SCHEMA_VERSION,
    documentApiName: manifest.documentApiName,
    targetApi: manifest.targetApi,
    finalApiConfirmation: manifest.finalApiConfirmation,
    sources: manifest.sources,
  };
}

export function refreshManifestFingerprint(manifest) {
  manifest.fingerprint = sha256Text(JSON.stringify(manifestFingerprintPayload(manifest)));
  return manifest.fingerprint;
}

export function assertV2Manifest(manifest) {
  if (!manifest || !ACCEPTED_MANIFEST_KINDS.has(manifest.kind)) {
    throw new Error('不是有效的 Ascend C 文档 manifest.json');
  }
  if (manifest.schemaVersion !== MANIFEST_SCHEMA_VERSION) {
    throw new Error('manifest.json 必须使用 v2 格式');
  }
  if (manifest.output !== undefined && (
    !manifest.output ||
    typeof manifest.output.documentFileStem !== 'string' ||
    manifest.output.documentFileStem.trim() === ''
  )) {
    throw new Error('manifest.json 的 output.documentFileStem 无效');
  }
  assertReferenceSources(manifest.sources);
  assertFinalApiConfirmation(manifest);
}
