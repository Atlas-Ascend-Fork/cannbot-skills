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
import {
  absentReference,
  assertFinalApiConfirmed,
  assertV2Manifest,
  classifyReferenceDocumentTarget,
  createCustomReference,
  refreshManifestFingerprint,
  replaceReferenceWithPath,
} from './lib/reference-materials.mjs';

async function main() {
  const args = parseArgs(process.argv.slice(2));
  requireValue(args.manifestPath, '--manifest');
  assertAction(args);

  const manifestPath = path.resolve(args.manifestPath);
  const manifest = await readManifest(manifestPath);
  assertManifest(manifest);
  const repoPath = manifest.sources.repo.path;
  if (args.documentTargetKind) assertFinalApiConfirmed(manifest);
  let updatedReference = null;

  if (args.referenceId) {
    const index = manifest.sources.references.findIndex(reference => reference.id === args.referenceId);
    if (index < 0) {
      throw new Error('未找到资料记录: ' + args.referenceId);
    }
    const current = manifest.sources.references[index];
    if (args.name && args.name !== current.name) {
      throw new Error('已有资料记录不能改名；请使用原名称或新增一条资料');
    }
    if (args.documentTargetKind) {
      updatedReference = classifyReferenceDocumentTarget(current, {
        kind: args.documentTargetKind,
        detectedApis: args.detectedApis || [],
        reason: args.reason,
      });
    } else {
      updatedReference = args.absent
        ? absentReference(current)
        : await replaceReferenceWithPath(current, args.path, { repoPath });
    }
    manifest.sources.references[index] = updatedReference;
  } else if (args.name) {
    updatedReference = await createCustomReference({
      references: manifest.sources.references,
      name: args.name,
      value: args.path,
      absent: args.absent,
      repoPath,
    });
    manifest.sources.references.push(updatedReference);
  }

  if (args.completeReferenceCollection) {
    manifest.sources.referenceCollectionComplete = true;
  }
  refreshManifestFingerprint(manifest);
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
  process.stdout.write(JSON.stringify({
    manifest: manifestPath,
    reference: updatedReference,
    referenceCollectionComplete: manifest.sources.referenceCollectionComplete,
  }, null, 2) + '\n');
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--manifest') result.manifestPath = argv[++index];
    else if (arg === '--reference-id') result.referenceId = argv[++index];
    else if (arg === '--name') result.name = argv[++index];
    else if (arg === '--path') result.path = argv[++index];
    else if (arg === '--absent') result.absent = true;
    else if (arg === '--complete-reference-collection') result.completeReferenceCollection = true;
    else if (arg === '--document-target') result.documentTargetKind = argv[++index];
    else if (arg === '--detected-api') {
      result.detectedApis ||= [];
      result.detectedApis.push(argv[++index]);
    }
    else if (arg === '--reason') result.reason = argv[++index];
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
    '用法: node tools/ascendc-record-reference.mjs --manifest <manifest.json> ' +
    '(--reference-id <已有资料 ID> | --name <新增资料名称>) ' +
    '(--path <文件> | --absent) [--complete-reference-collection]\n' +
    '或: node tools/ascendc-record-reference.mjs --manifest <manifest.json> ' +
    '--reference-id <已有资料 ID> --document-target <unclassified|current-api|mixed-api|peer-api|downstream-api|other-api|needs-confirmation|excluded> ' +
    '[--detected-api <API 名称>]... [--reason <判定原因>]\n' +
    '或: node tools/ascendc-record-reference.mjs --manifest <manifest.json> --complete-reference-collection'
  );
}

function assertAction(args) {
  const hasClassificationAction = Boolean(args.documentTargetKind);
  const hasReferenceAction = Boolean(args.referenceId || args.name || args.path || args.absent || hasClassificationAction);
  if (!hasReferenceAction && !args.completeReferenceCollection) {
    throw new Error('请提供资料路径、标记资料不存在、记录资料对应的 API，或确认已完成其他资料收集');
  }
  if (args.referenceId && args.name) {
    throw new Error('--reference-id 与 --name 不能同时使用');
  }
  if (args.absent && args.path) {
    throw new Error('--absent 与 --path 不能同时使用');
  }
  if (hasClassificationAction) {
    if (!args.referenceId) {
      throw new Error('记录资料对应的 API 时必须提供 --reference-id');
    }
    if (args.name || args.path || args.absent) {
      throw new Error('--document-target 不能与 --name、--path 或 --absent 同时使用');
    }
    return;
  }
  if (args.detectedApis || args.reason) {
    throw new Error('--detected-api 和 --reason 只能与 --document-target 一起使用');
  }
  if ((args.referenceId || args.name) && !args.absent && !args.path) {
    throw new Error('请提供 --path，或使用 --absent 明确标记资料不存在');
  }
  if ((args.path || args.absent) && !args.referenceId && !args.name) {
    throw new Error('请指定 --reference-id 更新已有资料，或用 --name 新增资料');
  }
}

function requireValue(value, flag) {
  if (!value) throw new Error('缺少必需参数: ' + flag);
}

async function readManifest(manifestPath) {
  await assertFile(manifestPath, 'manifest.json');
  try {
    return JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  } catch (error) {
    throw new Error('manifest.json 无法解析: ' + manifestPath + '；' + error.message);
  }
}

function assertManifest(manifest) {
  assertV2Manifest(manifest);
  if (!manifest.documentApiName || !manifest.targetApi || !manifest.sources?.repo) {
    throw new Error('manifest.json 缺少已选 API 或代码仓信息');
  }
}

async function assertFile(filePath, label) {
  const stat = await fs.stat(filePath).catch(() => null);
  if (!stat?.isFile()) throw new Error(label + '不存在或不是文件: ' + filePath);
}

main().catch(error => {
  console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
  process.exit(1);
});
