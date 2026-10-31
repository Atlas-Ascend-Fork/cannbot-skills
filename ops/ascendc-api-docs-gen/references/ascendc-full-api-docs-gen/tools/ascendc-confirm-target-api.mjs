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
  assertV2Manifest,
  confirmFinalApi,
  refreshManifestFingerprint,
} from './lib/reference-materials.mjs';

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.manifestPath) throw new Error('缺少必需参数: --manifest');
  if (!args.apiId) throw new Error('缺少必需参数: --api-id');

  const manifestPath = path.resolve(args.manifestPath);
  const manifest = await readManifest(manifestPath);
  assertV2Manifest(manifest);
  const confirmation = confirmFinalApi(manifest, args.apiId);
  refreshManifestFingerprint(manifest);
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n', 'utf8');

  process.stdout.write(JSON.stringify({
    manifest: manifestPath,
    finalApi: summarizeFinalApi(manifest.targetApi),
    confirmation,
  }, null, 2) + '\n');
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--manifest') result.manifestPath = argv[++index];
    else if (arg === '--api-id') result.apiId = argv[++index];
    else if (arg === '--help' || arg === '-h') {
      console.log('用法: node tools/ascendc-confirm-target-api.mjs --manifest <manifest.json> --api-id <targetApi.id>');
      process.exit(0);
    } else {
      throw new Error('未知参数: ' + arg);
    }
  }
  return result;
}

async function readManifest(manifestPath) {
  try {
    return JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  } catch (error) {
    throw new Error('manifest.json 无法读取或解析: ' + manifestPath + '；' + error.message);
  }
}

function summarizeFinalApi(targetApi) {
  return {
    id: targetApi.id,
    name: targetApi.name,
    apiType: targetApi.apiType,
    callingConvention: targetApi.callingConvention,
    members: (targetApi.members || []).map(member => typeof member === 'string' ? member : member.name),
    files: targetApi.files || [],
  };
}

main().catch(error => {
  console.error('错误: ' + (error instanceof Error ? error.message : String(error)));
  process.exit(1);
});
