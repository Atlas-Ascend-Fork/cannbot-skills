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
import { assertFinalApiConfirmed, assertV2Manifest } from './lib/reference-materials.mjs';

const require = createRequire(import.meta.url);
const { buildApiRelationPlan } = require('../scripts/lib/ascendc-api-relations.js');

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.manifestPath) throw new Error('缺少必需参数: --manifest');
  const manifestPath = path.resolve(args.manifestPath);
  const manifest = JSON.parse(await fs.readFile(manifestPath, 'utf8'));
  assertV2Manifest(manifest);
  assertFinalApiConfirmed(manifest);

  const repoPath = manifest.sources?.repo?.path;
  if (typeof repoPath !== 'string' || !repoPath.trim()) {
    throw new Error('manifest.json 缺少代码仓路径');
  }
  const documentApiName = requiredString(manifest.documentApiName, 'documentApiName');
  if (!manifest.targetApi || typeof manifest.targetApi !== 'object') {
    throw new Error('manifest.json 缺少 targetApi');
  }

  const plan = buildApiRelationPlan({
    repoPath,
    documentApiName,
    targetApi: manifest.targetApi
  });
  const text = JSON.stringify(plan, null, 2) + '\n';
  if (args.outputPath) {
    const outputPath = path.resolve(args.outputPath);
    await fs.mkdir(path.dirname(outputPath), { recursive: true });
    await fs.writeFile(outputPath, text, 'utf8');
    process.stderr.write('已写入临时调用关系定位结果: ' + outputPath + '\n');
  } else {
    process.stdout.write(text);
  }
}

function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === '--manifest') result.manifestPath = argv[++index];
    else if (arg === '--output') result.outputPath = argv[++index];
    else if (arg === '--help' || arg === '-h') {
      console.log('用法: node tools/ascendc-api-relation-plan.mjs --manifest <manifest.json> [--output <临时结果.json>]');
      process.exit(0);
    } else {
      throw new Error('未知参数: ' + arg);
    }
  }
  return result;
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
