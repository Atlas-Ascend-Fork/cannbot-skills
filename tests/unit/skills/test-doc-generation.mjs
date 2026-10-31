// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { after, before, test } from 'node:test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const acl = path.join(root, 'ops/ascendc-docs-gen/references/aclnn-full-api-docs-gen');
const asc = path.join(root, 'ops/ascendc-api-docs-gen/references/ascendc-full-api-docs-gen');
const require = createRequire(import.meta.url);
const validator = require(path.join(asc, 'scripts/lib/final-document-validator.js'));
const aclProductPolicy = require(path.join(acl, 'tools/scripts/lib/product-document-policy.js'));
const ascProductPolicy = require(path.join(asc, 'scripts/lib/product-document-policy.js'));
const signatures = require(path.join(asc, 'scripts/lib/ascendc-signatures.js'));
const relations = require(path.join(asc, 'scripts/lib/ascendc-api-relations.js'));
const ascFinalize = await import(pathToFileURL(path.join(asc, 'tools/ascendc-finalize.mjs')));
const dependency = require(path.join(asc, 'scripts/lib/markdown-it-dependency.json'));
const originalEnvironment = {
  cache: process.env.CANNDOC_DEPENDENCY_CACHE,
  nodeOptions: process.env.NODE_OPTIONS,
  fetch: globalThis.fetch
};
let regressionDirectory;

before(() => {
  const preparationHelp = '请先设置 CANNDOC_DEPENDENCY_CACHE，并单独运行模块内 markdown-it-loader.js --prepare；回归不自动下载依赖。';
  assert.ok(originalEnvironment.cache, preparationHelp);
  const source = path.join(path.resolve(originalEnvironment.cache), dependency.name, dependency.version);
  const files = dependency.files.map(file => {
    let bytes;
    try {
      bytes = fs.readFileSync(path.join(source, file.name));
    } catch {
      throw new Error(`测试依赖未准备或不可读：${file.name}。${preparationHelp}`);
    }
    assert.equal(createHash('sha256').update(bytes).digest('hex'), file.sha256,
      `测试依赖摘要不匹配：${file.name}。${preparationHelp}`);
    return { name: file.name, bytes };
  });
  regressionDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'canndoc-regression-cache-'));
  const isolatedRoot = path.join(regressionDirectory, 'cache');
  for (const file of files) write(path.join(isolatedRoot, dependency.name, dependency.version, file.name), file.bytes);
  const preload = path.join(regressionDirectory, 'offline.cjs');
  write(preload, 'globalThis.fetch = async () => { throw new Error("CDN network disabled during deterministic regression"); };');
  process.env.CANNDOC_DEPENDENCY_CACHE = isolatedRoot;
  process.env.NODE_OPTIONS = `${originalEnvironment.nodeOptions || ''} --require "${preload.replaceAll('\\', '/')}"`.trim();
  globalThis.fetch = async () => { throw new Error('CDN network disabled during deterministic regression'); };
});

after(() => {
  for (const [name, value] of [['CANNDOC_DEPENDENCY_CACHE', originalEnvironment.cache], ['NODE_OPTIONS', originalEnvironment.nodeOptions]]) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
  globalThis.fetch = originalEnvironment.fetch;
  if (regressionDirectory) {
    const resolved = fs.realpathSync(regressionDirectory);
    assert.equal(path.dirname(resolved).toLowerCase(), fs.realpathSync(os.tmpdir()).toLowerCase());
    assert.ok(path.basename(resolved).startsWith('canndoc-regression-cache-'));
    fs.rmSync(resolved, { recursive: true });
  }
});

function productSupport(type) {
  const entries = [
    ['950', 'Ascend 950PR&950DT系列产品'],
    ['A3', 'Atlas A3系列产品'],
    ['910b', 'Atlas A2系列产品'],
    ['310b', 'Atlas 200I/500 A2推理产品'],
    ...(type === 'ascendc'
      ? [['310p', 'Atlas推理系列产品AI Core'], ['310p', 'Atlas推理系列产品Vector Core']]
      : [['310p', 'Atlas推理系列产品']]),
    ['910', 'Atlas训练系列产品']
  ];
  return '## 产品支持情况\n' + entries.map(([npu, product], i) =>
    `<!-- npu="${npu}" id${i + 1} -->\n- ${type === 'aclnn' ? `<term>${product}</term>` : product}：支持\n<!-- end id${i + 1} -->`).join('\n');
}

function fixture(t) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'canndoc-skill-test-'));
  t.after(() => {
    const resolved = fs.realpathSync(dir);
    assert.equal(path.dirname(resolved).toLowerCase(), fs.realpathSync(os.tmpdir()).toLowerCase());
    assert.ok(path.basename(resolved).startsWith('canndoc-skill-test-'));
    fs.rmSync(resolved, { recursive: true });
  });
  return dir;
}

function run(script, args = [], options = {}) {
  const result = spawnSync(process.execPath, [script, ...args], {
    encoding: 'utf8', timeout: 30000, windowsHide: true, ...options
  });
  assert.ifError(result.error);
  return result;
}

function write(file, content) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
}

function offlineDependencyEnvironment(t, cacheRoot = fixture(t)) {
  const preload = path.join(fixture(t), 'offline.cjs');
  write(preload, 'globalThis.fetch = async () => { throw new Error("offline dependency test"); };');
  return {
    ...process.env,
    CANNDOC_DEPENDENCY_CACHE: cacheRoot,
    NODE_OPTIONS: `${process.env.NODE_OPTIONS || ''} --require "${preload.replaceAll('\\', '/')}"`.trim()
  };
}

test('regression uses an isolated cache and disables unexpected CDN requests in parent and child processes', async () => {
  assert.notEqual(path.resolve(process.env.CANNDOC_DEPENDENCY_CACHE), path.resolve(originalEnvironment.cache));
  await assert.rejects(fetch('https://cdn.jsdelivr.net/'), /CDN network disabled/);
  const child = run('--eval', [
    'fetch("https://cdn.jsdelivr.net/").then(()=>process.exit(1)).catch(error=>{' +
    'if(!error.message.includes("CDN network disabled"))process.exit(2);});'
  ]);
  assert.equal(child.status, 0, child.stderr);
});

test('missing or corrupt explicit test dependencies fail at setup without attempting a download', t => {
  const missingRoot = path.join(fixture(t), 'not-prepared');
  const corruptRoot = fixture(t);
  const marker = path.join(corruptRoot, 'unverified-code-executed');
  write(path.join(corruptRoot, dependency.name, dependency.version, dependency.files[0].name),
    `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'unsafe');`);
  for (const cache of [undefined, missingRoot, corruptRoot]) {
    const env = { ...process.env };
    // The negative control starts a standalone runner rather than inheriting the parent test context.
    delete env.NODE_TEST_CONTEXT;
    if (cache === undefined) delete env.CANNDOC_DEPENDENCY_CACHE;
    else env.CANNDOC_DEPENDENCY_CACHE = cache;
    const child = run('--test', ['--test-name-pattern=eval definitions retain', fileURLToPath(import.meta.url)], { env });
    assert.notEqual(child.status, 0, child.stdout + child.stderr);
    assert.match(child.stdout + child.stderr, /回归不自动下载依赖/);
    assert.doesNotMatch(child.stdout + child.stderr, /CDN network disabled/);
  }
  assert.ok(!fs.existsSync(missingRoot));
  assert.ok(!fs.existsSync(marker));
});

function setupPreprocess(dir, behavior) {
  const op = path.join(dir, 'operator');
  const scripts = path.join(dir, 'scripts');
  write(path.join(op, 'op_api/aclnn_demo.h'), 'aclnnStatus aclnnDemo(void);');
  for (const name of ['prototype', 'cross-dir', 'chip-evidence', 'error-codes', 'call-example']) {
    write(path.join(scripts, `extract-${name}.js`), behavior);
  }
  return { op, scripts, output: path.join(op, 'outputs/aclnnDemo') };
}

const successScript = `const fs = require('fs');
fs.writeFileSync(process.argv[process.argv.indexOf('--output') + 1], '# Evidence\\n');`;

test('all scripts in both skills parse successfully', () => {
  function check(dir) {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) check(file);
      else if (/\.(?:js|mjs)$/.test(file)) {
        const result = run('--check', [file]);
        assert.equal(result.status, 0, `${file}\n${result.stderr}`);
      }
    }
  }
  check(acl);
  check(asc);
});

test('preprocessing fails when all children fail or omit outputs', t => {
  for (const behavior of ['process.exit(2);', 'process.exit(0);']) {
    const { op, scripts, output } = setupPreprocess(fixture(t), behavior);
    const result = run(path.join(acl, 'tools/aclnn-preprocess.mjs'),
      [op, '--interface', 'aclnnDemo', '--scripts-dir', scripts]);
    assert.notEqual(result.status, 0);
    assert.equal((result.stdout.match(/\[FAILED\]/g) || []).length, 5);
    assert.ok(!result.stdout.includes('预处理完成'));
    assert.ok(!fs.existsSync(path.join(output, '0_extract_plan.json')));
  }
});

test('new preprocessing archives previous stages and uses the installed skill configuration', t => {
  const dir = fixture(t);
  const installed = path.join(dir, '中文 skill with spaces');
  fs.cpSync(acl, installed, { recursive: true });
  const { op, scripts, output } = setupPreprocess(dir, successScript);
  write(path.join(output, '3_formula.md'), 'OLD FORMULA');
  write(path.join(output, '5_review_patch.json'), '{"status":"pass"}');
  const result = run(path.join(installed, 'tools/aclnn-preprocess.mjs'),
    [op, '--interface', 'aclnnDemo', '--scripts-dir', scripts]);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(result.stdout.includes(path.join(installed, 'config/product_mapping.yml')));
  assert.ok(fs.existsSync(path.join(output, '0_extract_plan.json')));
  assert.ok(!fs.existsSync(path.join(output, '3_formula.md')));
  const archived = fs.readdirSync(output).find(name => name.startsWith('.previous-'));
  assert.ok(archived);
  assert.equal(fs.readFileSync(path.join(output, archived, '3_formula.md'), 'utf8'), 'OLD FORMULA');
  const batch = run(path.join(installed, 'tools/aclnn-batch.mjs'), ['detect', op]);
  assert.equal(batch.status, 0, batch.stderr);
});

function prepareReview(op, output, patch) {
  const prepared = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo', '--prepare']);
  assert.equal(prepared.status, 0, prepared.stderr);
  if (patch === undefined) return;
  try {
    const value = JSON.parse(patch);
    value.reviewedDocumentSha256 = JSON.parse(fs.readFileSync(path.join(output, '4_review_input.json'), 'utf8')).reviewedDocumentSha256;
    patch = JSON.stringify(value);
  } catch { /* Malformed review inputs are intentionally tested. */ }
  write(path.join(output, '5_review_patch.json'), patch);
}

function setupFinalize(dir, patch) {
  const op = path.join(dir, 'operator');
  const output = path.join(op, 'outputs/aclnnDemo');
  write(path.join(op, 'op_api/aclnn_demo.h'), 'aclnnStatus aclnnDemo(void);');
  write(path.join(output, '4_generate_skeleton.md'), [
    '# aclnnDemo', productSupport('aclnn'), '## 功能说明', '执行示例接口。', '## 函数原型', '{{ACLNN_PROTOTYPE_BLOCK_PLACEHOLDER}}',
    '{{ACLNN_FORMULA_BLOCK_PLACEHOLDER}}', '{{ACLNN_EXECUTE_INTERFACE_BLOCK_PLACEHOLDER}}', '## 约束说明', '无'
  ].join('\n'));
  write(path.join(output, '3_formula.md'), '## 最终文档公式块\n无');
  write(path.join(output, '0_prototype.md'), '```cpp\naclnnStatus aclnnDemo(void);\n```');
  write(path.join(output, '0_call_example.md'), '示例。\n```cpp\nvoid run() { aclnnDemo(); }\n```');
  const header = path.join(op, 'op_api/aclnn_demo.h');
  const sha256 = createHash('sha256').update(fs.readFileSync(header)).digest('hex');
  write(path.join(output, '0_extract_plan.json'), JSON.stringify({
    prototypeReportSha256: createHash('sha256').update(fs.readFileSync(path.join(output, '0_prototype.md'))).digest('hex'),
    prototypeSources: [{ file: header, kind: 'source', sha256 }], sourceSnapshot: [{ file: header, sha256 }]
  }));
  prepareReview(op, output, patch);
  return { op, output };
}

test('finalize rejects missing, malformed, failed and inapplicable review patches without publishing', t => {
  for (const patch of [
    undefined, '{broken', '{"status":"fail","failureReason":"sensitive content"}',
    JSON.stringify({ status: 'fix', skeletonEdits: [
      { section: '不存在的章节', oldText: 'NOT PRESENT', newText: 'new', reason: 'test' }
    ], formulaBlockEdits: [] })
  ]) {
    const { op, output } = setupFinalize(fixture(t), patch);
    const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
    assert.notEqual(result.status, 0);
    // 4_generate.md is a prepared draft, never the final delivery.
    assert.ok(!fs.existsSync(path.join(output, '5_review.md')));
    assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
  }
});

test('finalize publishes a valid pass patch', t => {
  const { op, output } = setupFinalize(fixture(t),
    '{"status":"pass","skeletonEdits":[],"formulaBlockEdits":[]}');
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
});

test('finalize applies a valid fix patch before publishing', t => {
  const { op, output } = setupFinalize(fixture(t), JSON.stringify({
    status: 'fix',
    skeletonEdits: [],
    formulaBlockEdits: [{ section: '最终文档公式块', oldText: '无', newText: '修正公式。', reason: 'test' }]
  }));
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  const final = fs.readdirSync(output).find(name => /^aclnnDemo_\d+\.md$/.test(name));
  assert.ok(fs.readFileSync(path.join(output, final), 'utf8').includes('修正公式。'));
});

test('ACLNN real preprocessing tools complete on a minimal operator', t => {
  const { op, output } = setupPreprocess(fixture(t), successScript);
  const result = run(path.join(acl, 'tools/aclnn-preprocess.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
  for (const name of ['prototype', 'cross_dir', 'chip_evidence', 'error_codes', 'call_example']) {
    assert.ok(fs.statSync(path.join(output, `0_${name}.md`)).size > 0);
  }
});

test('batch finalize returns failure for incomplete inputs', t => {
  const { op } = setupFinalize(fixture(t));
  const result = run(path.join(acl, 'tools/aclnn-batch.mjs'), ['finalize', op]);
  assert.notEqual(result.status, 0);
  assert.ok(result.stdout.includes('5_review_patch.json'));
});

test('AscendC full validation rejects private content but accepts public API syntax', () => {
  const prototype = '## 函数原型\n```cpp\n__aicore__ inline void Demo();\n```';
  const doc = `# Demo\n\n${productSupport('ascendc')}\n\n## 功能说明\nPUBLIC_TEXT\n\n${prototype}
\n## 参数说明\n无\n## 返回值说明\n无\n## 约束说明\n无\n## 调用示例\n\`\`\`cpp\n#include "kernel_operator.h"\nvoid run() { Demo(); }\n\`\`\`
结果示例如下：
\`\`\`text
完成
\`\`\``;
  const validate = text => validator.validateFinalDocument({
    apiName: 'Demo', prototypeDoc: prototype, finalDoc: doc.replace('PUBLIC_TEXT', text), evidenceIds: ['K-001', 'R-123']
  });
  assert.doesNotThrow(() => validate('公开接口 <code>Demo</code>。'));
  for (const text of [
    'C:\\Users\\example\\internal\\kernel.cpp:42', '/home/user/private/code.h',
    '\\\\server\\private\\file.h', '__NPU_ARCH__', '__CCE_AICORE__',
    'K-001', '证据 R-123', 'op_kernel/private.h',
    '```cpp\n// C:\\Users\\example\\secret.h\n```'
  ]) assert.throws(() => validate(text), undefined, text);
});

test('AscendC supported entry creates a v2 manifest from a Chinese path', t => {
  const repo = path.join(fixture(t), '中文 repo');
  const header = path.join(repo, 'include/c_api/demo.h');
  write(header, '__aicore__ inline void Demo();\n');
  const result = run(path.join(asc, 'tools/ascendc-create-manifest.mjs'),
    ['--repo', repo, '--interface-file', header, '--api-name', 'Demo']);
  assert.equal(result.status, 0, result.stderr);
  const manifest = JSON.parse(fs.readFileSync(path.join(repo, 'outputs/Demo/manifest.json'), 'utf8'));
  assert.equal(manifest.schemaVersion, 2);
  assert.equal(manifest.documentApiName, 'Demo');
  assert.equal(manifest.finalApiConfirmation.status, 'pending');
});

test('September product names normalize without repeated suffixes or broadening Core scope', () => {
  const cases = [
    ['Ascend 950PR/Ascend 950DT', 'Ascend 950PR&950DT系列产品'],
    ['Ascend 950PR', 'Ascend 950PR系列产品'],
    ['Ascend 950DT', 'Ascend 950DT系列产品'],
    ['Atlas A3 训练系列产品/Atlas A3 推理系列产品', 'Atlas A3系列产品'],
    ['Atlas A3训练系列产品/Atlas A3推理系列产品', 'Atlas A3系列产品'],
    ['Atlas A2 训练系列产品/Atlas A2 推理系列产品', 'Atlas A2系列产品'],
    ['Atlas A2训练系列产品/Atlas A2推理系列产品', 'Atlas A2系列产品'],
    ['Atlas A3 训练系列产品', 'Atlas A3训练系列产品'],
    ['Atlas A3 推理系列产品', 'Atlas A3推理系列产品'],
    ['Atlas A2 训练系列产品', 'Atlas A2训练系列产品'],
    ['Atlas A2 推理系列产品', 'Atlas A2推理系列产品'],
    ['Atlas 训练系列产品', 'Atlas训练系列产品'],
    ['Atlas 200I/500 A2 推理产品', 'Atlas 200I/500 A2推理产品'],
    ['Atlas 推理系列产品AI Core', 'Atlas推理系列产品AI Core'],
    ['Atlas 推理系列产品Vector Core', 'Atlas推理系列产品Vector Core']
  ];
  for (const [policy, type] of [[aclProductPolicy, 'aclnn'], [ascProductPolicy, 'ascendc']]) {
    for (const [oldName, newName] of cases) {
      assert.equal(policy.normalizeProductNames(oldName, type), newName);
      assert.equal(policy.normalizeProductNames(newName, type), newName);
    }
    assert.equal(policy.normalizeProductNames('Atlas 推理系列产品', type), type === 'aclnn'
      ? 'Atlas推理系列产品' : 'Atlas推理系列产品AI Core、Atlas推理系列产品Vector Core');
  }
});

test('product lists and paragraph tags enforce pairing, scope, syntax and document-wide IDs', () => {
  for (const [policy, type] of [[aclProductPolicy, 'aclnn'], [ascProductPolicy, 'ascendc']]) {
    const base = '# Demo\n' + productSupport(type) + '\n## 约束说明\n';
    const paragraph = '<!-- npu="950" id8 -->\nAscend 950PR系列产品仅支持输入连续。\n<!-- end id8 -->';
    assert.doesNotThrow(() => policy.validateProductDocument(base + paragraph, type));
    assert.doesNotThrow(() => policy.validateProductDocument(base +
      '| 产品 | 约束 |\n| --- | --- |\n| Atlas A2系列产品 | 输入连续 |', type));
    assert.doesNotThrow(() => policy.validateProductDocument(base +
      '<table><tr><td>Atlas A2系列产品</td><td>输入连续</td></tr></table>', type));
    assert.doesNotThrow(() => policy.validateProductDocument(base +
      '<!-- npu="A3" id8 -->Atlas A3系列产品支持此模式。<!-- end id8 -->', type));
    for (const invalid of [
      paragraph.replace('end id8', 'end id9'),
      paragraph.replace('<!-- end id8 -->', ''),
      paragraph.replace('id8', 'id1').replace('end id8', 'end id1'),
      paragraph.replace('npu="950"', 'npu="A3"'),
      paragraph.replace('npu="950"', 'npu="a3"'),
      paragraph.replace('npu="950"', 'npu="999"'),
      paragraph.replace('npu="950" id8', 'npu="950"  id8'),
      paragraph.replace('<!-- npu', '<!--npu'),
      '<!-- npu="950" id8 --><!-- end id8 -->',
      paragraph + '\n' + paragraph,
      '<!-- npu="950" id9 -->\n' + paragraph + '\n<!-- end id9 -->',
      paragraph.replace('Ascend 950PR系列产品', 'Ascend 950PR')
    ]) assert.throws(() => policy.validateProductDocument(base + invalid, type), undefined, invalid);
    assert.throws(() => policy.validateProductDocument(
      '# Demo\n## 产品支持情况\n| 产品 | 支持 |\n| --- | --- |\n| Atlas A2系列产品 | 支持 |', type));
    assert.throws(() => policy.validateProductDocument(base.replace('## 产品支持情况', '## 功能说明'), type));
  }
});

test('ACLNN finalize refuses mismatched product scopes and duplicate IDs before publishing', t => {
  for (const extra of [
    '\n## 约束说明\n<!-- npu="950" id8 -->Atlas A2系列产品仅支持连续输入。<!-- end id8 -->',
    '\n## 约束说明\n<!-- npu="950" id1 -->\nAscend 950PR系列产品支持此模式。\n<!-- end id1 -->'
  ]) {
    const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
    fs.appendFileSync(path.join(output, '4_generate_skeleton.md'), extra);
    const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
    assert.notEqual(result.status, 0);
    assert.ok(!fs.existsSync(path.join(output, '5_review.md')));
    assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
  }
});

 test('standalone spacing preserves canonical product names and paired comments', t => {
  const dir = fixture(t);
  const input = path.join(dir, 'input.md');
  const content = productSupport('aclnn') + '\n## 功能说明\n<!-- npu="950" id1000 -->\n<term>Ascend 950PR</term>支持此功能。\n<!-- end id1000 -->\n';
  write(input, content);
  const result = run(path.join(acl, 'tools/scripts/normalize-doc-spacing.js'), [input]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /<term>Ascend 950PR系列产品<\/term>/);
  aclProductPolicy.validateProductDocument(result.stdout, 'aclnn');
});

test('architecture-specific call examples carry paired product tags', t => {
  const dir = fixture(t);
  write(path.join(dir, 'examples/arch3510/example.cpp'), 'int main() { return 0; }');
  const result = run(path.join(acl, 'tools/scripts/extract-call-example.js'), [dir]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /<!-- npu="950" id2000 -->/);
  aclProductPolicy.validateProductDocument(productSupport('aclnn') + '\n## 调用示例\n' + result.stdout, 'aclnn');
});


test('product normalization preserves code and resource targets while updating visible labels', () => {
  const protectedParts = [
    '```cpp\nconst char* model = "Ascend 950PR";\n```',
    '~~~cpp\nconst char* model = "Ascend 950PR";\n~~~~',
    '``code ` Ascend 950PR``',
    '<code>Ascend 950PR</code>',
    '<pre>Ascend 950PR</pre>',
    '[ref]: <./Ascend 950PR.md> "Ascend 950PR"',
    '<a href="./Ascend 950PR.md">文档</a>'
  ];
  for (const [policy, type] of [[aclProductPolicy, 'aclnn'], [ascProductPolicy, 'ascendc']]) {
    for (const original of protectedParts) {
      assert.equal(policy.normalizeProductNames(original, type), original);
      const document = productSupport(type) + '\n## 功能说明\n' + original;
      assert.doesNotThrow(() => policy.validateProductDocument(document, type));
    }
    const link = '[Ascend 950PR](./Ascend 950PR(test).md "标题")';
    const expected = '[Ascend 950PR系列产品](./Ascend 950PR(test).md "标题")';
    assert.equal(policy.normalizeProductNames(link, type), expected);
    assert.equal(policy.normalizeProductNames(expected, type), expected);
  }
});

test('common product statements do not require invented scopes but support lists still require tags', () => {
  for (const [policy, type] of [[aclProductPolicy, 'aclnn'], [ascProductPolicy, 'ascendc']]) {
    const base = productSupport(type) + '\n## 功能说明\n';
    assert.doesNotThrow(() => policy.validateProductDocument(base +
      'Atlas A2系列产品与Atlas A3系列产品的行为完全一致，无产品差异。', type));
    assert.throws(() => policy.validateProductDocument(base.replace('<!-- npu="950" id1 -->', '')
      .replace('<!-- end id1 -->', ''), type), /产品支持列表缺少/);
  }
});

test('table exemptions respect Markdown delimiters and exact HTML boundaries', () => {
  for (const [policy, type] of [[aclProductPolicy, 'aclnn'], [ascProductPolicy, 'ascendc']]) {
    const base = productSupport(type) + '\n## 约束说明\n';
    const start = '<!-- npu="950" id8 -->';
    const end = '<!-- end id8 -->';
    // Deliberately use another NPU inside tables to prove that only cells are exempt.
    for (const table of [
      '产品 | 约束\n--- | ---\nAtlas A2系列产品 | 输入连续',
      '| 产品 |\n| --- |\n| Atlas A2系列产品 |',
      '| 产品 | 约束 |\n| --- | --- |\n| Atlas A2系列产品 | 输入连续 |',
      '<table><tr><td>Atlas A2系列产品</td></tr></table>'
    ]) {
      assert.doesNotThrow(() => policy.validateProductDocument(base + start + '\n' + table + '\n' + end, type));
      assert.throws(() => policy.validateProductDocument(base + start + '\n' + table +
        '\n\nAtlas A2系列产品仅支持 float。' + end, type), /产品名称与 npu/);
    }
    assert.throws(() => policy.validateProductDocument(base + start +
      '<table><tr><td>通用参数</td></tr></table>Atlas A2系列产品仅支持 float。' + end, type), /产品名称与 npu/);
    assert.throws(() => policy.validateProductDocument(base + '<table>\n正文', type), /HTML 表格未闭合/);
    assert.throws(() => policy.validateProductDocument(base + '</table>', type), /HTML 表格结束标签/);
    assert.doesNotThrow(() => policy.validateProductDocument(base + '<!-- <table> -->\n普通说明', type));
  }
});

test('ACLNN final artifact preserves code literals and link destinations', t => {
  const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
  const example = '```cpp\nconst char* model = "Ascend 950PR";\n```';
  write(path.join(output, '0_call_example.md'), example + '\n```cpp\naclnnDemo();\n```');
  fs.appendFileSync(path.join(output, '4_generate_skeleton.md'), '\n[产品文档](./Ascend 950PR.md)\n');
  prepareReview(op, output, '{"status":"pass"}');
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  const final = fs.readdirSync(output).find(name => /^aclnnDemo_\d+\.md$/.test(name));
  const document = fs.readFileSync(path.join(output, final), 'utf8');
  assert.ok(document.includes(example));
  assert.ok(document.includes('(./Ascend 950PR.md)'));
});

test('ACLNN review patches preserve literal dollar sequences in formulas and document edits', t => {
  const { op, output } = setupFinalize(fixture(t), JSON.stringify({
    status: 'fix', formulaBlockEdits: [{ section: '最终文档公式块', oldText: '无',
      newText: '$$\nA = B\n$$', reason: 'formula correction' }],
    documentEdits: [{ section: '功能说明', oldText: '## 功能说明',
      newText: '## 功能说明\n字面量：$&、$`、$\'。', reason: 'literal text' }]
  }));
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  const final = fs.readdirSync(output).find(name => /^aclnnDemo_\d+\.md$/.test(name));
  const doc = fs.readFileSync(path.join(output, final), 'utf8');
  assert.ok(doc.includes('$$\nA = B\n$$'));
  assert.ok(doc.includes('$&、$`、$\''));
});

test('ACLNN rejects unreviewed changes to examples and stale review digests', t => {
  for (const change of ['example', 'digest']) {
    const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
    if (change === 'example') fs.appendFileSync(path.join(output, '0_call_example.md'), '\nNEW CONTENT');
    else {
      const patch = JSON.parse(fs.readFileSync(path.join(output, '5_review_patch.json'), 'utf8'));
      patch.reviewedDocumentSha256 = 'outdated';
      write(path.join(output, '5_review_patch.json'), JSON.stringify(patch));
    }
    const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /候选|review/);
    assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
  }
});

test('ACLNN reviewed document edits cover the assembled call example', t => {
  const { op, output } = setupFinalize(fixture(t), JSON.stringify({ status: 'fix', documentEdits: [
    { section: '调用示例', oldText: '示例。', newText: '审阅后的示例说明。', reason: 'review example' }
  ] }));
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  const doc = fs.readFileSync(path.join(output, '5_review.md'), 'utf8');
  assert.match(doc, /审阅后的示例说明/);
});

test('ACLNN document review cannot change a verified function declaration', t => {
  const { op, output } = setupFinalize(fixture(t), JSON.stringify({ status: 'fix', documentEdits: [
    { section: '函数原型', oldText: 'aclnnStatus aclnnDemo(void)',
      newText: 'aclnnStatus aclnnDemo(int inventedParameter)', reason: 'invalid correction' }
  ] }));
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /原型.*不一致/);
  assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
});

test('example extraction matches exact API calls and preserves all source comments', t => {
  const dir = fixture(t);
  const old = 'void run() { aclnnAddGetWorkspaceSize(); aclnnAdd(); }';
  const current = '/* doc */ int helper() { return 1; } /* Copyright notice */\n' +
    'void run() { aclnnAddV3GetWorkspaceSize(); aclnnAddV3(); }';
  write(path.join(dir, 'examples/demo.cpp'), old + '\n// aclnnAddV3();\nconst char* name = "aclnnAddV3(";');
  const script = path.join(acl, 'tools/scripts/extract-call-example.js');
  let result = run(script, [dir, 'aclnnAddV3']);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /需用户补齐/);
  assert.ok(!result.stdout.includes(old));
  write(path.join(dir, 'examples/current.cpp'), current);
  result = run(script, [dir, 'aclnnAddV3']);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(result.stdout.includes(current));
  write(path.join(dir, 'examples/demo.cpp'), '// old example removed');
  result = run(script, [dir, 'aclnnAdd']);
  assert.match(result.stdout, /需用户补齐/);
});

test('unknown full architecture values cannot inherit a public product from their prefix', t => {
  const dir = fixture(t);
  for (const arch of ['3199', '3599']) {
    write(path.join(dir, `examples/arch_${arch}/example.cpp`), 'PRIVATE_ARCH_EXAMPLE();');
  }
  write(path.join(dir, 'examples/arch_3510/example.cpp'), 'PUBLIC_ARCH_EXAMPLE();');
  const result = run(path.join(acl, 'tools/scripts/extract-call-example.js'), [dir]);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(!result.stdout.includes('PRIVATE_ARCH_EXAMPLE'));
  assert.match(result.stdout, /PUBLIC_ARCH_EXAMPLE/);
});

test('chip evidence recognizes default and named AddConfig overloads', t => {
  const dir = fixture(t);
  write(path.join(dir, 'op_host/demo_def.cpp'),
    'AddConfig("ascend310p");\nAddConfig("ascend910b", configA2);');
  const result = run(path.join(acl, 'tools/scripts/extract-chip-evidence.js'), [dir, dir]);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(!result.stdout.includes('未找到 AddConfig'));
  assert.match(result.stdout, /ascend310p/);
  assert.match(result.stdout, /configA2/);
});

test('preprocessing infers an include-resolving repository root for nested operators', t => {
  const repo = fixture(t);
  const op = path.join(repo, 'loss/mse_loss');
  write(path.join(op, 'op_api/aclnn_demo.h'), 'aclnnStatus aclnnDemo(void);');
  write(path.join(op, 'op_host/demo.cpp'), '#include "loss/common/check.h"\nvoid f(){ CheckDtype(); }');
  write(path.join(repo, 'loss/common/check.h'), 'inline void CheckDtype() {}');
  const result = run(path.join(acl, 'tools/aclnn-preprocess.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  const report = fs.readFileSync(path.join(op, 'outputs/aclnnDemo/0_cross_dir.md'), 'utf8');
  assert.match(report, /loss\/common\/check.h/);
});

test('missing generated declarations report missing evidence rather than nonexistent interfaces', t => {
  const dir = fixture(t);
  write(path.join(dir, 'op_host/demo_def.cpp'), 'ACLNNTYPE(aclnn);');
  const result = run(path.join(acl, 'tools/aclnn-preprocess.mjs'), [dir, '--interface', 'aclnnGenerated']);
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /opbuild/);
  assert.match(result.stderr, /同一源码版本/);
});

test('batch finalize discovers the same nested operators and rejects missing filter values', t => {
  const repo = fixture(t);
  const { op } = setupFinalize(path.join(repo, 'math/add'), '{"status":"pass"}');
  const script = path.join(acl, 'tools/aclnn-batch.mjs');
  const result = run(script, ['finalize', repo]);
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.ok(fs.readdirSync(path.join(op, 'outputs/aclnnDemo')).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
  for (const option of ['--include', '--exclude']) {
    const invalid = run(script, ['detect', repo, option]);
    assert.notEqual(invalid.status, 0);
    assert.match(invalid.stderr, /缺少参数值/);
  }
  const invalidPath = run(path.join(acl, 'tools/scripts/detect-interfaces.js'), [path.join(repo, 'missing')]);
  assert.notEqual(invalidPath.status, 0);
  assert.match(invalidPath.stderr, /路径不存在/);
});

test('signature identity normalizes operators and parses nested or long template headers once', () => {
  const source = 'Value operator +(Value lhs, Value rhs);\nValue operator+(Value lhs, Value rhs);\n' +
    `template <typename T, template <class> class Alloc, ${Array.from({ length: 20 }, (_, i) => `typename LongType${i} = T, `).join('')}typename U = T>\n` +
    '__aicore__ inline void Demo(T value);';
  const parsed = signatures.parseAscendcSignaturesFromContent(source);
  assert.equal(parsed.filter(item => item.funcName === 'operator+').length, 1);
  assert.equal(parsed.filter(item => item.funcName === 'Demo').length, 1);
  assert.match(parsed.find(item => item.funcName === 'Demo').templateParams, /template <class> class Alloc/);
  assert.ok(signatures.hasApiSignature(source, 'operator +'));
  for (const [name, code] of [
    ['operator+', 'a.operator +(b);'], ['operator+', 'auto c = a + b;'],
    ['operator[]', 'auto x = a[0];'], ['Demo', 'Demo<int>(value);']
  ]) assert.ok(validator.hasMemberInvocation(code, name), code);
  assert.ok(!validator.hasMemberInvocation('DemoV2();', 'Demo'));
});

test('AscendC examples distinguish calls from declarations, literals and unevaluated uses', () => {
  const prototypeDoc = '## 函数原型\n```cpp\n__aicore__ inline void Demo();\n```';
  const document = example => '# Demo\n' + productSupport('ascendc') + '\n## 功能说明\n执行接口。\n' +
    prototypeDoc + '\n## 参数说明\n无\n## 返回值说明\n无\n## 约束说明\n无\n## 调用示例\n' +
    example + '\n结果示例如下：\n```text\n无返回值。\n```';
  const validate = example => validator.validateFinalDocument({ apiName: 'Demo', prototypeDoc, finalDoc: document(example) });
  const rejected = [
    'void Demo();', '__aicore__ inline void Demo() {}', 'const Value* Demo();',
    'void Other(), Demo();', 'int value = 0, Demo();', 'Value<T> Other(), Demo();',
    'template<int N = 1> int* Demo();', 'decltype(value)* Demo();', '[[nodiscard]] int* Demo();',
    'template<class T> auto Demo(T value) -> T;', 'extern "C" void Demo();',
    'void AscendC::Demo();', 'void Container<int>::Demo();', 'void (Demo)();', 'const char* text = "Demo()";',
    'const char* text = R"tag(a quote " and Demo())tag";',
    'const char* text = u8R"(Demo())";', String.raw`const char* text = "escaped \" Demo()";`,
    '// Demo();', '/* Demo(); */', '#define EXAMPLE Demo()',
    'using Result = decltype(Demo());', 'static_assert(noexcept(Demo()));',
    'void run() { (void)sizeof(Demo()); }', 'void run() { DemoV2(); }'
  ];
  for (const code of rejected) assert.throws(() => validate('```cpp\n' + code + '\n```'), /调用示例/, code);
  const accepted = [
    'Demo();', 'void run() { Demo(); }', 'void Demo();\nvoid run() { Demo(); }',
    'auto run() { return Demo(); }', 'auto result = ::Demo();', 'AscendC::Demo();',
    'Container<Container<int>>::Demo();', '(Demo)();', '(::Demo)();',
    'auto value = scale * Demo();', 'auto value = ready && Demo();',
    'auto value = scale<T>() * Demo();', 'auto value = (scale) * Demo();',
    'object.Demo();', 'pointer->Demo();', 'object.template Demo<int>(value);',
    'Demo<Container<Container<int>>>(value);', 'consume(Demo());', 'if (ready) Demo();',
    'Other(), Demo();', 'consume(value, Demo());', 'int first = Other(), second = Demo();',
    'const char* url = "https://example.com"; Demo();',
    'const char* text = "/* not a comment */"; Demo();',
    'const char* text = R"tag(// " /* literal */)tag"; Demo();',
    String.raw`const char* text = "escaped \" and //"; Demo();`,
    'auto character = \'/\'; Demo();', 'Demo /* comment */ ();', 'Demo(1\'000);'
  ];
  for (const code of accepted) {
    for (const [opening, closing] of [['```cpp title="example"', '```'], ['~~~c++', '~~~'], ['```cxx', '```']]) {
      assert.doesNotThrow(() => validate(opening + '\n' + code + '\n' + closing), code);
    }
  }
  for (const example of ['```text\nDemo();\n```', '```Demo()\nno call\n```',
    '```cpp\nDemo\n```\n```cpp\n();\n```']) assert.throws(() => validate(example), /调用示例/, example);
  assert.throws(() => validate('```cpp\nDemo(); /* unclosed\n```'), /未闭合/);
  validator.validateExampleSafety('```cpp\n// forbidden\nDemo();\n```',
    [{ member: 'Demo', subject: 'value', staticPatterns: ['forbidden'] }]);
  assert.throws(() => validator.validateExampleSafety('```cpp\nconst char* value = "forbidden"; Demo();\n```',
    [{ member: 'Demo', subject: 'value', staticPatterns: ['forbidden'] }]), /受限制/);
});

test('AscendC operator coverage uses expressions rather than strings or declarators', () => {
  assert.equal(validator.hasMemberInvocation('static_assert(Demo());', 'Demo'), true);
  const precedingCalls = Array.from({ length: 64 }, (_, index) => `Other${index}()`).join(', ');
  assert.equal(validator.hasMemberInvocation(precedingCalls + ', Demo();', 'Demo'), true);
  assert.equal(validator.hasMemberInvocation('void ' + precedingCalls + ', Demo();', 'Demo'), false);
  for (const [name, valid, invalid] of [
    ['operator+', ['a.operator +(b);', 'auto c = a + b;'], ['Value operator+(Value a, Value b);', 'const char* text = "a + b";']],
    ['operator[]', ['auto x = a[0];'], ['int a[4];', 'const char* text = "a[0]";']],
    ['operator()', ['auto x = callable(value);'], ['void callable();', 'const char* text = "callable(value)";']]
  ]) {
    for (const code of valid) assert.equal(validator.hasMemberInvocation(code, name), true, code);
    for (const code of invalid) assert.equal(validator.hasMemberInvocation(code, name), false, code);
  }
});

test('public output checks cover both skills and preserve public install paths', t => {
  for (const policy of [
    require(path.join(acl, 'tools/scripts/lib/public-document-boundary.js')),
    require(path.join(asc, 'scripts/lib/public-document-boundary.js'))
  ]) {
    for (const text of ['/root/work/code.h', '/workspace/project/code.cpp',
      '<table><tr><td>DAV_C220</td></tr></table>']) {
      assert.throws(() => policy.validatePublicDocumentBoundary(text), undefined, text);
    }
    assert.doesNotThrow(() => policy.validatePublicDocumentBoundary(
      '```cpp\n#include "kernel_operator.h"\n```\n安装路径 /usr/local/Ascend/ascend-toolkit。'));
    assert.throws(() => policy.validatePublicDocumentBoundary('/srv/local-repo/include/demo.h',
      { localRoots: ['/srv/local-repo'] }));
    assert.throws(() => policy.validatePublicDocumentBoundary('K-123', { evidenceIds: ['K-123'] }));
  }
  const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
  write(path.join(output, '0_call_example.md'), '```cpp\n// /root/private/code.h\n```');
  prepareReview(op, output, '{"status":"pass"}');
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.notEqual(result.status, 0);
  assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
});

test('AscendC decisions and source paths are matched exactly', () => {
  for (const colon of [':', '：']) {
    assert.throws(() => ascFinalize.assertNoPendingActiveDecision(`状态${colon} 待确认`));
  }
  assert.doesNotThrow(() => ascFinalize.assertNoPendingActiveDecision('状态: 已确认\n## 决策变更记录\n状态: 待确认'));
  assert.ok(!ascFinalize.containsExactToken('`impl/a.hpp`', 'impl/a.h'));
  assert.ok(ascFinalize.containsExactToken('`impl/a.h`', 'impl/a.h'));
  assert.ok(ascFinalize.containsExactToken('`impl with spaces/a.h`', 'impl with spaces/a.h'));
});

function markdownTable(heading, headers, rows) {
  return `## ${heading}\n| ${headers.join(' | ')} |\n| ${headers.map(() => '---').join(' | ')} |\n` +
    rows.map(row => `| ${row.join(' | ')} |`).join('\n') + '\n';
}

test('AscendC public fact mappings cannot substitute a prefix-related fact ID', () => {
  const evidence = markdownTable('对外事实候选', ['事实 ID', '类别', '候选对外事实', '适用范围',
    '来源层与透传关系', '建议正文位置', '必含标识符'], [['K-1', '约束', '输入有效', '全部', '当前接口', '约束说明', 'value']]);
  const resolution = basis => markdownTable('对外事实映射', ['事实 ID', '正文位置', '适用范围',
    '来源与透传关系', '必含标识符', '正文处理', '依据'], [['K-1', '约束说明', '全部', '当前接口', 'value', '写入正文', basis]]);
  assert.throws(() => ascFinalize.assertPublicFactMappings(evidence, resolution('K-12、K-100')));
  assert.doesNotThrow(() => ascFinalize.assertPublicFactMappings(evidence, resolution('`K-1`、K-12')));
});

function createReadyAscendcSession(t, { family, name, source, extraFiles = {} }) {
  const repo = fixture(t);
  for (const [file, content] of Object.entries(extraFiles)) write(path.join(repo, file), content);
  const header = path.join(repo, `include/${family}/demo.h`);
  write(header, source);
  const created = run(path.join(asc, 'tools/ascendc-create-manifest.mjs'),
    ['--repo', repo, '--interface-file', header, '--api-name', name]);
  assert.equal(created.status, 0, created.stderr);
  const out = path.dirname(JSON.parse(created.stdout).manifest);
  const manifestPath = path.join(out, 'manifest.json');
  let manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const confirmed = run(path.join(asc, 'tools/ascendc-confirm-target-api.mjs'),
    ['--manifest', manifestPath, '--api-id', manifest.targetApi.id]);
  assert.equal(confirmed.status, 0, confirmed.stderr);
  for (const reference of manifest.sources.references) {
    const result = run(path.join(asc, 'tools/ascendc-record-reference.mjs'),
      ['--manifest', manifestPath, '--reference-id', reference.id, '--absent']);
    assert.equal(result.status, 0, result.stderr);
  }
  const completed = run(path.join(asc, 'tools/ascendc-record-reference.mjs'),
    ['--manifest', manifestPath, '--complete-reference-collection']);
  assert.equal(completed.status, 0, completed.stderr);
  const reading = run(path.join(asc, 'tools/ascendc-code-reading-plan.mjs'), ['--manifest', manifestPath]);
  assert.equal(reading.status, 0, reading.stderr);
  manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const plan = relations.buildApiRelationPlan({ repoPath: repo, documentApiName: manifest.documentApiName,
    targetApi: manifest.targetApi });
  const paths = plan.codeClosure.candidateFiles.map(item => item.path);
  return { repo, out, manifestPath, manifest, plan, paths };
}

test('AscendC operator API reaches finalize with confirmed sources and frozen decisions', t => {
  const { repo, out, manifestPath, manifest, plan, paths } = createReadyAscendcSession(t, {
    family: 'c_api', name: 'operator+',
    source: 'struct Value { __aicore__ inline Value operator +(const Value& rhs) { return rhs; } };\n'
  });
  const prototype = '## 函数原型\n```cpp\n__aicore__ inline Value operator+(const Value& rhs);\n```\n';
  const evidence = prototype + markdownTable('源码处理记录', ['候选文件', '进入原因', '实际核验范围', '处理结果',
    '关联事实或缺口'], paths.map(item => [item, '目标接口', '完整实现', '已核验并形成事实', '代码证据'])) +
    `\n## 自动索引与候选代码闭包\n- 初始待核验源码文件数：${paths.length}\n` +
    `- 阅读中新增直接证据链文件数：0\n- 最终直接证据链文件数：${paths.length}\n` +
    markdownTable('对外事实候选', ['事实 ID', '类别', '候选对外事实', '适用范围',
      '来源层与透传关系', '建议正文位置', '必含标识符'], [Array(7).fill('无')]);
  write(path.join(out, 'code-evidence.md'), evidence);
  const resolution = `---\nstatus: frozen\nunresolvedConflicts: 0\npendingReferenceApprovals: 0\nmanifestFingerprint: ${manifest.fingerprint}\n---\n` +
    markdownTable('覆盖与待补证', ['槽位', '适用范围', '状态', '依据'], [['原型', '全部', '已由代码证实', '代码证据']]) +
    markdownTable('下层资料适用性', ['下层资料事实', '当前成员与条件', '代码判断', '对当前用户的处理', '依据'],
      [['无', '全部', '无下层调用', '不适用', '代码证据']]) +
    markdownTable('调用示例组成依据', ['示例部分', '使用依据', '允许使用的资料'],
      [['示例来源', '当前 API 实现与同类 API 结构', '代码证据']]) +
    markdownTable('示例使用限制', ['成员', '受限制对象', '示例只能使用', '不得使用', '校验写法', '依据'],
      [['operator+', '无', '无', '无', '无', '代码证据']]) +
    markdownTable('对外事实映射', ['事实 ID', '正文位置', '适用范围', '来源与透传关系',
      '必含标识符', '正文处理', '依据'], [Array(7).fill('无')]);
  write(path.join(out, 'resolution.md'), resolution);
  const doc = '# operator+\n' + productSupport('ascendc') + '\n## 功能说明\n返回右侧对象。\n' + prototype +
    '\n## 参数说明\n<code>rhs</code>：右侧对象。\n## 返回值说明\n返回右侧对象。\n## 约束说明\n无\n' +
    '\n## 调用示例\n```cpp\nvoid run(Value a, Value b) { auto result = a.operator+(b); }\n```\n' +
    '结果示例如下：\n```text\n返回右侧对象。\n```\n';
  const document = path.join(out, 'operator_add_202610230000.md');
  write(document, doc);
  const offline = run(path.join(asc, 'tools/ascendc-finalize.mjs'),
    ['--output-dir', out, '--document', document], { env: offlineDependencyEnvironment(t) });
  assert.notEqual(offline.status, 0);
  assert.match(offline.stderr, /无法准备 Markdown 解析依赖/);
  assert.equal(fs.readFileSync(document, 'utf8'), doc);
  for (const args of [['--check'], []]) {
    const result = run(path.join(asc, 'tools/ascendc-finalize.mjs'),
      ['--output-dir', out, '--document', document, ...args]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
  }
  const stale = structuredClone(manifest.targetApi);
  stale.members.push({ name: 'Other' });
  assert.ok(!relations.matchesTargetSelection(plan, { repoPath: repo,
    documentApiName: manifest.documentApiName, targetApi: stale }));
  const planPath = path.join(repo, 'plan.json');
  write(planPath, JSON.stringify(plan));
  manifest.targetApi = stale;
  write(manifestPath, JSON.stringify(manifest));
  const invalid = run(path.join(asc, 'tools/ascendc-find-existing-document.mjs'),
    ['--manifest', manifestPath, '--relation-plan', planPath]);
  assert.notEqual(invalid.status, 0);
  assert.match(invalid.stderr, /不匹配/);
});

test('eval definitions retain semantic criteria and supported assertions under the repository parser', () => {
  const supported = new Set(['contains', 'not_contains', 'file_exists', 'file_list', 'file_contains', 'skill_activated']);
  for (const skill of [acl, asc]) {
    const evalPath = path.join(skill, '../../evals/evals.json');
    const data = JSON.parse(fs.readFileSync(evalPath, 'utf8'));
    assert.equal(data.eval_mode, 'file_based');
    for (const entry of data.evals) {
      assert.ok(entry.title && entry.config && Array.isArray(entry.files));
      assert.ok(entry.expected_output?.trim() || entry.expectations.length > 0);
      for (const expectation of entry.expectations) {
        assert.ok(supported.has(expectation.type));
        assert.equal(typeof expectation.pattern, 'string');
      }
    }
    const parser = path.join(root, 'tests/system/scripts/evals_json_parser.py');
    if (fs.existsSync(parser)) {
      const result = spawnSync('python', ['-B', '-c',
        'import importlib.util,sys,pathlib,json; s=importlib.util.spec_from_file_location("p",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); p=pathlib.Path(sys.argv[2]); raw=json.loads(p.read_text(encoding="utf-8")); d=m.parse_evals_json(p); assert len(d["evals"])==len(raw["evals"]); assert all(actual["expected_output"]==original["expected_output"] and actual["expectations"]==[{"type": e["type"], "pattern": e["pattern"]} for e in original["expectations"]] for original,actual in zip(raw["evals"],d["evals"]))',
        parser, evalPath], { encoding: 'utf8', windowsHide: true });
      assert.ifError(result.error);
      assert.equal(result.status, 0, result.stderr);
    }
  }
});

test('independently distributed policy copies stay consistent', () => {
  assert.deepEqual(fs.readFileSync(path.join(acl, 'tools/scripts/lib/public-document-boundary.js')),
    fs.readFileSync(path.join(asc, 'scripts/lib/public-document-boundary.js')));
  assert.deepEqual(fs.readFileSync(path.join(acl, 'tools/scripts/lib/markdown-resource-links.js')),
    fs.readFileSync(path.join(asc, 'scripts/lib/markdown-resource-links.js')));
  assert.deepEqual(fs.readFileSync(path.join(acl, 'tools/scripts/lib/source-snapshot.js')),
    fs.readFileSync(path.join(asc, 'scripts/lib/source-snapshot.js')));
  for (const file of ['markdown-it-loader.js', 'markdown-it-dependency.json']) {
    assert.deepEqual(fs.readFileSync(path.join(acl, 'tools/scripts/lib', file)),
      fs.readFileSync(path.join(asc, 'scripts/lib', file)));
  }
});

test('CDN dependency pins parser and license bytes and reuses a verified cache offline', async t => {
  const bases = [path.join(acl, 'tools/scripts/lib'), path.join(asc, 'scripts/lib')];
  const expected = JSON.parse(fs.readFileSync(path.join(bases[0], 'markdown-it-dependency.json'), 'utf8'));
  assert.equal(expected.name, 'markdown-it');
  assert.equal(expected.license, 'MIT');
  for (const base of bases) {
    assert.ok(!fs.existsSync(path.join(base, 'vendor/markdown-it')));
    assert.deepEqual(JSON.parse(fs.readFileSync(path.join(base, 'markdown-it-dependency.json'), 'utf8')), expected);
    const loader = require(path.join(base, 'markdown-it-loader.js'));
    loader.loadMarkdownIt();
    let downloads = 0;
    const directory = fixture(t);
    await loader.prepareDependencyCache(directory, async file => {
      downloads++;
      assert.ok(file.url.startsWith(`https://cdn.jsdelivr.net/npm/markdown-it@${expected.version}/`));
      return fs.readFileSync(path.join(loader.cacheDirectory(), file.name));
    });
    assert.equal(downloads, 2);
    await loader.prepareDependencyCache(directory, async () => { throw new Error('offline'); });
    for (const file of expected.files) {
      assert.equal(createHash('sha256').update(fs.readFileSync(path.join(directory, file.name))).digest('hex'), file.sha256);
    }
    assert.match(fs.readFileSync(path.join(directory, 'LICENSE'), 'utf8'), /Permission is hereby granted/);
    const MarkdownIt = require(path.join(directory, expected.files[0].name));
    assert.equal(new MarkdownIt().parse('# Example', {})[0].type, 'heading_open');
  }
});

test('CDN preparation rejects corrupt or unavailable downloads before publishing any new files', async t => {
  const base = path.join(asc, 'scripts/lib');
  const loader = require(path.join(base, 'markdown-it-loader.js'));
  const dependency = require(path.join(base, 'markdown-it-dependency.json'));
  loader.loadMarkdownIt();
  for (const brokenName of [dependency.files[0].name, 'LICENSE']) {
    const directory = fixture(t);
    await assert.rejects(loader.prepareDependencyCache(directory, async file =>
      file.name === brokenName ? Buffer.from('corrupt') : fs.readFileSync(path.join(loader.cacheDirectory(), file.name))),
    /摘要不匹配/);
    assert.deepEqual(fs.readdirSync(directory), []);
  }
  const directory = fixture(t);
  write(path.join(directory, dependency.files[0].name), 'damaged cached source');
  await assert.rejects(loader.prepareDependencyCache(directory, async () => { throw new Error('offline'); }), /offline/);
  assert.equal(fs.readFileSync(path.join(directory, dependency.files[0].name), 'utf8'), 'damaged cached source');
  await loader.prepareDependencyCache(directory, async file => fs.readFileSync(path.join(loader.cacheDirectory(), file.name)));
  assert.match(fs.readFileSync(path.join(directory, 'LICENSE'), 'utf8'), /Permission is hereby granted/);
});

test('concurrent CDN preparations publish identical verified cache files', async t => {
  const loader = require(path.join(asc, 'scripts/lib/markdown-it-loader.js'));
  const dependency = require(path.join(asc, 'scripts/lib/markdown-it-dependency.json'));
  loader.loadMarkdownIt();
  const directory = fixture(t);
  const download = async file => fs.readFileSync(path.join(loader.cacheDirectory(), file.name));
  await Promise.all([loader.prepareDependencyCache(directory, download), loader.prepareDependencyCache(directory, download)]);
  assert.deepEqual(fs.readdirSync(directory).sort(), dependency.files.map(file => file.name).sort());
  await loader.prepareDependencyCache(directory, async () => { throw new Error('must remain offline'); });
});

test('ACLNN dependency failure never executes corrupt cached code or overwrites an existing delivery', t => {
  const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
  const document = path.join(op, 'docs/aclnnDemo.md');
  write(document, 'existing delivery');
  const cacheRoot = fixture(t);
  const dependency = require(path.join(asc, 'scripts/lib/markdown-it-dependency.json'));
  const marker = path.join(cacheRoot, 'unverified-code-executed');
  write(path.join(cacheRoot, dependency.name, dependency.version, dependency.files[0].name),
    `require('node:fs').writeFileSync(${JSON.stringify(marker)}, 'unsafe');`);
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo'],
    { env: offlineDependencyEnvironment(t, cacheRoot) });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /无法准备 Markdown 解析依赖/);
  assert.equal(fs.readFileSync(document, 'utf8'), 'existing delivery');
  assert.ok(!fs.existsSync(marker));
  assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
});

test('built-in ACLNN mappings and standalone examples use only ACLNN product targets', t => {
  const mapping = require(path.join(acl, 'tools/scripts/lib/mapping-config.js'));
  const allowed = new Set(aclProductPolicy.supportProducts('aclnn').map(([name]) => name));
  for (const configFile of [undefined, path.join(acl, 'config/product_mapping.yml')]) {
    const config = mapping.loadProductMappingConfig(configFile);
    for (const item of [...config.chipMapping, ...config.npuArchMapping]) {
      for (const product of item.product.split('、')) assert.ok(allowed.has(product), product);
    }
    assert.deepEqual(mapping.mapChipIdentifier('UNKNOWN_MODEL', config), []);
    for (const chip of ['Ascend 310P AIC', 'Ascend 310P AIV', '__NPU_ARCH__==2002']) {
      assert.deepEqual(mapping.mapChipIdentifier(chip, config), ['Atlas推理系列产品']);
    }
  }
  const dir = fixture(t);
  write(path.join(dir, 'examples/arch2002/example.cpp'), 'int main() { return 0; }');
  const result = run(path.join(acl, 'tools/scripts/extract-call-example.js'), [dir]);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /<!-- npu="310p" id2000 -->/);
  assert.match(result.stdout, /Atlas推理系列产品/);
  assert.ok(!result.stdout.includes('AI Core') && !result.stdout.includes('Vector Core'));
  aclProductPolicy.validateProductDocument(productSupport('aclnn') + '\n## 调用示例\n' + result.stdout, 'aclnn');
});

test('parent installation includes runnable workflows without nested Skill discovery entries', t => {
  for (const module of [acl, asc]) {
    const parent = path.resolve(module, '../..');
    const installed = path.join(fixture(t), '中文 installed', path.basename(parent));
    fs.cpSync(parent, installed, { recursive: true });
    const installedModule = path.join(installed, 'references', path.basename(module));
    assert.ok(fs.existsSync(path.join(installed, 'SKILL.md')));
    assert.ok(!fs.existsSync(path.join(installed, 'reference')));
    assert.ok(fs.existsSync(path.join(installedModule, 'WORKFLOW.md')));
    assert.ok(!fs.existsSync(path.join(installedModule, 'SKILL.md')));
    assert.ok(!fs.existsSync(path.join(installedModule, 'references/common_terminology.md')));
    const workflowLink = `references/${path.basename(module)}/WORKFLOW.md`;
    assert.ok(fs.readFileSync(path.join(installed, 'SKILL.md'), 'utf8').includes(workflowLink));
    if (module === acl) {
      const { op } = setupPreprocess(fixture(t), successScript);
      const result = run(path.join(installedModule, 'tools/aclnn-preprocess.mjs'),
        [op, '--interface', 'aclnnDemo']);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.ok(result.stdout.includes(path.join(installedModule, 'config/product_mapping.yml')));
    } else {
      const repo = fixture(t);
      const header = path.join(repo, 'include/simt_api/demo.h');
      write(header, '__SIMT_DEVICE_FUNCTIONS_DECL__ inline void Demo(int count);');
      const result = run(path.join(installedModule, 'tools/ascendc-create-manifest.mjs'),
        ['--repo', repo, '--interface-file', header, '--api-name', 'Demo']);
      assert.equal(result.status, 0, result.stdout + result.stderr);
      const manifest = JSON.parse(fs.readFileSync(path.join(repo, 'outputs/Demo/manifest.json'), 'utf8'));
      assert.equal(manifest.targetApi.apiType, 'simt_api');
      assert.equal(manifest.targetApi.callingConvention, '__SIMT_DEVICE_FUNCTIONS_DECL__');
    }
  }
  const common = path.join(root, 'tests/system/scripts/common.py');
  if (fs.existsSync(common)) {
    const result = spawnSync('python', ['-B', '-c',
      'import sys,pathlib; sys.path.insert(0,str(pathlib.Path(sys.argv[1]).parent)); import common; names=common.discover_all_entities("skill_dirs","skill_whitelist"); assert {"ascendc-docs-gen","ascendc-api-docs-gen"}.issubset(names); assert "aclnn-full-api-docs-gen" not in names and "ascendc-full-api-docs-gen" not in names', common],
    { encoding: 'utf8', windowsHide: true });
    assert.ifError(result.error);
    assert.equal(result.status, 0, result.stderr);
  }
});

function writeScalarFinalizationCase(session, qualifier) {
  const { out, manifest, paths } = session;
  const prototype = `## 函数原型\n\`\`\`cpp\n${qualifier} inline void Demo(int count);\n\`\`\`\n`;
  const evidence = prototype + '\n## 代码事实\n- K-001：目标原型只有 int count 参数，返回 void；见目标声明。\n' +
    markdownTable('源码处理记录', ['候选文件', '进入原因', '实际核验范围', '处理结果', '关联事实或缺口'],
      paths.map(item => [item, '目标声明', '完整原型', '已核验并形成事实', 'K-001'])) +
    `\n## 自动索引与候选代码闭包\n- 初始待核验源码文件数：${paths.length}\n` +
    `- 阅读中新增直接证据链文件数：0\n- 最终直接证据链文件数：${paths.length}\n` +
    markdownTable('对外事实候选', ['事实 ID', '类别', '候选对外事实', '适用范围',
      '来源层与透传关系', '建议正文位置', '必含标识符'], [Array(7).fill('无')]);
  write(path.join(out, 'code-evidence.md'), evidence);
  const resolution = `---\nstatus: frozen\nunresolvedConflicts: 0\npendingReferenceApprovals: 0\nmanifestFingerprint: ${manifest.fingerprint}\n---\n` +
    markdownTable('覆盖与待补证', ['槽位', '适用范围', '状态', '依据'], [
      ['原型', 'Demo', '已由代码证实', 'K-001'],
      ['TPosition', 'Demo.count', '不适用', 'K-001；参数为标量 int，不包含 Tensor 位置']
    ]) +
    markdownTable('下层资料适用性', ['下层资料事实', '当前成员与条件', '代码判断', '对当前用户的处理', '依据'],
      [['无', 'Demo', '无下层资料', '不适用', 'K-001']]) +
    markdownTable('调用示例组成依据', ['示例部分', '使用依据', '允许使用的资料'],
      [['示例来源', '当前 API 实现与同类 API 结构', 'K-001']]) +
    markdownTable('示例使用限制', ['成员', '受限制对象', '示例只能使用', '不得使用', '校验写法', '依据'],
      [['Demo', '无', '无', '无', '无', 'K-001']]) +
    markdownTable('对外事实映射', ['事实 ID', '正文位置', '适用范围', '来源与透传关系',
      '必含标识符', '正文处理', '依据'], [Array(7).fill('无')]);
  write(path.join(out, 'resolution.md'), resolution);
  const finalDoc = `# Demo\n${productSupport('ascendc')}\n## 功能说明\n接收标量参数。\n${prototype}\n## 参数说明\n<code>count</code>：标量参数。\n## 返回值说明\n无\n## 约束说明\n无\n## 调用示例\n\`\`\`cpp\n${qualifier} inline void Example() { Demo(1); }\n\`\`\`\n结果示例如下：\n\`\`\`text\n本示例无返回值。\n\`\`\``;
  const document = path.join(out, 'Demo_202610230000.md');
  write(document, finalDoc);
  return { document, resolution, evidence };
}

test('AscendC finalization rejects examples without calls before writing the document', t => {
  const session = createReadyAscendcSession(t, { family: 'basic_api', name: 'Demo',
    source: '__aicore__ inline void Demo(int count) {}' });
  const { document } = writeScalarFinalizationCase(session, '__aicore__');
  const complete = fs.readFileSync(document, 'utf8');
  const originalExample = '__aicore__ inline void Example() { Demo(1); }';
  for (const replacement of ['void Demo(int count);', 'const char* text = "Demo(1)";',
    'void Demo(int count) {}', 'const char* text = R"tag(Demo(1))tag";']) {
    const invalid = complete.replace(originalExample, replacement);
    write(document, invalid);
    const result = run(path.join(asc, 'tools/ascendc-finalize.mjs'), ['--output-dir', session.out, '--document', document]);
    assert.notEqual(result.status, 0, replacement);
    assert.match(result.stderr, /未识别到文档成员的调用表达式/);
    assert.equal(fs.readFileSync(document, 'utf8'), invalid);
  }
  write(document, complete);
  const result = run(path.join(asc, 'tools/ascendc-finalize.mjs'), ['--output-dir', session.out, '--document', document]);
  assert.equal(result.status, 0, result.stdout + result.stderr);
});

test('directory routing smoke reaches finalization with evidence-backed non-applicable slots', t => {
  let last;
  for (const [family, qualifier] of [
    ['adv_api', '__aicore__'], ['basic_api', '__aicore__'], ['tensor_api', '__aicore__'],
    ['c_api', '__simd_callee__'], ['simt_api', '__SIMT_DEVICE_FUNCTIONS_DECL__']
  ]) {
    const session = createReadyAscendcSession(t, { family, name: 'Demo',
      source: `${qualifier} inline void Demo(int count) {}` });
    assert.equal(session.manifest.targetApi.apiType, family);
    assert.equal(session.manifest.targetApi.callingConvention, qualifier);
    const files = writeScalarFinalizationCase(session, qualifier);
    const result = run(path.join(asc, 'tools/ascendc-finalize.mjs'),
      ['--output-dir', session.out, '--document', files.document]);
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.ok(!fs.readFileSync(files.document, 'utf8').includes('TPosition'));
    last = { ...session, ...files };
  }
  for (const basis of ['无', 'K-001', 'K-001；原因：无', 'K-999；参数为标量，不涉及位置']) {
    const invalid = last.resolution.replace('K-001；参数为标量 int，不包含 Tensor 位置', basis);
    write(path.join(last.out, 'resolution.md'), invalid);
    const result = run(path.join(asc, 'tools/ascendc-finalize.mjs'),
      ['--output-dir', last.out, '--document', last.document, '--check']);
    assert.notEqual(result.status, 0, basis);
    assert.match(result.stderr, /不适用槽位/);
  }
  const pending = last.resolution.replace('| 不适用 |', '| 待确认 |');
  assert.throws(() => ascFinalize.assertFrozenResolution(pending, last.manifest.fingerprint, last.evidence), /未完成槽位/);
  write(path.join(last.out, 'resolution.md'), last.resolution);
  const complete = fs.readFileSync(last.document, 'utf8');
  for (const heading of ['功能说明', '参数说明', '返回值说明', '约束说明']) {
    const incomplete = complete.replace(new RegExp('^## ' + heading + '\\r?\\n[\\s\\S]*?(?=^## |$(?![\\s\\S]))', 'm'), '');
    write(last.document, incomplete);
    const failed = run(path.join(asc, 'tools/ascendc-finalize.mjs'),
      ['--output-dir', last.out, '--document', last.document, '--check']);
    assert.notEqual(failed.status, 0, heading);
    assert.match(failed.stderr, /必需章节/);
    assert.equal(fs.readFileSync(last.document, 'utf8'), incomplete);
  }
});

test('module diagnostics validate complete-format documents read-only and reject private content or changed prototypes', t => {
  for (const [module, type, prototypeText] of [
    [acl, 'aclnn', 'aclnnStatus aclnnDemo(int count);'],
    [asc, 'ascendc', '__aicore__ inline void Demo(int count);']
  ]) {
    const parent = path.resolve(module, '../..');
    const dir = fixture(t);
    const document = path.join(dir, 'doc.md');
    const prototype = path.join(dir, 'prototype.md');
    const protoDoc = `## 函数原型\n\`\`\`cpp\n${prototypeText}\n\`\`\`\n`;
    const content = `# Demo\n${productSupport(type)}\n## 功能说明\n公开描述。\n${protoDoc}`;
    write(document, content);
    write(prototype, protoDoc);
    const args = ['--document', document, '--prototype-file', prototype];
    if (type === 'ascendc') args.push('--api-name', 'Demo');
    const result = run(path.join(module, 'tools/check-api-document.mjs'), args, { cwd: parent });
    assert.equal(result.status, 0, result.stdout + result.stderr);
    assert.equal(fs.readFileSync(document, 'utf8'), content);
    for (const invalid of [content + '\n/root/private/code.h', content.replace('int count', 'float count'),
      content.replace('end id1', 'end id99')]) {
      write(document, invalid);
      const failed = run(path.join(module, 'tools/check-api-document.mjs'), args, { cwd: parent });
      assert.notEqual(failed.status, 0);
      assert.equal(fs.readFileSync(document, 'utf8'), invalid);
    }
    const missing = run(path.join(module, 'tools/check-api-document.mjs'), ['--document'], { cwd: parent });
    assert.notEqual(missing.status, 0);
    assert.match(missing.stderr, /缺少参数值/);
  }
});

test('ACLNN parent command sequence completes preprocessing, candidate assembly, review and delivery', t => {
  const parent = path.resolve(acl, '../..');
  const tools = 'references/aclnn-full-api-docs-gen/tools/';
  const op = fixture(t);
  write(path.join(op, 'op_api/aclnn_demo.h'), [
    'aclnnStatus aclnnDemoGetWorkspaceSize(const aclTensor* self, uint64_t* workspaceSize, aclOpExecutor** executor);',
    'aclnnStatus aclnnDemo(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);'
  ].join('\n'));
  write(path.join(op, 'examples/demo.cpp'), 'void run() { aclnnDemoGetWorkspaceSize(self, &size, &executor); aclnnDemo(workspace, size, executor, stream); }');
  const preprocessed = run(tools + 'aclnn-preprocess.mjs', [op, '--interface', 'aclnnDemo'], { cwd: parent });
  assert.equal(preprocessed.status, 0, preprocessed.stdout + preprocessed.stderr);
  const output = path.join(op, 'outputs/aclnnDemo');
  // Fixtures replace only the LLM-authored artifacts, not the deterministic command chain.
  write(path.join(output, '3_formula.md'), '## 最终文档公式块\n无');
  write(path.join(output, '4_generate_skeleton.md'), [
    '# aclnnDemo', productSupport('aclnn'), '## 功能说明', '执行示例接口。', '{{ACLNN_FORMULA_BLOCK_PLACEHOLDER}}',
    '## 函数原型', '{{ACLNN_PROTOTYPE_BLOCK_PLACEHOLDER}}', '## aclnnDemoGetWorkspaceSize', '获取工作空间和执行器。',
    '{{ACLNN_EXECUTE_INTERFACE_BLOCK_PLACEHOLDER}}', '## 约束说明', '无'
  ].join('\n'));
  const prepared = run(tools + 'aclnn-finalize.mjs', [op, '--interface', 'aclnnDemo', '--prepare'], { cwd: parent });
  assert.equal(prepared.status, 0, prepared.stderr);
  const digest = JSON.parse(fs.readFileSync(path.join(output, '4_review_input.json'), 'utf8')).reviewedDocumentSha256;
  write(path.join(output, '5_review_patch.json'), JSON.stringify({ status: 'pass', reviewedDocumentSha256: digest }));
  const finalized = run(tools + 'aclnn-finalize.mjs', [op, '--interface', 'aclnnDemo'], { cwd: parent });
  assert.equal(finalized.status, 0, finalized.stdout + finalized.stderr);
  const file = fs.readdirSync(output).find(name => /^aclnnDemo_\d+\.md$/.test(name));
  assert.ok(file);
  const checked = run(path.join(acl, 'tools/check-api-document.mjs'), ['--document', path.join(op, 'docs/aclnnDemo.md'),
    '--prototype-file', path.join(output, '0_prototype.md')], { cwd: parent });
  assert.equal(checked.status, 0, checked.stdout + checked.stderr);
});

test('member suffixes survive extraction and changed or unsupported suffixes cannot pass prototype checks', t => {
  const repo = fixture(t);
  const header = path.join(repo, 'include/tensor_api/value.h');
  const declarations = [
    '__aicore__ inline int GetValue() const;',
    '__aicore__ inline int GetValue();',
    '__aicore__ inline int GetValue() volatile & noexcept;',
    '__aicore__ inline int GetValue() && noexcept(noexcept(Helper()));'
  ];
  write(header, 'struct Value {\n' + declarations.map((line, index) => `/** Overload ${index}. */\n${line}`).join('\n') + '\n};');
  const extracted = run(path.join(asc, 'scripts/extract-prototype.js'),
    [repo, 'GetValue', 'tensor_api', '--input', header]);
  assert.equal(extracted.status, 0, extracted.stderr);
  const parsed = signatures.parseAscendcSignaturesFromContent(extracted.stdout);
  assert.equal(parsed.length, declarations.length);
  assert.deepEqual(parsed.map(item => item.suffix), ['const', '', 'volatile & noexcept', '&& noexcept(noexcept(Helper()))']);
  const contracts = require(path.join(asc, 'scripts/extract-public-contracts.js')).collectPublicContracts({
    repoPath: repo, sourceFiles: [header], members: ['GetValue'], callingConvention: '__aicore__'
  }).contracts;
  assert.equal(contracts.length, declarations.length);
  assert.deepEqual(contracts.map(item => item.declaration), declarations.map(line => line.slice(0, -1)));
  const parent = path.resolve(asc, '../..');
  const prototype = path.join(repo, 'prototype.md');
  const document = path.join(repo, 'GetValue.md');
  for (const declaration of declarations.filter(line => line !== declarations[1])) {
    write(prototype, '```cpp\n' + declaration + '\n```');
    const valid = '# GetValue\n' + productSupport('ascendc') + '\n## 函数原型\n```cpp\n' + declaration + '\n```';
    const args = ['--document', document, '--prototype-file', prototype, '--api-name', 'GetValue'];
    write(document, valid);
    const accepted = run(path.join(asc, 'tools/check-api-document.mjs'), args, { cwd: parent });
    assert.equal(accepted.status, 0, accepted.stderr);
    const invalid = valid.replace(declaration, declarations[1]);
    write(document, invalid);
    const rejected = run(path.join(asc, 'tools/check-api-document.mjs'), args, { cwd: parent });
    assert.notEqual(rejected.status, 0);
    assert.match(rejected.stderr, /原型不完整/);
    assert.equal(fs.readFileSync(document, 'utf8'), invalid);
  }
  const trailing = '__aicore__ inline auto GetValue() const noexcept -> int;';
  write(header, trailing);
  const preserved = run(path.join(asc, 'scripts/extract-prototype.js'), [repo, 'GetValue', 'tensor_api', '--input', header]);
  assert.equal(preserved.status, 0, preserved.stderr);
  assert.ok(preserved.stdout.includes(trailing));
  const trailingPrototype = '## 函数原型\n```cpp\n' + trailing + '\n```';
  assert.doesNotThrow(() => validator.validatePrototypeCompleteness({ apiName: 'GetValue', prototypeDoc: trailingPrototype, finalDoc: trailingPrototype }));
  assert.throws(() => validator.validatePrototypeCompleteness({ apiName: 'GetValue', prototypeDoc: trailingPrototype,
    finalDoc: trailingPrototype.replace('-> int', '-> float') }), /原型不完整/);
  const unknown = '__aicore__ inline int GetValue() noexcept(unclosed;';
  assert.throws(() => validator.validatePrototypeCompleteness({ apiName: 'GetValue',
    prototypeDoc: '```cpp\n' + unknown + '\n```', finalDoc: '## 函数原型\n```cpp\n' + unknown + '\n```' }), /后缀未闭合/);
});

test('full document structure rejects missing, empty, repeated or reordered chapters and wrong titles', () => {
  const headings = ['产品支持情况', '功能说明', '函数原型', '参数说明', '返回值说明', '约束说明', '调用示例'];
  const valid = '# Demo\n' + headings.map(heading => '## ' + heading + '\n内容。\n').join('\n');
  assert.doesNotThrow(() => validator.validateDocumentStructure(valid, 'Demo'));
  assert.doesNotThrow(() => validator.validateDocumentStructure(valid + '\n```text\n# Other\n## Fake\n```', 'Demo'));
  for (const heading of headings) {
    assert.throws(() => validator.validateDocumentStructure(valid.replace('## ' + heading + '\n内容。\n', ''), 'Demo'), /必需章节/);
    assert.throws(() => validator.validateDocumentStructure(valid.replace('## ' + heading + '\n内容。', '## ' + heading + '\n<!-- empty -->'), 'Demo'), /不能为空/);
  }
  for (const invalid of [valid + '\n## 功能说明\n重复。', valid.replace('## 功能说明', '## 参数说明').replace('## 参数说明\n内容。\n\n## 返回值说明', '## 功能说明\n内容。\n\n## 返回值说明')]) {
    assert.throws(() => validator.validateDocumentStructure(invalid, 'Demo'), /必需章节/);
  }
  assert.throws(() => validator.validateDocumentStructure(valid.replace('# Demo', '# Other'), 'Demo'), /一级标题/);
  assert.throws(() => validator.validateDocumentStructure(valid + '\n# Demo', 'Demo'), /一级标题/);
});

test('public boundaries use session provenance and report ambiguous references without treating names as secrets', t => {
  const modules = [require(path.join(acl, 'tools/scripts/lib/public-document-boundary.js')),
    require(path.join(asc, 'scripts/lib/public-document-boundary.js'))];
  for (const boundary of modules) {
    const check = boundary.validatePublicDocumentBoundary;
    for (const text of ['长度为 K-1。', '长度为 K-100。', '$K-1$', '$$\ny=\\sum_{k=0}^{K-1}x_k\n$$',
      '\\(R-2\\)', '\\[D-100\\]', '```cpp\n// K-001 is a public code literal\n```', '读取 model.json。']) {
      assert.doesNotThrow(() => check(text), text);
    }
    for (const text of ['K-001', '证据 K-1', '依据 R-12', '引用 C-2']) {
      assert.ok(check(text).length > 0);
    }
    assert.throws(() => check('结论对应 S-2。', { evidenceIds: ['S-2'] }), /证据 ID/);
    assert.ok(check('依据 $\\text{K-001}$。', { evidenceIds: ['K-001'] }).length > 0);
    const dir = fixture(t);
    const session = path.join(dir, 'outputs/Demo');
    const document = path.join(session, 'Demo.md');
    write(path.join(session, 'manifest.json'), '{}');
    write(path.join(session, 'code-evidence.md'), '### K-001 — 事实\n输入长度为 K-100。');
    write(path.join(dir, 'input/manifest.json'), '{}');
    const context = boundary.loadBoundaryContext('ascendc', session, document);
    assert.deepEqual(context.evidenceIds, ['K-001']);
    assert.throws(() => check('依据 K-001。', context), /证据 ID/);
    assert.throws(() => check('[清单](manifest.json)', context), /内部阶段文件/);
    assert.throws(() => check('<a href="manifest.json">清单</a>', context), /内部阶段文件/);
    assert.deepEqual(check('[用户配置](../../input/manifest.json)', context), []);
    assert.deepEqual(check('[公开资料](https://example.org/manifest.json)', context), []);
    assert.ok(check('依据 manifest.json。', context).length > 0);
    assert.deepEqual(check('依据 manifest.json。', { workflow: 'aclnn' }), []);
  }
  const prototype = '## 函数原型\n```cpp\n__aicore__ inline void Demo();\n```';
  const doc = '# Demo\n' + productSupport('ascendc') + '\n## 功能说明\n$$\ny=\\sum_{k=0}^{K-1}x_k\n$$\n' +
    prototype + '\n## 参数说明\n无\n## 返回值说明\n无\n## 约束说明\n无\n## 调用示例\n```cpp\nvoid run(){ Demo(); }\n```\n结果示例如下：\n```text\n完成\n```';
  assert.doesNotThrow(() => validator.validateFinalDocument({ apiName: 'Demo', prototypeDoc: prototype, finalDoc: doc }));
});

test('module session diagnostics distinguish internal artifacts from same-named user resources without rewriting documents', t => {
  for (const [module, workflow, stageFile, evidenceFile] of [
    [acl, 'aclnn', '0_prototype.md', '1_extract.md'],
    [asc, 'ascendc', 'manifest.json', 'code-evidence.md']
  ]) {
    const repo = fixture(t);
    const session = path.join(repo, 'outputs/Demo');
    const document = path.join(session, 'Demo.md');
    write(path.join(session, stageFile), 'stage content');
    write(path.join(session, evidenceFile), '- K-001：原型事实。');
    write(path.join(repo, 'input', stageFile), 'public input');
    const parent = path.resolve(module, '../..');
    const args = ['--document', document, '--session-dir', session];
    const base = '# Demo\n' + productSupport(workflow) + '\n## 功能说明\n';
    for (const [body, passes, warning] of [
      [`读取 [用户配置](../../input/${stageFile})。`, true, false],
      [`依据 [过程产物](${stageFile})。`, false, false],
      [`依据 [过程产物](${stageFile} "阶段清单")。`, false, false],
      [`依据 [过程产物][session]\n\n[session]: ${stageFile} "阶段清单"`, false, false],
      [`读取 [用户配置](<../../input/${stageFile}> "配置文件")。`, true, false],
      [`读取 [用户配置][session]\n\n[session]: ../../input/${stageFile} "配置文件"`, true, false],
      [`读取 [配置](${stageFile} "未闭合标题)。`, true, true],
      ['依据 K-001。', false, false],
      ['依据 $\\text{K-001}$。', true, true],
      ['元素数量为 K-100。', true, true],
      ['元素数量为 $K-100$。', true, false]
    ]) {
      const content = base + body;
      write(document, content);
      const result = run(path.join(module, 'tools/check-api-document.mjs'), args, { cwd: parent });
      assert.equal(result.status === 0, passes, result.stdout + result.stderr);
      if (warning) assert.match(result.stdout, /另有 .* 项审阅提示/);
      else if (passes) assert.ok(!result.stderr.includes('审阅提示'));
      assert.equal(fs.readFileSync(document, 'utf8'), content);
    }
  }
});

test('resource inspection resolves Markdown destinations and titles without treating public names as internal paths', t => {
  for (const boundary of [require(path.join(acl, 'tools/scripts/lib/public-document-boundary.js')),
    require(path.join(asc, 'scripts/lib/public-document-boundary.js'))]) {
    const dir = fixture(t);
    const documentPath = path.join(dir, 'Demo.md');
    const internalArtifacts = ['manifest.json', 'phase (draft).md', 'phase(draft).md'].map(name => path.join(dir, name));
    const context = { workflow: 'ascendc', documentPath, internalArtifacts };
    for (const text of [
      '[清单](manifest.json "阶段清单")', "[清单](manifest.json '阶段清单')",
      '[清单](manifest.json (阶段清单))', '[清单](<manifest.json> "阶段清单")',
      '[清单](manifest.json\n"阶段清单")', '[清单](manifest.json#section "阶段清单")',
      '[清单][SESSION]\n\n[session]: <manifest.json> "阶段清单"',
      '[清单][]\n\n[清单]: manifest.json', '[清单]\n\n[清单]: manifest.json',
      '[清单][session label]\n\n[ Session   Label ]: manifest.json\n  "阶段清单"',
      '[草稿](<phase (draft).md> "阶段文件")', '[草稿](phase\\(draft\\).md "阶段文件")',
      '[草稿](phase%20(draft).md "阶段文件")'
    ]) assert.throws(() => boundary.validatePublicDocumentBoundary(text, context), /内部阶段文件/, text);
    for (const text of [
      '[清单](../public/manifest.json "用户配置")', '[清单](<../public/manifest.json> (用户配置))',
      '[清单][session]\n\n[session]: ../public/manifest.json "用户配置"',
      '[清单][]\n\n[清单]: https://example.org/manifest.json "公开清单"',
      '[清单]\n\n[清单]: https://example.org/manifest.json',
      '[清单](../public/manifest.json "title has (parentheses)")',
      '[清单](../public/manifest.json "title has \\"quotes\\"")',
      '[清单][session]\n\n[session]: ../public/manifest.json\n[SESSION]: manifest.json'
    ]) assert.deepEqual(boundary.validatePublicDocumentBoundary(text, context), [], text);
    for (const text of ['[清单](manifest.json "未闭合标题)', '[清单][missing]', '[清单]: <manifest.json',
      '[草稿](phase(draft name).md)', '[草稿](phase\\ name.md)']) {
      assert.ok(boundary.validatePublicDocumentBoundary(text, context).some(warning => /需审阅/.test(warning)), text);
    }
  }
});

test('rendered-resource matrix distinguishes active links from code across encodings and block contexts', t => {
  // Self-authored matrix independently checked against markdown-it 12.3.2. Runtime
  // assertions use expected resource identities, not the runtime parser's answer.
  const destinations = new Set(['manifest.json', 'manifest&period;json', 'manifest.json?download=true#section']);
  for (const [index, char] of [...'manifest.json'].entries()) {
    const before = 'manifest.json'.slice(0, index), after = 'manifest.json'.slice(index + 1);
    destinations.add(before + '&#' + char.codePointAt(0) + ';' + after);
    destinations.add(before + '&#x' + char.codePointAt(0).toString(16) + ';' + after);
    destinations.add(before + '%' + char.codePointAt(0).toString(16) + after);
  }
  const active = [
    value => `[清单](${value})`, value => `[清单](${value} "标题")`,
    value => `[清单](<${value}> '标题')`, value => `[清单][s]\n\n[s]: ${value}`,
    value => `> [清单][s]\n>\n> [s]: ${value}`, value => `- [清单][s]\n\n  [s]: ${value}`,
    value => `说明\n    [清单](${value})`, value => `![图片](${value})`
  ];
  const code = [
    value => `    [清单](${value})`, value => `\t[清单](${value})`,
    value => '```md\n[清单](' + value + ')\n```', value => '~~~~md\n[清单](' + value + ')\n~~~~',
    value => '`[清单](' + value + ')`', value => `>     [清单](${value})`
  ];
  for (const boundary of [require(path.join(acl, 'tools/scripts/lib/public-document-boundary.js')),
    require(path.join(asc, 'scripts/lib/public-document-boundary.js'))]) {
    const dir = fixture(t), documentPath = path.join(dir, 'Demo.md');
    write(path.join(dir, 'manifest.json'), '{}');
    const context = { workflow: 'ascendc', documentPath, internalArtifacts: [path.join(dir, 'manifest.json')] };
    for (const value of destinations) {
      for (const format of active) {
        assert.throws(() => boundary.validatePublicDocumentBoundary(format(value), context), /内部阶段文件/, format(value));
        assert.deepEqual(boundary.validatePublicDocumentBoundary(format('../public/' + value), context), [], format(value));
      }
      for (const format of code) assert.deepEqual(boundary.validatePublicDocumentBoundary(format(value), context), [], format(value));
    }
    assert.throws(() => boundary.validatePublicDocumentBoundary('<a href="manif&#101;st.json">清单</a>', context), /内部阶段文件/);
    assert.throws(() => boundary.validatePublicDocumentBoundary('<a href="file:manifest.json">清单</a>', context), /内部阶段文件/);
    assert.doesNotThrow(() => boundary.validatePublicDocumentBoundary('<span>&lt;a href="manifest.json"&gt;</span>', context));
    assert.deepEqual(boundary.validatePublicDocumentBoundary('<!-- <a href="manifest.json">清单</a> -->', context), []);
    if (process.platform === 'win32') {
      assert.throws(() => boundary.validatePublicDocumentBoundary('[清单](MANIFEST.JSON)', context), /内部阶段文件/);
    }
    const reviewOnly = boundary.validatePublicDocumentBoundary('<script>const value = "<a href=manifest.json>";</script>', context);
    assert.ok(reviewOnly.some(warning => warning.includes('原始 HTML')));
  }
});

test('repository documentation excerpts preserve Tensor templates and register API signatures', t => {
  // Source excerpts test parser behavior, not SDK correctness or NPU execution.
  const samples = [
    { source: 'ops/ascendc-api-best-practices/references/api-arithmetic.md', name: 'Adds',
      code: 'template <typename T, bool isSetMask = true>\n__aicore__ inline void Adds(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);',
      convention: '__aicore__', params: ['dst', 'src', 'scalarValue', 'count'] },
    { source: 'ops/ascendc-api-best-practices/references/api-arithmetic.md', name: 'Muls',
      code: 'template <typename T, bool isSetMask = true>\n__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, const T& scalarValue, const int32_t& count);',
      convention: '__aicore__', params: ['dst', 'src', 'scalarValue', 'count'] },
    { source: 'ops/ascendc-regbase-best-practice/references/paradigms/common/vf-programming-rules.md', name: 'UpdateMask',
      code: 'template <typename T>\n__simd_callee__ inline MaskReg UpdateMask(uint32_t& scalarValue);',
      convention: '__simd_callee__', params: ['scalarValue'] }
  ];
  const repo = fixture(t);
  for (const sample of samples) {
    const source = path.join(root, sample.source);
    if (fs.existsSync(source)) {
      const blocks = [...fs.readFileSync(source, 'utf8').matchAll(/```cpp\s*\n([\s\S]*?)```/g)].map(match => match[1]);
      const original = blocks.flatMap(code => signatures.parseAscendcSignaturesFromContent(code))
        .find(item => item.funcName === sample.name);
      assert.ok(original, sample.source);
      assert.equal(original.key, signatures.parseAscendcSignaturesFromContent(sample.code)[0].key);
    }
    const header = path.join(repo, 'include/basic_api', sample.name + '.h');
    write(header, sample.code);
    const extracted = run(path.join(asc, 'scripts/extract-prototype.js'),
      [repo, sample.name, 'basic_api', '--input', header]);
    assert.equal(extracted.status, 0, extracted.stderr);
    const parsed = signatures.parseAscendcSignaturesFromContent(extracted.stdout);
    assert.equal(parsed.length, 1);
    assert.equal(parsed[0].callingConvention, sample.convention);
    assert.deepEqual(signatures.getParameterNames(parsed[0].rawParams), sample.params);
    const prototypeDoc = '## 函数原型\n```cpp\n' + sample.code + '\n```';
    const finalDoc = '## 函数原型\n```cpp\n' + require(path.join(asc, 'scripts/extract-prototype.js')).formatSignature(parsed[0]) + '\n```';
    assert.doesNotThrow(() => validator.validatePrototypeCompleteness({ apiName: sample.name, prototypeDoc, finalDoc }));
    assert.throws(() => validator.validatePrototypeCompleteness({ apiName: sample.name, prototypeDoc,
      finalDoc: finalDoc.replace(sample.name === 'UpdateMask' ? 'uint32_t&' : 'const LocalTensor<T>& dst',
        sample.name === 'UpdateMask' ? 'uint16_t&' : 'const GlobalTensor<T>& dst') }), /原型不完整/);
  }
});


function generatedPrototypeFixture(t, interfaceName = 'aclnnGenerated') {
  const dir = fixture(t);
  const repo = path.join(dir, 'source');
  const op = path.join(repo, 'activation/generated');
  write(path.join(op, 'op_host/registration.cpp'), '// Interface declarations are supplied by the build artifact.\n');
  const git = (...args) => {
    const result = spawnSync('git', ['-C', repo, ...args], { encoding: 'utf8', windowsHide: true });
    assert.ifError(result.error);
    assert.equal(result.status, 0, result.stderr);
    return result.stdout.trim();
  };
  git('init', '-q');
  git('add', '.');
  git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'source fixture');
  const header = path.join(dir, 'artifacts/public-interface.h');
  write(header, `aclnnStatus ${interfaceName}GetWorkspaceSize(const aclTensor *input, uint64_t *workspaceSize, aclOpExecutor **executor);\naclnnStatus ${interfaceName}(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);\n`);
  const entry = { operatorDir: path.relative(dir, op), interfaces: [interfaceName], kind: 'generated',
    sourceRevision: git('rev-parse', 'HEAD'), headers: [{ path: path.relative(dir, header),
      sha256: createHash('sha256').update(fs.readFileSync(header)).digest('hex') }] };
  const manifest = path.join(dir, 'sources.json');
  const save = () => write(manifest, JSON.stringify({ schemaVersion: 1, entries: [entry] }));
  save();
  return { dir, repo, op, header, entry, manifest, save, git, interfaceName };
}

test('generated and installed ACLNN headers enter shared discovery, extraction and preprocessing', t => {
  const f = generatedPrototypeFixture(t);
  for (const kind of ['generated', 'installed']) {
    f.entry.kind = kind;
    f.save();
    const detected = run(path.join(acl, 'tools/scripts/detect-interfaces.js'), [f.op, '--prototype-manifest', f.manifest]);
    assert.equal(detected.status, 0, detected.stderr);
    assert.match(detected.stdout, /aclnnGenerated/);
    const preprocessed = run(path.join(acl, 'tools/aclnn-preprocess.mjs'),
      [f.op, '--interface', f.interfaceName, '--prototype-manifest', f.manifest]);
    assert.equal(preprocessed.status, 0, preprocessed.stderr + preprocessed.stdout);
    const output = path.join(f.op, 'outputs', f.interfaceName);
    const prototype = fs.readFileSync(path.join(output, '0_prototype.md'), 'utf8');
    assert.match(prototype, /aclnnGeneratedGetWorkspaceSize/);
    assert.match(prototype, /aclnnGenerated\(/);
    const plan = JSON.parse(fs.readFileSync(path.join(output, '0_extract_plan.json'), 'utf8'));
    assert.equal(plan.prototypeSources[0].kind, kind);
    assert.equal(plan.prototypeSources[0].sourceRevision, f.entry.sourceRevision);
    assert.equal(plan.prototypeSources[0].sha256, f.entry.headers[0].sha256);
    assert.equal(plan.prototypeReportSha256, createHash('sha256').update(prototype).digest('hex'));
  }
});

test('external ACLNN prototypes reject missing evidence, version or digest mismatches and conflicting declarations', t => {
  for (const scenario of ['missing', 'revision', 'digest', 'conflict', 'wrong-api', 'dirty-source', 'wrong-root']) {
    const f = generatedPrototypeFixture(t);
    const args = [f.op, '--interface', f.interfaceName];
    if (scenario !== 'missing') args.push('--prototype-manifest', f.manifest);
    if (scenario === 'revision') f.entry.sourceRevision = '0'.repeat(40);
    if (scenario === 'digest') f.entry.headers[0].sha256 = '0'.repeat(64);
    if (scenario === 'wrong-api') f.entry.interfaces = ['aclnnOther'];
    if (scenario === 'conflict') {
      write(path.join(f.op, 'op_api/aclnn_generated.h'), `aclnnStatus ${f.interfaceName}(int different);\n`);
    }
    if (scenario === 'dirty-source') write(path.join(f.op, 'op_host/registration.cpp'), '// changed source\n');
    if (scenario === 'wrong-root') {
      const other = path.join(f.dir, 'unrelated-checkout');
      const cloned = spawnSync('git', ['clone', '-q', '--shared', f.repo, other], { encoding: 'utf8', windowsHide: true });
      assert.equal(cloned.status, 0, cloned.stderr);
      args.push('--project-root', other);
    }
    f.save();
    const result = run(path.join(acl, 'tools/aclnn-preprocess.mjs'), args);
    assert.notEqual(result.status, 0, scenario);
    assert.ok(!fs.existsSync(path.join(f.op, 'outputs', f.interfaceName, '0_extract_plan.json')), scenario);
    assert.match(result.stderr, /来源|原型|修订|摘要|声明|接口|工作树/);
  }
});

test('batch generated-header discovery binds each interface to its declared operator', t => {
  const f = generatedPrototypeFixture(t);
  const second = path.join(f.repo, 'activation/second');
  write(path.join(second, 'op_host/registration.cpp'), '// another operator\n');
  f.git('add', '.');
  f.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'second operator');
  f.entry.sourceRevision = f.git('rev-parse', 'HEAD');
  const otherHeader = path.join(f.dir, 'artifacts/second.h');
  write(otherHeader, 'aclnnStatus aclnnSecond(void);\n');
  const other = { ...f.entry, operatorDir: path.relative(f.dir, second), interfaces: ['aclnnSecond'],
    headers: [{ path: path.relative(f.dir, otherHeader), sha256: createHash('sha256').update(fs.readFileSync(otherHeader)).digest('hex') }] };
  write(f.manifest, JSON.stringify({ schemaVersion: 1, entries: [f.entry, other] }));
  const detected = run(path.join(acl, 'tools/aclnn-batch.mjs'), ['detect', f.repo, '--prototype-manifest', f.manifest]);
  assert.equal(detected.status, 0, detected.stderr);
  assert.match(detected.stdout, /合计接口数: 2/);
  const result = run(path.join(acl, 'tools/aclnn-batch.mjs'), ['preprocess', f.repo, '--prototype-manifest', f.manifest]);
  assert.equal(result.status, 0, result.stderr + result.stdout);
  for (const [op, name] of [[f.op, f.interfaceName], [second, 'aclnnSecond']]) {
    const output = path.join(op, 'outputs');
    assert.deepEqual(fs.readdirSync(output), [name]);
    assert.ok(fs.existsSync(path.join(output, name, '0_extract_plan.json')));
  }
});

test('source snapshot stays bound through generated-header review and final delivery', t => {
  for (const mutation of ['none', 'header', 'report', 'revision']) {
    const f = generatedPrototypeFixture(t, 'aclnnDemo');
    const result = run(path.join(acl, 'tools/aclnn-preprocess.mjs'), [f.op, '--prototype-manifest', f.manifest]);
    assert.equal(result.status, 0, result.stderr + result.stdout);
    const output = path.join(f.op, 'outputs/aclnnDemo');
    write(path.join(output, '4_generate_skeleton.md'), [
      '# aclnnDemo', productSupport('aclnn'), '## 功能说明', '执行示例接口。',
      '## 函数原型', '{{ACLNN_PROTOTYPE_BLOCK_PLACEHOLDER}}',
      '{{ACLNN_FORMULA_BLOCK_PLACEHOLDER}}', '## aclnnDemoGetWorkspaceSize', '获取工作空间和执行器。',
      '{{ACLNN_EXECUTE_INTERFACE_BLOCK_PLACEHOLDER}}', '## 约束说明', '无'
    ].join('\n'));
    write(path.join(output, '3_formula.md'), '## 最终文档公式块\n无');
    write(path.join(output, '0_call_example.md'), '## 调用示例\n```cpp\naclnnDemoGetWorkspaceSize(input, &workspaceSize, &executor);\naclnnDemo(workspace, workspaceSize, executor, stream);\n```');
    prepareReview(f.op, output, '{"status":"pass","skeletonEdits":[],"formulaBlockEdits":[]}');
    if (mutation === 'header') fs.appendFileSync(f.header, '\n// artifact changed\n');
    if (mutation === 'report') fs.appendFileSync(path.join(output, '0_prototype.md'), '\nchanged report\n');
    if (mutation === 'revision') {
      f.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '--allow-empty', '-qm', 'new revision');
    }
    const final = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [f.op, '--interface', 'aclnnDemo']);
    assert.equal(final.status === 0, mutation === 'none', final.stderr);
    assert.equal(fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)), mutation === 'none');
  }
});

test('prototype identity tolerates parameter renaming without mixing versions or neighboring APIs', t => {
  const f = generatedPrototypeFixture(t);
  write(path.join(f.op, 'op_api/aclnn_generated.h'), fs.readFileSync(f.header, 'utf8').replaceAll('workspaceSize', 'bytes'));
  // Untracked local declaration is intentionally used only for declaration conflict comparison.
  const extracted = run(path.join(acl, 'tools/scripts/extract-prototype.js'),
    [f.op, f.interfaceName, '--prototype-manifest', f.manifest]);
  assert.equal(extracted.status, 0, extracted.stderr);
  assert.match(extracted.stdout, /workspaceSize/);
  assert.ok(!extracted.stdout.includes(' bytes'));
  const unknown = run(path.join(acl, 'tools/scripts/extract-prototype.js'),
    [f.op, f.interfaceName + 'V2', '--prototype-manifest', f.manifest]);
  assert.notEqual(unknown.status, 0);
  write(f.header, `aclnnStatus ${f.interfaceName}(int);\n`);
  f.entry.headers[0].sha256 = createHash('sha256').update(fs.readFileSync(f.header)).digest('hex');
  f.save();
  write(path.join(f.op, 'op_api/aclnn_generated.h'), `aclnnStatus ${f.interfaceName}(float);\n`);
  const unnamed = run(path.join(acl, 'tools/scripts/extract-prototype.js'),
    [f.op, f.interfaceName, '--prototype-manifest', f.manifest]);
  assert.notEqual(unnamed.status, 0);
  assert.match(unnamed.stderr, /原型冲突/);
});

test('official references replace local knowledge copies with reachable source navigation', () => {
  for (const module of [acl, asc]) {
    assert.ok(!fs.existsSync(path.join(module, 'references/terminology_dict.yml')));
    const workflow = fs.readFileSync(path.join(module, 'WORKFLOW.md'), 'utf8');
    assert.ok(workflow.includes('references/official-sources.md'));
    assert.match(fs.readFileSync(path.join(module, 'references/official-sources.md'), 'utf8'),
      /https:\/\/gitcode\.com\/cann\/asc-devkit\/blob\/master\/docs\/zh\/guide\/technical_appendix\/concepts_and_terms\/glossary\.md/);
  }
  for (const old of ['broadcast关系.md', 'quantization_introduction.md', 'sparse_mode参数说明.md', '数据类型.md', 'figures']) {
    assert.ok(!fs.existsSync(path.join(acl, 'references', old)));
  }
});


test('independently distributed adapters import without network and share a verified offline cache', t => {
  const cacheRoot = fixture(t);
  const env = offlineDependencyEnvironment(t, cacheRoot);
  const base = path.join(asc, 'scripts/lib');
  const loader = require(path.join(base, 'markdown-it-loader.js'));
  const dependency = require(path.join(base, 'markdown-it-dependency.json'));
  loader.loadMarkdownIt();
  // A .cjs cache payload must load even when its user's directory is an ESM package.
  write(path.join(cacheRoot, 'package.json'), '{"type":"module"}');
  const installed = [fixture(t), fixture(t)];
  for (const dir of installed) {
    for (const file of ['markdown-resource-links.js', 'markdown-it-loader.js', 'markdown-it-dependency.json']) {
      write(path.join(dir, file), fs.readFileSync(path.join(base, file)));
    }
    const imported = run('--eval', [`require(${JSON.stringify(path.join(dir, 'markdown-resource-links.js'))});`], { env });
    assert.equal(imported.status, 0, imported.stderr);
  }
  assert.ok(!fs.existsSync(path.join(cacheRoot, dependency.name)));
  for (const file of dependency.files) {
    write(path.join(cacheRoot, dependency.name, dependency.version, file.name),
      fs.readFileSync(path.join(loader.cacheDirectory(), file.name)));
  }
  for (const dir of installed) {
    const parsed = run('--eval', [
      `const {documentHeadings}=require(${JSON.stringify(path.join(dir, 'markdown-resource-links.js'))});` +
      'if(documentHeadings("# Offline")[0].title!=="Offline")process.exit(1);'
    ], { env });
    assert.equal(parsed.status, 0, parsed.stderr);
  }
});

test('ACLNN canonical delivery follows the parent template and preserves other documents', t => {
  const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
  const document = path.join(op, 'docs/aclnnDemo.md');
  const other = path.join(op, 'docs/aclnnOther.md');
  const readme = path.join(op, 'README.md');
  write(other, '[Other](docs/aclnnOther.md)');
  write(readme, '[Demo](docs/aclnnDemo.md)\n[Other](docs/aclnnOther.md)');
  write(path.join(output, '0_call_example.md'), '```cpp\nvoid example() { aclnnDemo(); }\n```');
  prepareReview(op, output, '{"status":"pass"}');
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  const complete = fs.readFileSync(document, 'utf8');
  const archive = fs.readdirSync(output).find(name => /^aclnnDemo_\d+\.md$/.test(name));
  assert.equal(complete, fs.readFileSync(path.join(output, archive), 'utf8'));
  assert.match(complete, /aclnnStatus aclnnDemo\(void\)\n```/);
  assert.match(complete, /void example\(\) \{ aclnnDemo\(\); \}/);
  const parentExample = fs.readFileSync(path.join(acl, '../example-aclnn-md.md'), 'utf8');
  for (const [name] of aclProductPolicy.supportProducts('aclnn')) {
    const product = `<term>${name}</term>`;
    assert.ok(parentExample.includes(product));
    assert.ok(complete.includes(product));
  }
  assert.equal(fs.readFileSync(other, 'utf8'), '[Other](docs/aclnnOther.md)');
  assert.equal(fs.readFileSync(readme, 'utf8'), '[Demo](docs/aclnnDemo.md)\n[Other](docs/aclnnOther.md)');
});

test('ACLNN failed review preserves a previously published document', t => {
  const { op, output } = setupFinalize(fixture(t), '{"status":"fail","failureReason":"review failed"}');
  const document = path.join(op, 'docs/aclnnDemo.md');
  write(document, 'previously reviewed document');
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.notEqual(result.status, 0);
  assert.equal(fs.readFileSync(document, 'utf8'), 'previously reviewed document');
  assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
});

test('local Unix path checks cover punctuation, file types and directories without blocking public paths', () => {
  const policies = [require(path.join(acl, 'tools/scripts/lib/public-document-boundary.js')),
    require(path.join(asc, 'scripts/lib/public-document-boundary.js'))];
  const roots = ['home', 'root', 'Users', 'private', 'tmp', 'var/tmp', 'data', 'mnt', 'workspace'];
  const formats = [value => value, value => `源码路径：${value}`, value => `目录“${value}”`,
    value => `path=${value}`, value => `\`${value}\``,
    value => `<table><tr><td>${value}</td></tr></table>`,
    value => `\`\`\`cpp\n// ${value}\n\`\`\``];
  const reviewCases = ['源码路径：/root/private/code.h', '数据文件 /data/private/input.json',
    '工作目录 /workspace/private/build'];
  for (const policy of policies) {
    const check = text => policy.validatePublicDocumentBoundary(text, { localRoots: ['/workspace/current-repo'] });
    for (const text of reviewCases) assert.throws(() => check(text), /本机.*路径/, text);
    for (const root of roots) {
      for (const suffix of ['', '/private/input.json', '/private/build', '/private/中文目录/说明.txt']) {
        for (const format of formats) {
          const text = format('/' + root + suffix);
          assert.throws(() => check(text), /本机.*路径/, text);
        }
      }
    }
    for (const target of ['/data/private/input.json', '%2Fworkspace%2Fprivate%2Fbuild',
      'file:///root/private/code.h', 'file://localhost/data/private/input.json',
      'file:///%77orkspace/private/build']) {
      assert.throws(() => check(`[文件](${target})`), /本机.*路径/, target);
    }
    assert.throws(() => check('<a href="&#47;data/private/input.json">文件</a>'), /本机.*路径/);
    for (const text of ['安装路径 /usr/local/Ascend/ascend-toolkit。', '/opt/Ascend/ascend-toolkit/include',
      '公开头文件 /usr/include/stdint.h', '../data/private/input.json', './root/code.h',
      '/public/data/input.json', '/workspace-tools/bin', '/data_backup/input.json',
      'https://example.org/root/manual.md',
      '[公开配置](https://example.org/document?path=/root/manual.md)',
      'https://example.org/document?source=/workspace/current-repo/include/demo.h',
      '```cpp\n#include "kernel_operator.h"\n```']) {
      assert.doesNotThrow(() => check(text), text);
    }
  }
});

test('AscendC complete document validation rejects the reported local paths', () => {
  const prototype = '## 函数原型\n```cpp\n__aicore__ inline void Demo();\n```';
  const document = body => '# Demo\n' + productSupport('ascendc') + '\n## 功能说明\n' + body + '\n' +
    prototype + '\n## 参数说明\n无\n## 返回值说明\n无\n## 约束说明\n无\n## 调用示例\n' +
    '```cpp\nvoid example() { Demo(); }\n```\n结果示例如下：\n```text\n无返回值。\n```';
  const validate = body => validator.validateFinalDocument({ apiName: 'Demo', prototypeDoc: prototype,
    finalDoc: document(body), localRoots: ['/workspace/current-repo'] });
  assert.doesNotThrow(() => validate('安装路径 /usr/local/Ascend/ascend-toolkit。'));
  for (const text of ['源码路径：/root/private/code.h', '数据文件 /data/private/input.json',
    '工作目录 /workspace/private/build']) {
    assert.throws(() => validate(text), /本机.*路径/, text);
  }
});

test('ACLNN finalization blocks reported local paths and preserves the published document', t => {
  for (const text of ['源码路径：/root/private/code.h', '数据文件 /data/private/input.json',
    '工作目录 /workspace/private/build']) {
    const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
    const document = path.join(op, 'docs/aclnnDemo.md');
    write(document, 'previously published document');
    write(path.join(output, '0_call_example.md'), '## 调用示例\n' + text + '\n```cpp\nvoid run() { aclnnDemo(); }\n```');
    prepareReview(op, output, '{"status":"pass"}');
    const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
    assert.notEqual(result.status, 0, text);
    assert.match(result.stderr, /本机.*路径/);
    assert.equal(fs.readFileSync(document, 'utf8'), 'previously published document');
    assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
  }
});

test('registered roots use absolute path boundaries across encodings and preserve public relative resources', () => {
  for (const base of [path.join(acl, 'tools/scripts/lib'), path.join(asc, 'scripts/lib')]) {
    const { validatePublicDocumentBoundary: check } = require(path.join(base, 'public-document-boundary.js'));
    for (const root of ['/srv/current-repo', '/opt/custom workspace']) {
      const context = { localRoots: [root], documentPath: path.resolve('public/Demo.md') };
      for (const text of [`代码仓：${root}`, `${root}/input.json`, `[源码](${encodeURIComponent(root + '/input.json')})`,
        `<img src="${encodeURIComponent(root + '/input.png')}">`, `file://${encodeURI(root).replace('srv', '%73rv')}/input.json`]) {
        assert.throws(() => check(text, context), /本机|本地|代码仓/, text);
      }
      for (const text of [`[示例](..${root.replaceAll(' ', '%20')}/input.json)`, root + '-public/input.json',
        `https://example.org/?path=${root}/input.json`]) assert.doesNotThrow(() => check(text, context), text);
    }
    for (const text of ['[文件](%43%3A%2Fprivate%2Finput.json)', '[文件](file:///%43%3A/private/input.json)']) {
      assert.throws(() => check(text), /本机.*路径/, text);
    }
    assert.throws(() => check('源码路径：&#47;root/private/input.json'), /本机.*路径/);
    assert.throws(() => check('依据 K&#45;001。', { evidenceIds: ['K-001'] }), /证据 ID/);
    assert.deepEqual(check('https://example.org/?path=C:/Users/public/demo.h'), []);
    assert.deepEqual(check('https://example.org/?path=\\\\server\\share\\demo.h'), []);
  }
});

test('HTML resources share Markdown image identity checks without treating code as active HTML', t => {
  for (const base of [path.join(acl, 'tools/scripts/lib'), path.join(asc, 'scripts/lib')]) {
    const { validatePublicDocumentBoundary: check } = require(path.join(base, 'public-document-boundary.js'));
    const dir = fixture(t), documentPath = path.join(dir, 'Demo.md');
    const context = { documentPath, internalArtifacts: [path.join(dir, 'manifest.json')] };
    for (const format of [value => `<img src="${value}">`, value => `<source src='${value}'>`,
      value => `<video poster="${value}"></video>`, value => `<object data="${value}"></object>`,
      value => `<svg><image xlink:href="${value}"/></svg>`,
      value => `<img srcset="public.png 1x, ${value} 2x">`]) {
      assert.throws(() => check(format('manif%65st.json'), context), /内部阶段文件/);
      assert.throws(() => check(format('%2Froot%2Fprivate%2Finput.png'), context), /本机.*路径/);
      assert.deepEqual(check(format('../public/manifest.json'), context), []);
      assert.deepEqual(check('```html\n' + format('manif%65st.json') + '\n```', context), []);
      assert.deepEqual(check('    ' + format('manif%65st.json'), context), []);
    }
    assert.deepEqual(check('<img src="../public/manifest.json" src="manifest.json">', context), []);
    assert.ok(check('<img srcset="public.png unknown">', context).length > 0);
    assert.ok(check('<span style="background:url(manifest.json)">示例</span>', context)
      .some(warning => warning.includes('CSS')));
  }
});

test('product and spacing normalization preserve CommonMark code blocks and literal code whitespace', t => {
  const literal = 'const char* value = "Atlas 推理系列产品";  \n\n\n\n// `keep`\n';
  for (const [type, policy] of [['aclnn', aclProductPolicy], ['ascendc', ascProductPolicy]]) {
    for (const code of ['    ' + literal.replace(/\n(?=.)/g, '\n    '), '\tconst char* name = "Ascend 950PR";\n',
      '~~~~cpp\n' + literal + '~~~~\n', '> ```cpp\n> const char* name = "Ascend 950PR";\n> ```\n',
      '- Example:\n\n      const char* name = "Ascend 950PR";\n']) {
      assert.equal(policy.normalizeProductNames(code, type), code);
      assert.equal(validator.normalizeInlineCodeOutsideFences(code), code);
      if (type === 'aclnn') {
        const dir = fixture(t), file = path.join(dir, 'code.md');
        write(file, code);
        const result = run(path.join(acl, 'tools/scripts/normalize-doc-spacing.js'), [file]);
        assert.equal(result.status, 0, result.stderr);
        assert.equal(result.stdout.trimEnd(), code.trimEnd());
      }
    }
  }
  const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
  const code = '```cpp\nvoid run() { aclnnDemo(); }\nconst char* sample = R"(\n$\n\n\n\nAtlas 推理系列产品\n)";  \n```';
  write(path.join(output, '0_call_example.md'), code);
  prepareReview(op, output, '{"status":"pass"}');
  const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
  assert.equal(result.status, 0, result.stderr);
  assert.ok(fs.readFileSync(path.join(op, 'docs/aclnnDemo.md'), 'utf8').includes(code));
});

test('ACLNN complete delivery rejects wrong titles, missing chapters and examples without a target call', t => {
  for (const mutation of ['title', 'missing', 'empty', 'example', 'comment-only', 'info-only', 'declaration-only', 'definition-only', 'text-only']) {
    const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
    const target = path.join(op, 'docs/aclnnDemo.md');
    write(target, 'previous published document');
    const skeleton = path.join(output, '4_generate_skeleton.md');
    let content = fs.readFileSync(skeleton, 'utf8');
    if (mutation === 'title') content = content.replace('# aclnnDemo', '# OtherApi');
    if (mutation === 'missing') content = content.replace('## 约束说明\n无', '');
    if (mutation === 'empty') content = content.replace('## 约束说明\n无', '## 约束说明\n<!-- empty -->');
    write(skeleton, content);
    if (mutation === 'example') write(path.join(output, '0_call_example.md'), '示例待补齐。');
    if (mutation === 'comment-only') write(path.join(output, '0_call_example.md'), '```cpp\n// aclnnDemo();\nconst char* name = "aclnnDemo()";\n```');
    if (mutation === 'info-only') write(path.join(output, '0_call_example.md'), '```aclnnDemo()\n// no call\n```');
    if (mutation === 'declaration-only') write(path.join(output, '0_call_example.md'), '```cpp\nextern aclnnStatus aclnnDemo(void);\n```');
    if (mutation === 'definition-only') write(path.join(output, '0_call_example.md'), '```cpp\naclnnStatus aclnnDemo(void) { return 0; }\n```');
    if (mutation === 'text-only') write(path.join(output, '0_call_example.md'), '```text\naclnnDemo();\n```');
    prepareReview(op, output, '{"status":"pass"}');
    const result = run(path.join(acl, 'tools/aclnn-finalize.mjs'), [op, '--interface', 'aclnnDemo']);
    assert.notEqual(result.status, 0, mutation);
    assert.match(result.stderr, /一级标题|必需章节|章节不能为空|实际调用/);
    assert.equal(fs.readFileSync(target, 'utf8'), 'previous published document');
    assert.ok(!fs.readdirSync(output).some(name => /^aclnnDemo_\d+\.md$/.test(name)));
  }
});

test('ACLNN example checks preserve valid calls after declarations and distinguish code from fence metadata', () => {
  const { validateAclnnDocument } = require(path.join(acl, 'tools/scripts/lib/aclnn-document-validator.js'));
  const prototype = '```cpp\naclnnStatus aclnnDemo(void)\n```';
  const base = '# aclnnDemo\n' + productSupport('aclnn') + '\n## 功能说明\n执行接口。\n## 函数原型\n' + prototype +
    '\n## 约束说明\n无\n## 调用示例\n';
  for (const code of ['void run() { aclnnDemo(); }', 'aclnnStatus run() { return aclnnDemo(); }',
    'extern aclnnStatus ACLNN_API aclnnDemo(void);\nvoid run() { aclnnStatus status = ::aclnnDemo(); }',
    'void run() { auto status = aclnnDemo /* comment */ (); }']) {
    for (const [open, close] of [['```cpp title="sample"', '```'], ['~~~c++', '~~~'], ['```cxx', '```']]) {
      assert.doesNotThrow(() => validateAclnnDocument(base + open + '\n' + code + '\n' + close, 'aclnnDemo', prototype));
    }
  }
});

test('file URL hosts retain network-share identity and malformed URLs produce review notes', () => {
  for (const base of [path.join(acl, 'tools/scripts/lib'), path.join(asc, 'scripts/lib')]) {
    const { validatePublicDocumentBoundary: check } = require(path.join(base, 'public-document-boundary.js'));
    for (const target of ['file://server/share/private.json', 'file://%73erver/share/private.json']) {
      for (const content of [target, `[文件](${target})`, `<a href="${target}">文件</a>`, `<img src="${target}">`]) {
        assert.throws(() => check(content), /网络共享路径/, content);
      }
    }
    for (const target of ['file:///srv/private%QQ', 'file://[invalid/share/file.json']) {
      assert.ok(check(target).some(note => note.includes('无法解析的文件 URL')), target);
    }
    assert.deepEqual(check('file://localhost/usr/local/Ascend/ascend-toolkit'), []);
    assert.deepEqual(check('https://example.org/?url=file://server/share/file.json'), []);
  }
});

test('AscendC evidence snapshots reject changed dependencies and cannot be refreshed over stale evidence', t => {
  const session = createReadyAscendcSession(t, { family: 'c_api', name: 'Demo',
    source: '#include "helper.h"\n__aicore__ inline void Demo(int count) { Helper(count); }\n',
    extraFiles: { 'include/c_api/helper.h': '__aicore__ inline void Helper(int count) {}\n' } });
  const { document } = writeScalarFinalizationCase(session, '__aicore__');
  const finalize = () => run(path.join(asc, 'tools/ascendc-finalize.mjs'), ['--output-dir', session.out, '--document', document, '--check']);
  assert.equal(finalize().status, 0);
  write(path.join(session.repo, 'unused.cpp'), 'unrelated change');
  assert.equal(finalize().status, 0);
  const original = fs.readFileSync(document, 'utf8');
  const manifest = fs.readFileSync(session.manifestPath, 'utf8');
  write(path.join(session.repo, 'include/c_api/helper.h'), '__aicore__ inline void Helper(int count) { if (count < 10) __builtin_trap(); }\n');
  const changed = finalize();
  assert.notEqual(changed.status, 0);
  assert.match(changed.stderr, /源码证据文件已变化/);
  assert.equal(fs.readFileSync(document, 'utf8'), original);
  const retry = run(path.join(asc, 'tools/ascendc-code-reading-plan.mjs'), ['--manifest', session.manifestPath]);
  assert.notEqual(retry.status, 0);
  assert.equal(fs.readFileSync(session.manifestPath, 'utf8'), manifest);
});

test('ACLNN preprocessing snapshots external validation code before candidate review', t => {
  const { op, output } = setupFinalize(fixture(t), '{"status":"pass"}');
  const skeleton = fs.readFileSync(path.join(output, '4_generate_skeleton.md'), 'utf8');
  const repo = path.dirname(op), helper = path.join(repo, 'common/check.h');
  write(helper, 'inline int Check() { return 0; }\n');
  write(path.join(op, 'op_api/aclnn_demo.cpp'), '#include "../../common/check.h"\naclnnStatus aclnnDemo(void) { return Check(); }\n');
  const prepared = run(path.join(acl, 'tools/aclnn-preprocess.mjs'), [op, '--project-root', repo, '--interface', 'aclnnDemo']);
  assert.equal(prepared.status, 0, prepared.stderr);
  const plan = JSON.parse(fs.readFileSync(path.join(output, '0_extract_plan.json'), 'utf8'));
  assert.ok(plan.sourceSnapshot.some(item => path.resolve(item.file) === helper));
  write(path.join(output, '4_generate_skeleton.md'), skeleton);
  write(path.join(output, '3_formula.md'), '## 最终文档公式块\n无');
  write(path.join(output, '0_call_example.md'), '```cpp\nvoid run() { aclnnDemo(); }\n```');
  prepareReview(op, output, '{"status":"pass"}');
  const args = [op, '--interface', 'aclnnDemo'];
  assert.equal(run(path.join(acl, 'tools/aclnn-finalize.mjs'), args).status, 0);
  const target = path.join(op, 'docs/aclnnDemo.md'), original = fs.readFileSync(target, 'utf8');
  write(helper, 'inline int Check() { return 1; }\n');
  const changed = run(path.join(acl, 'tools/aclnn-finalize.mjs'), args);
  assert.notEqual(changed.status, 0);
  assert.match(changed.stderr, /源码证据文件已变化/);
  assert.equal(fs.readFileSync(target, 'utf8'), original);
});
