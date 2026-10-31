#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

/**
 * AscendC interface detector.
 *
 * Usage:
 *   node scripts/detect-interfaces.js <AscendC file or directory> [projectRoot] --output <jsonFile>
 *
 * The detector intentionally follows the same broad interface-discovery rules used by
 * tools/ascendc-create-manifest.mjs: documented APIs, basic_api device declarations,
 * reg_compute __simd_callee__ declarations, and simt_api __SIMT_DEVICE_FUNCTIONS_DECL__ declarations.
 */

const fs = require('fs');
const path = require('path');
const { buildDocumentFamilies } = require('./lib/ascendc-api-family');
const { parseAscendcSignaturesFromContent } = require('./lib/ascendc-signatures');

const API_TYPES = ['adv_api', 'basic_api', 'c_api', 'simt_api', 'tensor_api'];

function parseArgs(argv) {
  const args = { pretty: true };
  for (let index = 0; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === '--output' || arg === '-o') {
      args.output = argv[++index];
    } else if (arg === '--compact') {
      args.pretty = false;
    } else if (arg === '--help' || arg === '-h') {
      args.help = true;
    } else if (!args.inputPath) {
      args.inputPath = arg;
    } else if (!args.projectRoot) {
      args.projectRoot = arg;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

function usage() {
  console.log('Usage: node scripts/detect-interfaces.js <AscendC file or directory> [projectRoot] --output <jsonFile> [--compact]');
  console.log('Example: node scripts/detect-interfaces.js <repo-root>/include <repo-root> --output <output-file>');
}

function removeComments(source) {
  return source
    .replace(/\/\/.*$/gm, '')
    .replace(/\/\*[\s\S]*?\*\//g, '');
}

function isSourceFile(filePath) {
  return /\.(h|hpp|hh|cpp|cc|cxx)$/i.test(filePath);
}

function relPath(filePath, base) {
  if (!base) return filePath;
  return path.relative(base, filePath).replace(/\\/g, '/');
}

function detectApiTypeFromPath(filePath) {
  const normalized = filePath.replace(/\\/g, '/');
  for (const apiType of API_TYPES) {
    if (normalized.includes(`/include/${apiType}/`) || normalized.endsWith(`/include/${apiType}`) || normalized.includes(`/impl/${apiType}/`) || normalized.endsWith(`/impl/${apiType}`)) return apiType;
  }
  return '';
}

function addApiName(names, name) {
  if (isIgnoredApiCandidateName(name)) return;
  names.add(name);
}

function isIgnoredApiCandidateName(name) {
  return [
    '_',
    '__in_pipe__',
    '__out_pipe__',
    '__inout_pipe__',
    'operator',
    'decltype',
    'ASSERT',
    'DEBUG_CODE',
    'EXTERN_IMPL_BUFPOOL'
  ].includes(name);
}

function collectSourceFiles(inputPath) {
  const stat = fs.statSync(inputPath);
  if (stat.isFile()) return isSourceFile(inputPath) ? [inputPath] : [];

  const result = [];
  const walk = (dir) => {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
    catch { return; }
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!['tests', 'test', 'build', 'output', 'outputs'].includes(entry.name.toLowerCase())) walk(fullPath);
      } else if (entry.isFile() && isSourceFile(fullPath)) {
        result.push(fullPath);
      }
    }
  };
  walk(inputPath);
  return result.sort((a, b) => a.localeCompare(b));
}

function parseInterfaceNames(content) {
  const source = removeComments(content);
  const names = new Set();

  const documentedRe = /\/\*(?:\s*!|\*)[\s\S]*?(?:\\brief|@brief)[\s\S]*?\*\/\s*(?:template\s*<[^;{}]*?>\s*)?(?:(?:__aicore__|__simd_callee__|__SIMT_DEVICE_FUNCTIONS_DECL__|static|inline|__in_pipe__\([^)]*\)|__out_pipe__\([^)]*\)|__inout_pipe__\([^)]*\))\s+)*(?:void|bool|auto|int\d+_t|uint\d+_t|float|double|[A-Za-z_][A-Za-z0-9_:<>*&\s]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;
  const qualifiedRe = /(?:template\s*<[^;{}]*?>\s*)?(?:(?:__aicore__|__simd_callee__|__SIMT_DEVICE_FUNCTIONS_DECL__|static|inline|__in_pipe__\([^)]*\)|__out_pipe__\([^)]*\)|__inout_pipe__\([^)]*\))\s+)*(?:__aicore__|__simd_callee__|__SIMT_DEVICE_FUNCTIONS_DECL__)\s+(?:(?:static|inline|__in_pipe__\([^)]*\)|__out_pipe__\([^)]*\)|__inout_pipe__\([^)]*\))\s+)*(?:void|bool|auto|int\d+_t|uint\d+_t|float|double|[A-Za-z_][A-Za-z0-9_:<>*&\s]*?)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g;

  let match;
  while ((match = documentedRe.exec(content)) !== null) addApiName(names, match[1]);
  while ((match = qualifiedRe.exec(source)) !== null) addApiName(names, match[1]);

  // The shared signature parser understands symbolic C++ operator names.
  // Keep interface detection and prototype extraction aligned instead of
  // treating the return type (for example decltype) as an API name.
  for (const signature of parseAscendcSignaturesFromContent(content)) {
    if (signature.callingConvention !== 'unqualified') addApiName(names, signature.funcName);
  }
  return [...names].sort((a, b) => a.localeCompare(b));
}

function identityId(apiType, callingConvention, name) {
  return `${apiType || 'unknown'}|${callingConvention || 'unknown'}|${name}`;
}

function fileReference(filePath, projectRoot, apiType) {
  return {
    path: filePath,
    relativePath: relPath(filePath, projectRoot),
    apiType
  };
}

function detectInterfaces(inputPath, projectRoot) {
  const files = collectSourceFiles(inputPath);
  const symbolMap = new Map();
  const byFile = [];
  const allSignatures = [];

  for (const filePath of files) {
    let content;
    try { content = fs.readFileSync(filePath, 'utf8'); }
    catch { continue; }

    const names = parseInterfaceNames(content);
    if (names.length === 0) continue;
    const apiType = detectApiTypeFromPath(filePath) || 'unknown';
    const publicNames = new Set(names);
    const signatures = parseAscendcSignaturesFromContent(content)
      .filter(signature => publicNames.has(signature.funcName))
      .map(signature => ({ ...signature, sourceFile: filePath, apiType }));
    allSignatures.push(...signatures);

    const fileSymbolIds = new Set();
    for (const signature of signatures) {
      const id = identityId(apiType, signature.callingConvention, signature.funcName);
      fileSymbolIds.add(id);
      if (!symbolMap.has(id)) {
        symbolMap.set(id, {
          id,
          name: signature.funcName,
          apiType,
          callingConvention: signature.callingConvention,
          files: new Map()
        });
      }
      symbolMap.get(id).files.set(filePath, fileReference(filePath, projectRoot, apiType));
    }

    for (const name of names) {
      if (signatures.some(signature => signature.funcName === name)) continue;
      const id = identityId(apiType, 'unknown', name);
      fileSymbolIds.add(id);
      if (!symbolMap.has(id)) {
        symbolMap.set(id, {
          id,
          name,
          apiType,
          callingConvention: 'unknown',
          files: new Map()
        });
      }
      symbolMap.get(id).files.set(filePath, fileReference(filePath, projectRoot, apiType));
    }

    byFile.push({
      path: filePath,
      relativePath: relPath(filePath, projectRoot),
      apiType,
      symbolIds: [...fileSymbolIds].sort()
    });
  }

  const symbols = [...symbolMap.values()]
    .map(symbol => ({
      id: symbol.id,
      name: symbol.name,
      apiType: symbol.apiType,
      callingConvention: symbol.callingConvention,
      fileCount: symbol.files.size,
      files: [...symbol.files.values()].sort((a, b) => a.relativePath.localeCompare(b.relativePath))
    }))
    .sort((a, b) => a.name.localeCompare(b.name) ||
      a.apiType.localeCompare(b.apiType) ||
      a.callingConvention.localeCompare(b.callingConvention));

  const signaturesByApiType = new Map();
  for (const signature of allSignatures) {
    if (!signaturesByApiType.has(signature.apiType)) signaturesByApiType.set(signature.apiType, []);
    signaturesByApiType.get(signature.apiType).push(signature);
  }

  const interfaces = [];
  const coveredSymbolIds = new Set();
  for (const [apiType, signatures] of signaturesByApiType) {
    for (const family of buildDocumentFamilies(signatures)) {
      const memberNames = new Set(family.members.map(member => member.name));
      const familyFiles = new Map();
      for (const signature of signatures) {
        if (signature.callingConvention !== family.callingConvention ||
          !memberNames.has(signature.funcName)) continue;
        familyFiles.set(
          signature.sourceFile,
          fileReference(signature.sourceFile, projectRoot, signature.apiType)
        );
        coveredSymbolIds.add(identityId(apiType, family.callingConvention, signature.funcName));
      }
      interfaces.push({
        id: identityId(apiType, family.callingConvention, family.name),
        name: family.name,
        apiType,
        callingConvention: family.callingConvention,
        members: family.members,
        fileCount: familyFiles.size,
        files: [...familyFiles.values()].sort((a, b) => a.relativePath.localeCompare(b.relativePath)),
        diagnostics: family.diagnostics
      });
    }
  }

  for (const symbol of symbols) {
    if (coveredSymbolIds.has(symbol.id)) continue;
    interfaces.push({
      id: symbol.id,
      name: symbol.name,
      apiType: symbol.apiType,
      callingConvention: symbol.callingConvention,
      members: [{ name: symbol.name, role: 'regular', relation: 'document-root' }],
      fileCount: symbol.fileCount,
      files: symbol.files,
      diagnostics: ['Shared signature parser did not resolve this interface; kept as a single-member document.']
    });
  }

  interfaces.sort((a, b) => a.name.localeCompare(b.name) ||
    a.apiType.localeCompare(b.apiType) ||
    a.callingConvention.localeCompare(b.callingConvention));

  const interfaceIdsByFile = new Map();
  for (const item of interfaces) {
    for (const file of item.files) {
      if (!interfaceIdsByFile.has(file.path)) interfaceIdsByFile.set(file.path, []);
      interfaceIdsByFile.get(file.path).push(item.id);
    }
  }
  const outputFiles = byFile.map(file => ({
    ...file,
    interfaceIds: [...new Set(interfaceIdsByFile.get(file.path) || [])].sort()
  }));

  const byApiType = new Map();
  for (const symbol of symbols) {
    byApiType.set(symbol.apiType, (byApiType.get(symbol.apiType) || 0) + 1);
  }

  const summary = {
    filesScanned: files.length,
    filesWithInterfaces: byFile.length,
    symbolCount: symbols.length,
    interfaceCount: interfaces.length,
    byApiType: Object.fromEntries([...byApiType.entries()].sort((a, b) => a[0].localeCompare(b[0])))
  };

  return { summary, interfaces, symbols, files: outputFiles };
}

function main() {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (args.help || !args.inputPath) {
      usage();
      process.exit(args.help ? 0 : 1);
    }

    const inputPath = path.resolve(args.inputPath);
    if (!fs.existsSync(inputPath)) throw new Error(`Input path does not exist: ${inputPath}`);

    const projectRoot = args.projectRoot ? path.resolve(args.projectRoot) : undefined;
    const outputPath = path.resolve(args.output || 'ascendc_interfaces.json');

    const detected = detectInterfaces(inputPath, projectRoot);
    const report = {
      schemaVersion: 3,
      kind: 'ascendc_interface_detection',
      generatedAt: new Date().toISOString(),
      inputPath,
      projectRoot: projectRoot || '',
      ...detected
    };

    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, JSON.stringify(report, null, args.pretty ? 2 : 0) + '\n', 'utf8');

    console.log(`接口检测完成: ${outputPath}`);
    console.log(`扫描文件数: ${report.summary.filesScanned}`);
    console.log(`含接口文件数: ${report.summary.filesWithInterfaces}`);
    console.log(`函数符号数: ${report.summary.symbolCount}`);
    console.log(`接口数: ${report.summary.interfaceCount}`);
  } catch (error) {
    console.error(`错误: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
}

module.exports = {
  collectSourceFiles,
  detectInterfaces,
  identityId,
  parseArgs,
  parseInterfaceNames
};

if (require.main === module) main();
