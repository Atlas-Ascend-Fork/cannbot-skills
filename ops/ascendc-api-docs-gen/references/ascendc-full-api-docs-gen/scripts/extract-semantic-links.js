#!/usr/bin/env node
// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

'use strict';

const fs = require('fs');
const path = require('path');
const { buildSemanticLinks, collectCodeFiles } = require('./lib/ascendc-semantic-tracker');
const { parseAscendcSignaturesFromContent } = require('./lib/ascendc-signatures');

function memberRole(name) {
    return name.endsWith('_sync') ? 'sync' : 'regular';
}

function parseArgs(argv) {
    const result = { projectRoot: argv[0], documentApiName: argv[1], memberApis: [] };
    for (let index = 2; index < argv.length; index++) {
        const arg = argv[index];
        if (arg === '--input') result.inputPath = argv[++index];
        else if (arg === '--calling-convention') result.callingConvention = argv[++index];
        else if (arg === '--member') {
            const name = argv[++index];
            result.memberApis.push({ name, role: memberRole(name) });
        } else if (!result.apiType) result.apiType = arg;
        else throw new Error(`Unknown argument: ${arg}`);
    }
    if (!result.memberApis.some(member => member.name === result.documentApiName)) {
        result.memberApis.unshift({ name: result.documentApiName, role: memberRole(result.documentApiName) });
    }
    return result;
}

function collectPublicSignatures(inputPath, memberApis, callingConvention) {
    const memberNames = new Set(memberApis.map(member => member.name));
    const signatures = [];
    for (const file of collectCodeFiles(inputPath)) {
        const content = fs.readFileSync(file, 'utf8');
        signatures.push(...parseAscendcSignaturesFromContent(content)
            .filter(signature => memberNames.has(signature.funcName) &&
                (!callingConvention || signature.callingConvention === callingConvention))
            .map(signature => ({ ...signature, sourceFile: path.resolve(file) })));
    }
    return signatures;
}

function main() {
    const options = parseArgs(process.argv.slice(2));
    if (!options.projectRoot || !options.documentApiName || !options.inputPath) {
        throw new Error('Usage: node extract-semantic-links.js <projectRoot> <documentApiName> [apiType] --input <fileOrDir> [--calling-convention <value>] [--member <apiName>]...');
    }
    const projectRoot = path.resolve(options.projectRoot);
    const inputPath = path.resolve(options.inputPath);
    const publicSignatures = collectPublicSignatures(inputPath, options.memberApis, options.callingConvention);
    if (publicSignatures.length === 0) throw new Error('No public signatures found for semantic tracking');

    const result = buildSemanticLinks({
        projectRoot,
        memberApis: options.memberApis,
        publicSignatures
    });
    process.stdout.write(JSON.stringify({
        schemaVersion: 1,
        kind: 'ascendc_semantic_links',
        generatedAt: new Date().toISOString(),
        documentApiName: options.documentApiName,
        apiType: options.apiType || '',
        callingConvention: options.callingConvention || '',
        memberApis: options.memberApis,
        sourceFiles: [...new Set(publicSignatures.map(signature => signature.sourceFile))],
        ...result
    }, null, 2) + '\n');
}

if (require.main === module) {
    try { main(); } catch (error) {
        console.error(`Error: ${error instanceof Error ? error.message : String(error)}`);
        process.exitCode = 1;
    }
}
