// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
'use strict';

const {
    parseAscendcSignaturesFromContent,
    signatureShapeKey
} = require('./ascendc-signatures');

const DOCUMENT_GROUP_MARKER = /^[ \t]*\/\/[ \t]*={3,}[ \t]*([A-Za-z_]\w*)[^\r\n]*?={3,}[ \t]*$/gm;

function memberRole(name) {
    return name.endsWith('_sync') ? 'sync' : 'regular';
}

function signatureShapes(signatures, name, callingConvention) {
    return new Set(signatures
        .filter(signature => signature.funcName === name &&
            (!callingConvention || signature.callingConvention === callingConvention))
        .map(signatureShapeKey));
}

function sourceFiles(signatures, name, callingConvention) {
    return new Set(signatures
        .filter(signature => signature.funcName === name &&
            (!callingConvention || signature.callingConvention === callingConvention))
        .map(signature => signature.sourceFile)
        .filter(Boolean));
}

function isSubset(left, right) {
    return left.size > 0 && [...left].every(item => right.has(item));
}

function setsCorrespond(left, right) {
    return isSubset(left, right) || isSubset(right, left);
}

function sharesSourceFile(left, right) {
    if (left.size === 0 || right.size === 0) return true;
    return [...left].some(item => right.has(item));
}

function canPairMembers(signatures, regularName, syncName, callingConvention) {
    return setsCorrespond(
        signatureShapes(signatures, regularName, callingConvention),
        signatureShapes(signatures, syncName, callingConvention)
    ) && sharesSourceFile(
        sourceFiles(signatures, regularName, callingConvention),
        sourceFiles(signatures, syncName, callingConvention)
    );
}

function hasMember(signatures, name, callingConvention) {
    return signatures.some(signature => signature.funcName === name &&
        (!callingConvention || signature.callingConvention === callingConvention));
}

function padFamilyRoot(name) {
    const match = name.match(/^(.*)_pad\d+(?:_sync)?$/);
    return match ? match[1] : name;
}

function padFamilyMembers(signatures, requestedApiName, callingConvention) {
    const rootName = padFamilyRoot(requestedApiName);
    if (!hasMember(signatures, rootName, callingConvention)) return null;

    const escapedRoot = rootName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const memberPattern = new RegExp(`^${escapedRoot}(?:_sync|_pad\\d+(?:_sync)?)?$`);
    const memberNames = [...new Set(signatures
        .filter(signature => (!callingConvention || signature.callingConvention === callingConvention) &&
            memberPattern.test(signature.funcName))
        .map(signature => signature.funcName))];
    if (!memberNames.some(name => /_pad\d+(?:_sync)?$/.test(name))) return null;
    return { rootName, memberNames };
}

function uniqueMembers(names, documentApiName, sourceDocumentGroup) {
    const groupedNames = new Set(sourceDocumentGroup?.memberNames || []);
    return [...new Set(names)].map(name => ({
        name,
        role: memberRole(name),
        relation: name === documentApiName ? 'document-root' :
            (name === `${documentApiName}_sync` ? 'sync-counterpart' :
                (groupedNames.has(name) ? 'source-document-group' : 'explicit-related'))
    }));
}

function sourceDocumentBlocks(content) {
    const markers = [...content.matchAll(DOCUMENT_GROUP_MARKER)].map(match => ({
        rootName: match[1],
        start: match.index,
        contentStart: match.index + match[0].length
    }));
    return markers.map((marker, index) => {
        const end = index + 1 < markers.length ? markers[index + 1].start : content.length;
        const signatures = parseAscendcSignaturesFromContent(content.slice(marker.contentStart, end));
        return {
            rootName: marker.rootName,
            memberNames: [...new Set(signatures.map(signature => signature.funcName))]
        };
    });
}

function discoverSourceDocumentGroup(content, requestedApiName) {
    const blocks = sourceDocumentBlocks(content);
    const containingBlocks = blocks.filter(block => block.memberNames.includes(requestedApiName));
    if (containingBlocks.length === 0) return null;

    const roots = [...new Set(containingBlocks.map(block => block.rootName))];
    const regularRequestedName = requestedApiName.endsWith('_sync')
        ? requestedApiName.slice(0, -'_sync'.length)
        : requestedApiName;
    const preferredRoot = roots.includes(requestedApiName)
        ? requestedApiName
        : (roots.includes(regularRequestedName) ? regularRequestedName : (roots.length === 1 ? roots[0] : null));
    if (!preferredRoot) return null;

    const memberNames = [];
    for (const block of blocks.filter(item => item.rootName === preferredRoot)) {
        for (const name of block.memberNames) {
            if (!memberNames.includes(name)) memberNames.push(name);
        }
    }
    if (!memberNames.includes(preferredRoot) || !memberNames.includes(requestedApiName)) return null;
    return { rootName: preferredRoot, memberNames };
}

function resolveDocumentFamily({
    requestedApiName,
    signatures,
    callingConvention,
    exactInterface = false,
    relatedInterfaces = [],
    sourceDocumentGroup = null
}) {
    requestedApiName = requestedApiName.replace(/\s+/g, '');
    relatedInterfaces = relatedInterfaces.map(name => name.replace(/\s+/g, ''));
    if (!hasMember(signatures, requestedApiName, callingConvention)) {
        throw new Error(`未找到接口 ${requestedApiName} 的 ${callingConvention || '目标'} 调用约定定义`);
    }

    const documentGroup = !exactInterface && !sourceDocumentGroup
        ? padFamilyMembers(signatures, requestedApiName, callingConvention)
        : sourceDocumentGroup;
    let documentApiName = requestedApiName;
    const members = [];
    const diagnostics = [];

    if (!exactInterface && documentGroup?.memberNames?.includes(requestedApiName)) {
        documentApiName = documentGroup.rootName;
        for (const name of documentGroup.memberNames) {
            if (hasMember(signatures, name, callingConvention)) members.push(name);
        }
    } else {
        members.push(requestedApiName);
    }

    if (!exactInterface && !documentGroup) {
        if (requestedApiName.endsWith('_sync')) {
            const regularName = requestedApiName.slice(0, -'_sync'.length);
            if (hasMember(signatures, regularName, callingConvention) &&
                canPairMembers(signatures, regularName, requestedApiName, callingConvention)) {
                documentApiName = regularName;
                members.unshift(regularName);
            }
        } else {
            const syncName = `${requestedApiName}_sync`;
            if (hasMember(signatures, syncName, callingConvention)) {
                if (canPairMembers(signatures, requestedApiName, syncName, callingConvention)) {
                    members.push(syncName);
                } else {
                    diagnostics.push(`检测到 ${syncName}，但其原型集合与 ${requestedApiName} 不存在对应关系，未自动合并到同一文档族。`);
                }
            }
        }
    }

    for (const relatedName of relatedInterfaces) {
        if (!hasMember(signatures, relatedName, callingConvention)) {
            throw new Error(`关联接口 ${relatedName} 不存在于当前调用约定范围内`);
        }
        members.push(relatedName);
    }

    return {
        requestedApiName,
        documentApiName,
        exactInterface,
        members: uniqueMembers(members, documentApiName, documentGroup),
        diagnostics
    };
}

function buildDocumentFamilies(signatures) {
    const families = [];
    const seen = new Set();
    const identities = [...new Set(signatures.map(signature =>
        `${signature.callingConvention}|${signature.funcName}`))].sort();

    for (const identity of identities) {
        if (seen.has(identity)) continue;
        const separator = identity.indexOf('|');
        const callingConvention = identity.slice(0, separator);
        const name = identity.slice(separator + 1);
        const family = resolveDocumentFamily({
            requestedApiName: name,
            signatures,
            callingConvention
        });
        for (const member of family.members) {
            seen.add(`${callingConvention}|${member.name}`);
        }
        families.push({
            name: family.documentApiName,
            callingConvention,
            members: family.members,
            diagnostics: family.diagnostics
        });
    }
    return families.sort((left, right) =>
        left.name.localeCompare(right.name) || left.callingConvention.localeCompare(right.callingConvention));
}

module.exports = {
    buildDocumentFamilies,
    canPairMembers,
    discoverSourceDocumentGroup,
    memberRole,
    resolveDocumentFamily,
    signatureShapes
};
