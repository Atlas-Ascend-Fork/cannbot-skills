// SPDX-License-Identifier: CANN-2.0
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
const OPERATOR_DIRECTORY_NAMES = new Map([
  ['operator+', 'operator_add'],
  ['operator-', 'operator_subtract'],
  ['operator*', 'operator_multiply'],
  ['operator/', 'operator_divide'],
  ['operator%', 'operator_modulo'],
  ['operator++', 'operator_increment'],
  ['operator--', 'operator_decrement'],
  ['operator==', 'operator_equal'],
  ['operator!=', 'operator_not_equal'],
  ['operator<', 'operator_less'],
  ['operator<=', 'operator_less_equal'],
  ['operator>', 'operator_greater'],
  ['operator>=', 'operator_greater_equal'],
  ['operator<=>', 'operator_three_way_compare'],
  ['operator!', 'operator_logical_not'],
  ['operator&&', 'operator_logical_and'],
  ['operator||', 'operator_logical_or'],
  ['operator~', 'operator_bitwise_not'],
  ['operator&', 'operator_bitwise_and'],
  ['operator|', 'operator_bitwise_or'],
  ['operator^', 'operator_bitwise_xor'],
  ['operator<<', 'operator_left_shift'],
  ['operator>>', 'operator_right_shift'],
  ['operator=', 'operator_assign'],
  ['operator+=', 'operator_add_assign'],
  ['operator-=', 'operator_subtract_assign'],
  ['operator*=', 'operator_multiply_assign'],
  ['operator/=', 'operator_divide_assign'],
  ['operator%=', 'operator_modulo_assign'],
  ['operator&=', 'operator_bitwise_and_assign'],
  ['operator|=', 'operator_bitwise_or_assign'],
  ['operator^=', 'operator_bitwise_xor_assign'],
  ['operator<<=', 'operator_left_shift_assign'],
  ['operator>>=', 'operator_right_shift_assign'],
  ['operator[]', 'operator_subscript'],
  ['operator()', 'operator_call'],
  ['operator,', 'operator_comma'],
  ['operator->', 'operator_arrow'],
  ['operator->*', 'operator_arrow_member'],
  ['operatornew[]', 'operator_new_array'],
  ['operatordelete[]', 'operator_delete_array'],
]);

function normalizedName(value) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error('API 名称必须是非空字符串');
  }
  return value.trim();
}

function operatorKey(name) {
  if (!name.startsWith('operator')) return null;
  const suffix = name.slice('operator'.length).trim();
  return { suffix, key: `operator${suffix}` };
}

/**
 * Returns a stable, Windows-safe, human-readable default output directory.
 * The API's original C++ spelling remains unchanged in manifest.json.
 */
export function outputDirectoryName(documentApiName) {
  const name = normalizedName(documentApiName);
  const operator = operatorKey(name);
  if (!operator) return name;

  const mapped = OPERATOR_DIRECTORY_NAMES.get(operator.key);
  if (mapped) return mapped;

  if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(operator.suffix)) {
    return `operator_${operator.suffix}`;
  }

  const codePoints = [...operator.suffix]
    .map(character => character.codePointAt(0).toString(16))
    .join('_');
  return `operator_symbol_${codePoints || 'unknown'}`;
}

/** Returns the stable, readable filename stem for generated documents. */
export function documentFileStem(documentApiName) {
  return outputDirectoryName(documentApiName);
}

/** Returns the pre-semantic-name directory used by older tasks. */
export function legacyOutputDirectoryName(documentApiName) {
  return encodeURIComponent(normalizedName(documentApiName));
}

/** Returns the current name first, followed by the legacy name when distinct. */
export function outputDirectoryNameCandidates(documentApiName) {
  const current = outputDirectoryName(documentApiName);
  const legacy = legacyOutputDirectoryName(documentApiName);
  return current === legacy ? [current] : [current, legacy];
}
