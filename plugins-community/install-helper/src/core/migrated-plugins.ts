// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import type { PluginEntry } from "../types/index.js";

// Plugins removed from this repository and migrated to the cannbot repo
// (gitcode.com/cann/cannbot). The managed @cannbot-plugin/cannbot installer
// follows the cannbot git mainline when it installs, but its discovery
// snapshot (dist/plugins) only refreshes on npm releases — a freshly
// migrated plugin stays invisible to findPlugin until the next installer
// release. These tombstone entries bridge that gap: discovery resolves them
// as cannbot-source plugins and delegateInstall() skips its snapshot
// precheck for them, letting the installer's mainline-following install
// decide authoritatively (it fails with its own "plugin not found" for
// plugins that truly do not exist upstream).
export const MIGRATED_PLUGINS: PluginEntry[] = [
  {
    id: "triton-op-generator",
    dir: "",
    displayName: "Triton 算子开发",
    script: "cannbot",
    aliases: ["triton"],
    skills: 8,
    agents: 0,
    description: "Triton-Ascend 算子 6 阶段自动生成与优化（描述→代码→验证→优化）；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
  {
    id: "ops-registry-invoke",
    dir: "",
    displayName: "AscendC 算子注册调用",
    script: "cannbot",
    aliases: ["ops-registry", "ascendc-registry", "registry"],
    skills: 7,
    agents: 0,
    description: "生产级 ACLNN 注册算子全流程：代码→UT/ST→上库；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
  {
    id: "collaborative-agent-kernel-evolution",
    dir: "",
    displayName: "CAKE 协同算子进化",
    script: "cannbot",
    aliases: ["cake"],
    skills: 0,
    agents: 0,
    description: "CAKE2 - Collaborative Ascend Kernel Evolution，AI 驱动的 AscendC 算子协同进化；已迁移至 cannbot 仓，经托管安装器安装",
    source: "cannbot",
  },
];

export function isMigratedPlugin(id: string): boolean {
  return MIGRATED_PLUGINS.some((p) => p.id === id);
}
