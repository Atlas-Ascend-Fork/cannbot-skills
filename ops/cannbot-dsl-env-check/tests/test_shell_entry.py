#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ENTRY = Path(__file__).resolve().parents[1] / "scripts/check_env.sh"
FAKE_INTERPRETER = """import json, os, signal, sys
from pathlib import Path
if sys.argv[1:] == ['-c', 'import yaml']:
    sys.exit(int(os.environ.get('ENTRY_YAML_STATUS', '0')))
Path(os.environ['ENTRY_ARGS']).write_text(json.dumps(sys.argv[1:]))
if os.environ.get('ENTRY_WAIT'):
    def stop(signum, frame):
        Path(os.environ['ENTRY_SIGNAL']).write_text(str(signum))
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    Path(os.environ['ENTRY_READY']).touch()
    while True:
        signal.pause()
sys.exit(int(os.environ.get('ENTRY_STATUS', '0')))
"""


class ShellEntryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        interpreter = self.root / "python with spaces"
        interpreter.write_text(f"#!{sys.executable}\n" + FAKE_INTERPRETER)
        interpreter.chmod(0o700)
        self.env = dict(os.environ, PYTHON=str(interpreter))
        for key in ("ARGS", "SIGNAL", "READY"):
            self.env[f"ENTRY_{key}"] = str(self.root / key.lower())

    def call(self, *args):
        return subprocess.run(
            ["bash", str(ENTRY), *args],
            env=self.env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

    def test_output_path_is_one_literal_argument(self):
        output = str(self.root / "output with spaces;$(touch unexpected)")
        result = self.call("--output-dir", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(Path(self.env["ENTRY_ARGS"]).read_text())
        self.assertEqual(
            args, [str(ENTRY.with_name("env_check.py")), "--output-dir", output]
        )
        self.assertFalse((self.root / "unexpected").exists())

    def test_probe_failure_status_is_preserved(self):
        self.env["ENTRY_STATUS"] = "17"
        result = self.call("--output-dir", "artifacts")
        self.assertEqual(result.returncode, 17)

    def test_invalid_arguments_do_not_launch_probe(self):
        result = self.call("--output-dir")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(Path(self.env["ENTRY_ARGS"]).exists())

    def test_missing_yaml_is_an_entry_error(self):
        self.env["ENTRY_YAML_STATUS"] = "1"
        result = self.call("--output-dir", "artifacts")
        self.assertEqual(result.returncode, 2)
        self.assertIn("PyYAML", result.stderr)
        self.assertFalse(Path(self.env["ENTRY_ARGS"]).exists())

    @unittest.skipUnless(os.name == "posix", "process groups require POSIX")
    def test_process_group_cancellation_reaches_probe(self):
        self.env["ENTRY_WAIT"] = "1"
        with subprocess.Popen(
            ["bash", str(ENTRY), "--output-dir", "artifacts"],
            env=self.env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            try:
                self.wait_for(Path(self.env["ENTRY_READY"]), process)
                os.killpg(process.pid, signal.SIGTERM)
                process.communicate(timeout=5)
                self.wait_for(Path(self.env["ENTRY_SIGNAL"]))
                self.assertEqual(
                    Path(self.env["ENTRY_SIGNAL"]).read_text(), str(signal.SIGTERM)
                )
                self.assertNotEqual(process.returncode, 0)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.communicate(timeout=5)

    def wait_for(self, path, process=None):
        deadline = time.monotonic() + 5
        while not path.exists():
            if time.monotonic() > deadline or (process and process.poll() is not None):
                self.fail(f"probe did not create {path.name}")
            time.sleep(0.02)


if __name__ == "__main__":
    unittest.main()
