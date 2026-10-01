#!/usr/bin/env python3
"""Stand-in for `claude -p --output-format json` used by the tests.

Replays replies from $FAKE_CLAUDE_SCRIPT (JSON list of strings) in order, and appends each invocation's
argv / stdin / environment variable names to $FAKE_CLAUDE_LOG.
"""
import json
import os
import sys

script = json.load(open(os.environ["FAKE_CLAUDE_SCRIPT"]))
log = os.environ["FAKE_CLAUDE_LOG"]
n = sum(1 for _ in open(log)) if os.path.exists(log) else 0
stdin = sys.stdin.read()
with open(log, "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "stdin": stdin, "env": sorted(os.environ)}) + "\n")
reply = script[n] if n < len(script) else "(script finished)"
print(json.dumps({"type": "result", "is_error": False, "result": reply, "session_id": "sess-1",
                  "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}}))
