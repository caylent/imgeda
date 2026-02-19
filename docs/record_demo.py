"""Drive the imgeda wizard with pexpect and write an asciinema v2 cast file.

Usage:
    uv run python docs/record_demo.py
    agg docs/demo.cast docs/demo.gif --font-size 14 --theme dracula
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time

import pexpect

COLS = 120
ROWS = 35
PAD = 1  # columns/rows of padding on all sides
CAST_PATH = "docs/demo.cast"


class CastLogger:
    """Logs pexpect output to both stdout and an asciinema v2 cast file."""

    def __init__(self, path: str, width: int, height: int, pad: int) -> None:
        self.f = open(path, "w")
        self.pad = pad
        self.first_write = True
        header = {
            "version": 2,
            "width": width + pad * 2,
            "height": height + pad * 2,
            "timestamp": int(time.time()),
            "env": {"SHELL": "/bin/zsh", "TERM": "xterm-256color"},
        }
        self.f.write(json.dumps(header) + "\n")
        self.start = time.time()

    def write(self, data: bytes) -> None:
        elapsed = time.time() - self.start
        text = data.decode("utf-8", errors="replace")
        # Add top padding on first write
        if self.first_write:
            top_pad = "\n" * self.pad + " " * self.pad
            self.f.write(json.dumps([round(elapsed, 6), "o", top_pad]) + "\n")
            self.first_write = False
        # Add left padding after each newline
        if self.pad:
            text = text.replace("\n", "\n" + " " * self.pad)
        self.f.write(json.dumps([round(elapsed, 6), "o", text]) + "\n")
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

    def flush(self) -> None:
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def main() -> None:
    env = os.environ.copy()
    env["PROMPT_TOOLKIT_NO_CPR"] = "1"

    # Clean up previous outputs
    for f in ["imgeda_manifest.jsonl", "imgeda_report.html"]:
        path = os.path.join("/Users/ranman/dev/caylent/imgeda", f)
        if os.path.exists(path):
            os.remove(path)
    plots_dir = "/Users/ranman/dev/caylent/imgeda/plots"
    if os.path.isdir(plots_dir):
        shutil.rmtree(plots_dir)

    logger = CastLogger(CAST_PATH, COLS, ROWS, PAD)

    child = pexpect.spawn(
        "/Users/ranman/dev/caylent/imgeda/.venv/bin/imgeda",
        timeout=60,
        cwd="/Users/ranman/dev/caylent/imgeda",
        dimensions=(ROWS, COLS),
        env=env,
    )
    # This makes pexpect send ALL output through our logger continuously
    child.logfile_read = logger

    # Step 1: directory path
    child.expect("Where are your images")
    time.sleep(0.5)
    child.send("\x01")  # Ctrl+A
    child.send("\x0b")  # Ctrl+K
    time.sleep(0.3)
    for c in "./docs/demo_images":
        child.send(c)
        time.sleep(0.04)
    time.sleep(0.3)
    child.sendline("")

    # Step 4: analyses — accept all defaults
    child.expect("What analyses", timeout=15)
    time.sleep(1.0)
    child.sendline("")

    # Step 5: workers
    child.expect("How many workers", timeout=10)
    time.sleep(0.8)
    child.sendline("")

    # Step 6: output path
    child.expect("Output manifest path", timeout=10)
    time.sleep(0.8)
    child.sendline("")

    # Step 7: generate report
    child.expect("Generate plots", timeout=10)
    time.sleep(0.8)
    child.sendline("Y")

    # Wait for completion
    child.expect("Done!", timeout=120)
    time.sleep(2.0)

    try:
        child.expect(pexpect.EOF, timeout=3)
    except pexpect.TIMEOUT:
        pass

    child.close()
    logger.close()
    print(f"\n\nCast written: {CAST_PATH} ({COLS}x{ROWS})")


if __name__ == "__main__":
    main()
