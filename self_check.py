"""Self-check: PASS/FAIL per milestone using the public contract tests.

    uv run python self_check.py            # all milestones
    uv run python self_check.py 3          # only milestone 3

No points here — just whether each milestone meets its contract. The report is
written to self_check_report.json and stamped with SELF_CHECK_KEY from .env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
MILESTONES = [
    ("1", "LLM + LangGraph engine", "test_m1_engine.py"),
    ("2", "LangSmith observability", "test_m2_observability.py"),
    ("3", "Chunking", "test_m3_chunking.py"),
    ("4", "Embeddings", "test_m4_embeddings.py"),
    ("5", "Vector store", "test_m5_store.py"),
    ("6", "Ingestion + upload indexing", "test_m6_ingest.py"),
    ("7", "Retriever + RAG with sources", "test_m7_retrieval.py"),
    ("8", "Tools", "test_m8_tools.py"),
]


def run_milestone(test_file: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"tests/contract/{test_file}",
            "-m",
            "contract",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--tb=line",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return proc.returncode == 0, (lines[-1] if lines else proc.stderr.strip()[-200:])


def main() -> int:
    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("SELF_CHECK_KEY", "").strip()
    if not key:
        print("SELF_CHECK_KEY is not set in .env — copy .env.example to .env and fill it in.")
        return 2

    wanted = set(sys.argv[1:])
    results = []
    print(f"Self-check for {key}\n")
    for number, title, test_file in MILESTONES:
        if wanted and number not in wanted:
            continue
        ok, summary = run_milestone(test_file)
        results.append({"milestone": number, "title": title, "status": "PASS" if ok else "FAIL", "summary": summary})
        print(f"  M{number:<2} {'PASS' if ok else 'FAIL'}  {title:<32} {summary}")

    report = {"self_check_key": key, "checked_at": datetime.now(UTC).isoformat(timespec="seconds"), "results": results}
    (ROOT / "self_check_report.json").write_text(json.dumps(report, indent=2))
    passed = sum(r["status"] == "PASS" for r in results)
    print(f"\n{passed}/{len(results)} milestones pass. Details: uv run pytest -m contract -v")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
