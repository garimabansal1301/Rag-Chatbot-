"""Run the offline evaluation (EXTERNAL CALLS to Gemini and, unless --local, LangSmith; may incur Gemini usage).

  uv run python scripts/run_evaluation.py                          # dry run: validate and show expected calls
  uv run python scripts/run_evaluation.py --ids q01,q17 --confirm  # a small LangSmith experiment
  uv run python scripts/run_evaluation.py --local --confirm        # Gemini only; results stay local

Reports are written to evaluation/reports/. The LangSmith dataset must exist first
(scripts/create_evaluation_dataset.py --confirm).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere: import the app package

from app.evaluation.cli import run_evaluation_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_evaluation_main())
