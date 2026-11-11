"""Upload the local evaluation dataset to LangSmith (EXTERNAL CALL; never run automatically).

  uv run python scripts/create_evaluation_dataset.py              # dry run: validate and show what would be sent
  uv run python scripts/create_evaluation_dataset.py --confirm    # create the dataset / add missing examples

Existing examples (same example_id) are skipped; changed ones are reported and never overwritten.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere: import the app package

from app.evaluation.cli import create_dataset_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(create_dataset_main())
