"""
fairness_gate.py

CI entrypoint for the "Fairness Gate." Trains the baseline model (via
src/train_model.py) and evaluates its Disparate Impact score. Exits with
a non-zero status code — failing the CI job — if DI falls below threshold.

This is what .github/workflows/fairness-gate.yml runs on every push/PR.

Usage:
    python scripts/fairness_gate.py --threshold 0.8
"""

import argparse
import json
import sys
from pathlib import Path

# Defensive fix: Windows consoles often default to cp1252, which can't
# encode certain unicode characters and crashes print(). Force UTF-8 output
# regardless of platform.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow running as a script

from src.train_model import train_and_save


def main():
    parser = argparse.ArgumentParser(description="Automated fairness gate for CI/CD (loan approval)")
    parser.add_argument("--threshold", type=float, default=0.8, help="Minimum acceptable Disparate Impact score")
    parser.add_argument("--output", type=str, default="fairness_report.json", help="Path to write the JSON report")
    args = parser.parse_args()

    model, di_score = train_and_save()

    passed = di_score >= args.threshold
    report = {
        "domain": "loan_approval",
        "protected_attribute": "Gender",
        "disparate_impact": round(float(di_score), 4),
        "threshold": args.threshold,
        "passed": bool(passed),
    }

    Path(args.output).write_text(json.dumps(report, indent=2))

    print("=" * 50)
    print("FAIRNESS GATE REPORT — Loan Approval Model")
    print("=" * 50)
    print(json.dumps(report, indent=2))
    print("=" * 50)

    if not passed:
        print(f"FAILED: Disparate Impact {di_score:.3f} is below threshold {args.threshold}")
        sys.exit(1)

    print(f"PASSED: Disparate Impact {di_score:.3f} meets threshold {args.threshold}")
    sys.exit(0)


if __name__ == "__main__":
    main()