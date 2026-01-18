# Sample usage:
# python scriptsv2/11_slice_frontier_report.py \
#   --preds outputs/preds.parquet \
#   --manifest data/processed/mimic_v2/mimic_manifest.parquet \
#   --out_dir outputs/frontier_report

from __future__ import annotations

import argparse
import subprocess


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, required=True)
    ap.add_argument("--manifest", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--min_slice_size", type=int, default=200)
    args = ap.parse_args()

    subprocess.check_call(
        [
            "python",
            "scriptsv2/09_controlled_shift_benchmark.py",
            "--preds",
            args.preds,
            "--out_dir",
            args.out_dir,
            "--min_slice_size",
            str(args.min_slice_size),
        ]
        + (["--manifest", args.manifest] if args.manifest else [])
    )

    subprocess.check_call(
        [
            "python",
            "scriptsv2/12_plot_frontier_and_appendix.py",
            "--out_dir",
            args.out_dir,
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
