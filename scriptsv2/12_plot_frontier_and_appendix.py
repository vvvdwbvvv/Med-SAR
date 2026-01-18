# Sample usage:
# python scriptsv2/12_plot_frontier_and_appendix.py \
#   --out_dir outputs/controlled_shift_v2 \
#   --guard_stats outputs/guard_stats.jsonl

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_guard_stats(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--guard_stats", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frontier_path = out_dir / "frontier_points.parquet"
    slice_path = out_dir / "slice_metrics.parquet"
    slices_meta_path = out_dir / "n10_slices.parquet"

    if frontier_path.exists():
        frontier = pd.read_parquet(frontier_path)
        plt.figure(figsize=(6, 5))
        plt.scatter(frontier["accuracy"], frontier["stability"], c=frontier["t"], s=40)
        plt.xlabel("Accuracy")
        plt.ylabel("Stability")
        plt.title("Fig.2 Frontier")
        plt.colorbar(label="t")
        plt.tight_layout()
        plt.savefig(out_dir / "fig2_frontier.png", dpi=200)

    if slice_path.exists() and slices_meta_path.exists():
        metrics = pd.read_parquet(slice_path)
        meta = pd.read_parquet(slices_meta_path)
        merged = metrics.merge(
            meta[["slice_id", "time_bucket"]], on="slice_id", how="left"
        )
        plt.figure(figsize=(7, 5))
        for tb, grp in merged.groupby("time_bucket"):
            series = grp.groupby("t")["accuracy"].mean().reset_index()
            plt.plot(series["t"], series["accuracy"], label=f"time_bucket={tb}")
        plt.xlabel("t")
        plt.ylabel("Accuracy")
        plt.title("Fig.3 Temporal Slices")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "fig3_temporal_slices.png", dpi=200)

    if args.guard_stats:
        guard_df = _load_guard_stats(Path(args.guard_stats))
        proxy_cols = [
            "newline_ratio",
            "colon_ratio",
            "digit_ratio",
            "header_density",
            "abbrev_ratio",
        ]
        if not guard_df.empty:
            for proxy in proxy_cols:
                if proxy not in guard_df.columns:
                    continue
                plt.figure(figsize=(6, 4))
                series = guard_df.groupby("t")[proxy].mean().reset_index()
                plt.plot(series["t"], series[proxy], marker="o")
                plt.xlabel("t")
                plt.ylabel(proxy)
                plt.title(f"Appendix A.6 {proxy}")
                plt.tight_layout()
                plt.savefig(out_dir / f"appendix_A6_{proxy}.png", dpi=200)

            if "tries" in guard_df.columns:
                plt.figure(figsize=(6, 4))
                tries = guard_df.groupby("t")["tries"].mean().reset_index()
                plt.plot(tries["t"], tries["tries"], marker="o")
                plt.xlabel("t")
                plt.ylabel("avg_resample_tries")
                plt.title("Appendix A.7 Resampling Pressure")
                plt.tight_layout()
                plt.savefig(out_dir / "appendix_A7_resampling.png", dpi=200)

    print(f"plots written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
