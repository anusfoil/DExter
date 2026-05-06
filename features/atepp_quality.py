"""Compute ATEPP per-piece alignment-quality metrics.

Background: ATEPP has automatic alignments via parangonar. Quality varies —
some pieces have near-perfect matches; others have many insertions/deletions.
A bad alignment poisons the codec extraction (the model learns from "label"
performance features that are mis-aligned to the wrong score notes), which
shows up downstream as the model rendering performances that sound vaguely
plausible but with wandering or implausible micro-timing.

This module reads each ATEPP ``align.csv`` and reports::

    match_ratio = N_match / (N_match + N_insertion + N_deletion)

Higher = cleaner alignment. ASAP and VIENNA422 are hand-aligned / curated and
don't need this filter.

CLI usage::

    python -m features.atepp_quality
    # -> writes data/atepp_match_ratios.csv with piece_name, match_ratio columns

Then at training time, ``load_data_from_hdf5(..., atepp_min_match_ratio=0.9)``
drops ATEPP segments below threshold while leaving ASAP / VIENNA422 untouched.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import partitura as pt
from tqdm import tqdm

from prepare_data import discover_atepp, load_paths

logger = logging.getLogger(__name__)


def _match_ratio(alignment_path: Path) -> float | None:
    """Return matches / (matches + insertions + deletions), or None on failure."""
    try:
        align = pt.io.importparangonada.load_parangonada_alignment(str(alignment_path))
    except Exception as e:
        logger.warning("alignment load failed for %s: %s", alignment_path, e)
        return None
    n_match = sum(1 for a in align if a["label"] == "match")
    n_ins = sum(1 for a in align if a["label"] == "insertion")
    n_del = sum(1 for a in align if a["label"] == "deletion")
    total = n_match + n_ins + n_del
    if total == 0:
        return None
    return n_match / total


def compute_match_ratios(paths) -> pd.DataFrame:
    """Compute ATEPP match ratios across all discovered pieces.

    Returns a DataFrame indexed by ``piece_name`` (the ATEPP segment ID like
    ``"11604"``) with columns: ``match_ratio``, ``n_match``, ``n_insertion``,
    ``n_deletion``, ``alignment_path``, ``score_path``.
    """
    pieces = discover_atepp(paths)
    rows = []
    for piece in tqdm(pieces, desc="atepp quality"):
        try:
            align = pt.io.importparangonada.load_parangonada_alignment(str(piece.alignment_path))
        except Exception as e:
            rows.append({
                "piece_name": piece.piece_name,
                "match_ratio": float("nan"),
                "n_match": 0, "n_insertion": 0, "n_deletion": 0,
                "alignment_path": str(piece.alignment_path),
                "score_path": str(piece.score_path),
                "error": str(e),
            })
            continue
        n_match = sum(1 for a in align if a["label"] == "match")
        n_ins = sum(1 for a in align if a["label"] == "insertion")
        n_del = sum(1 for a in align if a["label"] == "deletion")
        total = n_match + n_ins + n_del
        rows.append({
            "piece_name": piece.piece_name,
            "match_ratio": (n_match / total) if total else float("nan"),
            "n_match": n_match,
            "n_insertion": n_ins,
            "n_deletion": n_del,
            "alignment_path": str(piece.alignment_path),
            "score_path": str(piece.score_path),
            "error": "",
        })
    return pd.DataFrame(rows).set_index("piece_name")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute ATEPP alignment-quality metrics")
    p.add_argument("--paths_config", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--datasets_root", type=str, default=None)
    p.add_argument("--out", type=str, default="data/atepp_match_ratios.csv")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    paths = load_paths(
        config_path=Path(args.paths_config) if args.paths_config else None,
        data_root=args.data_root,
        datasets_root=args.datasets_root,
    )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = paths.data_root.parent / args.out  # default: ./data/atepp_match_ratios.csv

    df = compute_match_ratios(paths)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)

    valid = df["match_ratio"].dropna()
    logger.info(
        "wrote %d ATEPP rows to %s (valid match_ratio for %d / %d). "
        "distribution: min=%.3f p10=%.3f p50=%.3f p90=%.3f max=%.3f",
        len(df), out_path, len(valid), len(df),
        valid.min(), valid.quantile(0.10), valid.quantile(0.50),
        valid.quantile(0.90), valid.max(),
    )
    logger.info(
        "with thresholds: 0.7→%d kept, 0.8→%d, 0.9→%d, 0.95→%d (out of %d)",
        (valid >= 0.7).sum(), (valid >= 0.8).sum(),
        (valid >= 0.9).sum(), (valid >= 0.95).sum(), len(valid),
    )


if __name__ == "__main__":
    main()
