"""Incrementally add the 6 metrical/key/time-sig columns to an existing
xml_features dataset, AND drop the old trailing pedal_down column.

This is a focused alternative to a full re-augment: the old slow path
recomputes articulations, slurs, dynamics, wedge progress, tempo
directions etc. — all heavy partitura iter_all() traversals that took
~50-90 min on the full corpus and choked on a few pathological pieces.
The metrical features come directly from note_array fields, so the
incremental pass is just dict lookups + numpy slicing.

Old layout (15 cols, ends with pedal_down):
    [..., active_tempo_kind_id, expected_bpm, pedal_down]

New layout (20 cols):
    [..., active_tempo_kind_id, expected_bpm,
     is_downbeat, rel_onset_in_measure, ts_beats, ts_beat_type,
     ks_fifths, ks_mode]

Usage::
    python -m features.add_metrical --hdf5 data/codec_N=200_mixup.hdf5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import partitura as pt
import partitura.score as ps
from tqdm import tqdm

from prepare_data import _apply_path_remap, _decode

logger = logging.getLogger(__name__)


# Number of leading xml_features columns to keep (drops pedal_down at index 14).
_OLD_KEEP = 14
# Number of new metrical columns appended.
_N_NEW = 6


def _load_score(score_path: str) -> ps.Score | None:
    try:
        score = pt.load_musicxml(score_path, force_note_ids="keep")
        # Mirror prepare_data.get_codecs unfolding logic so snote_ids match.
        return score
    except Exception as e:
        logger.warning("score load failed for %s: %s", score_path, e)
        return None


def _maybe_unfold(score: ps.Score, sample_id: str) -> ps.Score:
    """Unfold the score if snote_ids carry repeat suffixes that the loaded
    score doesn't. Same logic as prepare_data.get_codecs."""
    sna = score.note_array(include_divs_per_quarter=False)
    if "-" in sample_id and "-" not in sna["id"][0]:
        score = ps.unfold_part_maximal(ps.merge_parts(score.parts))
    return score


def _build_metrical_lookup(score: ps.Score) -> dict[str, tuple]:
    """Map note_id → (is_downbeat, rel_in_meas, ts_beats, ts_bt, ks_fif, ks_mode)."""
    sna = score.parts[0].note_array(
        include_pitch_spelling=True,
        include_key_signature=True,
        include_time_signature=True,
        include_metrical_position=True,
        include_grace_notes=True,
        include_staff=True,
        include_divs_per_quarter=True,
    )
    out = {}
    for row in sna:
        tot = int(row["tot_measure_div"]) if row["tot_measure_div"] else 0
        rel = float(row["rel_onset_div"]) / tot if tot > 0 else 0.0
        out[str(row["id"])] = (
            int(row["is_downbeat"]),
            rel,
            int(row["ts_beats"]),
            int(row["ts_beat_type"]),
            int(row["ks_fifths"]),
            int(row["ks_mode"]),
        )
    return out


def add_metrical(hdf5_path: Path, errors_path: Path,
                 path_remap: dict | None = None) -> None:
    score_cache: dict[str, dict[str, tuple]] = {}
    n_added = n_skipped = n_failed = 0

    with h5py.File(hdf5_path, "a") as f, errors_path.open("a") as errf:
        groups = list(f.keys())
        for gname in tqdm(groups, desc="metrical"):
            g = f[gname]
            if "error" in g:
                continue
            if "xml_features" not in g:
                n_skipped += 1
                continue
            existing = np.array(g["xml_features"], dtype=np.float32)
            S, T, F = existing.shape

            # Already migrated? leave alone.
            if F >= _OLD_KEEP + _N_NEW:
                n_skipped += 1
                continue

            score_path = _apply_path_remap(_decode(np.array(g["score_path"])[0]), path_remap)
            snote_id_paths = [_apply_path_remap(_decode(p), path_remap)
                              for p in np.array(g["snote_id_path"])]

            # Load + cache score per piece (multiple segments share scores).
            if score_path not in score_cache:
                score = _load_score(score_path)
                if score is None:
                    errf.write(json.dumps({"group": gname, "stage": "score_load",
                                            "score_path": score_path}) + "\n")
                    n_failed += 1
                    score_cache[score_path] = None
                    continue
                # use first segment's snote_ids to decide unfold
                first_ids = np.load(snote_id_paths[0])
                if len(first_ids) > 0:
                    score = _maybe_unfold(score, str(first_ids[0]))
                try:
                    score_cache[score_path] = _build_metrical_lookup(score)
                except Exception as e:
                    errf.write(json.dumps({"group": gname, "stage": "metrical_lookup",
                                            "score_path": score_path, "error": str(e)}) + "\n")
                    n_failed += 1
                    score_cache[score_path] = None
                    continue
            lookup = score_cache[score_path]
            if lookup is None:
                n_failed += 1
                continue

            # Build (S, T, 6) metrical block aligned with existing segments.
            new_block = np.zeros((S, T, _N_NEW), dtype=np.float32)
            for s_idx, sid_path in enumerate(snote_id_paths):
                try:
                    sids = np.load(sid_path)
                except Exception:
                    continue
                for n_idx, nid in enumerate(sids):
                    if n_idx >= T:
                        break
                    tup = lookup.get(str(nid))
                    if tup is None:
                        continue
                    new_block[s_idx, n_idx, :] = tup

            # Drop trailing pedal_down (col 14) if present, append metrical block.
            head = existing[..., :_OLD_KEEP]
            merged = np.concatenate([head, new_block], axis=-1).astype(np.float32)

            # Replace dataset.
            del g["xml_features"]
            g.create_dataset("xml_features", data=merged, dtype="f4")
            n_added += 1

    logger.info("metrical augment: %d added, %d already done / skipped, %d failed",
                n_added, n_skipped, n_failed)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", required=True)
    p.add_argument("--errors", default="data/metrical_errs.jsonl")
    p.add_argument("--remap", action="append", default=[],
                   help="prefix substitution as 'old=new', repeatable. Use for "
                        "both Datasets root and DExter_/data root since HDF5 "
                        "embeds absolute paths from the regen machine.")
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    path_remap: dict[str, str] = {}
    for entry in args.remap:
        if "=" not in entry:
            raise SystemExit(f"--remap '{entry}' missing '=' separator")
        old, new = entry.split("=", 1)
        path_remap[old] = new
    Path(args.errors).parent.mkdir(parents=True, exist_ok=True)
    add_metrical(Path(args.hdf5), Path(args.errors),
                 path_remap=path_remap or None)


if __name__ == "__main__":
    main()
