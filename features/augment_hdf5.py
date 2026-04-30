"""Augment an existing codec HDF5 with structured XML features.

The base ``prepare_data.py --compute_codec`` writes ``p_codec`` /
``s_codec`` / ``c_codec`` per piece. v2 wants the richer score features
from ``features.xml_features.extract_xml_features`` packed into the
``s_codec`` dimension. This script reads the existing HDF5, loads each
piece's score (via the embedded ``score_path``), extracts XML features
per segment matching the existing ``snote_id_path`` slicing, and writes
a new ``xml_features`` dataset alongside the existing ones.

Result: an additive HDF5 augmentation. Original v1 fields are untouched;
``load_data_from_hdf5(..., include_xml_features=True)`` then concatenates
the new columns onto ``s_codec`` at load time.

Skips:

* groups with an ``error`` dataset
* groups that already have ``xml_features`` (idempotent)
* pieces whose score file can't be loaded (logged to errors.jsonl)

Output dimensions: ``(n_segments, seg_len, n_xml_features)`` where
``n_xml_features = len(FEATURE_SPEC) = 14``.

Usage::

    python -m features.augment_hdf5 \\
        --hdf5 data/codec_N=200_mixup.hdf5 \\
        --datasets_root_orig "/Users/huanz/01Acdemics/PhD/Research/Datasets/" \\
        --datasets_root_new  "/Users/huanz/01Acdemics/PhD/Research/Datasets/"

The ``_orig``/``_new`` pair is the same path-remap mechanism used by
``load_data_from_hdf5``: pass them when running on a host different from
where the codec was generated. Both can be omitted on the regen host.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import partitura as pt
from tqdm import tqdm

from features.xml_features import FEATURE_SPEC, PieceFeatureCache, extract_xml_features

logger = logging.getLogger(__name__)


# Stable ordering matching FEATURE_SPEC for downstream embedding tables.
FEATURE_NAMES = [name for (name, _kind, _vocab) in FEATURE_SPEC]
N_XML_FEATURES = len(FEATURE_NAMES)


def _remap(path: str, src: Optional[str], dst: Optional[str]) -> str:
    if not src or not dst:
        return path
    return dst + path[len(src):] if path.startswith(src) else path


def _maybe_unfold_score(score: pt.score.Score, snote_ids: list[str]) -> pt.score.Score:
    """Replicate the unfolding logic from prepare_data.get_codecs.

    The codec generator unfolds repeated sections when the alignment carries
    IDs with a "-N" suffix but the bare score doesn't. We don't have access
    to the alignment here, so we infer the same condition from the
    snote_ids: if they contain "-" and the score's bare IDs don't, unfold.
    """
    if not snote_ids:
        return score
    has_unfolded = "-" in snote_ids[0]
    score_ids = score.parts[0].note_array(include_divs_per_quarter=False)["id"]
    if has_unfolded and len(score_ids) and "-" not in score_ids[0]:
        score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts))
    return score


class _PieceContext:
    """Per-piece resources reused across all segments of that piece.

    Building note_array_full + the id→idx map is O(notes-in-piece) and the
    PieceFeatureCache rebuilds note_by_id + direction-span lists. Cache them
    once per piece — the augment loop usually visits all segments of a piece
    consecutively because hdf5 group order matches discovery order.
    """

    __slots__ = ("score_path", "note_array_full", "id_to_idx", "feature_cache")

    def __init__(self, score: pt.score.Score, score_path: str):
        self.score_path = score_path
        part = score if isinstance(score, pt.score.Part) else score.parts[0]
        self.note_array_full = part.note_array(
            include_pitch_spelling=True,
            include_key_signature=True,
            include_time_signature=True,
            include_metrical_position=True,
            include_grace_notes=True,
            include_staff=True,
            include_divs_per_quarter=True,
        )
        self.id_to_idx = {nid: i for i, nid in enumerate(self.note_array_full["id"])}
        self.feature_cache = PieceFeatureCache(score)


def _segment_features(ctx: _PieceContext, snote_ids: list[str], seg_len: int) -> np.ndarray:
    """Extract XML features for a list of snote_ids, padded to seg_len."""
    seg_idxs = [ctx.id_to_idx[nid] for nid in snote_ids if nid in ctx.id_to_idx]
    if not seg_idxs:
        return np.zeros((seg_len, N_XML_FEATURES), dtype=np.float32)
    seg_note_array = ctx.note_array_full[seg_idxs]
    feats = extract_xml_features(ctx.feature_cache.part, seg_note_array, cache=ctx.feature_cache)

    out = np.zeros((seg_len, N_XML_FEATURES), dtype=np.float32)
    n_real = min(len(seg_idxs), seg_len)
    for col, name in enumerate(FEATURE_NAMES):
        out[:n_real, col] = feats[name][:n_real]
    return out


def augment(
    hdf5_path: Path,
    seg_len: int,
    datasets_root_orig: Optional[str],
    datasets_root_new: Optional[str],
    errors_path: Path,
    overwrite: bool = False,
    limit: Optional[int] = None,
):
    n_done = n_skipped = n_failed = 0
    with h5py.File(hdf5_path, "a") as hf, errors_path.open("a") as err_log:
        group_names = list(hf.keys())
        if limit:
            group_names = group_names[:limit]

        # Per-piece context cache. Keyed by (score_path, has_unfolded_ids) so
        # folded vs unfolded versions of the same score don't collide.
        ctx_cache: dict[tuple, _PieceContext] = {}
        ctx_lru_order: list[tuple] = []
        CTX_CACHE_MAX = 4   # bound memory: a few full-piece contexts at a time

        for name in tqdm(group_names, desc="augmenting"):
            grp = hf[name]
            if "error" in grp:
                continue
            if "xml_features" in grp:
                if not overwrite:
                    n_skipped += 1
                    continue
                del grp["xml_features"]

            try:
                score_path = _remap(
                    grp["score_path"][0].decode() if hasattr(grp["score_path"][0], "decode") else str(grp["score_path"][0]),
                    datasets_root_orig, datasets_root_new,
                )
                snote_id_paths = [
                    _remap(
                        s.decode() if hasattr(s, "decode") else str(s),
                        datasets_root_orig, datasets_root_new,
                    )
                    for s in grp["snote_id_path"]
                ]
            except Exception as e:
                err_log.write(json.dumps({"group": name, "stage": "path_decode", "error": str(e)}) + "\n")
                n_failed += 1
                continue

            # Peek at the first segment's IDs to decide if we need an unfolded score.
            try:
                first_snote_ids = np.load(snote_id_paths[0]).tolist()
            except Exception as e:
                err_log.write(json.dumps({"group": name, "stage": "snote_id_load", "snote_id_path": snote_id_paths[0], "error": str(e)}) + "\n")
                n_failed += 1
                continue
            has_unfolded = bool(first_snote_ids) and "-" in first_snote_ids[0]
            ctx_key = (score_path, has_unfolded)

            if ctx_key not in ctx_cache:
                try:
                    score = pt.load_musicxml(score_path, force_note_ids="keep")
                    if has_unfolded:
                        bare_ids = score.parts[0].note_array(include_divs_per_quarter=False)["id"]
                        if len(bare_ids) and "-" not in bare_ids[0]:
                            score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts))
                    ctx_cache[ctx_key] = _PieceContext(score, score_path)
                    ctx_lru_order.append(ctx_key)
                    while len(ctx_lru_order) > CTX_CACHE_MAX:
                        ctx_cache.pop(ctx_lru_order.pop(0))
                except Exception as e:
                    err_log.write(json.dumps({"group": name, "stage": "score_load", "score_path": score_path, "error": str(e)}) + "\n")
                    n_failed += 1
                    continue
            ctx = ctx_cache[ctx_key]

            xml_per_seg = []
            for sip in snote_id_paths:
                try:
                    snote_ids = np.load(sip).tolist()
                    xml_per_seg.append(_segment_features(ctx, snote_ids, seg_len))
                except Exception as e:
                    err_log.write(json.dumps({"group": name, "stage": "feature_extract", "snote_id_path": sip, "error": str(e)}) + "\n")
                    xml_per_seg.append(np.zeros((seg_len, N_XML_FEATURES), dtype=np.float32))

            xml_arr = np.stack(xml_per_seg, axis=0)
            grp.create_dataset("xml_features", data=xml_arr, compression="gzip", compression_opts=4)
            n_done += 1

    logger.info("augment: %d added, %d already had xml_features, %d failed", n_done, n_skipped, n_failed)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Augment a DExter codec HDF5 with XML features")
    p.add_argument("--hdf5", type=Path, required=True, help="path to codec_N=...mixup.hdf5")
    p.add_argument("--seg_len", type=int, default=200)
    p.add_argument("--datasets_root_orig", type=str, default=None,
                   help="prefix in score_path strings as embedded in the HDF5")
    p.add_argument("--datasets_root_new", type=str, default=None,
                   help="prefix to substitute for datasets_root_orig (cross-host runs)")
    p.add_argument("--errors", type=Path, default=Path("data/xml_augment_errors.jsonl"))
    p.add_argument("--overwrite", action="store_true",
                   help="re-extract even when xml_features already exists")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N groups — useful for smoke-testing")
    p.add_argument("--log_level", default="INFO")
    return p


def main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    augment(
        hdf5_path=args.hdf5,
        seg_len=args.seg_len,
        datasets_root_orig=args.datasets_root_orig,
        datasets_root_new=args.datasets_root_new,
        errors_path=args.errors,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
