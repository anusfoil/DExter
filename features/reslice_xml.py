"""Re-slice xml_features from a source HDF5 (e.g. N=200) into a destination
HDF5 with a different segmentation length (e.g. N=2000).

Background: the per-piece work in extract_xml_features (iter_all over the
score for slurs/dynamics/wedges/tempo directions/etc.) does not depend on
seg_len — only the final segment slicing does. So once we've augmented at
N=200, re-augmenting at N=2000 is "literally a reslice" (per user). This
script avoids the redundant 1-2h second pass over scores.

Strategy:
  1. Open SOURCE hdf5 (N=200, augmented). For each group, walk its segments
     and pair every snote_id (loaded from the segment's .npy) with the
     corresponding row of xml_features. Build per-piece dict
     ``{note_id: feature_row}`` keyed by score note id.
  2. Open DEST hdf5 (N=2000, NOT augmented). For each group, walk its
     segments, load the new (longer) snote_id arrays, look up each note's
     feature row in the per-piece dict, build (S, T, F) xml_features array,
     write to the group.

Usage::
    python -m features.reslice_xml \\
        --src data/codec_N=200_mixup.hdf5 \\
        --dst data/codec_N=2000_mixup.hdf5 \\
        --src-snote-root /Users/huanz/01Acdemics/PhD/Research/DExter_/data/snote_ids/N=200 \\
        --dst-snote-root /Users/huanz/01Acdemics/PhD/Research/DExter_/data/snote_ids/N=2000

Limitations:
  - Source must already have xml_features at the desired schema width.
    If not, run features/augment_hdf5.py on the source first.
  - Destination groups missing from source (or marked 'error' in source)
    are skipped — no features written.
  - Mixup variants in source/dest must be paired one-to-one by group key.
    The discoverer sort order is deterministic, so 'ASAP_42' refers to the
    same piece in both seg_len HDF5s.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from prepare_data import _decode

logger = logging.getLogger(__name__)


def _path_swap_root(p: str, old_root: str, new_root: str) -> str:
    """Replace the snote-id root prefix. Tolerates path-encoding mismatches."""
    if p.startswith(old_root):
        return new_root + p[len(old_root):]
    return p


def _build_piece_lookup(group: h5py.Group,
                        src_snote_root: str | None,
                        dst_snote_root: str | None) -> dict[str, np.ndarray] | None:
    """Walk a source group's segments and assemble note_id → feature_row.

    Returns None if the group's xml_features dataset is unreadable (HDF5
    corruption from past killed writes); caller skips the group.
    """
    try:
        xml = np.array(group["xml_features"], dtype=np.float32)   # (S, T, F)
        snote_paths = [_decode(p) for p in np.array(group["snote_id_path"])]
    except (OSError, KeyError) as e:
        logger.warning("source group %s unreadable: %s", group.name, e)
        return None
    out: dict[str, np.ndarray] = {}
    for s_idx, sp in enumerate(snote_paths):
        if src_snote_root and dst_snote_root:
            sp = _path_swap_root(sp, src_snote_root, dst_snote_root)
        try:
            sids = np.load(sp)
        except FileNotFoundError:
            # try without remap
            try:
                sids = np.load(snote_paths[s_idx])
            except FileNotFoundError:
                continue
        feats_seg = xml[s_idx]                                # (T, F)
        for n_idx, nid in enumerate(sids):
            if n_idx >= feats_seg.shape[0]:
                break
            nid_s = str(nid)
            if not nid_s or nid_s == "0":
                continue   # padding row
            # Don't overwrite — first occurrence wins. Mixup variants share
            # the same snote_ids as the original, so deterministic.
            if nid_s not in out:
                out[nid_s] = feats_seg[n_idx]
    return out


def reslice(src_hdf5: Path, dst_hdf5: Path,
            src_snote_root: str | None,
            dst_snote_root: str | None,
            dry_run: bool = False) -> None:
    n_done = n_skipped_no_src = n_skipped_dst_error = n_partial = 0
    n_features_unknown = 0

    dst_mode = "r" if dry_run else "r+"
    with h5py.File(src_hdf5, "r") as src, h5py.File(dst_hdf5, dst_mode) as dst:
        # Discover schema width from any augmented source group.
        sample_F = None
        for k in src:
            if "xml_features" in src[k]:
                sample_F = src[k]["xml_features"].shape[-1]
                break
        if sample_F is None:
            raise RuntimeError(f"no xml_features found in {src_hdf5}")
        logger.info("source schema: F=%d xml columns", sample_F)

        for gname in tqdm(list(dst), desc="reslice"):
            dgroup = dst[gname]
            if "error" in dgroup:
                n_skipped_dst_error += 1
                continue
            if gname not in src:
                n_skipped_no_src += 1
                continue
            sgroup = src[gname]
            if "xml_features" not in sgroup:
                n_skipped_no_src += 1
                continue

            lookup = _build_piece_lookup(sgroup, src_snote_root, dst_snote_root)
            if lookup is None:
                n_skipped_no_src += 1
                continue

            dst_snote_paths = [_decode(p) for p in np.array(dgroup["snote_id_path"])]
            # dst paths point at N=2000 snote_ids. apply remap if needed (the
            # paths are absolute; user supplies dst_snote_root for portability).
            dst_p_codec = np.array(dgroup["p_codec"], dtype=np.float32)
            S, T, _ = dst_p_codec.shape
            new_xml = np.zeros((S, T, sample_F), dtype=np.float32)

            unknown = 0
            for s_idx, sp in enumerate(dst_snote_paths):
                if dst_snote_root and src_snote_root:
                    # strip and re-add for any path that came in with the wrong root
                    sp = _path_swap_root(sp, src_snote_root, dst_snote_root)
                try:
                    sids = np.load(sp)
                except FileNotFoundError:
                    continue
                for n_idx, nid in enumerate(sids):
                    if n_idx >= T:
                        break
                    nid_s = str(nid)
                    if not nid_s:
                        continue
                    feat = lookup.get(nid_s)
                    if feat is None:
                        unknown += 1
                        continue
                    new_xml[s_idx, n_idx, :] = feat
            n_features_unknown += unknown

            if dry_run:
                if unknown:
                    n_partial += 1
                n_done += 1
                continue

            if "xml_features" in dgroup:
                del dgroup["xml_features"]
            dgroup.create_dataset("xml_features", data=new_xml, dtype="f4")
            n_done += 1
            if unknown:
                n_partial += 1

    logger.info("reslice: done=%d  partial=%d  skipped_no_src=%d  skipped_dst_error=%d  total_unknown_notes=%d",
                n_done, n_partial, n_skipped_no_src, n_skipped_dst_error, n_features_unknown)


def main() -> None:
    p = argparse.ArgumentParser(description="Re-slice xml_features into a different-seg_len HDF5.")
    p.add_argument("--src", required=True, help="source HDF5 (must be augmented)")
    p.add_argument("--dst", required=True, help="destination HDF5 (gets xml_features written)")
    p.add_argument("--src-snote-root", default=None,
                   help="prefix replacement for snote_id .npy paths in the source HDF5 (optional)")
    p.add_argument("--dst-snote-root", default=None,
                   help="prefix replacement for snote_id .npy paths in the destination HDF5 (optional)")
    p.add_argument("--dry-run", action="store_true",
                   help="do everything except writing xml_features to dst")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    reslice(Path(args.src), Path(args.dst),
            args.src_snote_root, args.dst_snote_root,
            dry_run=args.dry_run)


if __name__ == "__main__":
    main()
