"""Compute performance / score / perceptual codecs from aligned datasets.

Produces a single HDF5 file under ``data_root`` (one group per piece, segmented
to ``MAX_NOTE_LEN``) plus per-segment ``snote_ids`` ``.npy`` files.

Path configuration precedence: CLI flags > environment variables > config/paths.yaml.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import numpy.lib.recfunctions as rfn
import pandas as pd
import partitura as pt
from omegaconf import OmegaConf
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

DEFAULT_PATHS_YAML = Path(__file__).parent / "config" / "paths.yaml"


@dataclass
class DatasetPaths:
    data_root: Path                 # where outputs (hdf5, snote_ids) are written
    datasets_root: Path             # parent of source corpora

    atepp_root: Path
    atepp_meta: Path
    asap_root: Path
    vienna_match_dir: Path
    vienna_musicxml_dir: Path
    vienna_cep_dir: Path


def load_paths(
    config_path: Optional[Path] = None,
    data_root: Optional[str] = None,
    datasets_root: Optional[str] = None,
) -> DatasetPaths:
    """Resolve dataset paths from yaml + environment + CLI overrides."""
    cfg = OmegaConf.load(config_path or DEFAULT_PATHS_YAML)

    resolved_data_root = Path(
        data_root
        or os.environ.get("DEXTER_DATA_ROOT")
        or cfg.data_root
    ).expanduser().resolve()

    resolved_datasets_root = Path(
        datasets_root
        or os.environ.get("DEXTER_DATASETS_ROOT")
        or cfg.datasets_root
    ).expanduser().resolve()

    return DatasetPaths(
        data_root=resolved_data_root,
        datasets_root=resolved_datasets_root,
        atepp_root=resolved_datasets_root / cfg.datasets.ATEPP.root,
        atepp_meta=resolved_datasets_root / cfg.datasets.ATEPP.metadata,
        asap_root=resolved_datasets_root / cfg.datasets.ASAP.root,
        vienna_match_dir=resolved_datasets_root / cfg.datasets.VIENNA422.match_dir,
        vienna_musicxml_dir=resolved_datasets_root / cfg.datasets.VIENNA422.musicxml_dir,
        vienna_cep_dir=resolved_datasets_root / cfg.datasets.VIENNA422.cep_dir,
    )


# ---------------------------------------------------------------------------
# Per-dataset path discovery
# ---------------------------------------------------------------------------

@dataclass
class PieceTuple:
    score_path: Path
    performance_path: Optional[Path]   # None for VIENNA422 (alignment carries the perf)
    alignment_path: Path
    cep_path: Path
    piece_name: str


def discover_vienna(paths: DatasetPaths) -> list[PieceTuple]:
    alignment_paths = sorted(glob.glob(str(paths.vienna_match_dir / "*[!e].match")))
    out = []
    for a in alignment_paths:
        a_p = Path(a)
        stem = a_p.name[:-10]               # strip "_pNN.match"
        out.append(PieceTuple(
            score_path=paths.vienna_musicxml_dir / f"{stem}.musicxml",
            performance_path=None,
            alignment_path=a_p,
            cep_path=paths.vienna_cep_dir / f"{a_p.stem}.csv",
            piece_name=stem,
        ))
    return out


def discover_asap(paths: DatasetPaths) -> list[PieceTuple]:
    perf_paths = sorted(glob.glob(str(paths.asap_root / "**/*[!e].mid"), recursive=True))
    out = []
    for p in perf_paths:
        p_p = Path(p)
        out.append(PieceTuple(
            score_path=p_p.parent / "xml_score.musicxml",
            performance_path=p_p,
            alignment_path=p_p.parent / f"{p_p.stem}_note_alignments" / "note_alignment.tsv",
            cep_path=p_p.parent / f"{p_p.stem}_cep_features.csv",
            piece_name=str(p_p.relative_to(paths.asap_root).parent).replace("/", "_"),
        ))
    return out


def discover_atepp(paths: DatasetPaths) -> list[PieceTuple]:
    """ATEPP layout: <piece_dir>/<seg_id>/align.csv with perf at <piece_dir>/<seg_id>.mid."""
    alignment_paths = sorted(glob.glob(str(paths.atepp_root / "**/[!z]*n.csv"), recursive=True))
    out = []
    for a in alignment_paths:
        a_p = Path(a)
        seg_id = a_p.parent.name
        piece_dir = a_p.parent.parent
        perf = piece_dir / f"{seg_id}.mid"
        score_candidates = glob.glob(str(piece_dir / "*.*l"))   # .xml, .mxl, .musicxml
        if not score_candidates:
            continue
        out.append(PieceTuple(
            score_path=Path(score_candidates[0]),
            performance_path=perf,
            alignment_path=a_p,
            cep_path=piece_dir / f"{seg_id}_cep_features.csv",
            piece_name=seg_id,
        ))
    return out


DISCOVERERS = {
    "VIENNA422": discover_vienna,
    "ASAP": discover_asap,
    "ATEPP": discover_atepp,
}


def get_atepp_overlap(paths: DatasetPaths) -> set[str]:
    """ATEPP score folders with multiple performances (mixup-eligible)."""
    meta = pd.read_csv(paths.atepp_meta)
    score_groups = meta.groupby(["score_path"])["midi_path"].count()
    return {
        str(paths.atepp_root / Path(score_path).parent)
        for score_path in score_groups.index
    }


# ---------------------------------------------------------------------------
# Codec extraction (per piece)
# ---------------------------------------------------------------------------

def get_codecs(score_path, alignment_path, cep_path, performance_path=None, score=None):
    """Encode a single (score, performance, alignment) triple into p_codec / c_codec."""
    score_path = str(score_path)
    alignment_path = str(alignment_path)
    cep_path = str(cep_path) if cep_path is not None else None

    if alignment_path.endswith(".match"):
        performance, alignment = pt.load_match(alignment_path)
    elif alignment_path.endswith(".tsv"):
        alignment = pt.io.importparangonada.load_alignment_from_ASAP(alignment_path)
    elif alignment_path.endswith(".csv"):                     # ATEPP / parangonar
        alignment = pt.io.importparangonada.load_parangonada_alignment(alignment_path)
        if score is None:
            score = pt.load_musicxml(score_path, force_note_ids="keep")
    else:
        raise ValueError(f"Unknown alignment format: {alignment_path}")

    if score is None:
        score = pt.load_musicxml(score_path)

    # if the alignment has unfolded score IDs but the score doesn't, unfold the score
    if (("score_id" in alignment[0])
            and ("-" in alignment[0]["score_id"])
            and ("-" not in score.note_array(include_divs_per_quarter=False)["id"][0])):
        score = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts))

    if performance_path is not None:
        performance = pt.load_performance(str(performance_path))

    parameters, snote_ids, _, m_score = pt.musicanalysis.encode_performance(
        score, performance, alignment,
    )
    p_codec = rfn.structured_to_unstructured(parameters)

    if cep_path and os.path.exists(cep_path):
        cep_feats = pd.read_csv(cep_path)
        c_codec = rfn.structured_to_unstructured(get_cep_codec(cep_feats, m_score))
    else:
        if cep_path:
            logger.info("No cep_features for %s", alignment_path)
        c_codec = np.zeros((len(p_codec), 7))

    return p_codec, c_codec, snote_ids, score


def get_cep_codec(cep_feats: pd.DataFrame, m_score) -> np.ndarray:
    """Align perceptual features with the performance, return structured array."""
    rows = []
    for note in m_score:
        window = cep_feats[cep_feats["frame_start_time"] <= note["p_onset"]]
        if len(window):
            rows.append(window.iloc[-1])
        else:
            rows.append(cep_feats.iloc[-1])

    c_codec_df = pd.DataFrame(rows).drop(columns=["frame_start_time"])
    records = c_codec_df.to_records(index=False)
    return np.array(records, dtype=records.dtype.descr)


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def hdf5_path(paths: DatasetPaths, max_note_len: int, mixup: bool) -> Path:
    suffix = "_mixup" if mixup else ""
    return paths.data_root / f"codec_N={max_note_len}{suffix}.hdf5"


def process_dataset_codec(
    paths: DatasetPaths,
    max_note_len: int,
    mix_up: bool = False,
    datasets: Iterable[str] = ("ATEPP", "ASAP", "VIENNA422"),
    test_split_p: float = 0.15,
    rng_seed: Optional[int] = None,
) -> Path:
    """Compute and persist codecs for the requested datasets.

    Returns the path to the written HDF5 file.
    Per-piece errors are logged to ``data_root / errors.jsonl``.
    """
    rng = np.random.default_rng(rng_seed)
    paths.data_root.mkdir(parents=True, exist_ok=True)
    snote_dir = paths.data_root / f"snote_ids/N={max_note_len}"
    snote_dir.mkdir(parents=True, exist_ok=True)

    out_path = hdf5_path(paths, max_note_len, mix_up)
    errors_path = paths.data_root / "errors.jsonl"

    atepp_overlap = get_atepp_overlap(paths) if "ATEPP" in datasets else set()

    with h5py.File(out_path, "a") as hdf5_file, errors_path.open("a") as err_log:
        for dataset in datasets:
            pieces = DISCOVERERS[dataset](paths)
            logger.info("[%s] discovered %d pieces", dataset, len(pieces))

            same_score_p, mixup_p = [], []
            same_score_c, mixup_c = [], []
            prev_score_path = None
            split = "train"

            for j, piece in enumerate(tqdm(pieces, desc=dataset)):
                group_key = f"{dataset}_{j}"
                if group_key in hdf5_file:
                    continue
                if not (piece.score_path.exists() and piece.alignment_path.exists()):
                    err_log.write(json.dumps({
                        "dataset": dataset, "idx": j, "reason": "missing_files",
                        "score": str(piece.score_path), "alignment": str(piece.alignment_path),
                    }) + "\n")
                    continue

                # ATEPP-only: drop alignments with <50% match
                if dataset == "ATEPP":
                    try:
                        alignment = pt.io.importparangonada.load_parangonada_alignment(str(piece.alignment_path))
                    except Exception as e:
                        err_log.write(json.dumps({
                            "dataset": dataset, "idx": j, "reason": "alignment_load_error",
                            "alignment": str(piece.alignment_path), "error": str(e),
                        }) + "\n")
                        continue
                    counts = {label: 0 for label in ("match", "insertion", "deletion")}
                    for a in alignment:
                        if a["label"] in counts:
                            counts[a["label"]] += 1
                    total = sum(counts.values())
                    if total == 0 or counts["match"] / total < 0.5:
                        err_log.write(json.dumps({
                            "dataset": dataset, "idx": j, "reason": "low_alignment_match_ratio",
                            "ratio": counts["match"] / total if total else 0.0,
                        }) + "\n")
                        continue

                reuse_score = (prev_score_path == piece.score_path)
                try:
                    p_codec, c_codec, snote_ids, score = get_codecs(
                        piece.score_path, piece.alignment_path, piece.cep_path,
                        performance_path=piece.performance_path,
                        score=(score if reuse_score else None),
                    )
                except Exception as e:
                    err_log.write(json.dumps({
                        "dataset": dataset, "idx": j, "reason": "codec_extraction_error",
                        "score": str(piece.score_path), "error": str(e),
                    }) + "\n")
                    continue

                # tempo sanity check
                if 60 / p_codec[:, 0].mean() > 200:
                    hdf5_file.create_group(group_key).create_dataset("error", data="tempo_outlier")
                    continue

                if not reuse_score:
                    same_score_p, mixup_p = [], []
                    same_score_c, mixup_c = [], []
                    split = "train" if rng.random() > test_split_p else "test"
                else:
                    eligible = (
                        dataset == "VIENNA422"
                        or (dataset == "ATEPP" and str(piece.score_path.parent) in atepp_overlap)
                    )
                    if mix_up and eligible:
                        mixup_p = [np.mean([p_codec, ssp], axis=0) for ssp in same_score_p]
                        mixup_c = [np.mean([c_codec, ssc], axis=0) for ssc in same_score_c]
                        same_score_p.append(p_codec)
                        same_score_c.append(c_codec)

                sna = score.note_array()
                sna = sna[np.in1d(sna["id"], snote_ids)]
                s_codec = rfn.structured_to_unstructured(
                    sna[["onset_div", "duration_div", "pitch", "voice"]]
                )

                if not (len(p_codec) == len(s_codec) == len(snote_ids)):
                    hdf5_file.create_group(group_key).create_dataset("error", data="length_mismatch")
                    err_log.write(json.dumps({
                        "dataset": dataset, "idx": j, "reason": "length_mismatch",
                        "p": len(p_codec), "s": len(s_codec), "snote_ids": len(snote_ids),
                    }) + "\n")
                    continue

                data = _segment_piece(
                    snote_dir=snote_dir,
                    dataset=dataset,
                    piece_name=piece.piece_name,
                    score_path=piece.score_path,
                    p_codecs=[p_codec, *mixup_p],
                    c_codecs=[c_codec, *mixup_c],
                    s_codec=s_codec,
                    snote_ids=snote_ids,
                    split=split,
                    max_note_len=max_note_len,
                )

                grp = hdf5_file.create_group(group_key)
                for key in ("p_codec", "s_codec", "c_codec", "snote_id_path",
                            "score_path", "piece_name"):
                    grp.create_dataset(key, data=data[key])
                grp.create_dataset(
                    "split", data=data["split"], dtype=h5py.string_dtype(encoding="utf-8")
                )

                prev_score_path = piece.score_path

    logger.info("Wrote %s", out_path)
    return out_path


def _segment_piece(
    *, snote_dir, dataset, piece_name, score_path,
    p_codecs, c_codecs, s_codec, snote_ids, split, max_note_len,
):
    """Segment a piece (and its mixup variants) into fixed-length windows.

    Pads short tail segments to ``max_note_len`` with zeros.
    """
    data = defaultdict(list)
    for variant_idx, (p_codec, c_codec) in enumerate(zip(p_codecs, c_codecs)):
        variant_name = piece_name + ("_mixup" if variant_idx > 0 else "")
        save_id_prefix = snote_dir / f"{dataset}_{piece_name}"

        for seg_idx, start in enumerate(range(0, len(p_codec), max_note_len)):
            end = start + max_note_len
            seg_p = p_codec[start:end]
            seg_s = s_codec[start:end]
            seg_c = c_codec[start:end]
            seg_ids = snote_ids[start:end]
            if len(seg_ids) == 0:
                continue

            if len(seg_p) < max_note_len:
                pad = max_note_len - len(seg_p)
                seg_p = np.pad(seg_p, ((0, pad), (0, 0)))
                seg_s = np.pad(seg_s, ((0, pad), (0, 0)))
                seg_c = np.pad(seg_c, ((0, pad), (0, 0)))

            seg_id_path = f"{save_id_prefix}_seg{seg_idx}.npy"
            np.save(seg_id_path, seg_ids)

            data["p_codec"].append(seg_p)
            data["s_codec"].append(seg_s)
            data["c_codec"].append(seg_c)
            data["snote_id_path"].append(seg_id_path)
            data["score_path"].append(str(score_path))
            data["piece_name"].append(variant_name)
            data["split"].append(split)
    return data


# ---------------------------------------------------------------------------
# HDF5 → in-memory loader (consumed by train.py)
# ---------------------------------------------------------------------------

def _decode(x) -> str:
    """h5py returns numpy bytes for string datasets; decode to plain str."""
    return x.decode() if hasattr(x, "decode") else str(x)


def _apply_path_remap(path: str, path_remap: Optional[dict]) -> str:
    """Prefix-substitute a path, first match wins. Pass-through if no remap matches."""
    if not path_remap:
        return path
    for old, new in path_remap.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_data_from_hdf5(
    hdf5_path,
    path_remap: Optional[dict] = None,
    include_xml_features: bool = False,
    dataset_filter: Optional[list] = None,
):
    """Load codec dicts from HDF5, optionally rewriting absolute paths.

    ``path_remap`` is a dict of ``{old_prefix: new_prefix}``; both ``score_path``
    and ``snote_id_path`` are rewritten when they start with one of the keys.
    Use this to consume an HDF5 produced on a different machine — score paths
    embedded by ``prepare_data.py`` are absolute, so cross-host portability needs
    a runtime override.

    If ``include_xml_features=True`` and groups carry a ``xml_features``
    dataset (added by ``features/augment_hdf5.py``), those columns are
    concatenated to ``s_codec`` so the consumer sees a wider score-codec
    array. Pieces missing the dataset are silently skipped — keeps the loader
    usable on partially-augmented hdf5 during incremental rollout.

    ``dataset_filter`` (e.g. ``['ASAP']``) keeps only groups whose name starts
    with one of the listed prefixes. Use this to train on a single dataset
    (typically ASAP, which has the cleanest alignment labels — VirtuosoNet
    showed onset-deviation σ went 7.369 → 0.053 quarter-notes after refining
    alignment, so label noise is a real ceiling on this task).
    """
    train_data, test_data = [], []
    logger.info("Loading data from %s", hdf5_path)
    if path_remap:
        logger.info("path_remap: %s", dict(path_remap))
    if include_xml_features:
        logger.info("Loading xml_features columns alongside s_codec")
    if dataset_filter:
        logger.info("dataset_filter: keeping only %s groups", list(dataset_filter))

    skipped_no_xml = 0
    skipped_filtered = 0
    with h5py.File(hdf5_path, "r") as hdf5_file:
        for group_name in tqdm(hdf5_file):
            group = hdf5_file[group_name]
            if "error" in group:
                continue
            if dataset_filter and not any(group_name.startswith(d + "_") for d in dataset_filter):
                skipped_filtered += 1
                continue
            if include_xml_features and "xml_features" not in group:
                skipped_no_xml += 1
                continue

            p_codec = np.array(group["p_codec"])
            s_codec = np.array(group["s_codec"])
            if include_xml_features:
                # xml_features carries continuous columns (cresc_progress, dim_progress)
                # that would be truncated to 0 if cast to s_codec's int dtype.
                # Promote both to float32 for the concat.
                xml_features = np.array(group["xml_features"], dtype=np.float32)
                s_codec = np.concatenate(
                    [s_codec.astype(np.float32, copy=False), xml_features],
                    axis=-1,
                )
            c_codec = np.array(group["c_codec"])
            snote_id_path = np.array(group["snote_id_path"])
            score_path = np.array(group["score_path"])
            piece_name = np.array(group["piece_name"])
            split = np.array(group["split"])

            piece_data = [{
                "p_codec": p_codec[i],
                "s_codec": s_codec[i],
                "c_codec": c_codec[i],
                "snote_id_path": _apply_path_remap(_decode(snote_id_path[i]), path_remap),
                "score_path": _apply_path_remap(_decode(score_path[i]), path_remap),
                "piece_name": _decode(piece_name[i]),
                "split": _decode(split[i]),
            } for i in range(len(p_codec))]

            if piece_data[0]["split"] == "train":
                train_data.extend(piece_data)
            else:
                test_data.extend(pd_ for pd_ in piece_data if "mixup" not in pd_["piece_name"])

    if include_xml_features and skipped_no_xml:
        logger.warning(
            "skipped %d groups missing xml_features — re-run features/augment_hdf5.py to fill",
            skipped_no_xml,
        )
    if dataset_filter and skipped_filtered:
        logger.info("skipped %d groups outside %s", skipped_filtered, list(dataset_filter))
    return train_data, test_data


# ---------------------------------------------------------------------------
# Transfer pairing (segment-level pairs of real performances of the same piece)
# ---------------------------------------------------------------------------

def make_transfer_pair(codec_data, paths: DatasetPaths, K=50000, N=200):
    """Generate K pairs of segments (same piece + same segment idx, different performance).

    Cached at ``data_root/codec_N={N}_mixup_paired_K={K}.npy`` and ``..._unpaired_K={K}.npy``.
    """
    paired_path = paths.data_root / f"codec_N={N}_mixup_paired_K={K}.npy"
    unpaired_path = paths.data_root / f"codec_N={N}_mixup_unpaired_K={K}.npy"
    if paired_path.exists() and unpaired_path.exists():
        return (
            np.load(paired_path, allow_pickle=True),
            np.load(unpaired_path, allow_pickle=True),
        )

    codec_data = np.array(codec_data)
    real_only = codec_data[[i for i, x in enumerate(codec_data) if "mu" not in x["piece_name"]]]
    np.random.shuffle(real_only)
    unpaired = codec_data
    transfer_pairs = []

    for cd in tqdm(real_only):
        if len(transfer_pairs) > K:
            break
        seg_id = cd["snote_id_path"].split("seg")[-1].split(".")[0]
        mask = [
            (x["score_path"] == cd["score_path"]
             and f"seg{seg_id}." in x["snote_id_path"]
             and (x["p_codec"] != cd["p_codec"]).any())
            for x in real_only
        ]
        same_piece = real_only[mask]
        if len(same_piece):
            for scd in same_piece:
                transfer_pairs.extend([cd, scd])
            unpaired = unpaired[[x["score_path"] != cd["score_path"] for x in unpaired]]

    transfer_pairs = np.array(transfer_pairs).reshape(2, -1, order="F")
    np.random.shuffle(transfer_pairs.T)
    transfer_pairs = transfer_pairs.ravel(order="F")

    np.save(paired_path, transfer_pairs)
    np.save(unpaired_path, unpaired)
    return transfer_pairs, unpaired


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DExter codec preprocessing")
    p.add_argument("--compute_codec", action="store_true",
                   help="compute and persist p/s/c codecs to HDF5")
    p.add_argument("--pairing", action="store_true",
                   help="build transfer pairs from a precomputed codec HDF5")
    p.add_argument("--MAX_NOTE_LEN", type=int, default=200,
                   help="segment length in notes")
    p.add_argument("--K", type=int, default=2000,
                   help="number of transfer pairs to draw")
    p.add_argument("--mixup", action="store_true", default=True,
                   help="enable mixup augmentation across same-score performances")
    p.add_argument("--no-mixup", dest="mixup", action="store_false")
    p.add_argument("--datasets", nargs="+", default=["ATEPP", "ASAP", "VIENNA422"],
                   choices=list(DISCOVERERS.keys()))
    p.add_argument("--data_root", type=str, default=None,
                   help="output directory; overrides $DEXTER_DATA_ROOT and config/paths.yaml")
    p.add_argument("--datasets_root", type=str, default=None,
                   help="parent of source corpora; overrides $DEXTER_DATASETS_ROOT")
    p.add_argument("--paths_config", type=str, default=None,
                   help="path to a paths.yaml override")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--log_level", default="INFO")
    return p


def main():
    args = _build_argparser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    paths = load_paths(
        config_path=Path(args.paths_config) if args.paths_config else None,
        data_root=args.data_root,
        datasets_root=args.datasets_root,
    )
    logger.info("data_root=%s  datasets_root=%s", paths.data_root, paths.datasets_root)

    if args.compute_codec:
        process_dataset_codec(
            paths,
            max_note_len=args.MAX_NOTE_LEN,
            mix_up=args.mixup,
            datasets=args.datasets,
            rng_seed=args.seed,
        )
    elif args.pairing:
        codec_path = paths.data_root / f"codec_N={args.MAX_NOTE_LEN}_mixup_test.npy"
        codec_data = np.load(codec_path, allow_pickle=True)
        make_transfer_pair(codec_data, paths, K=args.K, N=args.MAX_NOTE_LEN)
    else:
        _build_argparser().print_help()


if __name__ == "__main__":
    main()
