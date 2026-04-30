"""XML-derived score features for DExter v2.

Extracts a richer per-note feature set than v1's ``(onset_div, duration_div,
pitch, voice)``, organized by natural scope:

- **note-scoped**: articulation, ornament, slur position, tie state, fermata,
  tuplet, grace status, staff, pitch spelling
- **beat/onset-time-scoped**: active dynamic marking (constant + increasing /
  decreasing wedge state), active tempo direction, sustain-pedal state
- **measure-scoped**: time signature, key signature, metrical position
- **piece-scoped**: opening tempo character (currently not extracted; left as a
  separate piece-level call)

The output is a dict of ``np.ndarray``\\ s, all aligned with the rows of the
input ``note_array`` so they can be concatenated or fed to embedding tables
downstream.

Usage::

    score = pt.load_musicxml(path, force_note_ids='keep')
    sna = score.note_array(include_pitch_spelling=True, ...)
    feats = extract_xml_features(score, sna)
    feats['articulation_id']  # shape (N,), int categorical
    feats['active_dynamic_id']  # shape (N,)
    feats['cresc_progress']    # shape (N,), float in [0,1] inside a wedge
    ...

The intent is to consume these alongside the existing s_codec — not replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import partitura as pt
import partitura.score as ps


# ---------------------------------------------------------------------------
# Vocabularies — mapping categorical strings to integer IDs.
#
# Each vocab includes 0 = "<none>" so a note with no articulation maps cleanly
# to id 0 and we can use a single embedding table per category.
# ---------------------------------------------------------------------------

ARTICULATION_VOCAB = {
    "<none>": 0,
    "staccato": 1,
    "staccatissimo": 2,
    "accent": 3,
    "marcato": 4,
    "tenuto": 5,
    "detached-legato": 6,
    "spiccato": 7,
    "stress": 8,
    "unstress": 9,
    "soft-accent": 10,
}

ORNAMENT_VOCAB = {
    "<none>": 0,
    "trill-mark": 1,
    "turn": 2,
    "inverted-turn": 3,
    "mordent": 4,
    "inverted-mordent": 5,
    "shake": 6,
    "schleifer": 7,
    "tremolo": 8,
}

SLUR_VOCAB = {
    "<none>": 0,
    "start": 1,
    "inside": 2,
    "stop": 3,
    "start-stop": 4,   # both start and stop on same note
}

TIE_VOCAB = {
    "<none>": 0,
    "start": 1,
    "continue": 2,
    "stop": 3,
}

# Loudness markings come both from explicit text ("p", "f") and as the level
# *during* a wedge. We map common labels; anything unknown falls back to <other>.
LOUDNESS_VOCAB = {
    "<none>": 0,
    "ppp": 1, "pp": 2, "p": 3, "mp": 4, "mf": 5, "f": 6, "ff": 7, "fff": 8,
    "fp": 9, "sf": 10, "sfz": 11, "sffz": 12, "rfz": 13,
    "<other>": 14,
}

# Tempo direction kind — orthogonal to the tempo text itself.
TEMPO_KIND_VOCAB = {
    "<none>": 0,
    "constant": 1,           # `Allegro`, `Lento`, `Andante`...
    "increasing": 2,         # `accel.`, `stringendo`
    "decreasing": 3,         # `rit.`, `ritenuto`, `rallentando`
    "reset": 4,              # `a tempo`, `Tempo I`
}


# ---------------------------------------------------------------------------
# Direction span helpers
# ---------------------------------------------------------------------------

@dataclass
class _Span:
    """A score-time span carrying a discrete label and a kind."""
    start: int           # divs
    end: int             # divs (may equal start for instantaneous marks)
    label: str           # e.g. "p", "crescendo", "lento"
    kind: str            # one of TEMPO_KIND_VOCAB / loudness kind


def _direction_to_span(d: ps.Direction, kind: str) -> _Span | None:
    """Convert a partitura Direction into a _Span, or None if unusable."""
    start = getattr(d.start, "t", None)
    end = getattr(d.end, "t", None) if d.end is not None else None
    if start is None:
        return None
    if end is None:
        end = start
    label = (getattr(d, "text", None) or getattr(d, "raw_text", None) or "").strip().lower()
    return _Span(start=start, end=end, label=label, kind=kind)


def _collect_loudness_spans(part: ps.Part) -> tuple[list[_Span], list[_Span], list[_Span]]:
    """Returns (constant_spans, increasing_spans, decreasing_spans)."""
    constants = [
        _direction_to_span(d, "constant")
        for d in part.iter_all(ps.ConstantLoudnessDirection)
    ]
    increasing = [
        _direction_to_span(d, "increasing")
        for d in part.iter_all(ps.IncreasingLoudnessDirection)
    ]
    decreasing = [
        _direction_to_span(d, "decreasing")
        for d in part.iter_all(ps.DecreasingLoudnessDirection)
    ]
    return (
        [s for s in constants if s is not None],
        [s for s in increasing if s is not None],
        [s for s in decreasing if s is not None],
    )


def _collect_tempo_spans(part: ps.Part) -> list[_Span]:
    spans: list[_Span] = []
    for klass, kind in [
        (ps.ConstantTempoDirection, "constant"),
        (ps.IncreasingTempoDirection, "increasing") if hasattr(ps, "IncreasingTempoDirection") else (None, None),
        (ps.DecreasingTempoDirection, "decreasing") if hasattr(ps, "DecreasingTempoDirection") else (None, None),
        (ps.ResetTempoDirection, "reset") if hasattr(ps, "ResetTempoDirection") else (None, None),
    ]:
        if klass is None:
            continue
        for d in part.iter_all(klass):
            sp = _direction_to_span(d, kind)
            if sp is not None:
                spans.append(sp)
    return spans


def _collect_pedal_spans(part: ps.Part) -> list[_Span]:
    """Sustain-pedal on/off spans. Returns a Span with label='down' for the
    pressed interval. Spans are non-overlapping per pedal."""
    spans: list[_Span] = []
    klass = getattr(ps, "SustainPedalDirection", None) or getattr(ps, "PedalDirection", None)
    if klass is None:
        return spans
    for d in part.iter_all(klass):
        sp = _direction_to_span(d, "pedal")
        if sp is not None:
            sp.label = "down"
            spans.append(sp)
    return spans


def _active_span_at(t: int, spans: Iterable[_Span]) -> _Span | None:
    """Return the span covering time ``t``; if multiple, pick the one with the
    latest start (most recent direction wins)."""
    candidates = [s for s in spans if s.start <= t < max(s.end, s.start + 1)]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.start)


def _wedge_progress(t: int, span: _Span | None) -> float:
    """Fractional position [0,1] inside a wedge, or 0 if not inside one."""
    if span is None or span.end <= span.start:
        return 0.0
    return float(np.clip((t - span.start) / (span.end - span.start), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Note-level extraction
# ---------------------------------------------------------------------------

def _slur_state(note: ps.Note) -> str:
    starts = bool(note.slur_starts)
    stops = bool(note.slur_stops)
    if starts and stops:
        return "start-stop"
    if starts:
        return "start"
    if stops:
        return "stop"
    return "<none>"   # caller will resolve "inside" by checking slur span coverage


def _tie_state(note: ps.Note) -> str:
    has_next = note.tie_next is not None
    has_prev = note.tie_prev is not None
    if has_next and has_prev:
        return "continue"
    if has_next:
        return "start"
    if has_prev:
        return "stop"
    return "<none>"


def _vocab_get(vocab: dict[str, int], key: str, default_key: str = "<other>") -> int:
    """Look up ``key`` (case-insensitive); fall back to ``<other>`` then 0."""
    v = vocab.get(key.lower())
    if v is None:
        v = vocab.get(default_key, 0)
    return int(v)


def extract_xml_features(
    score: ps.Score | ps.Part,
    note_array: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Extract structured XML features aligned with ``note_array``'s rows.

    Args:
        score: a partitura Score or single Part.
        note_array: pre-computed note_array. If None, computed from the score
            with all include_* flags on. Order must match the iteration order
            of ``part.iter_all(Note)``.

    Returns:
        Dict of features. Each value is an ``np.ndarray`` of length N (number
        of notes), with dtype int (categorical IDs) or float (continuous).
        Keys:

        * ``articulation_id``, ``ornament_id``, ``slur_state_id``,
          ``tie_state_id``, ``has_fermata``, ``is_grace``, ``in_tuplet``
        * ``active_loudness_id``, ``in_cresc``, ``cresc_progress``,
          ``in_dim``, ``dim_progress``
        * ``active_tempo_kind_id`` (constant / inc / dec / reset / none)
        * ``pedal_down`` (0/1)
    """
    part = score.parts[0] if isinstance(score, ps.Score) else score

    if note_array is None:
        note_array = part.note_array(
            include_pitch_spelling=True,
            include_key_signature=True,
            include_time_signature=True,
            include_metrical_position=True,
            include_grace_notes=True,
            include_staff=True,
            include_divs_per_quarter=True,
        )

    note_by_id = {n.id: n for n in part.iter_all(ps.Note)}

    constant_loud, inc_loud, dec_loud = _collect_loudness_spans(part)
    tempo_spans = _collect_tempo_spans(part)
    pedal_spans = _collect_pedal_spans(part)

    N = len(note_array)
    feats: dict[str, np.ndarray] = {
        "articulation_id":      np.zeros(N, dtype=np.int32),
        "ornament_id":          np.zeros(N, dtype=np.int32),
        "slur_state_id":        np.zeros(N, dtype=np.int32),
        "tie_state_id":         np.zeros(N, dtype=np.int32),
        "has_fermata":          np.zeros(N, dtype=np.int32),
        "is_grace":             np.zeros(N, dtype=np.int32),
        "in_tuplet":            np.zeros(N, dtype=np.int32),
        "active_loudness_id":   np.zeros(N, dtype=np.int32),
        "in_cresc":             np.zeros(N, dtype=np.int32),
        "cresc_progress":       np.zeros(N, dtype=np.float32),
        "in_dim":               np.zeros(N, dtype=np.int32),
        "dim_progress":         np.zeros(N, dtype=np.float32),
        "active_tempo_kind_id": np.zeros(N, dtype=np.int32),
        "pedal_down":           np.zeros(N, dtype=np.int32),
    }

    for i, row in enumerate(note_array):
        nid = row["id"]
        note = note_by_id.get(nid)
        if note is None:
            # row exists in note_array but not in iter_all(Note) — rare; skip.
            continue

        # --- per-note categorical
        if note.articulations:
            feats["articulation_id"][i] = _vocab_get(
                ARTICULATION_VOCAB, note.articulations[0]
            )
        if note.ornaments:
            first_orn = next(iter(note.ornaments))
            feats["ornament_id"][i] = _vocab_get(ORNAMENT_VOCAB, first_orn)
        feats["slur_state_id"][i] = SLUR_VOCAB[_slur_state(note)]
        feats["tie_state_id"][i]  = TIE_VOCAB[_tie_state(note)]
        feats["has_fermata"][i]   = int(note.fermata is not None)
        feats["is_grace"][i]      = int(getattr(note, "is_grace", False))
        feats["in_tuplet"][i]     = int(bool(note.tuplet_starts) or bool(note.tuplet_stops))

        # --- onset-time-scoped (look up active span at note's onset_div)
        t = int(row["onset_div"])
        cl = _active_span_at(t, constant_loud)
        if cl is not None:
            feats["active_loudness_id"][i] = _vocab_get(LOUDNESS_VOCAB, cl.label)
        ic = _active_span_at(t, inc_loud)
        if ic is not None:
            feats["in_cresc"][i] = 1
            feats["cresc_progress"][i] = _wedge_progress(t, ic)
        dc = _active_span_at(t, dec_loud)
        if dc is not None:
            feats["in_dim"][i] = 1
            feats["dim_progress"][i] = _wedge_progress(t, dc)
        tp = _active_span_at(t, tempo_spans)
        if tp is not None:
            feats["active_tempo_kind_id"][i] = TEMPO_KIND_VOCAB.get(tp.kind, 0)
        ped = _active_span_at(t, pedal_spans)
        feats["pedal_down"][i] = int(ped is not None)

    return feats


# Compact list of (name, dtype, vocab_size) for downstream embedding tables.
FEATURE_SPEC = [
    ("articulation_id",      "categorical", len(ARTICULATION_VOCAB)),
    ("ornament_id",          "categorical", len(ORNAMENT_VOCAB)),
    ("slur_state_id",        "categorical", len(SLUR_VOCAB)),
    ("tie_state_id",         "categorical", len(TIE_VOCAB)),
    ("has_fermata",          "binary",      2),
    ("is_grace",             "binary",      2),
    ("in_tuplet",            "binary",      2),
    ("active_loudness_id",   "categorical", len(LOUDNESS_VOCAB)),
    ("in_cresc",             "binary",      2),
    ("cresc_progress",       "continuous",  None),
    ("in_dim",               "binary",      2),
    ("dim_progress",         "continuous",  None),
    ("active_tempo_kind_id", "categorical", len(TEMPO_KIND_VOCAB)),
    ("pedal_down",           "binary",      2),
]
