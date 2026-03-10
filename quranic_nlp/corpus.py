"""
quranic_nlp.corpus
~~~~~~~~~~~~~~~~~~
Cross-verse continuous-sequence pattern matcher for the entire Quran.

The full corpus (~128 K tokens from *morphologhy.csv*) is loaded into
flat numpy arrays and inverted indexes, enabling O(log N) per-step
lookup during pattern search.  The cascade engine is fully vectorised:
no Python-level loops over candidate arrays.

Pattern syntax is identical to :class:`~quranic_nlp.query.VerseMatcher`
(see ``query.py``), but patterns can freely span verse and surah
boundaries.

POS / TAG notation (Quranic Treebank)::

    N    – noun              V     – verb
    P    – preposition       PRON  – pronoun
    PN   – proper noun       CONJ  – conjunction
    DET  – determiner        ADJ   – adjective
    REL  – relative pronoun  NEG   – negation particle
    REM  – resumption        T     – time adverb
    … (see morphologhy.csv Tag column for the full inventory)

Quick start
-----------
::

    from quranic_nlp.corpus import CorpusIndex

    # Build once (~1–2 s) and cache to disk
    idx = CorpusIndex.build(save=True)

    # Load cached index on subsequent calls (~0.04 s)
    idx = CorpusIndex.load()

    # ── Single condition ───────────────────────────────────────────────
    matches = idx.search([{'ROOT': 'رحم'}])

    # ── Consecutive pair ───────────────────────────────────────────────
    matches = idx.search([{'ROOT': 'رحم'}, {'ROOT': 'علم'}])

    # ── Proximity / SKIP (spans verse boundaries) ──────────────────────
    matches = idx.search([
        {'ROOT': 'رحم'},
        {'ROOT': 'علم', 'SKIP': 5},
    ])

    # ── POS + ROOT combination ─────────────────────────────────────────
    matches = idx.search([
        {'TAG': 'N', 'ROOT': 'صبر'},
        {'TAG': 'V', 'SKIP': 3},
    ])

    # ── Optional element (OP='?') ──────────────────────────────────────
    matches = idx.search([
        {'ROOT': 'علم'},
        {'TAG': 'DET', 'OP': '?'},
        {'TAG': 'N'},
    ])

    # ── Any-of values ──────────────────────────────────────────────────
    matches = idx.search([{'ROOT': {'IN': ['رحم', 'علم']}}])
"""

from __future__ import annotations

import os
import re as _re
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quranic_nlp import constant, utils


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_STRUCTURAL = frozenset(['OP', 'SKIP'])

_DEFAULT_CACHE = os.path.join(
    os.path.dirname(__file__), 'data', '_corpus_index.pkl'
)

# Attribute name (upper) → flat-array column name
_ATTR_COL: Dict[str, str] = {
    'TEXT':   'text',
    'LOWER':  'simple',
    'SIMPLE': 'simple',
    'LEMMA':  'lemma',
    'ROOT':   'root',
    'TAG':    'tag',
    'POS':    'tag',
}

_INV_COLS = ('text', 'simple', 'lemma', 'root', 'tag')

_EMPTY = np.array([], dtype=np.int32)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

class CorpusToken:
    """One token in the flat Quranic corpus."""

    __slots__ = ('pos', 'soure', 'ayeh', 'tok_i', 'text', 'simple',
                 'lemma', 'root', 'tag')

    def __init__(self, pos, soure, ayeh, tok_i, text, simple, lemma, root, tag):
        self.pos    = pos     #: flat 0-based index
        self.soure  = soure   #: surah number (1–114)
        self.ayeh   = ayeh    #: verse number (1-based)
        self.tok_i  = tok_i   #: position within verse (0-based)
        self.text   = text    #: surface form with diacritics
        self.simple = simple  #: surface form without diacritics
        self.lemma  = lemma   #: canonical lemma  ('' if unknown)
        self.root   = root    #: trilateral root  ('' if unknown)
        self.tag    = tag     #: POS tag (Quranic Treebank notation)

    def __repr__(self) -> str:
        return (
            f'CorpusToken(pos={self.pos}, ref={self.soure}:{self.ayeh},'
            f' tok={self.tok_i}, text={self.text!r},'
            f' root={self.root!r}, tag={self.tag!r})'
        )


@dataclass
class CorpusMatch:
    """A pattern match over one or more consecutive corpus positions."""

    pattern_key: str
    tokens: List[CorpusToken]

    @property
    def start(self) -> int:
        return self.tokens[0].pos

    @property
    def end(self) -> int:
        return self.tokens[-1].pos + 1

    @property
    def refs(self) -> List[Tuple[int, int]]:
        """Unique ``(soure, ayeh)`` pairs covered by this match."""
        seen: list = []
        for t in self.tokens:
            ref = (t.soure, t.ayeh)
            if ref not in seen:
                seen.append(ref)
        return seen

    @property
    def text(self) -> str:
        return ' '.join(t.text for t in self.tokens)

    def __repr__(self) -> str:
        refs_str = ', '.join(f'{s}:{a}' for s, a in self.refs)
        return (
            f'CorpusMatch(key={self.pattern_key!r},'
            f' refs=[{refs_str}], text={self.text!r})'
        )


# ---------------------------------------------------------------------------
# Condition matching helpers
# ---------------------------------------------------------------------------

def _to_str(val) -> str:
    if val is None:
        return ''
    if isinstance(val, float) and (val != val):
        return ''
    return str(val)


def _cond_matches_arrays(col_arrays: Dict[str, np.ndarray], idx: int, cond: dict) -> bool:
    for key, value in cond.items():
        k = key.upper()
        if k in _STRUCTURAL:
            continue
        negate = k.startswith('NOT_')
        attr = k[4:] if negate else k
        col = _ATTR_COL.get(attr)
        if col is None:
            continue
        tok_val = str(col_arrays[col][idx])

        if isinstance(value, dict):
            if 'IN' in value:
                match = tok_val in [_to_str(v) for v in value['IN']]
            elif 'NOT_IN' in value:
                match = tok_val not in [_to_str(v) for v in value['NOT_IN']]
            elif 'REGEX' in value:
                match = bool(_re.search(value['REGEX'], tok_val))
            else:
                match = False
        elif isinstance(value, (list, tuple)):
            match = tok_val in [_to_str(v) for v in value]
        else:
            match = tok_val == _to_str(value)

        if negate:
            match = not match
        if not match:
            return False
    return True


def _positions_for_cond(
    inv: Dict[str, Dict[str, np.ndarray]],
    col_arrays: Dict[str, list],
    cond: dict,
    N: int,
) -> np.ndarray:
    """Return sorted ``int32`` positions satisfying *cond* via inverted index."""
    indexed_sets: List[np.ndarray] = []
    post_filter: dict = {}

    for key, value in cond.items():
        k = key.upper()
        if k in _STRUCTURAL:
            continue
        negate = k.startswith('NOT_')
        attr = k[4:] if negate else k
        col = _ATTR_COL.get(attr)

        if col is None or col not in inv:
            post_filter[key] = value
            continue

        if negate or (isinstance(value, dict) and 'NOT_IN' in value):
            post_filter[key] = value
            continue

        if isinstance(value, dict):
            if 'IN' in value:
                parts = [inv[col].get(_to_str(v), _EMPTY) for v in value['IN']]
                merged = np.unique(np.concatenate(parts)) if parts else _EMPTY
                indexed_sets.append(merged)
            elif 'REGEX' in value:
                pat = _re.compile(value['REGEX'])
                parts = [arr for v, arr in inv[col].items() if pat.search(v)]
                merged = np.unique(np.concatenate(parts)) if parts else _EMPTY
                indexed_sets.append(merged)
            else:
                post_filter[key] = value
        elif isinstance(value, (list, tuple)):
            parts = [inv[col].get(_to_str(v), _EMPTY) for v in value]
            merged = np.unique(np.concatenate(parts)) if parts else _EMPTY
            indexed_sets.append(merged)
        else:
            arr = inv[col].get(_to_str(value), _EMPTY)
            indexed_sets.append(arr)

    if indexed_sets:
        result = indexed_sets[0]
        for other in indexed_sets[1:]:
            result = np.intersect1d(result, other, assume_unique=True)
    elif post_filter:
        result = np.arange(N, dtype=np.int32)
    else:
        return np.arange(N, dtype=np.int32)

    if post_filter and len(result) > 0:
        mask = np.fromiter(
            (_cond_matches_arrays(col_arrays, int(p), post_filter) for p in result),
            dtype=bool,
            count=len(result),
        )
        result = result[mask]

    return result.astype(np.int32)


# ---------------------------------------------------------------------------
# Vectorised cascade engine
# ---------------------------------------------------------------------------

# A "group" is a list of parallel int32 arrays — one array per matched
# pattern element.  All arrays in a group have the same length.
# groups[0][i] = flat position of element-0 match in chain i
# groups[1][i] = flat position of element-1 match in chain i, etc.
_Group = List[np.ndarray]


def _extend_group(group: _Group, cands: np.ndarray, skip: int) -> Optional[_Group]:
    """
    Extend each chain in *group* with one new position from *cands*.

    For ``skip == 0``: the new position must be exactly ``prev + 1``.
    For ``skip > 0``:  the new position must be in ``[prev+1, prev+1+skip]``;
                       one chain may expand into multiple (one per valid match).

    Returns ``None`` if no chains can be extended.
    """
    if len(group[0]) == 0 or len(cands) == 0:
        return None

    prev = group[-1]          # last matched positions, shape (M,)
    lo   = prev + np.int32(1)

    if skip == 0:
        # ── Consecutive: exactly prev+1 must be in cands ──────────────
        lo_idx  = np.searchsorted(cands, lo)
        safe    = np.minimum(lo_idx, len(cands) - 1)
        valid   = (lo_idx < len(cands)) & (cands[safe] == lo)
        if not np.any(valid):
            return None
        new_group = [c[valid] for c in group]
        new_group.append(cands[lo_idx[valid]])
        return new_group

    else:
        # ── SKIP: any position in [prev+1, prev+1+skip] ───────────────
        hi     = prev + np.int32(1 + skip)
        lo_idx = np.searchsorted(cands, lo)
        hi_idx = np.searchsorted(cands, hi + np.int32(1))
        counts = (hi_idx - lo_idx).astype(np.int32)
        valid  = counts > 0
        if not np.any(valid):
            return None

        vi    = np.where(valid)[0]
        reps  = counts[vi]
        total = int(reps.sum())

        # Expand all existing chain arrays
        new_group = [np.repeat(c[vi], reps) for c in group]

        # Vectorised arange: inner offsets [0,1,...,c₀-1, 0,1,...,c₁-1, …]
        cumreps  = np.concatenate([[0], np.cumsum(reps[:-1])])
        inner    = np.arange(total, dtype=np.int32) - np.repeat(cumreps, reps)
        base_idx = np.repeat(lo_idx[vi], reps)
        new_group.append(cands[base_idx + inner])
        return new_group


def _cascade_vectorized(
    elements: List[Tuple[dict, int, str]],
    cand_arrays: List[np.ndarray],
    max_results: Optional[int],
) -> List[tuple]:
    """
    Fully vectorised multi-element cascade.

    Supports:
    * default ``OP``  — exactly one match
    * ``OP='?'``      — zero or one match (splits into two groups)
    * ``OP='!'``      — token must NOT match; advances without consuming
    * ``SKIP``        — allow up to N tokens gap

    Returns a list of position-tuples ``(p0, p1, …, pK)``.
    """
    if not elements:
        return []

    # Start: one group seeded with element-0 candidates
    # Each group is a list of parallel int32 arrays
    active: List[_Group] = [[cand_arrays[0].astype(np.int32)]]

    for pi in range(1, len(elements)):
        _, skip, op = elements[pi]
        cands = cand_arrays[pi]

        next_active: List[_Group] = []

        for group in active:
            if len(group[0]) == 0:
                continue

            if op == '?':
                # Branch A: skip element pi (keep group unchanged)
                next_active.append([c.copy() for c in group])
                # Branch B: match element pi
                extended = _extend_group(group, cands, skip)
                if extended is not None:
                    next_active.append(extended)

            elif op == '!':
                # Negative lookahead: next position must NOT be in cands
                # Advance to prev+1 only for positions where cands ∌ prev+1
                prev    = group[-1]
                lo      = prev + np.int32(1)
                lo_idx  = np.searchsorted(cands, lo)
                safe    = np.minimum(lo_idx, len(cands) - 1)
                no_hit  = (lo_idx >= len(cands)) | (cands[safe] != lo)
                if np.any(no_hit):
                    new_group = [c[no_hit] for c in group]
                    new_group.append(lo[no_hit])  # consume the non-matching token
                    next_active.append(new_group)

            else:
                # Regular match (with optional SKIP)
                extended = _extend_group(group, cands, skip)
                if extended is not None:
                    next_active.append(extended)

        active = next_active
        if not active:
            return []

    if not active:
        return []

    # Merge all groups into flat arrays
    merged = [
        np.concatenate([grp[i] for grp in active])
        for i in range(len(active[0]))
    ]

    if max_results:
        merged = [c[:max_results] for c in merged]

    if len(merged[0]) == 0:
        return []

    return list(zip(*[c.tolist() for c in merged]))


# ---------------------------------------------------------------------------
# CorpusIndex
# ---------------------------------------------------------------------------

class CorpusIndex:
    """
    Pre-built flat index over all Quranic tokens (~128 K from morphologhy.csv).

    Build once with :meth:`build` (≈1–2 s), then query in microseconds
    to milliseconds via :meth:`search`.
    """

    def __init__(self) -> None:
        self.N: int = 0
        self._soure: np.ndarray = np.empty(0, dtype=np.uint8)
        self._ayeh:  np.ndarray = np.empty(0, dtype=np.uint16)
        self._tok_i: np.ndarray = np.empty(0, dtype=np.uint8)
        self._col:   Dict[str, list] = {c: [] for c in _INV_COLS}
        self._inv:   Dict[str, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Build / save / load
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, save: bool = False, cache_path: str = _DEFAULT_CACHE,
              verbose: bool = True) -> 'CorpusIndex':
        """
        Build from *morphologhy.csv*.  Pass ``save=True`` to cache to disk.
        """
        if verbose:
            print('Building CorpusIndex from morphologhy.csv …')

        df = pd.read_csv(constant.MORPHOLOGY, dtype=str)
        df['soure'] = df['soure'].astype(int)
        df['ayeh']  = df['ayeh'].astype(int)

        N  = len(df)
        ix = cls()
        ix.N = N

        ix._soure = df['soure'].to_numpy(dtype=np.uint8)
        ix._ayeh  = df['ayeh'].to_numpy(dtype=np.uint16)

        tok_i = []
        for _, grp in df.groupby(['soure', 'ayeh'], sort=False):
            tok_i.extend(range(len(grp)))
        ix._tok_i = np.array(tok_i, dtype=np.uint8)

        texts   = [_to_str(v) for v in df['word'].tolist()]
        simples = [utils.strip_diacritics(t) for t in texts]
        lemmas  = [_to_str(v) for v in df['Lemma'].tolist()]
        roots   = [_to_str(v) for v in df['Root'].tolist()]
        tags    = [_to_str(v) for v in df['Tag'].tolist()]

        # Store as numpy object arrays for fast vectorised fancy-indexing
        ix._col = {
            'text':   np.array(texts,   dtype=object),
            'simple': np.array(simples, dtype=object),
            'lemma':  np.array(lemmas,  dtype=object),
            'root':   np.array(roots,   dtype=object),
            'tag':    np.array(tags,    dtype=object),
        }

        if verbose:
            print('  Building inverted indexes …')

        for col in _INV_COLS:
            bucket: Dict[str, list] = {}
            for pos, val in enumerate(ix._col[col]):
                if val:
                    bucket.setdefault(val, []).append(pos)
            ix._inv[col] = {
                v: np.array(positions, dtype=np.int32)
                for v, positions in bucket.items()
            }

        if verbose:
            counts = {col: len(ix._inv[col]) for col in _INV_COLS}
            print(f'  Done. {N:,} tokens, index sizes: {counts}')

        if save:
            ix.save(cache_path)
            if verbose:
                print(f'  Cached → {cache_path}')

        return ix

    @classmethod
    def load(cls, cache_path: str = _DEFAULT_CACHE) -> 'CorpusIndex':
        """Load from cache; builds if cache missing."""
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return cls.build(save=True, cache_path=cache_path)

    def save(self, cache_path: str = _DEFAULT_CACHE) -> None:
        """Pickle this index to *cache_path*."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ------------------------------------------------------------------
    # Token access
    # ------------------------------------------------------------------

    def token(self, pos: int) -> CorpusToken:
        """Return the :class:`CorpusToken` at flat position *pos*."""
        return CorpusToken(
            pos    = pos,
            soure  = int(self._soure[pos]),
            ayeh   = int(self._ayeh[pos]),
            tok_i  = int(self._tok_i[pos]),
            text   = self._col['text'][pos],
            simple = self._col['simple'][pos],
            lemma  = self._col['lemma'][pos],
            root   = self._col['root'][pos],
            tag    = self._col['tag'][pos],
        )

    def _batch_tokens(self, positions: np.ndarray) -> List[CorpusToken]:
        """Vectorised bulk token construction from a sorted int32 position array."""
        pos_list  = positions.tolist()
        soures    = self._soure[positions].tolist()
        ayehs     = self._ayeh[positions].tolist()
        tok_is    = self._tok_i[positions].tolist()
        texts     = self._col['text'][positions].tolist()
        simples   = self._col['simple'][positions].tolist()
        lemmas    = self._col['lemma'][positions].tolist()
        roots     = self._col['root'][positions].tolist()
        tags      = self._col['tag'][positions].tolist()
        return [
            CorpusToken(p, s, a, ti, tx, si, le, ro, tg)
            for p, s, a, ti, tx, si, le, ro, tg
            in zip(pos_list, soures, ayehs, tok_is,
                   texts, simples, lemmas, roots, tags)
        ]

    def tokens_for_verse(self, soure: int, ayeh: int) -> List[CorpusToken]:
        """All tokens for a given verse."""
        mask = (self._soure == soure) & (self._ayeh == ayeh)
        return self._batch_tokens(np.where(mask)[0].astype(np.int32))

    # ------------------------------------------------------------------
    # Inverted-index lookup
    # ------------------------------------------------------------------

    def _positions_for(self, cond: dict) -> np.ndarray:
        base = {k: v for k, v in cond.items() if k.upper() not in _STRUCTURAL}
        return _positions_for_cond(self._inv, self._col, base, self.N)

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------

    def search(
        self,
        pattern: List[dict],
        key: str = 'match',
        max_results: Optional[int] = None,
    ) -> List[CorpusMatch]:
        """
        Search the entire Quran for *pattern*.

        Parameters
        ----------
        pattern:
            List of condition dicts.  Supported keys: ``TEXT``, ``LOWER``,
            ``LEMMA``, ``ROOT``, ``TAG`` / ``POS``, ``OP``
            (``"?"``, ``"!"``, ``"+"``), ``SKIP`` (int).
        key:
            Label assigned to every returned :class:`CorpusMatch`.
        max_results:
            Stop after this many matches (``None`` = all).

        Returns
        -------
        list of :class:`CorpusMatch`
        """
        if not pattern:
            return []

        # Parse elements
        elements: List[Tuple[dict, int, str]] = []
        for cond in pattern:
            skip = int(cond.get('SKIP', 0))
            op   = (cond.get('OP', '') or '').strip()
            base = {k: v for k, v in cond.items() if k.upper() not in _STRUCTURAL}
            elements.append((base, skip, op))

        # Build per-element candidate arrays
        cand_arrays: List[np.ndarray] = [
            _positions_for_cond(self._inv, self._col, base, self.N)
            for base, _, _ in elements
        ]

        if len(cand_arrays[0]) == 0:
            return []

        # Single-element shortcut
        if len(elements) == 1:
            positions = cand_arrays[0]
            if max_results:
                positions = positions[:max_results]
            tok_list = self._batch_tokens(positions)
            return [CorpusMatch(key, [t]) for t in tok_list]

        # Multi-element vectorised cascade
        chains = _cascade_vectorized(elements, cand_arrays, max_results)
        if not chains:
            return []

        # Build all tokens in one vectorised batch per element column
        n_elem  = len(chains[0])
        n_match = len(chains)

        # Interleaved flat array: [p0_chain0, p1_chain0, p0_chain1, p1_chain1, …]
        all_pos  = np.array(chains, dtype=np.int32).ravel()   # (n_match * n_elem,)
        all_toks = self._batch_tokens(all_pos)

        return [
            CorpusMatch(key, all_toks[i * n_elem:(i + 1) * n_elem])
            for i in range(n_match)
        ]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def find_root(
        self,
        root: str,
        tag: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[CorpusMatch]:
        """All occurrences of *root* (optionally filtered by *tag*)."""
        cond: dict = {'ROOT': root}
        if tag:
            cond['TAG'] = tag
        return self.search([cond], key=f'ROOT:{root}', max_results=max_results)

    def find_lemma(
        self,
        lemma: str,
        tag: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[CorpusMatch]:
        """All occurrences of *lemma*."""
        cond: dict = {'LEMMA': lemma}
        if tag:
            cond['TAG'] = tag
        return self.search([cond], key=f'LEMMA:{lemma}', max_results=max_results)

    def find_root_near_root(
        self,
        root1: str,
        root2: str,
        max_dist: int = 5,
        max_results: Optional[int] = None,
    ) -> List[CorpusMatch]:
        """
        Find *root1* within *max_dist* tokens of *root2*
        (cross-verse allowed).
        """
        return self.search(
            [{'ROOT': root1}, {'ROOT': root2, 'SKIP': max_dist}],
            key=f'ROOT:{root1}+ROOT:{root2}',
            max_results=max_results,
        )

    def __repr__(self) -> str:
        return f'CorpusIndex(N={self.N:,})'


# ---------------------------------------------------------------------------
# Module-level shared index
# ---------------------------------------------------------------------------

_shared_index: Optional[CorpusIndex] = None


def get_index(cache_path: str = _DEFAULT_CACHE) -> CorpusIndex:
    """
    Return the module-level :class:`CorpusIndex`, building on first call.
    Cached in memory across calls within the same process.
    """
    global _shared_index
    if _shared_index is None:
        _shared_index = CorpusIndex.load(cache_path)
    return _shared_index
