"""
quranic_nlp.corpus
~~~~~~~~~~~~~~~~~~
Cross-verse continuous-sequence pattern matcher for the entire Quran.

The full corpus (~128 K tokens from *morphologhy.csv*) is loaded into
flat numpy arrays and inverted indexes, enabling O(log N) per-step
lookup during pattern search.

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

    # Build once (~1–2 s) and optionally cache to disk
    idx = CorpusIndex.build(save=True)

    # Load cached index on subsequent calls
    idx = CorpusIndex.load()

    # ── Single condition ────────────────────────────────────────────────
    matches = idx.search([{'ROOT': 'رحم'}])
    for m in matches[:5]:
        print(m)

    # ── Consecutive pair ────────────────────────────────────────────────
    matches = idx.search([{'ROOT': 'رحم'}, {'ROOT': 'علم'}])

    # ── Proximity / SKIP (spans verse boundaries) ───────────────────────
    matches = idx.search([
        {'ROOT': 'رحم'},
        {'ROOT': 'علم', 'SKIP': 5},   # within 5 tokens
    ])

    # ── POS + ROOT combination ──────────────────────────────────────────
    matches = idx.search([
        {'TAG': 'N', 'ROOT': 'صبر'},
        {'TAG': 'V', 'SKIP': 3},
    ])

    # ── Optional element (OP='?') ────────────────────────────────────────
    matches = idx.search([
        {'ROOT': 'علم'},
        {'TAG': 'DET', 'OP': '?'},
        {'TAG': 'N'},
    ])

    # ── Any-of values ────────────────────────────────────────────────────
    matches = idx.search([{'ROOT': {'IN': ['رحم', 'علم']}}])

    # Limit results
    matches = idx.search([{'ROOT': 'رحم'}], max_results=20)

    # Display helpers
    print(matches[0].refs)      # [(soure, ayeh), ...]
    print(matches[0].text)      # Arabic surface form
    for t in matches[0].tokens:
        print(t.soure, t.ayeh, t.tok_i, t.text, t.root, t.tag)
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

# Columns to include in the inverted index (subset of _ATTR_COL values)
_INV_COLS = ('text', 'simple', 'lemma', 'root', 'tag')


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class CorpusToken:
    """One token in the flat Quranic corpus."""

    pos:    int    #: flat 0-based index (0 … N-1)
    soure:  int    #: surah number (1–114)
    ayeh:   int    #: verse number (1-based)
    tok_i:  int    #: position within the verse (0-based)
    text:   str    #: surface form with diacritics
    simple: str    #: surface form *without* diacritics
    lemma:  str    #: canonical lemma (empty if unknown)
    root:   str    #: trilateral root  (empty if unknown)
    tag:    str    #: POS tag (Quranic Treebank notation)

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
        """Flat position of the first token."""
        return self.tokens[0].pos

    @property
    def end(self) -> int:
        """Flat position *after* the last token (exclusive)."""
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
        """Surface text of all matched tokens joined by spaces."""
        return ' '.join(t.text for t in self.tokens)

    def __repr__(self) -> str:
        refs_str = ', '.join(f'{s}:{a}' for s, a in self.refs)
        return (
            f'CorpusMatch(key={self.pattern_key!r},'
            f' refs=[{refs_str}], text={self.text!r})'
        )


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _to_str(val) -> str:
    """Convert a value (possibly NaN/None/float) to a str, or '' if missing."""
    if val is None:
        return ''
    if isinstance(val, float) and (val != val):   # NaN check
        return ''
    return str(val)


def _cond_matches_arrays(
    col_arrays: Dict[str, list],
    idx: int,
    cond: dict,
) -> bool:
    """Return ``True`` if the token at flat *idx* satisfies *cond*."""
    for key, value in cond.items():
        k = key.upper()
        if k in _STRUCTURAL:
            continue
        negate = k.startswith('NOT_')
        attr = k[4:] if negate else k
        col = _ATTR_COL.get(attr)
        if col is None:
            continue
        tok_val = col_arrays[col][idx]

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
    """
    Return a *sorted* ``int32`` array of corpus positions satisfying *cond*.

    Uses inverted indexes for indexed attributes and falls back to linear
    scan for negations, ``NOT_IN``, and regex conditions.
    """
    indexed_sets: List[np.ndarray] = []
    post_filter_cond: dict = {}

    for key, value in cond.items():
        k = key.upper()
        if k in _STRUCTURAL:
            continue
        negate = k.startswith('NOT_')
        attr = k[4:] if negate else k
        col = _ATTR_COL.get(attr)

        if col is None or col not in inv:
            post_filter_cond[key] = value
            continue

        # Negation and NOT_IN require post-filter
        if negate or (isinstance(value, dict) and ('NOT_IN' in value)):
            post_filter_cond[key] = value
            continue

        if isinstance(value, dict):
            if 'IN' in value:
                parts = [
                    inv[col].get(_to_str(v), _EMPTY)
                    for v in value['IN']
                ]
                merged = np.unique(np.concatenate(parts)) if parts else _EMPTY
                indexed_sets.append(merged)
            elif 'REGEX' in value:
                pat = _re.compile(value['REGEX'])
                parts = [arr for v, arr in inv[col].items() if pat.search(v)]
                merged = np.unique(np.concatenate(parts)) if parts else _EMPTY
                indexed_sets.append(merged)
            else:
                post_filter_cond[key] = value
        elif isinstance(value, (list, tuple)):
            parts = [
                inv[col].get(_to_str(v), _EMPTY)
                for v in value
            ]
            merged = np.unique(np.concatenate(parts)) if parts else _EMPTY
            indexed_sets.append(merged)
        else:
            arr = inv[col].get(_to_str(value), _EMPTY)
            indexed_sets.append(arr)

    # Intersect indexed sets
    if indexed_sets:
        result = indexed_sets[0]
        for other in indexed_sets[1:]:
            result = np.intersect1d(result, other, assume_unique=True)
    elif post_filter_cond:
        result = np.arange(N, dtype=np.int32)
    else:
        # Completely empty condition — match every position
        return np.arange(N, dtype=np.int32)

    # Post-filter for conditions not handled via inverted index
    if post_filter_cond and len(result) > 0:
        mask = np.fromiter(
            (_cond_matches_arrays(col_arrays, int(p), post_filter_cond)
             for p in result),
            dtype=bool,
            count=len(result),
        )
        result = result[mask]

    return result.astype(np.int32)


_EMPTY = np.array([], dtype=np.int32)


# ---------------------------------------------------------------------------
# CorpusIndex
# ---------------------------------------------------------------------------

class CorpusIndex:
    """
    Pre-built flat index over all Quranic tokens.

    Build once with :meth:`build` (≈1–2 s), then query repeatedly
    via :meth:`search`.

    Attributes
    ----------
    N : int
        Total number of tokens in the corpus.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self.N: int = 0
        # Positional metadata as numpy arrays (indices: 0 … N-1)
        self._soure: np.ndarray = np.empty(0, dtype=np.uint8)
        self._ayeh:  np.ndarray = np.empty(0, dtype=np.uint16)
        self._tok_i: np.ndarray = np.empty(0, dtype=np.uint8)
        # String columns as Python lists (faster str lookup than numpy)
        self._col: Dict[str, list] = {c: [] for c in _INV_COLS}
        # Inverted index: col → {value_str → sorted int32 positions}
        self._inv: Dict[str, Dict[str, np.ndarray]] = {}

    @classmethod
    def build(cls, save: bool = False, cache_path: str = _DEFAULT_CACHE,
              verbose: bool = True) -> 'CorpusIndex':
        """
        Build the corpus index from *morphologhy.csv*.

        Parameters
        ----------
        save:
            If ``True``, save the built index to *cache_path* so that
            subsequent calls to :meth:`load` are instant.
        cache_path:
            Where to write the cache file (default: ``data/_corpus_index.pkl``).
        verbose:
            Print progress messages.

        Returns
        -------
        CorpusIndex
        """
        if verbose:
            print('Building CorpusIndex from morphologhy.csv …')

        df = pd.read_csv(constant.MORPHOLOGY, dtype=str)
        # Ensure numeric columns are correct after str dtype
        df['soure'] = df['soure'].astype(int)
        df['ayeh']  = df['ayeh'].astype(int)

        N = len(df)
        idx = cls()
        idx.N = N

        # --- Positional arrays -------------------------------------------
        idx._soure = df['soure'].to_numpy(dtype=np.uint8)
        idx._ayeh  = df['ayeh'].to_numpy(dtype=np.uint16)

        # tok_i: 0-based position within each (soure, ayeh) group
        tok_i_list = []
        for _, grp in df.groupby(['soure', 'ayeh'], sort=False):
            tok_i_list.extend(range(len(grp)))
        idx._tok_i = np.array(tok_i_list, dtype=np.uint8)

        # --- String columns ----------------------------------------------
        texts   = [_to_str(v) for v in df['word'].tolist()]
        simples = [utils.strip_diacritics(t) for t in texts]
        lemmas  = [_to_str(v) for v in df['Lemma'].tolist()]
        roots   = [_to_str(v) for v in df['Root'].tolist()]
        tags    = [_to_str(v) for v in df['Tag'].tolist()]

        idx._col = {
            'text':   texts,
            'simple': simples,
            'lemma':  lemmas,
            'root':   roots,
            'tag':    tags,
        }

        # --- Inverted indexes --------------------------------------------
        if verbose:
            print('  Building inverted indexes …')

        for col in _INV_COLS:
            inv_col: Dict[str, list] = {}
            for pos, val in enumerate(idx._col[col]):
                if val:
                    inv_col.setdefault(val, []).append(pos)
            idx._inv[col] = {
                v: np.array(positions, dtype=np.int32)
                for v, positions in inv_col.items()
            }

        if verbose:
            print(f'  Done. {N:,} tokens indexed.')

        if save:
            idx.save(cache_path)
            if verbose:
                print(f'  Index cached to {cache_path}')

        return idx

    @classmethod
    def load(cls, cache_path: str = _DEFAULT_CACHE) -> 'CorpusIndex':
        """
        Load a previously saved index from *cache_path*.

        If the file does not exist, builds the index from scratch
        (equivalent to :meth:`build`).
        """
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

    def tokens_for_verse(self, soure: int, ayeh: int) -> List[CorpusToken]:
        """Return all :class:`CorpusToken` objects for a given verse."""
        mask = (self._soure == soure) & (self._ayeh == ayeh)
        return [self.token(int(p)) for p in np.where(mask)[0]]

    # ------------------------------------------------------------------
    # Inverted-index helpers
    # ------------------------------------------------------------------

    def _positions_for(self, cond: dict) -> np.ndarray:
        """Return sorted int32 positions satisfying *cond* (ignoring OP/SKIP)."""
        base = {k: v for k, v in cond.items() if k.upper() not in _STRUCTURAL}
        return _positions_for_cond(self._inv, self._col, base, self.N)

    # ------------------------------------------------------------------
    # Core cascade search
    # ------------------------------------------------------------------

    def _cascade(
        self,
        elements: List[Tuple[dict, int, str]],   # (cond, skip, op)
        cand_arrays: List[np.ndarray],
        max_results: Optional[int],
    ) -> List[List[int]]:
        """
        Cascade-search for multi-element patterns.

        Returns a list of position-chains ``[p0, p1, …, pK]``.
        """
        results: List[List[int]] = []

        def extend(chain: list, pi: int) -> None:
            if max_results and len(results) >= max_results:
                return
            if pi == len(elements):
                results.append(list(chain))
                return

            _, skip, op = elements[pi]
            last = chain[-1]
            cands = cand_arrays[pi]

            lo = last + 1
            hi = last + 1 + skip   # inclusive

            # Find the slice of cands that falls in [lo, hi]
            lo_idx = int(np.searchsorted(cands, lo))
            hi_idx = int(np.searchsorted(cands, hi + 1))

            if op == '!':
                # Negative lookahead: advance without consuming if cond NOT met
                if lo_idx >= hi_idx or cands[lo_idx] != lo:
                    extend(chain, pi + 1)
                return

            if op == '?':
                # Option A: skip this element entirely
                extend(chain, pi + 1)

            # Try all candidate positions in the window
            for ci in range(lo_idx, hi_idx):
                pos = int(cands[ci])
                chain.append(pos)
                extend(chain, pi + 1)
                chain.pop()
                if max_results and len(results) >= max_results:
                    return

        for start_p in cand_arrays[0]:
            extend([int(start_p)], 1)
            if max_results and len(results) >= max_results:
                break

        return results

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
            List of condition dicts, one per token to match.  Supports
            the same syntax as :class:`~quranic_nlp.query.VerseMatcher`:
            attribute keys (``TEXT``, ``LOWER``, ``LEMMA``, ``ROOT``,
            ``TAG`` / ``POS``), ``OP`` (``"?"``, ``"!"``, ``"+"``),
            ``SKIP`` (int).
        key:
            Label assigned to every :class:`CorpusMatch` returned.
        max_results:
            Stop after finding this many matches (``None`` = all).

        Returns
        -------
        list of :class:`CorpusMatch`
        """
        if not pattern:
            return []

        # --- Parse elements: (cond_without_structural, skip, op) --------
        elements: List[Tuple[dict, int, str]] = []
        for cond in pattern:
            skip = int(cond.get('SKIP', 0))
            op   = (cond.get('OP', '') or '').strip()
            base = {k: v for k, v in cond.items() if k.upper() not in _STRUCTURAL}
            elements.append((base, skip, op))

        # --- Build per-element candidate arrays from inverted index ------
        cand_arrays: List[np.ndarray] = []
        for base, skip, op in elements:
            if op == '?':
                # Optional: we still need candidates to try matching
                arr = _positions_for_cond(self._inv, self._col, base, self.N)
            elif op == '!':
                # Negative: candidates are positions that DO match (to exclude)
                arr = _positions_for_cond(self._inv, self._col, base, self.N)
            else:
                arr = _positions_for_cond(self._inv, self._col, base, self.N)
            cand_arrays.append(arr)

        if len(cand_arrays[0]) == 0:
            return []

        # --- Single-element shortcut ----------------------------------------
        if len(elements) == 1:
            _, skip, op = elements[0]
            positions = cand_arrays[0]
            if max_results:
                positions = positions[:max_results]
            return [
                CorpusMatch(key, [self.token(int(p))])
                for p in positions
            ]

        # --- Multi-element cascade ------------------------------------------
        chains = self._cascade(elements, cand_arrays, max_results)

        return [
            CorpusMatch(key, [self.token(p) for p in chain])
            for chain in chains
        ]

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def find_root(
        self,
        root: str,
        tag: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[CorpusMatch]:
        """Return all occurrences of *root* (optionally filtered by *tag*)."""
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
        """Return all occurrences of *lemma*."""
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
        Find all locations where *root1* appears within *max_dist*
        tokens of *root2* (cross-verse allowed).
        """
        return self.search(
            [{'ROOT': root1}, {'ROOT': root2, 'SKIP': max_dist}],
            key=f'ROOT:{root1}+ROOT:{root2}',
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f'CorpusIndex(N={self.N:,})'


# ---------------------------------------------------------------------------
# Module-level convenience: lazily build/load the shared index
# ---------------------------------------------------------------------------

_shared_index: Optional[CorpusIndex] = None


def get_index(cache_path: str = _DEFAULT_CACHE) -> CorpusIndex:
    """
    Return the module-level :class:`CorpusIndex`, building it on first call.

    The result is cached in memory across calls within the same process.
    """
    global _shared_index
    if _shared_index is None:
        _shared_index = CorpusIndex.load(cache_path)
    return _shared_index
