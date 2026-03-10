"""
quranic_nlp.query
~~~~~~~~~~~~~~~~~
spaCy-style token-pattern matching across Quranic verses.

Pattern syntax (per token element dict)
----------------------------------------
Attribute keys  (case-insensitive):
    TEXT   — exact surface form (with diacritics)
    LOWER  — lowercase surface form
    LEMMA  — canonical lemma
    POS    — Universal POS tag  (e.g. ``'NOUN'``, ``'VERB'``, ``'ADJ'``)
    DEP    — dependency relation label
    ROOT   — trilateral Arabic root  (e.g. ``'رحم'``, ``'علم'``)
    ARC    — dependency arc direction  (``'LTR'`` or ``'RTL'``)

Attribute values:
    str             — exact match
    list / tuple    — any-of  (token value must be in the list)
    dict            — ``{"IN": [...]}`` / ``{"NOT_IN": [...]}`` /
                      ``{"REGEX": "pattern"}``

Quantifier key (``OP``):
    omitted / ``""``  — exactly one  (default)
    ``"?"``           — zero or one
    ``"*"``           — zero or more
    ``"+"``           — one or more
    ``"!"``           — must NOT match; advances the pattern without
                        consuming a token

Gap key (``SKIP``)  — int, default 0:
    Allow up to this many arbitrary tokens to be skipped between the
    previous matched element and this one.  Enables proximity matching.

Examples
--------
::

    from quranic_nlp import language, query

    nlp = language.Pipeline('pos,root,lem,dep')

    matcher = query.VerseMatcher(nlp)

    # Verses containing a NOUN with root رحم
    matcher.add('MERCY_NOUN', [[{'ROOT': 'رحم', 'POS': 'NOUN'}]])

    # Verses where رحم root appears within 5 tokens of lemma الله
    matcher.add('MERCY_NEAR_ALLAH', [[
        {'ROOT': 'رحم'},
        {'LEMMA': 'الله', 'SKIP': 5},
    ]])

    # VERB followed within 3 tokens by a NOUN
    matcher.add('VERB_THEN_NOUN', [[
        {'POS': 'VERB'},
        {'POS': 'NOUN', 'SKIP': 3},
    ]])

    # Root رحم OR root علم in same verse
    matcher.add('ROOT_EITHER', [
        [{'ROOT': 'رحم'}],
        [{'ROOT': 'علم'}],
    ])

    # Search a single surah
    for doc, matches in matcher.search(surah=1):
        for key, start, end in matches:
            print(key, doc._.ayah, doc[start:end])

    # Search pre-computed docs  (faster: pipeline already ran)
    docs = language.surah_docs(nlp, 'بقره')
    for doc, matches in matcher.search(docs=docs):
        for key, start, end in matches:
            print(key, doc._.surah, doc._.ayah, doc[start:end])

    # --- Convenience functions ---
    # All verses with root رحم as a NOUN
    results = query.find_by_root(nlp, 'رحم', pos='NOUN', surah=1)

    # All verses containing lemma الله
    results = query.find_by_lemma(nlp, 'الله', surah=2)

    # رحم within 5 tokens of الله  (in either direction)
    results = query.find_near(nlp, {'ROOT': 'رحم'}, {'LEMMA': 'الله'}, max_dist=5)

    # Verses that contain ALL of: a VERB, a NOUN with root رحم, and lemma الله
    results = query.find_verses(nlp,
        [{'POS': 'VERB'}, {'ROOT': 'رحم', 'POS': 'NOUN'}, {'LEMMA': 'الله'}],
        mode='AND')
"""

import re as _re
from itertools import islice

from quranic_nlp import utils


# ---------------------------------------------------------------------------
# Token-level attribute access
# ---------------------------------------------------------------------------

_GETTERS = {
    'TEXT':  lambda t: t.text,
    'LOWER': lambda t: t.text.lower(),
    'LEMMA': lambda t: t.lemma_,
    'POS':   lambda t: t.pos_,
    'DEP':   lambda t: t.dep_,
    'ROOT':  lambda t: str(t._.root) if t._.root else '',
    'ARC':   lambda t: str(t._.dep_arc) if t._.dep_arc else '',
}

_STRUCTURAL = {'OP', 'SKIP'}


def _attr_val(token, attr: str):
    """Return the attribute value for *token*, or ``None`` if unknown."""
    return _GETTERS.get(attr.upper(), lambda _: None)(token)


def _token_matches(token, cond: dict) -> bool:
    """Return ``True`` if *token* satisfies every attribute constraint in *cond*."""
    for key, value in cond.items():
        k = key.upper()
        if k in _STRUCTURAL:
            continue

        # Negation shorthand: NOT_TEXT, NOT_POS, NOT_ROOT …
        negate = k.startswith('NOT_')
        attr = k[4:] if negate else k

        tok_val = _attr_val(token, attr)
        if tok_val is None:
            # Unknown attribute — skip silently
            continue

        # Resolve value constraint
        if isinstance(value, dict):
            match = True
            if 'IN' in value and tok_val not in value['IN']:
                match = False
            if 'NOT_IN' in value and tok_val in value['NOT_IN']:
                match = False
            if 'REGEX' in value and not _re.search(value['REGEX'], tok_val):
                match = False
        elif isinstance(value, (list, tuple)):
            match = tok_val in value
        else:
            match = (tok_val == value)

        if negate:
            match = not match
        if not match:
            return False

    return True


# ---------------------------------------------------------------------------
# Sequential pattern matching over a token list
# ---------------------------------------------------------------------------

def _find_matches(tokens, pattern: list) -> list:
    """
    Find all non-overlapping (start, end) spans where *pattern* matches
    in *tokens* (end is exclusive).

    Supports:
    - ``OP``: ``None``/``""`` (exactly one), ``"?"``, ``"*"``, ``"+"``,
      ``"!"`` (negative lookahead — matches without consuming).
    - ``SKIP``: gap of up to N arbitrary tokens before this element.
    """
    n = len(tokens)

    def _match(ti: int, pi: int):
        """Yield end indices matching pattern[pi:] starting at tokens[ti]."""
        if pi == len(pattern):
            yield ti
            return

        cond = pattern[pi]
        op   = cond.get('OP', '') or ''
        skip = int(cond.get('SKIP', 0))

        # --- SKIP: try gaps 0 .. skip before matching this element ----------
        if skip > 0:
            base = {k: v for k, v in cond.items() if k != 'SKIP'}
            max_gap = min(skip, n - ti)
            for gap in range(max_gap + 1):
                j = ti + gap
                if j >= n:
                    break
                if _token_matches(tokens[j], base):
                    yield from _match(j + 1, pi + 1)
            return

        # --- OP: negation ----------------------------------------------------
        if op == '!':
            if ti >= n or not _token_matches(tokens[ti], cond):
                yield from _match(ti, pi + 1)
            return

        # --- OP: exactly one (default) --------------------------------------
        if op in ('', None):
            if ti < n and _token_matches(tokens[ti], cond):
                yield from _match(ti + 1, pi + 1)
            return

        # --- OP: zero or one ------------------------------------------------
        if op == '?':
            yield from _match(ti, pi + 1)          # skip
            if ti < n and _token_matches(tokens[ti], cond):
                yield from _match(ti + 1, pi + 1)  # consume
            return

        # --- OP: zero or more -----------------------------------------------
        if op == '*':
            yield from _match(ti, pi + 1)           # zero
            i = ti
            while i < n and _token_matches(tokens[i], cond):
                yield from _match(i + 1, pi + 1)
                i += 1
            return

        # --- OP: one or more ------------------------------------------------
        if op == '+':
            i = ti
            while i < n and _token_matches(tokens[i], cond):
                yield from _match(i + 1, pi + 1)
                i += 1
            return

    results = []
    seen = set()
    for start in range(n):
        for end in _match(start, 0):
            if end > start and (start, end) not in seen:
                results.append((start, end))
                seen.add((start, end))
                break  # non-overlapping: take first end for this start
    return results


# ---------------------------------------------------------------------------
# VerseMatcher
# ---------------------------------------------------------------------------

class VerseMatcher:
    """
    Match spaCy-style token patterns across Quranic verses.

    Parameters
    ----------
    nlp : QuranicNLP
        A pipeline created with ``language.Pipeline(...)`` or
        ``language.load_pipeline(...)``.  Must include the components
        required by the patterns (e.g. ``'pos'``, ``'root'``, ``'lem'``).

    Example
    -------
    ::

        from quranic_nlp import language, query

        nlp     = language.Pipeline('pos,root,lem,dep')
        matcher = query.VerseMatcher(nlp)

        matcher.add('MERCY_NOUN', [[{'ROOT': 'رحم', 'POS': 'NOUN'}]])
        matcher.add('NEAR', [[
            {'ROOT': 'رحم'},
            {'LEMMA': 'الله', 'SKIP': 5},
        ]])

        for doc, matches in matcher.search(surah=1):
            for key, start, end in matches:
                print(key, doc._.ayah, doc[start:end])
    """

    def __init__(self, nlp):
        self._nlp = nlp
        self._patterns: dict[str, list[list[dict]]] = {}

    # ------------------------------------------------------------------
    def add(self, key: str, patterns: list):
        """
        Register one or more patterns under *key*.

        Parameters
        ----------
        key : str
            Name for this pattern group (returned in match results).
        patterns : list[list[dict]]
            List of alternative patterns.  Each alternative is a list of
            token-condition dicts.  A doc matches *key* if **any**
            alternative matches.

        Example
        -------
        ::

            # Two alternatives for the same key
            matcher.add('GOD_NAMES', [
                [{'LEMMA': 'الله'}],
                [{'LEMMA': 'الرحمن'}],
            ])
        """
        self._patterns[key] = patterns

    def remove(self, key: str):
        """Remove the pattern registered under *key*."""
        self._patterns.pop(key, None)

    def __contains__(self, key: str) -> bool:
        return key in self._patterns

    # ------------------------------------------------------------------
    def __call__(self, doc) -> list:
        """
        Apply all registered patterns to a single *doc*.

        Returns
        -------
        list[tuple[str, int, int]]
            ``[(key, start, end), ...]`` — all matches, in token-index order.
        """
        tokens = list(doc)
        results = []
        for key, alternatives in self._patterns.items():
            for pattern in alternatives:
                for start, end in _find_matches(tokens, pattern):
                    results.append((key, start, end))
        results.sort(key=lambda x: x[1])
        return results

    # ------------------------------------------------------------------
    def search(self, surah=None, docs=None, max_results: int = None):
        """
        Search for pattern matches across Quranic verses.

        Parameters
        ----------
        surah : int or str, optional
            Restrict search to this surah (index or Arabic name).
            If omitted and *docs* is not provided, all 6,236 verses
            are searched (slow — consider providing *docs* instead).
        docs : iterable of Doc, optional
            Pre-computed verse docs to search.  Fastest option when
            the pipeline has already been run (e.g. via
            ``language.surah_docs``).
        max_results : int, optional
            Stop after returning this many matching docs.

        Yields
        ------
        tuple[Doc, list[tuple[str, int, int]]]
            ``(doc, [(key, start, end), ...])`` for every verse that
            has at least one match.

        Example
        -------
        ::

            for doc, matches in matcher.search(surah=2, max_results=10):
                for key, start, end in matches:
                    print(key, doc._.ayah, doc[start:end])
        """
        verse_iter = self._resolve_iter(surah, docs)
        if max_results is not None:
            verse_iter = islice(verse_iter, max_results * 10)  # over-fetch, filter below

        found = 0
        for doc in verse_iter:
            matches = self(doc)
            if matches:
                yield doc, matches
                found += 1
                if max_results is not None and found >= max_results:
                    return

    # ------------------------------------------------------------------
    def _resolve_iter(self, surah, docs):
        """Return an iterator over processed verse docs."""
        if docs is not None:
            return iter(docs)
        if surah is not None:
            from quranic_nlp import language
            return iter(language.surah_docs(self._nlp, surah))
        return self._all_verses()

    def _all_verses(self):
        """Lazily yield all 6,236 verse docs through the pipeline."""
        index = utils._semantic_index()
        for soure_idx, files in enumerate(index):
            soure = soure_idx + 1
            for ayeh in range(1, len(files) + 1):
                yield self._nlp(f'{soure}#{ayeh}')


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def find_by_root(nlp, root: str, surah=None, docs=None,
                 pos: str = None, dep: str = None,
                 max_results: int = None) -> list:
    """
    Find verses containing at least one token with the given Arabic *root*.

    Parameters
    ----------
    nlp : QuranicNLP
        Pipeline (must include ``'root'``).
    root : str
        Trilateral or quadrilateral Arabic root, e.g. ``'رحم'``.
    surah : int or str, optional
        Restrict to this surah.
    docs : iterable of Doc, optional
        Pre-computed docs to search.
    pos : str, optional
        Also require the matching token to have this POS tag.
    dep : str, optional
        Also require this dependency relation.
    max_results : int, optional
        Maximum number of docs to return.

    Returns
    -------
    list[Doc]

    Example
    -------
    ::

        docs = query.find_by_root(nlp, 'رحم', pos='NOUN', surah=1)
    """
    cond = {'ROOT': root}
    if pos:
        cond['POS'] = pos
    if dep:
        cond['DEP'] = dep
    matcher = VerseMatcher(nlp)
    matcher.add('_', [[cond]])
    return [doc for doc, _ in matcher.search(surah=surah, docs=docs,
                                              max_results=max_results)]


def find_by_lemma(nlp, lemma: str, surah=None, docs=None,
                  pos: str = None, dep: str = None,
                  max_results: int = None) -> list:
    """
    Find verses containing at least one token with the given *lemma*.

    Parameters
    ----------
    nlp : QuranicNLP
        Pipeline (must include ``'lem'``).
    lemma : str
        Canonical lemma form.
    surah : int or str, optional
    docs : iterable of Doc, optional
    pos : str, optional
        Also require this POS tag on the matching token.
    dep : str, optional
        Also require this dependency relation.
    max_results : int, optional

    Returns
    -------
    list[Doc]
    """
    cond = {'LEMMA': lemma}
    if pos:
        cond['POS'] = pos
    if dep:
        cond['DEP'] = dep
    matcher = VerseMatcher(nlp)
    matcher.add('_', [[cond]])
    return [doc for doc, _ in matcher.search(surah=surah, docs=docs,
                                              max_results=max_results)]


def find_by_pos(nlp, pos: str, surah=None, docs=None,
                max_results: int = None) -> list:
    """
    Find verses containing at least one token with the given POS tag.

    Parameters
    ----------
    nlp : QuranicNLP
        Pipeline (must include ``'pos'``).
    pos : str
        Universal POS tag, e.g. ``'VERB'``, ``'NOUN'``, ``'ADJ'``.
    surah : int or str, optional
    docs : iterable of Doc, optional
    max_results : int, optional

    Returns
    -------
    list[Doc]
    """
    matcher = VerseMatcher(nlp)
    matcher.add('_', [[{'POS': pos}]])
    return [doc for doc, _ in matcher.search(surah=surah, docs=docs,
                                              max_results=max_results)]


def find_near(nlp, cond1: dict, cond2: dict, max_dist: int = 5,
              surah=None, docs=None, directed: bool = False,
              max_results: int = None) -> list:
    """
    Find verses where two token patterns occur within *max_dist* tokens.

    Parameters
    ----------
    nlp : QuranicNLP
        Pipeline with components required by *cond1* / *cond2*.
    cond1 : dict
        Token condition for the first token.
    cond2 : dict
        Token condition for the second token.
    max_dist : int
        Maximum number of tokens between the two matches (inclusive).
    surah : int or str, optional
    docs : iterable of Doc, optional
    directed : bool
        If ``True``, require cond1 to appear *before* cond2.
        If ``False`` (default), accept either order.
    max_results : int, optional

    Returns
    -------
    list[tuple[Doc, int, int, int, int]]
        Each entry is ``(doc, start1, end1, start2, end2)`` where
        *start/end* are token indices of each match.

    Example
    -------
    ::

        # رحم root within 5 tokens of الله lemma, either order
        results = query.find_near(nlp,
            {'ROOT': 'رحم'}, {'LEMMA': 'الله'}, max_dist=5)
        for doc, s1, e1, s2, e2 in results:
            print(doc._.ayah, doc[s1:e1], '...', doc[s2:e2])
    """
    matcher = VerseMatcher(nlp)
    # Forward: cond1 then cond2 with gap
    fwd_pattern = [cond1, dict(cond2, SKIP=max_dist)]
    matcher.add('_fwd', [fwd_pattern])
    if not directed:
        bwd_pattern = [cond2, dict(cond1, SKIP=max_dist)]
        matcher.add('_bwd', [bwd_pattern])

    results = []
    for doc, matches in matcher.search(surah=surah, docs=docs):
        tokens = list(doc)
        for key, start, end in matches:
            # Identify positions of the two conditions within the span
            span_tokens = tokens[start:end]
            if key == '_fwd':
                c_a, c_b = cond1, cond2
            else:
                c_a, c_b = cond2, cond1
            pos_a = next((i for i, t in enumerate(span_tokens) if _token_matches(t, c_a)), None)
            pos_b = next((i for i, t in enumerate(reversed(span_tokens)) if _token_matches(t, c_b)), None)
            if pos_a is not None and pos_b is not None:
                abs_a = start + pos_a
                abs_b = end - 1 - pos_b
                results.append((doc, abs_a, abs_a + 1, abs_b, abs_b + 1))
        if max_results is not None and len(results) >= max_results:
            break

    return results[:max_results] if max_results else results


def find_verses(nlp, conditions: list, mode: str = 'AND',
                surah=None, docs=None, max_results: int = None) -> list:
    """
    Find verses satisfying a set of token conditions.

    Parameters
    ----------
    nlp : QuranicNLP
        Pipeline with required components.
    conditions : list[dict]
        Each dict is a token-attribute condition (same syntax as a
        single pattern element without ``OP``/``SKIP``).
    mode : ``'AND'`` or ``'OR'``
        * ``'AND'`` — every condition must be satisfied by at least one
          token in the verse  (default).
        * ``'OR'``  — at least one condition must be satisfied.
    surah : int or str, optional
    docs : iterable of Doc, optional
    max_results : int, optional

    Returns
    -------
    list[Doc]

    Example
    -------
    ::

        # Verses with BOTH رحم root and علم root
        docs = query.find_verses(nlp, [
            {'ROOT': 'رحم'},
            {'ROOT': 'علم'},
        ], mode='AND')

        # Verses with رحم OR غفر root
        docs = query.find_verses(nlp, [
            {'ROOT': 'رحم'},
            {'ROOT': 'غفر'},
        ], mode='OR')
    """
    if mode.upper() not in ('AND', 'OR'):
        raise ValueError(f"mode must be 'AND' or 'OR', got {mode!r}")

    matcher = VerseMatcher(nlp)
    for i, cond in enumerate(conditions):
        matcher.add(f'_c{i}', [[cond]])

    results = []
    for doc in matcher._resolve_iter(surah, docs):
        tokens = list(doc)
        if mode.upper() == 'AND':
            hit = all(
                any(_token_matches(t, cond) for t in tokens)
                for cond in conditions
            )
        else:
            hit = any(
                any(_token_matches(t, cond) for t in tokens)
                for cond in conditions
            )
        if hit:
            results.append(doc)
            if max_results is not None and len(results) >= max_results:
                break

    return results


def concordance(nlp, cond: dict, context: int = 2,
                surah=None, docs=None, max_results: int = None) -> list:
    """
    Return a KWIC (keyword-in-context) concordance for a token condition.

    Parameters
    ----------
    nlp : QuranicNLP
    cond : dict
        Token condition to match (same syntax as a single pattern element).
    context : int
        Number of tokens to show on each side of the match.
    surah : int or str, optional
    docs : iterable of Doc, optional
    max_results : int, optional

    Returns
    -------
    list[dict]
        Each entry has keys:
        ``surah``, ``ayah``, ``left``, ``match``, ``right``, ``doc``.

    Example
    -------
    ::

        rows = query.concordance(nlp, {'ROOT': 'رحم'}, context=3, surah=1)
        for row in rows:
            left  = ' '.join(t.text for t in row['left'])
            match = row['match'].text
            right = ' '.join(t.text for t in row['right'])
            print(f"{row['surah']}:{row['ayah']}  {left} [{match}] {right}")
    """
    matcher = VerseMatcher(nlp)
    matcher.add('_kw', [[cond]])

    results = []
    for doc, matches in matcher.search(surah=surah, docs=docs):
        tokens = list(doc)
        for _, start, end in matches:
            for pos in range(start, end):
                left  = tokens[max(0, pos - context): pos]
                right = tokens[pos + 1: pos + 1 + context]
                results.append({
                    'surah': doc._.surah,
                    'ayah':  doc._.ayah,
                    'left':  left,
                    'match': tokens[pos],
                    'right': right,
                    'doc':   doc,
                })
                if max_results is not None and len(results) >= max_results:
                    return results

    return results
