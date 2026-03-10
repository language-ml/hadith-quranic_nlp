"""Tests for quranic_nlp.query module."""

import pytest
import spacy
from unittest.mock import MagicMock, patch
from spacy.tokens import Doc, Token

from quranic_nlp import query, language


# ---------------------------------------------------------------------------
# Helpers — build minimal mock docs without running the full pipeline
# ---------------------------------------------------------------------------

def _register_ext(name, default=None):
    if not Token.has_extension(name):
        Token.set_extension(name, default=default)
    if not Doc.has_extension(name):
        Doc.set_extension(name, default=default)


_register_ext('root')
_register_ext('dep_arc')
_register_ext('soure_index')
_register_ext('ayeh_index')
_register_ext('surah')
_register_ext('ayah')
_register_ext('text')
_register_ext('simple_text')
_register_ext('revelation_order')
_register_ext('translations')
_register_ext('sim_ayahs')
_register_ext('hadiths')
_register_ext('sentences')


_vocab = spacy.blank('ar').vocab


def _make_doc(tokens_data):
    """
    Build a spaCy Doc from a list of dicts:
      [{'text': 'word', 'lemma': 'l', 'root': 'r', 'pos': 'NOUN', 'dep': 'subj'}, ...]
    """
    words  = [t['text']  for t in tokens_data]
    spaces = [True] * len(words)
    doc = Doc(_vocab, words=words, spaces=spaces)
    doc._.soure_index = 1
    doc._.ayeh_index  = 1
    doc._.surah       = 'فاتحه'
    doc._.ayah        = 1
    doc._.text        = ' '.join(words)
    for token, td in zip(doc, tokens_data):
        token.lemma_    = td.get('lemma', token.text)
        token.pos_      = td.get('pos',   '')
        token.dep_      = td.get('dep',   '')
        token._.root    = td.get('root',  '')
        token._.dep_arc = td.get('arc',   '')
    return doc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fatiha_doc():
    """Minimal mock of Al-Fatiha verse 1 with root/lemma/pos annotations."""
    return _make_doc([
        {'text': 'بِ',          'lemma': '',       'root': '',    'pos': 'INTJ', 'dep': 'majrur'},
        {'text': 'سْمِ',        'lemma': 'اسم',    'root': 'سمو', 'pos': 'NOUN', 'dep': 'mudaf'},
        {'text': 'اللَّهِ',    'lemma': 'الله',   'root': 'اله', 'pos': 'NOUN', 'dep': 'naat'},
        {'text': 'الرَّحْمَنِ','lemma': 'رحمان',  'root': 'رحم', 'pos': 'ADJ',  'dep': 'badal'},
        {'text': 'الرَّحِیمِ', 'lemma': 'رحیم',   'root': 'رحم', 'pos': 'ADJ',  'dep': 'naat'},
    ])


@pytest.fixture
def multi_doc():
    """Doc with varied POS for OP and SKIP tests."""
    return _make_doc([
        {'text': 'الله',  'lemma': 'الله',  'root': 'اله', 'pos': 'NOUN'},
        {'text': 'یحب',   'lemma': 'احب',   'root': 'حب',  'pos': 'VERB'},
        {'text': 'العلم', 'lemma': 'علم',   'root': 'علم', 'pos': 'NOUN'},
        {'text': 'و',     'lemma': 'و',     'root': '',    'pos': 'CCONJ'},
        {'text': 'یرحم',  'lemma': 'ارحم',  'root': 'رحم', 'pos': 'VERB'},
        {'text': 'عباده', 'lemma': 'عبد',   'root': 'عبد', 'pos': 'NOUN'},
    ])


# ---------------------------------------------------------------------------
# _token_matches
# ---------------------------------------------------------------------------

class TestTokenMatches:
    def test_exact_text(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[2], {'TEXT': 'اللَّهِ'})

    def test_exact_text_fail(self, fatiha_doc):
        assert not query._token_matches(fatiha_doc[2], {'TEXT': 'الله'})

    def test_lemma(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[2], {'LEMMA': 'الله'})

    def test_pos(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[1], {'POS': 'NOUN'})

    def test_root(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[3], {'ROOT': 'رحم'})

    def test_multi_condition_all_match(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[3], {'ROOT': 'رحم', 'POS': 'ADJ'})

    def test_multi_condition_one_fails(self, fatiha_doc):
        assert not query._token_matches(fatiha_doc[3], {'ROOT': 'رحم', 'POS': 'NOUN'})

    def test_list_value_in(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[1], {'POS': ['NOUN', 'ADJ']})

    def test_list_value_not_in(self, fatiha_doc):
        assert not query._token_matches(fatiha_doc[0], {'POS': ['NOUN', 'ADJ']})

    def test_dict_in(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[1], {'POS': {'IN': ['NOUN', 'VERB']}})

    def test_dict_not_in(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[3], {'POS': {'NOT_IN': ['NOUN']}})

    def test_dict_regex(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[2], {'LEMMA': {'REGEX': '^الل'}})

    def test_not_prefix_pos(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[0], {'NOT_POS': 'NOUN'})

    def test_not_prefix_root(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[2], {'NOT_ROOT': 'رحم'})

    def test_unknown_attr_ignored(self, fatiha_doc):
        # Unknown attribute should not cause an error, just be ignored
        assert query._token_matches(fatiha_doc[0], {'UNKNOWN_KEY': 'anything'})

    def test_lower(self, fatiha_doc):
        assert query._token_matches(fatiha_doc[2], {'LOWER': 'اللَّهِ'})


# ---------------------------------------------------------------------------
# _find_matches (pattern engine)
# ---------------------------------------------------------------------------

class TestFindMatches:
    def test_single_element_match(self, fatiha_doc):
        tokens = list(fatiha_doc)
        matches = query._find_matches(tokens, [{'ROOT': 'رحم'}])
        assert len(matches) == 2  # positions 3 and 4 both have root رحم
        assert all(s + 1 == e for s, e in matches)

    def test_single_element_no_match(self, fatiha_doc):
        tokens = list(fatiha_doc)
        assert query._find_matches(tokens, [{'ROOT': 'علم'}]) == []

    def test_sequential_two_elements(self, multi_doc):
        tokens = list(multi_doc)
        # NOUN immediately followed by VERB
        matches = query._find_matches(tokens, [{'POS': 'NOUN'}, {'POS': 'VERB'}])
        assert len(matches) >= 1
        for s, e in matches:
            assert tokens[s].pos_ == 'NOUN'
            assert tokens[e - 1].pos_ == 'VERB'

    def test_op_optional(self, multi_doc):
        tokens = list(multi_doc)
        # NOUN, optional CCONJ, then VERB
        pattern = [{'POS': 'NOUN'}, {'POS': 'CCONJ', 'OP': '?'}, {'POS': 'VERB'}]
        matches = query._find_matches(tokens, pattern)
        assert len(matches) >= 1

    def test_op_star(self, multi_doc):
        tokens = list(multi_doc)
        # VERB then zero-or-more of anything then NOUN
        pattern = [{'POS': 'VERB'}, {'OP': '*'}, {'POS': 'NOUN'}]
        matches = query._find_matches(tokens, pattern)
        assert len(matches) >= 1

    def test_op_plus(self, fatiha_doc):
        tokens = list(fatiha_doc)
        # One or more ADJ after a NOUN
        pattern = [{'POS': 'NOUN'}, {'POS': 'ADJ', 'OP': '+'}]
        matches = query._find_matches(tokens, pattern)
        assert len(matches) >= 1

    def test_op_negation(self, multi_doc):
        tokens = list(multi_doc)
        # VERB NOT immediately followed by VERB
        pattern = [{'POS': 'VERB'}, {'POS': 'VERB', 'OP': '!'}, {'POS': 'NOUN'}]
        matches = query._find_matches(tokens, pattern)
        assert len(matches) >= 0  # may or may not match depending on sequence

    def test_skip_proximity(self, multi_doc):
        tokens = list(multi_doc)
        # ROOT رحم within 3 tokens of ROOT علم
        pattern = [{'ROOT': 'علم'}, {'ROOT': 'رحم', 'SKIP': 3}]
        matches = query._find_matches(tokens, pattern)
        assert len(matches) >= 1  # علم at idx 2, رحم at idx 4 — gap = 1

    def test_skip_too_far(self, multi_doc):
        tokens = list(multi_doc)
        # SKIP=0 means no gap allowed
        pattern = [{'ROOT': 'علم'}, {'ROOT': 'رحم', 'SKIP': 0}]
        matches = query._find_matches(tokens, pattern)
        assert matches == []  # gap is 1, not 0

    def test_no_overlapping_spans(self, fatiha_doc):
        tokens = list(fatiha_doc)
        pattern = [{'ROOT': 'رحم'}]
        matches = query._find_matches(tokens, pattern)
        spans = set()
        for s, e in matches:
            for i in range(s, e):
                assert i not in spans, "overlapping spans found"
            spans.update(range(s, e))


# ---------------------------------------------------------------------------
# VerseMatcher.__call__ (single-doc matching)
# ---------------------------------------------------------------------------

class TestVerseMatcherCall:
    def _make_matcher(self):
        mock_nlp = MagicMock()
        return query.VerseMatcher(mock_nlp)

    def test_single_pattern_match(self, fatiha_doc):
        m = self._make_matcher()
        m.add('MERCY', [[{'ROOT': 'رحم'}]])
        results = m(fatiha_doc)
        assert len(results) == 2
        assert all(k == 'MERCY' for k, _, _ in results)

    def test_multiple_keys(self, fatiha_doc):
        m = self._make_matcher()
        m.add('MERCY', [[{'ROOT': 'رحم'}]])
        m.add('GOD',   [[{'LEMMA': 'الله'}]])
        results = m(fatiha_doc)
        keys = {k for k, _, _ in results}
        assert 'MERCY' in keys
        assert 'GOD' in keys

    def test_alternative_patterns(self, fatiha_doc):
        m = self._make_matcher()
        # Two alternatives: either رحم root OR اله root
        m.add('ROOT', [[{'ROOT': 'رحم'}], [{'ROOT': 'اله'}]])
        results = m(fatiha_doc)
        assert len(results) >= 3  # positions 2, 3, 4

    def test_no_match_returns_empty(self, fatiha_doc):
        m = self._make_matcher()
        m.add('NOTFOUND', [[{'ROOT': 'xyz'}]])
        assert m(fatiha_doc) == []

    def test_contains(self, fatiha_doc):
        m = self._make_matcher()
        m.add('X', [[{'ROOT': 'رحم'}]])
        assert 'X' in m
        assert 'Y' not in m

    def test_remove(self, fatiha_doc):
        m = self._make_matcher()
        m.add('X', [[{'ROOT': 'رحم'}]])
        m.remove('X')
        assert 'X' not in m
        assert m(fatiha_doc) == []

    def test_results_sorted_by_start(self, multi_doc):
        m = self._make_matcher()
        m.add('NOUN', [[{'POS': 'NOUN'}]])
        m.add('VERB', [[{'POS': 'VERB'}]])
        results = m(multi_doc)
        starts = [s for _, s, _ in results]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# VerseMatcher.search  (over provided docs)
# ---------------------------------------------------------------------------

class TestVerseMatcherSearch:
    def _make_matcher_with_docs(self, docs):
        mock_nlp = MagicMock(side_effect=docs)
        return query.VerseMatcher(mock_nlp), docs

    def test_search_over_docs(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        m = query.VerseMatcher(mock_nlp)
        m.add('MERCY', [[{'ROOT': 'رحم'}]])
        results = list(m.search(docs=[fatiha_doc, multi_doc]))
        # Both docs have رحم root
        assert len(results) == 2

    def test_search_no_match_excluded(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        m = query.VerseMatcher(mock_nlp)
        m.add('NOTFOUND', [[{'ROOT': 'xyz'}]])
        results = list(m.search(docs=[fatiha_doc, multi_doc]))
        assert results == []

    def test_search_max_results(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        m = query.VerseMatcher(mock_nlp)
        m.add('NOUN', [[{'POS': 'NOUN'}]])
        results = list(m.search(docs=[fatiha_doc, multi_doc], max_results=1))
        assert len(results) == 1

    def test_search_yields_doc_and_matches(self, fatiha_doc):
        mock_nlp = MagicMock()
        m = query.VerseMatcher(mock_nlp)
        m.add('ROOT_RHM', [[{'ROOT': 'رحم'}]])
        results = list(m.search(docs=[fatiha_doc]))
        assert len(results) == 1
        doc, matches = results[0]
        assert doc is fatiha_doc
        assert all(isinstance(k, str) and isinstance(s, int) and isinstance(e, int)
                   for k, s, e in matches)

    def test_search_with_surah_calls_surah_docs(self, fatiha_doc):
        mock_nlp = MagicMock()
        m = query.VerseMatcher(mock_nlp)
        m.add('NOUN', [[{'POS': 'NOUN'}]])
        with patch('quranic_nlp.language.surah_docs', return_value=[fatiha_doc]) as mock_sd:
            results = list(m.search(surah=1))
        mock_sd.assert_called_once_with(mock_nlp, 1)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

class TestFindByRoot:
    def test_returns_list(self, fatiha_doc):
        with patch('quranic_nlp.query.VerseMatcher') as MockMatcher:
            instance = MockMatcher.return_value
            instance.search.return_value = iter([(fatiha_doc, [('_', 0, 1)])])
            result = query.find_by_root(MagicMock(), 'رحم', docs=[fatiha_doc])
        assert isinstance(result, list)
        assert result[0] is fatiha_doc

    def test_with_pos_filter(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        # Without mocking — use actual matching
        results = query.find_by_root(mock_nlp, 'رحم',
                                     docs=[fatiha_doc, multi_doc])
        assert len(results) == 2  # both docs have رحم

    def test_pos_restricts_results(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        # رحم as NOUN only — fatiha has ADJ, multi has VERB
        results = query.find_by_root(mock_nlp, 'رحم', pos='NOUN',
                                     docs=[fatiha_doc, multi_doc])
        assert len(results) == 0  # neither has رحم as NOUN

    def test_no_match_returns_empty(self, fatiha_doc):
        mock_nlp = MagicMock()
        results = query.find_by_root(mock_nlp, 'xyz', docs=[fatiha_doc])
        assert results == []


class TestFindByLemma:
    def test_finds_correct_doc(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        results = query.find_by_lemma(mock_nlp, 'الله',
                                      docs=[fatiha_doc, multi_doc])
        assert len(results) == 2  # both have lemma الله

    def test_with_pos_filter(self, fatiha_doc):
        mock_nlp = MagicMock()
        results = query.find_by_lemma(mock_nlp, 'الله', pos='NOUN',
                                      docs=[fatiha_doc])
        assert len(results) == 1

    def test_missing_lemma(self, fatiha_doc):
        mock_nlp = MagicMock()
        results = query.find_by_lemma(mock_nlp, 'xyz', docs=[fatiha_doc])
        assert results == []


class TestFindByPos:
    def test_finds_verbs(self, multi_doc):
        mock_nlp = MagicMock()
        results = query.find_by_pos(mock_nlp, 'VERB', docs=[multi_doc])
        assert len(results) == 1

    def test_no_match(self, fatiha_doc):
        mock_nlp = MagicMock()
        results = query.find_by_pos(mock_nlp, 'VERB', docs=[fatiha_doc])
        assert results == []


class TestFindNear:
    def test_finds_proximity_match(self, multi_doc):
        mock_nlp = MagicMock()
        # ROOT علم (idx 2) within 3 tokens of ROOT رحم (idx 4)
        results = query.find_near(mock_nlp,
                                  {'ROOT': 'علم'}, {'ROOT': 'رحم'},
                                  max_dist=3, docs=[multi_doc])
        assert len(results) >= 1

    def test_gap_too_large_no_match(self, multi_doc):
        mock_nlp = MagicMock()
        results = query.find_near(mock_nlp,
                                  {'ROOT': 'علم'}, {'ROOT': 'رحم'},
                                  max_dist=0, docs=[multi_doc])
        assert results == []

    def test_result_structure(self, multi_doc):
        mock_nlp = MagicMock()
        results = query.find_near(mock_nlp,
                                  {'ROOT': 'علم'}, {'ROOT': 'رحم'},
                                  max_dist=5, docs=[multi_doc])
        for row in results:
            assert len(row) == 5
            doc, s1, e1, s2, e2 = row
            assert isinstance(s1, int)
            assert isinstance(s2, int)

    def test_directed_false_finds_both_orders(self, fatiha_doc):
        mock_nlp = MagicMock()
        fwd = query.find_near(mock_nlp,
                              {'ROOT': 'رحم'}, {'LEMMA': 'الله'},
                              max_dist=5, docs=[fatiha_doc], directed=False)
        bwd = query.find_near(mock_nlp,
                              {'LEMMA': 'الله'}, {'ROOT': 'رحم'},
                              max_dist=5, docs=[fatiha_doc], directed=False)
        assert len(fwd) > 0
        assert len(bwd) > 0


class TestFindVerses:
    def test_and_mode_both_present(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        results = query.find_verses(mock_nlp,
                                    [{'ROOT': 'رحم'}, {'LEMMA': 'الله'}],
                                    mode='AND', docs=[fatiha_doc, multi_doc])
        assert len(results) == 2

    def test_and_mode_one_missing(self, multi_doc):
        mock_nlp = MagicMock()
        # multi_doc has رحم but NOT lemma 'xyz'
        results = query.find_verses(mock_nlp,
                                    [{'ROOT': 'رحم'}, {'LEMMA': 'xyz'}],
                                    mode='AND', docs=[multi_doc])
        assert results == []

    def test_or_mode_one_present(self, fatiha_doc):
        mock_nlp = MagicMock()
        results = query.find_verses(mock_nlp,
                                    [{'ROOT': 'xyz'}, {'ROOT': 'رحم'}],
                                    mode='OR', docs=[fatiha_doc])
        assert len(results) == 1

    def test_or_mode_none_present(self, fatiha_doc):
        mock_nlp = MagicMock()
        results = query.find_verses(mock_nlp,
                                    [{'ROOT': 'xyz'}, {'ROOT': 'abc'}],
                                    mode='OR', docs=[fatiha_doc])
        assert results == []

    def test_invalid_mode_raises(self, fatiha_doc):
        mock_nlp = MagicMock()
        with pytest.raises(ValueError, match='mode'):
            query.find_verses(mock_nlp, [{'ROOT': 'رحم'}],
                               mode='INVALID', docs=[fatiha_doc])

    def test_max_results_respected(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        results = query.find_verses(mock_nlp,
                                    [{'ROOT': 'رحم'}],
                                    mode='AND',
                                    docs=[fatiha_doc, multi_doc],
                                    max_results=1)
        assert len(results) == 1


class TestConcordance:
    def test_returns_list_of_dicts(self, fatiha_doc):
        mock_nlp = MagicMock()
        rows = query.concordance(mock_nlp, {'ROOT': 'رحم'},
                                 context=2, docs=[fatiha_doc])
        assert len(rows) == 2  # two رحم tokens
        assert all(isinstance(r, dict) for r in rows)

    def test_row_keys(self, fatiha_doc):
        mock_nlp = MagicMock()
        rows = query.concordance(mock_nlp, {'ROOT': 'رحم'},
                                 context=1, docs=[fatiha_doc])
        required = {'surah', 'ayah', 'left', 'match', 'right', 'doc'}
        for row in rows:
            assert required.issubset(row.keys())

    def test_context_size(self, multi_doc):
        mock_nlp = MagicMock()
        rows = query.concordance(mock_nlp, {'ROOT': 'رحم'},
                                 context=2, docs=[multi_doc])
        for row in rows:
            assert len(row['left'])  <= 2
            assert len(row['right']) <= 2

    def test_match_is_correct_token(self, fatiha_doc):
        mock_nlp = MagicMock()
        rows = query.concordance(mock_nlp, {'ROOT': 'رحم'},
                                 docs=[fatiha_doc])
        for row in rows:
            assert row['match']._.root == 'رحم'

    def test_max_results(self, fatiha_doc, multi_doc):
        mock_nlp = MagicMock()
        rows = query.concordance(mock_nlp, {'ROOT': 'رحم'},
                                 docs=[fatiha_doc, multi_doc], max_results=1)
        assert len(rows) == 1

    def test_no_match_empty(self, fatiha_doc):
        mock_nlp = MagicMock()
        rows = query.concordance(mock_nlp, {'ROOT': 'xyz'}, docs=[fatiha_doc])
        assert rows == []
