"""Tests for quranic_nlp.language module."""

import json
import pytest
from unittest.mock import patch, MagicMock
from spacy.tokens import Doc, Token
from quranic_nlp import language


class TestExtensionRegistration:
    def test_token_extensions_registered(self):
        """Custom Token extensions should be registered."""
        assert Token.has_extension('dep_arc')
        assert Token.has_extension('root')

    def test_doc_extensions_registered(self):
        """Custom Doc extensions should be registered."""
        for name in ('sentences', 'revelation_order', 'surah', 'ayah',
                     'sim_ayahs', 'text', 'translations', 'hadiths',
                     'soure_index', 'ayeh_index'):
            assert Doc.has_extension(name), f"Doc extension '{name}' not registered"

    def test_register_extensions_idempotent(self):
        """Calling _register_extensions() multiple times should not raise."""
        language._register_extensions()
        language._register_extensions()


class TestToJson:
    def _make_doc(self, words):
        import spacy
        nlp = spacy.blank('ar')
        return nlp.make_doc(' '.join(words))

    def test_basic_output_structure(self):
        doc = self._make_doc(['بِسْمِ', 'اللَّهِ'])
        result = language.to_json('', doc)
        assert len(result) == 2
        assert result[0]['id'] == 1
        assert result[1]['id'] == 2

    def test_text_is_string(self):
        """to_json must produce JSON-serializable output (no Token objects)."""
        doc = self._make_doc(['بِسْمِ', 'اللَّهِ'])
        result = language.to_json('dep,pos,root,lem', doc)
        # Must not raise
        serialized = json.dumps(result, ensure_ascii=False)
        assert 'بِسْمِ' in serialized

    def test_head_is_string(self):
        """head field must be a string, not a Token object."""
        doc = self._make_doc(['word', 'word2'])
        result = language.to_json('dep', doc)
        assert isinstance(result[0]['head'], str)

    def test_no_pipeline_only_id_and_text(self):
        doc = self._make_doc(['word'])
        result = language.to_json('', doc)
        assert 'id' in result[0]
        assert 'text' in result[0]
        assert 'root' not in result[0]
        assert 'pos' not in result[0]

    def test_root_in_output_when_root_pipeline(self):
        doc = self._make_doc(['word'])
        result = language.to_json('root', doc)
        assert 'root' in result[0]
        assert isinstance(result[0]['root'], str)

    def test_pos_in_output_when_pos_pipeline(self):
        doc = self._make_doc(['word'])
        result = language.to_json('pos', doc)
        assert 'pos' in result[0]

    def test_lemma_in_output_when_lem_pipeline(self):
        doc = self._make_doc(['word'])
        result = language.to_json('lem', doc)
        assert 'lemma' in result[0]

    def test_dep_fields_in_output_when_dep_pipeline(self):
        doc = self._make_doc(['word', 'word2'])
        result = language.to_json('dep', doc)
        assert 'rel' in result[0]
        assert 'arc' in result[0]
        assert 'head' in result[0]

    def test_empty_doc(self):
        doc = self._make_doc([])
        result = language.to_json('dep,pos,root,lem', doc)
        assert result == []


class TestQuranicNLPWrapper:
    """Tests for the QuranicNLP wrapper behavior."""

    def _make_mock_doc(self, surah, ayah):
        import spacy
        nlp = spacy.blank('ar')
        doc = nlp.make_doc('test')
        doc._.soure_index = surah
        doc._.ayeh_index = ayah
        return doc

    def test_hash_reference_returns_single_doc(self):
        """'1#1' input should return a single Doc, not a list."""
        mock_doc = self._make_mock_doc(1, 1)
        wrapper = language.QuranicNLP(MagicMock(return_value=mock_doc))
        result = wrapper('1#1')
        assert isinstance(result, Doc)

    def test_free_text_returns_list(self):
        """Free Arabic text (no #) should return a list of Docs."""
        mock_doc = self._make_mock_doc(1, 1)
        inner_nlp = MagicMock(return_value=mock_doc)
        wrapper = language.QuranicNLP(inner_nlp)
        with patch('quranic_nlp.utils.search_all_in_quran',
                   return_value=[(1, 1), (2, 5)]):
            result = wrapper('رب العالمین')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_getattr_delegates_to_inner_nlp(self):
        """Attribute access should delegate to the wrapped spaCy Language."""
        inner_nlp = MagicMock()
        inner_nlp.vocab = 'test_vocab'
        wrapper = language.QuranicNLP(inner_nlp)
        assert wrapper.vocab == 'test_vocab'


class TestSearchAll:
    def test_returns_list(self):
        mock_doc = MagicMock()
        mock_nlp = MagicMock(return_value=mock_doc)
        wrapper = language.QuranicNLP(mock_nlp)
        with patch('quranic_nlp.utils.search_all_in_quran',
                   return_value=[(1, 1), (1, 2)]):
            result = language.search_all(wrapper, 'test')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_max_results_respected(self):
        mock_doc = MagicMock()
        mock_nlp = MagicMock(return_value=mock_doc)
        wrapper = language.QuranicNLP(mock_nlp)
        with patch('quranic_nlp.utils.search_all_in_quran',
                   return_value=[(1, 1), (1, 2), (1, 3), (2, 1)]):
            result = language.search_all(wrapper, 'test', max_results=2)
        assert len(result) == 2
