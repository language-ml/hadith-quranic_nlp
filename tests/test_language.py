"""Tests for quranic_nlp.language module."""

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
