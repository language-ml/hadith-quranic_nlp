"""Tests for quranic_nlp.graph module."""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import MagicMock, patch

from quranic_nlp import graph, language


def _make_mock_doc(ayah, surah='فاتحه', text='test', words=('w1', 'w2')):
    """Create a minimal mock doc for graph testing."""
    import spacy
    nlp = spacy.blank('ar')
    doc = nlp.make_doc(' '.join(words))
    doc._.soure_index = 1
    doc._.ayeh_index = ayah
    doc._.ayah = ayah
    doc._.surah = surah
    doc._.text = text
    for token in doc:
        token.lemma_ = token.text
        token._.root = token.text
    return doc


@pytest.fixture
def mock_docs():
    # Words share 'الله' so TF-IDF produces non-zero similarity for connected graph tests
    return [
        _make_mock_doc(1, text='بِسْمِ اللَّهِ', words=('بسم', 'الله', 'الرحمن')),
        _make_mock_doc(2, text='الْحَمْدُ لِلَّهِ', words=('الحمد', 'الله', 'الرحيم')),
        _make_mock_doc(3, text='الرَّحْمَنِ الرَّحِيمِ', words=('الرحمن', 'الرحيم', 'الله')),
    ]


class TestBuildGraph:
    def test_returns_networkx_graph(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        assert isinstance(G, nx.Graph)

    def test_node_count_matches_docs(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        assert G.number_of_nodes() == len(mock_docs)

    def test_nodes_have_attributes(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        for i in range(len(mock_docs)):
            assert 'ayah' in G.nodes[i]
            assert 'surah' in G.nodes[i]
            assert 'text' in G.nodes[i]

    def test_edges_have_weight(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        for u, v, d in G.edges(data=True):
            assert 'weight' in d
            assert 0.0 <= d['weight'] <= 1.0

    def test_threshold_reduces_edges(self, mock_docs):
        G_all = graph.build_graph(mock_docs, rep='tfidf', threshold=0.0)
        G_thresh = graph.build_graph(mock_docs, rep='tfidf', threshold=0.5)
        assert G_thresh.number_of_edges() <= G_all.number_of_edges()

    def test_embedding_rep_with_model(self, mock_docs):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(len(mock_docs), 16)
        G = graph.build_graph(mock_docs, rep='embedding', model=mock_model)
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(mock_docs)

    def test_embedding_without_model_raises(self, mock_docs):
        with pytest.raises(ValueError, match='model'):
            graph.build_graph(mock_docs, rep='embedding')

    def test_unknown_rep_raises(self, mock_docs):
        with pytest.raises(ValueError, match='rep'):
            graph.build_graph(mock_docs, rep='unknown')


class TestMST:
    def test_returns_graph(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        T = graph.mst(G)
        assert isinstance(T, nx.Graph)

    def test_mst_is_spanning_tree(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        T = graph.mst(G)
        assert T.number_of_nodes() == G.number_of_nodes()
        assert T.number_of_edges() == G.number_of_nodes() - 1
        assert nx.is_connected(T)


class TestCentralVerse:
    def test_returns_doc_and_scores(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        doc, scores = graph.central_verse(G, mock_docs, method='pagerank')
        assert doc in mock_docs
        assert isinstance(scores, dict)
        assert len(scores) == len(mock_docs)

    @pytest.mark.parametrize('method', ['pagerank', 'degree', 'betweenness', 'eigenvector', 'mst'])
    def test_all_methods_return_valid_doc(self, mock_docs, method):
        G = graph.build_graph(mock_docs, rep='tfidf')
        doc, scores = graph.central_verse(G, mock_docs, method=method)
        assert doc in mock_docs
        assert len(scores) == len(mock_docs)

    def test_unknown_method_raises(self, mock_docs):
        G = graph.build_graph(mock_docs, rep='tfidf')
        with pytest.raises(ValueError, match='method'):
            graph.central_verse(G, mock_docs, method='unknown')


class TestSurahDocs:
    def test_surah_docs_by_index(self):
        mock_doc = MagicMock()
        mock_nlp = MagicMock(return_value=mock_doc)
        wrapper = language.QuranicNLP(mock_nlp)
        with patch('quranic_nlp.utils._semantic_index', return_value=[['a', 'b', 'c']] + [[]] * 113):
            docs = language.surah_docs(wrapper, 1)
        assert len(docs) == 3

    def test_surah_docs_by_string_number(self):
        mock_doc = MagicMock()
        mock_nlp = MagicMock(return_value=mock_doc)
        wrapper = language.QuranicNLP(mock_nlp)
        with patch('quranic_nlp.utils._semantic_index', return_value=[['a', 'b']] + [[]] * 113):
            docs = language.surah_docs(wrapper, '1')
        assert len(docs) == 2

    def test_surah_docs_by_name(self):
        mock_doc = MagicMock()
        mock_nlp = MagicMock(return_value=mock_doc)
        wrapper = language.QuranicNLP(mock_nlp)
        with patch('quranic_nlp.utils.get_index_soure_from_name_soure', return_value=1), \
             patch('quranic_nlp.utils._semantic_index', return_value=[['a', 'b', 'c', 'd']] + [[]] * 113):
            docs = language.surah_docs(wrapper, 'فاتحه')
        assert len(docs) == 4


class TestSurahDoc:
    def _make_surah_doc(self, mock_docs):
        return language.SurahDoc(mock_docs, 'فاتحه', 1)

    def test_repr(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        assert 'فاتحه' in repr(sd)
        assert str(len(mock_docs)) in repr(sd)

    def test_len(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        assert len(sd) == len(mock_docs)

    def test_iter(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        assert list(sd) == mock_docs

    def test_getitem(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        assert sd[0] is mock_docs[0]
        assert sd[-1] is mock_docs[-1]

    def test_graph_lazy_build(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        assert sd._graph is None
        G = sd.graph  # triggers lazy build
        assert isinstance(G, nx.Graph)
        assert sd._graph is G

    def test_build_graph_explicit(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        G = sd.build_graph(rep='tfidf')
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(mock_docs)

    def test_build_graph_embedding(self, mock_docs):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(len(mock_docs), 16)
        sd = self._make_surah_doc(mock_docs)
        G = sd.build_graph(rep='embedding', model=mock_model)
        assert isinstance(G, nx.Graph)

    def test_central_verse_returns_doc_and_scores(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        sd.build_graph(rep='tfidf')
        doc, scores = sd.central_verse(method='pagerank')
        assert doc in mock_docs
        assert isinstance(scores, dict)

    def test_mst_returns_graph(self, mock_docs):
        sd = self._make_surah_doc(mock_docs)
        sd.build_graph(rep='tfidf')
        T = sd.mst()
        assert isinstance(T, nx.Graph)

    @pytest.mark.parametrize('method', ['pagerank', 'degree', 'betweenness', 'eigenvector', 'mst'])
    def test_all_centrality_methods(self, mock_docs, method):
        sd = self._make_surah_doc(mock_docs)
        sd.build_graph(rep='tfidf')
        doc, scores = sd.central_verse(method=method)
        assert doc in mock_docs


class TestQuranicNLPDispatch:
    def _make_wrapper(self):
        mock_nlp = MagicMock(return_value=MagicMock())
        return language.QuranicNLP(mock_nlp), mock_nlp

    def test_hash_format_returns_doc(self):
        wrapper, mock_nlp = self._make_wrapper()
        result = wrapper('1#1')
        mock_nlp.assert_called_once_with('1#1')
        assert result is mock_nlp.return_value

    def test_surah_name_with_flag_returns_surah_doc(self):
        wrapper, mock_nlp = self._make_wrapper()
        with patch('quranic_nlp.language.surah_docs', return_value=[MagicMock()] * 7), \
             patch('quranic_nlp.utils.get_sourah_name_from_soure_index', return_value='فاتحه'):
            result = wrapper('فاتحه', surah=True)
        assert isinstance(result, language.SurahDoc)
        assert result.surah == 'فاتحه'
        assert len(result.docs) == 7

    def test_integer_surah_with_flag_returns_surah_doc(self):
        wrapper, mock_nlp = self._make_wrapper()
        with patch('quranic_nlp.language.surah_docs', return_value=[MagicMock()] * 7), \
             patch('quranic_nlp.utils.get_sourah_name_from_soure_index', return_value='فاتحه'):
            result = wrapper('1', surah=True)
        assert isinstance(result, language.SurahDoc)

    def test_surah_name_without_flag_returns_list(self):
        wrapper, mock_nlp = self._make_wrapper()
        with patch('quranic_nlp.language.search_all', return_value=[MagicMock()]) as mock_search:
            result = wrapper('فاتحه')
        assert isinstance(result, list)
        mock_search.assert_called_once()

    def test_free_text_returns_list(self):
        wrapper, mock_nlp = self._make_wrapper()
        with patch('quranic_nlp.language.search_all', return_value=[MagicMock(), MagicMock()]) as mock_search:
            result = wrapper('رب العالمین')
        assert isinstance(result, list)
        mock_search.assert_called_once()
