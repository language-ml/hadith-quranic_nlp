"""Tests for quranic_nlp.utils module."""

import pytest
from unittest.mock import patch, MagicMock
from quranic_nlp import utils, constant


class TestSearchInQuran:
    def test_numeric_reference(self):
        """'1#1' should return surah 1, ayah 1 without any API call."""
        soure, ayeh = utils.search_in_quran('1#1')
        assert soure == 1
        assert ayeh == 1

    def test_numeric_reference_surah3(self):
        soure, ayeh = utils.search_in_quran('3#200')
        assert soure == 3
        assert ayeh == 200

    def test_numeric_reference_last_surah(self):
        soure, ayeh = utils.search_in_quran('114#6')
        assert soure == 114
        assert ayeh == 6

    def test_arabic_surah_name_calls_api(self):
        """Arabic surah name input should attempt an API call."""
        with patch('quranic_nlp.utils.requests.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {'output': 'حمد'}
            mock_post.return_value = mock_resp
            with patch('quranic_nlp.utils.get_index_soure_from_name_soure', return_value=1):
                soure, ayeh = utils.search_in_quran('حمد#1')
        assert soure == 1
        assert ayeh == 1

    def test_free_text_calls_api(self):
        """Free Arabic text (no #) should call the quranic extraction API."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'output': {'regex_qe': {'quran_id': [['1##1']]}}
        }
        with patch('quranic_nlp.utils.requests.post', return_value=mock_response):
            soure, ayeh = utils.search_in_quran('بسم الله')
        assert soure == 1
        assert ayeh == 1

    def test_free_text_not_found_raises(self):
        """A failed free-text search should raise ValueError."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {'output': {'quran_id': []}}
        with patch('quranic_nlp.utils.requests.post', return_value=mock_response):
            with pytest.raises(ValueError):
                utils.search_in_quran('xyz not arabic')


class TestSearchAllInQuran:
    def test_returns_list_of_tuples(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'output': {'regex_qe': {'quran_id': [['1##2', '5##28']]}}
        }
        with patch('quranic_nlp.utils.requests.post', return_value=mock_response):
            result = utils.search_all_in_quran('رب العالمین')
        assert isinstance(result, list)
        assert (1, 2) in result
        assert (5, 28) in result

    def test_raises_when_no_matches(self):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {'output': {'quran_id': []}}
        with patch('quranic_nlp.utils.requests.post', return_value=mock_response):
            with pytest.raises(ValueError):
                utils.search_all_in_quran('xyz')


class TestGetIndexSoureFromNameSoure:
    def test_known_surah_name(self):
        """'حمد' should resolve to surah 1."""
        with patch('quranic_nlp.utils.requests.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {'output': 'حمد'}
            mock_post.return_value = mock_resp
            idx = utils.get_index_soure_from_name_soure('حمد')
        assert idx == 1

    def test_unknown_surah_name_raises(self):
        """An unknown surah name should raise ValueError."""
        with patch('quranic_nlp.utils.requests.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {'output': 'notasurah'}
            mock_post.return_value = mock_resp
            with pytest.raises(ValueError):
                utils.get_index_soure_from_name_soure('notasurah')


class TestGetText:
    def test_returns_string_for_valid_verse(self):
        """get_text should return a non-empty string for a known verse."""
        text = utils.get_text(1, 1)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_returns_none_for_invalid_verse(self):
        """get_text should return None for a non-existent verse."""
        result = utils.get_text(999, 999)
        assert result is None

    def test_fatiha_verse1_content(self):
        """Surah 1, Ayah 1 should contain 'بسم'."""
        text = utils.get_text(1, 1)
        assert 'بسم' in text or 'بِسْمِ' in text

    def test_different_verses_differ(self):
        """Two different verses should return different text."""
        assert utils.get_text(1, 1) != utils.get_text(1, 2)


class TestGetRevelationOrder:
    def test_returns_value_for_valid_surah(self):
        order = utils.get_revelation_order(1)
        assert order is not None

    def test_different_surahs_may_differ(self):
        """Revelation orders for surah 1 and 2 should be different numbers."""
        assert utils.get_revelation_order(1) != utils.get_revelation_order(2)


class TestGetSourahNameFromSoureIndex:
    def test_surah1_name(self):
        name = utils.get_sourah_name_from_soure_index(1)
        assert isinstance(name, str)
        assert len(name) > 0

    def test_different_surahs_have_different_names(self):
        assert utils.get_sourah_name_from_soure_index(1) != utils.get_sourah_name_from_soure_index(2)


class TestGetSimAyahs:
    def test_returns_list_of_tuples(self):
        """get_sim_ayahs should return a list of (ref, score) tuples."""
        result = utils.get_sim_ayahs(1, 2)
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 2

    def test_ref_is_string(self):
        """Each ref should be a 'surah#ayah' string."""
        result = utils.get_sim_ayahs(1, 2)
        ref, score = result[0]
        assert isinstance(ref, str)
        assert '#' in ref

    def test_score_is_float(self):
        """Each score should be a float."""
        result = utils.get_sim_ayahs(1, 2)
        ref, score = result[0]
        assert isinstance(score, float)

    def test_scores_are_positive(self):
        """All similarity scores should be positive."""
        result = utils.get_sim_ayahs(1, 2)
        assert all(score > 0 for _, score in result)

    def test_unknown_verse_returns_empty(self):
        """A verse with no similarity data should return an empty list."""
        result = utils.get_sim_ayahs(999, 999)
        assert result == []


class TestGetWordsAndSpaces:
    def test_returns_tuple_of_two(self):
        words, spaces = utils.get_words_and_spaces(1, 1)
        assert words is not None
        assert spaces is not None

    def test_words_and_spaces_same_length(self):
        words, spaces = utils.get_words_and_spaces(1, 1)
        assert len(words) == len(spaces)

    def test_words_are_strings(self):
        words, _ = utils.get_words_and_spaces(1, 1)
        assert all(isinstance(w, str) for w in words)

    def test_spaces_are_booleans(self):
        import numpy as np
        _, spaces = utils.get_words_and_spaces(1, 1)
        assert spaces.dtype == np.bool_

    def test_fatiha_verse1_word_count(self):
        """Surah 1, Ayah 1 should have 7 words."""
        words, _ = utils.get_words_and_spaces(1, 1)
        assert len(words) == 7

    def test_invalid_verse_raises(self):
        with pytest.raises((FileNotFoundError, IndexError)):
            utils.get_words_and_spaces(999, 999)


class TestGetHadiths:
    def test_returns_list_on_success(self):
        ids_resp = MagicMock()
        ids_resp.ok = True
        ids_resp.raise_for_status = MagicMock()
        ids_resp.json.return_value = [1, 2, 3]

        hadiths_resp = MagicMock()
        hadiths_resp.ok = True
        hadiths_resp.raise_for_status = MagicMock()
        hadiths_resp.json.return_value = {
            '1': {'narrators': 'A', 'hadith': 'B', 'translation_text': 'C'},
        }

        with patch('quranic_nlp.utils.requests.post',
                   side_effect=[ids_resp, hadiths_resp]):
            result = utils.get_hadiths(1, 1)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_returns_none_on_network_error(self):
        with patch('quranic_nlp.utils.requests.post',
                   side_effect=Exception('network error')):
            result = utils.get_hadiths(1, 1)
        assert result is None


class TestPrintAllTranslations:
    def test_no_exception_raised(self, capsys):
        """print_all_translations should not raise even for edge-case entries."""
        utils.print_all_translations()
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_outputs_fa_translations(self, capsys):
        utils.print_all_translations()
        out = capsys.readouterr().out
        assert 'fa' in out
        assert 'en' in out


class TestGetTranslations:
    def test_returns_empty_for_none_input(self):
        assert utils.get_translations(None, 1, 1) == ''

    def test_returns_empty_for_empty_string(self):
        assert utils.get_translations('', 1, 1) == ''

    def test_single_translator_returns_string(self):
        """'fa#1' should return a single string translation."""
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value='1|1|ترجمه\n1|2|بعدی\n'))
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        with patch('builtins.open', mock_open):
            result = utils.get_translations('fa#1', 1, 1)
        assert isinstance(result, str)

    def test_all_translators_returns_dict(self):
        """'fa' (no index) should return a dict keyed by translator name."""
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value='2|1|ترجمه\n2|2|بعدی\n'))
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        with patch('builtins.open', mock_open):
            result = utils.get_translations('fa', 2, 1)
        assert isinstance(result, dict)

    def test_all_translators_dict_keys_are_strings(self):
        mock_open = MagicMock()
        mock_open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(read=MagicMock(return_value='2|1|ترجمه\n2|2|بعدی\n'))
        )
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        with patch('builtins.open', mock_open):
            result = utils.get_translations('fa', 2, 1)
        assert all(isinstance(k, str) for k in result.keys())
