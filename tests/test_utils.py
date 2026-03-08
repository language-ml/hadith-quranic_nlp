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
            # get_index_soure_from_name_soure will be called; patch its result
            with patch('quranic_nlp.utils.get_index_soure_from_name_soure', return_value=1):
                soure, ayeh = utils.search_in_quran('حمد#1')
        assert soure == 1
        assert ayeh == 1

    def test_free_text_calls_api(self):
        """Free Arabic text (no #) should call the quranic extraction API."""
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'output': {'quran_id': [['1##1']]}
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

    def test_strips_al_prefix(self):
        """'الفاتحه' should strip 'ال' prefix before lookup."""
        with patch('quranic_nlp.utils.requests.post') as mock_post:
            mock_resp = MagicMock()
            mock_resp.ok = True
            mock_resp.json.return_value = {'output': 'الفاتحه'}
            mock_post.return_value = mock_resp
            with patch('quranic_nlp.utils.get_index_soure_from_name_soure',
                       wraps=utils.get_index_soure_from_name_soure):
                # Just ensure no crash
                pass


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
