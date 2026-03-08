"""Tests for quranic_nlp.constant module."""

import pytest
from quranic_nlp import constant


def test_pos_fa_uni_keys():
    """POS_FA_UNI should map all standard Arabic POS labels."""
    expected = {'اسم', 'فعل', 'حرف', 'صفت', 'تصدیق', 'قید', 'کمکی',
                'ربط هماهنگ کننده', 'تعیین کننده', 'عدد', 'ذره', 'ضمیر',
                'اسم خاص', 'نقطه گذاری', 'ربط تبعی', 'نماد', 'دیگر'}
    assert set(constant.POS_FA_UNI.keys()) == expected


def test_pos_uni_fa_keys():
    """POS_UNI_FA should map all Universal Dependencies POS tags."""
    expected = {'NOUN', 'VERB', 'INTJ', 'ADJ', 'ADP', 'ADV', 'AUX',
                'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
                'SCONJ', 'SYM', 'X'}
    assert set(constant.POS_UNI_FA.keys()) == expected


def test_pos_mappings_are_inverse():
    """POS_UNI_FA and POS_FA_UNI should be inverse mappings of each other."""
    for fa, uni in constant.POS_FA_UNI.items():
        assert constant.POS_UNI_FA[uni] == fa


def test_translation_dict_has_expected_languages():
    """TRANSLATION dict should contain common language codes."""
    for lang in ('fa', 'en', 'ar', 'tr', 'ur'):
        assert lang in constant.TRANSLATION, f"Missing language: {lang}"


def test_translation_entries_are_lists():
    """Each TRANSLATION entry should be a non-empty list of strings."""
    for lang, entries in constant.TRANSLATION.items():
        assert isinstance(entries, list), f"{lang} entries should be a list"
        assert len(entries) > 0, f"{lang} entries should be non-empty"
        for entry in entries:
            assert isinstance(entry, str)


def test_ayeh_index_has_114_surahs():
    """AYEH_INDEX should have exactly 114 entries (one per surah)."""
    assert len(constant.AYEH_INDEX) == 114


def test_ayeh_index_first_surah():
    """First entry in AYEH_INDEX should refer to Al-Fatiha."""
    assert 'حمد' in constant.AYEH_INDEX[0]


def test_ayeh_index_last_surah():
    """Last entry in AYEH_INDEX should refer to An-Nas."""
    assert 'ناس' in constant.AYEH_INDEX[113]


def test_get_data_file_path_returns_string():
    """get_data_file_path should return a string path."""
    path = constant.get_data_file_path('quran.xml')
    assert isinstance(path, str)
    assert path.endswith('quran.xml')
