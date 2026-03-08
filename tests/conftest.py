"""
Pytest configuration and shared fixtures for quranic_nlp tests.
"""

import pytest


@pytest.fixture
def sample_verse_numeric():
    """A verse reference using numeric surah/ayah notation."""
    return '1#1'


@pytest.fixture
def sample_verse_surah3():
    """A verse reference for Surah 3, verse 200."""
    return '3#200'
