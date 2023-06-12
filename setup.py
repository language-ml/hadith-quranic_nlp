"""Python setup.py for project_name package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="quranic_nlp",
    version="1.1.8",
    description="quarnic nlp",
    url="https://github.com/language-ml/hadith-quranic_nlp/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    # author="author_name",
    # packages=find_packages(exclude=["tests", ".github"]),
    install_requires=['spacy', 'pandas', 'numpy', 'openpyxl'],
    # data_files=[('data', '*')]
    packages=['quranic_nlp'],
    package_dir={'quranic_nlp': 'src/quranic_nlp'},
    package_data={'quranic_nlp': [ 'data/*',
                                   'data/qSyntaxSemantic/*',
                                   'data/syntax_data/*',
                                   'data/translate_quran/am/*',
                                   'data/translate_quran/az/*',
                                   'data/translate_quran/bg/*',
                                   'data/translate_quran/bs/*',
                                   'data/translate_quran/de/*',
                                   'data/translate_quran/en/*',
                                   'data/translate_quran/fa/*',
                                   'data/translate_quran/hi/*',
                                   'data/translate_quran/it/*',
                                   'data/translate_quran/ko/*',
                                   'data/translate_quran/ml/*',
                                   'data/translate_quran/nl/*',
                                   'data/translate_quran/pl/*',                  
                                   'data/translate_quran/pt/*',
                                   'data/translate_quran/ru/*',
                                   'data/translate_quran/so/*',
                                   'data/translate_quran/sv/*',
                                   'data/translate_quran/ta/*',
                                   'data/translate_quran/th/*',
                                   'data/translate_quran/tt/*',
                                   'data/translate_quran/ur/*',
                                   'data/translate_quran/zh/*',
                                   'data/translate_quran/ber/*',
                                   'data/translate_quran/ar/*',                                        
                                   'data/translate_quran/bn/*',
                                   'data/translate_quran/cs/*',
                                   'data/translate_quran/dv/*',
                                   'data/translate_quran/es/*',
                                   'data/translate_quran/ha/*',
                                   'data/translate_quran/id/*',
                                   'data/translate_quran/jp/*',
                                   'data/translate_quran/ku/*',
                                   'data/translate_quran/ms/*',
                                   'data/translate_quran/no/*',
                                   'data/translate_quran/ps/*',
                                   'data/translate_quran/ro/*',
                                   'data/translate_quran/sq/*',
                                   'data/translate_quran/sw/*',
                                   'data/translate_quran/tg/*',
                                   'data/translate_quran/tr/*',
                                   'data/translate_quran/ug/*',
                                   'data/translate_quran/uz/*',]},
)