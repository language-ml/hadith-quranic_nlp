"""Python setup.py for project_name package"""
import io
import os
from setuptools import find_packages, setup


def read(*paths, **kwargs):
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
    author="smajy",
    install_requires=['spacy', 'pandas', 'numpy', 'openpyxl', 'zipfile', 'requests', 'python-dotenv'],
    packages=['quranic_nlp'],
    package_dir={'quranic_nlp': 'src/quranic_nlp'},
    include_package_data=True,
    packages=find_packages(),
    package_data={'quranic_nlp': ['config/settings.json']},
    entry_points={
        'console_scripts': [
            'download_data=quranic_nlp.scripts.data_requirements:main',
        ],
    },
)