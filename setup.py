# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2022 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from setuptools import setup


CLASSIFIERS = [ 'Development Status :: 5 - Production/Stable', 'Environment :: Console', 'Intended Audience :: Developers', 'Intended Audience :: Education', 'Intended Audience :: Information Technology', 'Intended Audience :: Science/Research', 'License :: OSI Approved :: MIT License', 'Natural Language :: Arabic', 'Operating System :: OS Independent', 'Programming Language :: Python :: 3', 'Programming Language :: Python :: 3 :: Only', 'Programming Language :: Python :: 3.7', 'Programming Language :: Python :: 3.8', 'Programming Language :: Python :: 3.9', 'Programming Language :: Python :: 3.10', 'Topic :: Scientific/Engineering', 'Topic :: Scientific/Engineering :: Artificial Intelligence', 'Topic :: Scientific/Engineering :: Information Analysis', 'Topic :: Text Processing' ]

INSTALL_REQUIRES = ['spacy', 'pandas', 'numpy', 'openpyxl', 'requests', 'clint','python-dotenv'],

DESCRIPTION = ("quarnic nlp")

######################  Version Package
VERSION_FILE = os.path.join(os.path.dirname(__file__),
                            'quranic_nlp',
                            'VERSION')
with open(VERSION_FILE, encoding='utf-8') as version_fp:
    VERSION = version_fp.read().strip()
######################


README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
with open(README_FILE, 'r', encoding='utf-8') as version_fp:
    LONG_DESCRIPTION = version_fp.read().strip()



setup(
    name = "quranic_nlp",
    version = VERSION,
    author="SMAJY",
    author_email='s.m.aref.j@gmail.com',
    maintainer='SMAJY',
    maintainer_email='s.m.aref.j@gmail.com',    
    packages=['quranic_nlp'],
    package_data={'quranic_nlp': ['config/settings.json']},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'quranic_data=quranic_nlp.data_requirements:main',
        ],
    },
    
    url='https://github.com/language-ml/hadith-quranic_nlp',
    license='MIT',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.7.0, <3.11'
)

# INSTALL_REQUIRES_NOT_WINDOWS = [
#     'camel-kenlm >= 2023.3.17.2 ; platform_system!="Windows"'
# ]
# if sys.platform != 'win32':
#     INSTALL_REQUIRES.extend(INSTALL_REQUIRES_NOT_WINDOWS)

