<!-- <h1 align="center">
  <img src="images/dadmatech.jpeg"  width="150"  />
   Dadmatools
</h1> -->

<h2 align="center">QuaranicTools: A Python NLP Library for Quranic NLP</h2>
<h3 align="center"><a href='language.ml'>Language Processing and Digital Humanities Lab (Language.ML)</a></h2>

<div align="center">
  <a href="https://pypi.org/project/quranic-nlp/"><img src="https://shields.io/pypi/v/quranic-nlp.svg"></a>
  <a href=""><img src="https://img.shields.io/badge/license-Apache%202-blue.svg"></a>
</div>

<div align="center">
  <h5>
      Part of Speech Tagging
    <span> | </span>
      Dependency Parsing
    <span> | </span>
      Lemmatizer
    <span> | </span>
      Multilingual Search    <br>
    <span> | </span>
      Quranic Extractions        
    <span> | </span>
      Revelation Order
    <span> | </span> <br>
      Embeddings (coming soon)
    <span> | </span>
      Translations    
  </h5>
</div>

# Quranic NLP

Quranic NLP is a computational toolbox to conduct various syntactic and semantic analyses of Quranic verses. The aim is to put together all available resources contributing to a better understanding/analysis of the Quran for everyone.

Contents:

- [Installation](#installation)
- [Pipline (dep,pos,lem,root)](#pipeline)
- [Example](#example)
- [Format inputs](#format-inputs)

## Installation

To get started using Quranic NLP in your python project, you may simply install it via the pip package.

### Install with pip

```bash
pip install quranic-nlp
```

You can check the `requirements.txt` file to see the required packages.

## Pipeline

The NLP pipeline contains morphological information e.g., Lemmatizer as well as POS Tagger and Dependancy Parser in a `Spacy`-like pipeline.

```python
from quranic_nlp import language

translation_translator = 'fa#1'
pips = 'dep,pos,root,lemma'
nlp = language.Pipeline(pips, translation_translator)
```

[`Doc`](https://spacy.io/api/doc) object has different extensions.
First, there are `sentences` in `doc` referring to the verses.
Second, there are `ayah` in `doc` which is indicate number ayeh in soure.
Third, there are `surah` in `doc` which is indicate name of soure.
Fourth, there are `revelation_order` in `doc` which is indicate order of revelation of the ayeh.
`doc` which is the list of [`Token`](https://spacy.io/api/token) also has its own extensions.
The pips is information to use from quranic_nlp.
The translation_translator is language for translate quran such that language (fa) or language along with \# along with number books.
For see all translate run below code
```python
from quranic_nlp import utils
utils.print_all_translations()
```
Quranic NLP has its own spacy extensions. If related pipeline is not called, that extension cannot be used.

## Format Inputs

There are three ways to format the input.
First, number surah along with \# along with number ayah.
Second, name surah along with \# along with number ayah.
Third, search text in quran.

Note The last two calls require access to the net for an API call.

```python
from quranic_nlp import language

translation_translator = 'fa#1'
pips = 'dep,pos,root,lemma'
nlp = language.Pipeline(pips, translation_translator)

doc = nlp('1#1')
doc = nlp('حمد#1')
doc = nlp('رب العالمین')
```

## Example

```python
from quranic_nlp import language

translation_translator = 'fa#1'
pips = 'dep,pos,root,lemma'
nlp = language.Pipeline(pips, translation_translator)

doc = nlp('1#1')

print(doc)
print(doc._.surah)
print(doc._.ayah)
print(doc._.revelation_order)
print(doc._.sim_ayahs)
print(doc._.translations)
```

```
الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِینَ *
فاتحه
1
63
['27#30', '1#3', '55#1', '41#2', '2#163', '59#22', '11#41', '12#92', '7#151', '24#20', '44#42', '6#118', '36#5', '26#191', '26#175', '26#159', '26#140', '26#122', '26#104', '26#68', '26#9', '26#217', '20#5', '19#88', '21#83', '24#10', '25#60', '15#49', '12#64', '19#93', '2#192', '4#96', '4#106', '30#5', '32#6', '9#99', '6#121', '78#37', '19#91', '22#34', '2#218', '19#85', '41#32', '22#28', '12#98', '19#96', '20#8', '19#18', '5#4', '2#64', '39#53', '22#40', '5#74', '3#157', '10#58', '52#28', '22#36', '19#78', '43#84', '50#33', '6#119', '5#98', '17#110', '19#87', '21#26', '9#27', '36#58', '49#10', '26#5', '16#18', '9#104', '7#180', '6#138', '3#129', '23#118', '7#49', '67#29', '20#109', '27#46', '19#92', '43#36', '67#28', '25#59', '19#69', '2#37', '21#112', '43#20', '24#14', '69#52', '56#96', '56#74', '11#73', '3#132', '24#5', '3#89', '42#5', '43#45', '36#15', '57#28', '48#14']
ستايش خدا را كه پروردگار جهانيان است.
```

```python
print(doc[1])
print(doc[1].head)
print(doc[1].dep_)
print(doc[1]._.dep_arc)
print(doc[1]._.root)
print(doc[1].lemma_)
print(doc[1].pos_)
```

```
حَمْدُ
*
خبر
LTR
حمد

NOUN
```

To jsonify the results you can use the following:

```python
dictionary = language.to_json(pips, doc)
print(dictionary)
```

```python
[{'id': 1, 'text': الْ, 'root': None, 'lemma': '', 'pos': 'INTJ', 'rel': 'تعریف', 'arc': 'RTL', 'head': حَمْدُ}, {'id': 2, 'text': حَمْدُ, 'root': 'حمد', 'lemma': '', 'pos': 'NOUN', 'rel': 'خبر', 'arc': 'LTR', 'head': *}, {'id': 3, 'text': لِ, 'root': None, 'lemma': '', 'pos': 'INTJ', 'rel': 'متعلق', 'arc': 'LTR', 'head': *}, {'id': 4, 'text': لَّهِ, 'root': 'أله', 'lemma': '', 'pos': 'NOUN', 'rel': 'نعت', 'arc': 'LTR', 'head': رَبِّ}, {'id': 5, 'text': رَبِّ, 'root': 'ربب', 'lemma': '', 'pos': 'NOUN', 'rel': 'مضاف الیه ', 'arc': 'LTR', 'head': عَالَمِینَ}, {'id': 6, 'text': الْ, 'root': None, 'lemma': '', 'pos': 'INTJ', 'rel': 'تعریف', 'arc': 'RTL', 'head': عَالَمِینَ}, {'id': 7, 'text': عَالَمِینَ, 'root': 'علم', 'lemma': '', 'pos': 'NOUN', 'rel': '', 'arc': None, 'head': عَالَمِینَ}, {'id': 8, 'text': *, 'root': None, 'lemma': '', 'pos': '', 'rel': '', 'arc': None, 'head': *}]
```
