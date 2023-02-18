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

doc = nlp('1#4')

print(doc)
print(doc._.surah)
print(doc._.ayah)
print(doc._.revelation_order)
print(doc._.sim_ayahs)
print(doc._.translations)
```

```
إِيَّاكَ نَعْبُدُ وَ إِيَّاكَ نَسْتَعِينُ نحن نحن 
فاتحه
4
63
['82#15', '83#11', '70#26', '51#12', '56#56', '82#17', '74#46', '37#20', '82#18', '15#35', '38#78', '26#82', '109#6', '51#6', '82#9', '107#1', '95#7', '40#16', '19#15', '19#33', '61#9', '9#33', '48#28', '21#103', '6#73', '3#26', '98#5', '83#5', '39#11', '40#14', '77#12', '50#42', '77#35', '77#13', '39#2', '36#71', '74#9', '85#2', '16#52', '30#30', '42#13', '75#1', '30#43', '75#6', '40#29', '39#14', '43#77', '5#3', '86#9', '26#189', '40#65', '26#87', '38#81', '15#38', '7#51', '23#113', '23#16', '79#6', '51#13', '77#14', '37#26', '9#11', '3#24', '114#2', '82#19', '11#103', '34#40', '26#135', '25#25', '70#8', '2#193', '9#29', '19#38', '2#132', '7#14', '29#65', '8#39', '64#9', '30#14', '45#27', '10#105', '110#2', '78#17', '79#35', '83#6', '77#38', '50#34', '38#79', '15#36', '37#21', '44#40', '52#9', '56#50', '90#14', '40#32', '9#36', '80#34', '26#88', '56#86', '50#20']
تنها تو را مى پرستيم و تنها از تو يارى مى‌جوييم.
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
نَعْبُدُ
وَ
معطوف بر محل
LTR
عبد

VERB
```

To jsonify the results you can use the following:

```python
dictionary = language.to_json(pips, doc)
print(dictionary)
```

```python
[{'id': 1, 'text': الْ, 'root': None, 'lemma': '', 'pos': 'INTJ', 'rel': 'تعریف', 'arc': 'RTL', 'head': حَمْدُ}, {'id': 2, 'text': حَمْدُ, 'root': 'حمد', 'lemma': '', 'pos': 'NOUN', 'rel': 'خبر', 'arc': 'LTR', 'head': *}, {'id': 3, 'text': لِ, 'root': None, 'lemma': '', 'pos': 'INTJ', 'rel': 'متعلق', 'arc': 'LTR', 'head': *}, {'id': 4, 'text': لَّهِ, 'root': 'أله', 'lemma': '', 'pos': 'NOUN', 'rel': 'نعت', 'arc': 'LTR', 'head': رَبِّ}, {'id': 5, 'text': رَبِّ, 'root': 'ربب', 'lemma': '', 'pos': 'NOUN', 'rel': 'مضاف الیه ', 'arc': 'LTR', 'head': عَالَمِینَ}, {'id': 6, 'text': الْ, 'root': None, 'lemma': '', 'pos': 'INTJ', 'rel': 'تعریف', 'arc': 'RTL', 'head': عَالَمِینَ}, {'id': 7, 'text': عَالَمِینَ, 'root': 'علم', 'lemma': '', 'pos': 'NOUN', 'rel': '', 'arc': None, 'head': عَالَمِینَ}, {'id': 8, 'text': *, 'root': None, 'lemma': '', 'pos': '', 'rel': '', 'arc': None, 'head': *}]
```

To jsonify the results you can use the following:
```python
from spacy import displacy
from quranic_nlp import language
from quranic_nlp import utils

translation_translator = 'fa#1'
pips = 'dep,pos,root,lemma'

nlp = language.Pipeline(pips, translation_translator)
doc = nlp('1#4')
displacy.serve(doc, style="dep")
options = {"compact": True, "bg": "#09a3d5",
           "color": "white", "font": "xb-niloofar"}
displacy.serve(doc, style="dep", options=options)

```
![](./data/fig1.png "")
![](./data/fig2.png "")