<!-- <h1 align="center">
  <img src="images/dadmatech.jpeg"  width="150"  />
   Dadmatools
</h1> -->

<h2 align="center">QuranicTools: A Python NLP Library for Quranic NLP</h2>

<div align="center">
  <a href="https://pypi.org/project/quranic-nlp/"><img src="https://img.shields.io/pypi/v/quranic-nlp?cache=0"></a>
  <a href=""><img src="https://img.shields.io/badge/license-Apache%202-blue.svg"></a>
  <a href="https://colab.research.google.com/github/language-ml/quranic-nlp/blob/main/notebooks/quranic_nlp_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
- [Pipeline](#pipeline)
- [Input Formats](#input-formats)
- [Verse Information](#verse-information)
- [Translations](#translations)
- [Similar Verses](#similar-verses)
- [Multiple Matches](#multiple-matches)
- [Word-level Analysis](#word-level-analysis)
- [JSON Output](#json-output)
- [Hadiths](#hadiths)
- [Visualization](#visualization)
- [Contributors](#contributors)
- [Contributing](#contributing)

## Installation

### Step 1 — Install the package

```bash
pip install quranic-nlp
```

### Step 2 — Download the data

The library requires data files (~97MB) that are downloaded separately from GitHub Releases:

```bash
quranic_data
```

Or from Python:

```python
from quranic_nlp.data_requirements import download_data
download_data()
```

Data is downloaded once and stored inside the package directory automatically.

## Development Setup

To set up a local development environment:

```bash
git clone https://github.com/language-ml/quranic-nlp.git
cd quranic-nlp
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -e .
quranic_data
```

## Pipeline

Available pipeline components:

| Key | Description |
|-----|-------------|
| `dep` | Dependency parsing |
| `pos` | Part-of-speech tagging |
| `root` | Root extraction |
| `lem` | Lemmatization |

```python
from quranic_nlp import language, utils, constant

pips = 'dep,pos,root,lem'

# Basic pipeline — no hadiths fetching (default)
nlp = language.Pipeline(pips, translation_lang='fa#1')

# With hadith fetching enabled (makes one HTTP request per verse — use for single-verse lookups)
nlp_with_hadiths = language.Pipeline(pips, translation_lang='fa#1', hadiths=True)
```

To see all available translation languages and translators:

```python
utils.print_all_translations()
```

## Input Formats

Four ways to reference a verse or surah:

```python
# 1. surah_number#ayah_number — single Doc (no internet required)
doc = nlp('1#1')

# 2. surah_name#ayah_number — single Doc (requires internet)
doc = nlp('حمد#1')

# 3. surah name or index with surah=True — SurahDoc (all verses of that surah)
surah = nlp('فاتحه', surah=True)   # by Arabic name
surah = nlp(1,       surah=True)   # by integer index
surah = nlp('1',     surah=True)   # by string index

# 4. Free Arabic text — list[Doc] of all matching verses (requires internet)
docs = nlp('رب العالمین')
```

## Verse Information

```python
doc = nlp('1#1')

print(doc._.text)              # بِسْمِ اللَّهِ الرَّحْمَـٰنِ الرَّحِيمِ  (full diacritics)
print(doc._.simple_text)       # بسم الله الرحمن الرحیم  (no diacritics)
print(doc._.surah)             # فاتحه
print(doc._.ayah)              # 1
print(doc._.revelation_order)  # 5
```

> **Note:** `str(doc)` returns the morphologically segmented tokens (e.g. `بِ سْمِ اللَّهِ ...`), not the original verse text. Use `doc._.text` for the full verse text with diacritics, or `doc._.simple_text` for text without diacritics.

## Translations

Pass `'<lang>#<index>'` for a single translator (returns a string):

```python
nlp_en = language.Pipeline(pips, 'en#16')   # Yusuf Ali
doc = nlp_en('1#1')
print(doc._.translations)
# In the name of Allah, the Beneficent, the Merciful.
```

Pass `'<lang>'` (no index) for all translators (returns a `dict` keyed by translator name):

```python
nlp_fa = language.Pipeline(pips, 'fa')
doc = nlp_fa('1#2')
print(doc._.translations)
# {
#   'ansarian': 'همه ستایش ها، ویژه خدا، مالک و مربّی جهانیان است.',
#   'ayati':    'ستايش خدا را كه پروردگار جهانيان است.',
#   'bahrampour': 'ستايش خداى را كه پروردگار جهانيان است',
#   ...   # 12 Persian translators total
# }
```

## Similar Verses

`doc._.sim_ayahs` returns a list of `(ref, score)` tuples sorted by similarity score:

```python
doc = nlp('1#2')
for ref, score in doc._.sim_ayahs[:5]:
    print(f'{ref:10s}  score={score:.4f}')
```
```
37#182      score=1.0000
6#45        score=0.5199
40#65       score=0.4620
10#10       score=0.3862
39#75       score=0.3793
```

## Multiple Matches

When free Arabic text matches multiple verses, `nlp(text)` returns a **list of docs**:

```python
docs = nlp('رب العالمین')
print(f'Found {len(docs)} matching verses')
for doc in docs[:3]:
    print(doc._.surah, doc._.ayah, '—', doc._.text)
```
```
فاتحه 2 — الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ
مائده 28 — لَئِن بَسَطتَ إِلَيَّ يَدَكَ...
انعام 45 — فَقُطِعَ دَابِرُ الْقَوْمِ...
```

You can also call `search_all` explicitly with a `max_results` cap:

```python
docs = language.search_all(nlp, 'رب العالمین', max_results=5)
```

## Word-level Analysis

```python
doc = nlp('1#1')
word = doc[2]  # third word: اللَّهِ

print(word)                            # اللَّهِ
print(word.pos_)                       # NOUN
print(constant.POS_UNI_FA[word.pos_]) # اسم
print(word.lemma_)                     # ٱللَّه
print(word._.root)                     # اله
print(word.dep_)                       # نعت
print(word._.dep_arc)                  # LTR  (Left-to-Right arc)
print(word.head)                       # رَّحِیمِ
```

Print a table of all words:

```python
print(f"{'Word':<20} {'POS':<8} {'Lemma':<15} {'Root':<10} {'Dep'}")
print('-' * 65)
for token in doc:
    print(f'{str(token):<20} {token.pos_:<8} {token.lemma_:<15} {str(token._.root):<10} {token.dep_}')
```

## JSON Output

```python
import json

result = language.to_json(pips, doc)
print(json.dumps(result, ensure_ascii=False, indent=2))
```
```json
[
  {"id": 1, "text": "بِ",      "root": "",    "lemma": "",       "pos": "INTJ", "rel": "مجرور",      "arc": "LTR", "head": "سْمِ"},
  {"id": 2, "text": "سْمِ",   "root": "سمو", "lemma": "ٱسْم",  "pos": "NOUN", "rel": "مضاف الیه", "arc": "LTR", "head": "اللَّهِ"},
  {"id": 3, "text": "اللَّهِ","root": "اله", "lemma": "ٱللَّه","pos": "NOUN", "rel": "نعت",        "arc": "LTR", "head": "رَّحِیمِ"},
  ...
]
```

## Surah-Level Graph Analysis

Pass `surah=True` to get a `SurahDoc` — an object containing all verse docs for the surah and tools for graph-based analysis.

```python
from quranic_nlp import language, graph

nlp = language.Pipeline('pos,root,lem', 'fa#1')

# Get all verses of a surah as a SurahDoc (surah=True required)
surah = nlp('فاتحه', surah=True)   # by Arabic name
# surah = nlp(1,   surah=True)     # by integer index
# surah = nlp('1', surah=True)     # by string index

print(f'{surah.surah}: {len(surah)} verses')

# Iterate over verse docs
for doc in surah:
    print(doc._.ayah, doc._.text)

# Build a verse-similarity graph (TF-IDF over surface + lemma + root)
G = surah.build_graph(rep='tfidf')

# Or with a sentence-embedding model (any model with .encode())
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('CAMeL-Lab/bert-base-arabic-camelbert-ca')
# G = surah.build_graph(rep='embedding', model=model, threshold=0.3)

# Find the most central verse
doc, scores = surah.central_verse(method='pagerank')
print(f'Most central: Ayah {doc._.ayah}')
print(doc._.text)
print(scores)

# All centrality methods
for method in ['pagerank', 'degree', 'betweenness', 'eigenvector', 'mst']:
    doc, _ = surah.central_verse(method=method)
    print(f'{method:12s} → Ayah {doc._.ayah}')

# Maximum Spanning Tree
T = surah.mst()
import networkx as nx
print(nx.info(T))

# Access the underlying NetworkX graph directly
G = surah.graph
print(f'Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}')
for u, v, data in G.edges(data=True):
    print(f'  Ayah {u+1} ↔ Ayah {v+1}: similarity = {data["weight"]:.3f}')
```

You can also use the lower-level `graph` module directly with any list of docs:

```python
from quranic_nlp import language, graph

nlp = language.Pipeline('pos,root,lem')
docs = language.surah_docs(nlp, 'فاتحه')   # or surah_docs(nlp, 1)

G = graph.build_graph(docs, rep='tfidf')
T = graph.mst(G)
doc, scores = graph.central_verse(G, docs, method='pagerank')
print(doc._.surah, doc._.ayah, doc._.text)
```

## Hadiths

Hadith fetching is **disabled by default** (it makes one HTTP request per verse, which is slow for surah-level processing). Enable it explicitly with `hadiths=True`:

```python
# Create a pipeline with hadith fetching enabled
nlp_h = language.Pipeline(pips, translation_lang='fa#1', hadiths=True)
doc = nlp_h('1#1')

hadiths = doc._.hadiths
if hadiths:
    print(f'Found {len(hadiths)} hadith(s)')
    print(hadiths[0])
else:
    print('No hadiths found or API unavailable.')
```

When `hadiths=False` (the default), `doc._.hadiths` is `None`.

## Visualization

Render the dependency parse tree using spaCy's displacy:

```python
from spacy import displacy

options = {'compact': True, 'bg': '#09a3d5', 'color': 'white', 'font': 'Arial'}
displacy.render(doc, style='dep', options=options, jupyter=True)
```

![](./docs/image_readme/fig.png "")
![](./docs/image_readme/fig2.png "")

## Contributors

- Seyyed Mohammad Aref Jahanmir
- Alireza Sahebi
- Doratossadat Dastgheyb
- Erfan Mohammadi
- Mahdi Ahmadi
- Ehsaneddin Asgari

📧 Contact: asgari [dot] berkeley [dot] edu

## Contributing

We warmly welcome contributions from the community! Whether you are a researcher, developer, linguist, or simply passionate about the Quran and NLP, there are many ways to get involved:

| Area | How to Help |
|------|-------------|
| New features | New pipeline components, morphological analyses, or language support |
| Data quality | Corrections to POS tags, dependency parses, lemmas, or roots |
| Translations | Add or improve Quranic translations for underrepresented languages |
| Testing | Help increase test coverage |
| Bug reports | Open an issue if something doesn't work as expected |
| Documentation | Clearer examples, tutorials, or API docs |

To contribute, fork the repository, make your changes, and open a pull request. For larger changes, please open an issue first to discuss your idea.

We believe open collaboration leads to better tools for everyone. Every contribution, big or small, is valued and appreciated.
