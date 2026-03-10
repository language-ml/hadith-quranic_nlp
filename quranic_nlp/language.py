"""
quranic_nlp.language
~~~~~~~~~~~~~~~~~~~~
spaCy-based pipeline for Quranic NLP.
"""

from spacy.language import Language
from spacy.tokens import Doc, Token
import spacy

from quranic_nlp import dependency_parsing as dp
from quranic_nlp import postagger as pt
from quranic_nlp import lemmatizer
from quranic_nlp import root
from quranic_nlp import utils
from quranic_nlp import constant


def _register_extensions():
    """Register spaCy custom extensions, skipping already-registered ones."""
    for name in ('dep_arc', 'root'):
        if not Token.has_extension(name):
            Token.set_extension(name, default=None)

    for name in ('sentences', 'revelation_order', 'surah', 'ayah',
                 'sim_ayahs', 'text', 'translations', 'hadiths',
                 'soure_index', 'ayeh_index'):
        if not Doc.has_extension(name):
            Doc.set_extension(name, default=None)


_register_extensions()

# Module-level vocab shared across pipeline components
_vocab = spacy.blank('ar').vocab
_translation_lang = None
_fetch_hadiths = False
_verbose = False
_depparser_model = None
_postagger_model = None
_lemma_model = None
_root_model = None


@Language.component('Quran')
def _init_quran(doc):
    text = doc.text
    soure, ayeh = utils.search_in_quran(text)
    if _verbose:
        print(f'surah: {soure}, ayah: {ayeh}')

    words, spaces = utils.get_words_and_spaces(soure, ayeh)
    sent = Doc(_vocab, words=words, spaces=spaces)

    sent._.soure_index = soure
    sent._.ayeh_index = ayeh
    sent._.revelation_order = utils.get_revelation_order(soure)
    sent._.surah = utils.get_sourah_name_from_soure_index(soure)
    sent._.ayah = ayeh
    sent._.text = utils.get_text(soure, ayeh)
    sent._.translations = utils.get_translations(_translation_lang, soure, ayeh)
    sent._.sim_ayahs = utils.get_sim_ayahs(soure, ayeh)
    sent._.hadiths = utils.get_hadiths(soure, ayeh) if _fetch_hadiths else None
    return sent


@Language.component('dependancyparser', assigns=['token.dep'])
def _dep_parser(doc):
    soure = doc._.soure_index
    ayeh = doc._.ayeh_index
    if soure is None:
        return doc
    output = dp.depparser(_depparser_model, soure, ayeh)
    if output:
        word_index = utils.get_indexes_from_words(soure, ayeh)
        for token, out in zip(doc, output):
            if 'head' in out:
                token.dep_ = out['rel']
                token._.dep_arc = out['arc']
                token.head = doc[word_index[out['head']]]
    return doc


@Language.component('postagger', assigns=['token.pos'])
def _post_tagger(doc):
    soure = doc._.soure_index
    ayeh = doc._.ayeh_index
    if soure is None:
        return doc
    output = pt.postagger(_postagger_model, soure, ayeh)
    if output:
        for token, tags in zip(doc, output):
            if 'pos' in tags:
                token.pos_ = constant.POS_FA_UNI[tags['pos']]
    return doc


@Language.component('lemmatize', assigns=['token.lemma'])
def _lemmatizer(doc):
    soure = doc._.soure_index
    ayeh = doc._.ayeh_index
    if soure is None:
        return doc
    output = lemmatizer.lemma(_lemma_model, soure, ayeh)
    if output:
        for token, tags in zip(doc, output):
            if 'lemma' in tags:
                token.lemma_ = tags['lemma']
    return doc


@Language.component('root')
def _rooter(doc):
    soure = doc._.soure_index
    ayeh = doc._.ayeh_index
    if soure is None:
        return doc
    output = root.root(_root_model, soure, ayeh)
    if output:
        for token, tags in zip(doc, output):
            if 'root' in tags:
                token._.root = tags['root']
    return doc


class NLP:
    """
    Quranic NLP pipeline wrapping a spaCy blank Arabic model.

    Parameters
    ----------
    pipelines : str
        Comma-separated pipeline component names. Valid values:
        ``dep`` (dependency parsing), ``pos`` (POS tagging),
        ``root`` (root extraction), ``lem`` (lemmatization).
    translation_lang : str, optional
        Translation language/index string, e.g. ``'fa#1'`` or ``'en#3'``.
        Run ``utils.print_all_translations()`` to see all options.
    hadiths : bool, optional
        Fetch related hadiths for each verse from the API. Default ``False``
        (skip fetching — use ``True`` only for single-verse lookups since it
        makes a network request per verse).
    verbose : bool, optional
        Print surah/ayah info for each processed verse. Default ``False``.
    """

    def __init__(self, pipelines: str, translation_lang: str = None,
                 hadiths: bool = False, verbose: bool = False):
        global _vocab, _translation_lang, _fetch_hadiths, _verbose
        global _depparser_model, _postagger_model, _lemma_model, _root_model

        nlp = spacy.blank('ar')
        _vocab = nlp.vocab
        _translation_lang = translation_lang
        _fetch_hadiths = hadiths
        _verbose = verbose
        self.nlp = nlp
        self.pipelines = pipelines

        self.nlp.add_pipe('Quran')

        if 'dep' in pipelines:
            _depparser_model = dp.load_model()
            self.nlp.add_pipe('dependancyparser')

        if 'pos' in pipelines:
            _postagger_model = pt.load_model()
            self.nlp.add_pipe('postagger')

        if 'lem' in pipelines:
            _lemma_model = lemmatizer.load_model()
            self.nlp.add_pipe('lemmatize')

        if 'root' in pipelines:
            _root_model = root.load_model()
            self.nlp.add_pipe('root')


class SurahDoc:
    """
    Represents all verses of a single surah with graph-based analysis tools.

    Returned by ``nlp('حمد')`` or ``nlp(1)`` when a surah name or index is given.

    Attributes
    ----------
    docs : list[Doc]
        All verse docs of the surah, in order.
    surah : str
        Arabic name of the surah.
    soure_index : int
        1-based surah index.

    Example
    -------
    ::

        from quranic_nlp import language, graph

        nlp = language.Pipeline('pos,root,lem', 'fa#1')
        surah = nlp('فاتحه')

        # Access verse docs
        for doc in surah.docs:
            print(doc._.ayah, doc._.text)

        # Build a similarity graph
        G = surah.build_graph(rep='tfidf')

        # Find the most central verse
        doc, scores = surah.central_verse(method='pagerank')
        print(doc._.ayah, doc._.text)

        # Maximum Spanning Tree
        T = surah.mst()

        # Use the graph directly with networkx
        import networkx as nx
        print(nx.info(surah.graph))
    """

    def __init__(self, docs, surah_name, soure_index):
        self.docs = docs
        self.surah = surah_name
        self.soure_index = soure_index
        self._graph = None

    def build_graph(self, rep='tfidf', model=None, threshold=0.0):
        """
        Build a verse-similarity graph and cache it.

        Parameters
        ----------
        rep : str
            ``'tfidf'`` or ``'embedding'``.
        model : optional
            Required when ``rep='embedding'``.
        threshold : float
            Minimum similarity for an edge.

        Returns
        -------
        networkx.Graph
        """
        from quranic_nlp import graph as _graph
        self._graph = _graph.build_graph(self.docs, rep=rep, model=model, threshold=threshold)
        return self._graph

    @property
    def graph(self):
        """Return the cached graph, building it with TF-IDF if not yet built."""
        if self._graph is None:
            self.build_graph()
        return self._graph

    def central_verse(self, method='pagerank'):
        """
        Return the most central verse.

        Parameters
        ----------
        method : str
            One of ``'pagerank'``, ``'degree'``, ``'betweenness'``,
            ``'eigenvector'``, ``'mst'``.

        Returns
        -------
        tuple[Doc, dict]
            ``(most_central_doc, {node_index: score, ...})``
        """
        from quranic_nlp import graph as _graph
        return _graph.central_verse(self.graph, self.docs, method=method)

    def mst(self):
        """
        Return the Maximum Spanning Tree of the similarity graph.

        Returns
        -------
        networkx.Graph
        """
        from quranic_nlp import graph as _graph
        return _graph.mst(self.graph)

    def __len__(self):
        return len(self.docs)

    def __iter__(self):
        return iter(self.docs)

    def __getitem__(self, idx):
        return self.docs[idx]

    def __repr__(self):
        return f"SurahDoc(surah={self.surah!r}, n_ayahs={len(self.docs)})"


def _local_surah_index(text: str):
    """Return 1-based surah index for *text* using local data only, or None."""
    # Try exact match, then with 'ال' prefix (e.g. 'فاتحه' → 'الفاتحه')
    candidates = {text}
    if not text.startswith('ال'):
        candidates.add('ال' + text)
    for idx, names in enumerate(constant.AYEH_INDEX):
        if candidates & set(names):
            return idx + 1
    return None


def _is_surah_input(text: str) -> bool:
    """Return True if *text* looks like a surah name/index (not free-text search)."""
    if text.isdigit():
        return True
    return _local_surah_index(text) is not None


class QuranicNLP:
    """
    Wrapper around a spaCy Language with smart dispatch based on input format:

    - ``'surah#ayah'`` (e.g. ``'1#1'``) → single ``Doc``
    - surah name/index (e.g. ``'فاتحه'`` or ``1``) → ``SurahDoc``
    - free Arabic text (e.g. ``'رب العالمین'``) → ``list[Doc]``

    Use ``Pipeline(...)`` or ``load_pipeline(...)`` to get an instance.
    """

    def __init__(self, nlp):
        self._nlp = nlp

    def __call__(self, text):
        """
        Process *text* and return:

        - A single ``Doc`` when *text* is a ``'surah#ayah'`` reference.
        - A ``SurahDoc`` when *text* is a surah name or integer index.
        - A ``list[Doc]`` for free Arabic text, returning all matching verses.
        """
        text_str = str(text).strip()
        if '#' in text_str:
            return self._nlp(text_str)
        if _is_surah_input(text_str):
            return self._make_surah_doc(text_str)
        return search_all(self, text_str)

    def _make_surah_doc(self, text: str) -> 'SurahDoc':
        if text.isdigit():
            soure = int(text)
        else:
            soure = _local_surah_index(text) or utils.get_index_soure_from_name_soure(text)
        surah_name = utils.get_sourah_name_from_soure_index(soure)
        docs = surah_docs(self, soure)
        return SurahDoc(docs, surah_name, soure)

    def __getattr__(self, name):
        return getattr(self._nlp, name)


class Pipeline:
    """
    Convenience factory that returns a configured ``QuranicNLP`` pipeline.

    Parameters
    ----------
    pipeline : str
        Comma-separated pipeline names (e.g. ``'dep,pos,root,lem'``).
    translation_lang : str, optional
        Translation language/index (e.g. ``'fa#1'``).

    Returns
    -------
    QuranicNLP

    Example
    -------
    ::

        from quranic_nlp import language

        nlp = language.Pipeline('dep,pos,root,lem', 'fa#1')

        # Single doc (surah#ayah reference)
        doc = nlp('1#1')
        print(doc._.surah)

        # List of docs (free text — all matching verses)
        docs = nlp('رب العالمین')
        for doc in docs:
            print(doc._.surah, doc._.ayah)
    """

    def __new__(cls, pipeline: str, translation_lang: str = None,
                hadiths: bool = False, verbose: bool = False):
        return QuranicNLP(NLP(pipeline, translation_lang, hadiths=hadiths, verbose=verbose).nlp)


def load_pipeline(pipelines: str, translation_lang: str = None,
                  hadiths: bool = False, verbose: bool = False):
    """Load and return a pipeline (alias for ``Pipeline``)."""
    return QuranicNLP(NLP(pipelines, translation_lang, hadiths=hadiths, verbose=verbose).nlp)


def search_all(nlp, text: str, max_results: int = None) -> list:
    """
    Return a list of processed docs for all verses matching the given text.

    Unlike calling ``nlp(text)`` directly (which returns only the first match),
    this function processes every matching verse and returns them all.

    Parameters
    ----------
    nlp : spacy.Language
        A pipeline created with ``Pipeline(...)`` or ``load_pipeline(...)``.
    text : str
        Free Arabic text to search for.
    max_results : int, optional
        Maximum number of docs to return. Returns all matches if not set.

    Returns
    -------
    list[Doc]
        A list of processed spaCy ``Doc`` objects, one per matching verse.

    Example
    -------
    ::

        from quranic_nlp import language

        nlp = language.Pipeline('dep,pos,root,lem', 'fa#1')
        docs = language.search_all(nlp, 'رب العالمین')
        for doc in docs:
            print(doc._.surah, doc._.ayah, doc._.translations)
    """
    matches = utils.search_all_in_quran(text)
    if max_results is not None:
        matches = matches[:max_results]
    return [nlp(f'{soure}#{ayeh}') for soure, ayeh in matches]


def surah_docs(nlp, surah) -> list:
    """
    Return a list of processed docs for every verse in a surah.

    Parameters
    ----------
    nlp : QuranicNLP
        Pipeline created with ``Pipeline(...)`` or ``load_pipeline(...)``.
    surah : int or str
        Surah index (e.g. ``1``) or Arabic name (e.g. ``'فاتحه'``).

    Returns
    -------
    list[Doc]
        One doc per verse, in order.

    Example
    -------
    ::

        from quranic_nlp import language, graph

        nlp = language.Pipeline('pos,root,lem', 'fa#1')

        docs = language.surah_docs(nlp, 'فاتحه')  # or surah_docs(nlp, 1)
        G = graph.build_graph(docs, rep='tfidf')
        doc, scores = graph.central_verse(G, docs, method='pagerank')
        print(doc._.surah, doc._.ayah, doc._.text)
    """
    if isinstance(surah, int):
        soure = surah
    elif str(surah).isdigit():
        soure = int(surah)
    else:
        soure = utils.get_index_soure_from_name_soure(str(surah))

    n_ayahs = len(utils._semantic_index()[soure - 1])
    return [nlp(f'{soure}#{a}') for a in range(1, n_ayahs + 1)]


def to_json(pipelines: str, doc) -> list:
    """
    Serialize a processed ``Doc`` to a list of per-token dictionaries.

    Parameters
    ----------
    pipelines : str
        Comma-separated pipeline names used during processing.
    doc :
        A processed spaCy ``Doc``.

    Returns
    -------
    list[dict]
        Each dict contains at minimum ``id`` and ``text``, plus optional
        fields ``root``, ``lemma``, ``pos``, ``rel``, ``arc``, ``head``
        depending on which pipelines were active.
    """
    result = []
    for i, token in enumerate(doc):
        entry = {'id': i + 1, 'text': str(token)}
        if 'root' in pipelines:
            entry['root'] = token._.root or ''
        if 'lem' in pipelines:
            entry['lemma'] = token.lemma_
        if 'pos' in pipelines:
            entry['pos'] = token.pos_
        if 'dep' in pipelines:
            entry['rel'] = token.dep_
            entry['arc'] = token._.dep_arc
            entry['head'] = str(token.head)
        result.append(entry)
    return result
