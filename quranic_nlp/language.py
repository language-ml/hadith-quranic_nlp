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
_depparser_model = None
_postagger_model = None
_lemma_model = None
_root_model = None


@Language.component('Quran')
def _init_quran(doc):
    text = doc.text
    soure, ayeh = utils.search_in_quran(text)
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
    sent._.hadiths = utils.get_hadiths(soure, ayeh)
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
    """

    def __init__(self, pipelines: str, translation_lang: str = None):
        global _vocab, _translation_lang
        global _depparser_model, _postagger_model, _lemma_model, _root_model

        nlp = spacy.blank('ar')
        _vocab = nlp.vocab
        _translation_lang = translation_lang
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


class Pipeline:
    """
    Convenience factory that returns a configured spaCy ``Language`` object.

    Parameters
    ----------
    pipeline : str
        Comma-separated pipeline names (e.g. ``'dep,pos,root,lem'``).
    translation_lang : str, optional
        Translation language/index (e.g. ``'fa#1'``).

    Returns
    -------
    spacy.Language

    Example
    -------
    ::

        from quranic_nlp import language

        nlp = language.Pipeline('dep,pos,root,lem', 'fa#1')
        doc = nlp('1#1')
        print(doc._.surah)
        print(doc._.translations)
    """

    def __new__(cls, pipeline: str, translation_lang: str = None):
        return NLP(pipeline, translation_lang).nlp


def load_pipeline(pipelines: str, translation_lang: str = None):
    """Load and return a pipeline (alias for ``Pipeline``)."""
    return NLP(pipelines, translation_lang).nlp


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
        entry = {'id': i + 1, 'text': token}
        if 'root' in pipelines:
            entry['root'] = token._.root
        if 'lem' in pipelines:
            entry['lemma'] = token.lemma_
        if 'pos' in pipelines:
            entry['pos'] = token.pos_
        if 'dep' in pipelines:
            entry['rel'] = token.dep_
            entry['arc'] = token._.dep_arc
            entry['head'] = token.head
        result.append(entry)
    return result
