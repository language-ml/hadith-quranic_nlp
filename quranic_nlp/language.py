from spacy.language import Language
from spacy.tokens import Doc, Token
import spacy

from quranic_nlp import dependency_parsing as dp
from quranic_nlp import postagger as pt
from quranic_nlp import lemmatizer
from quranic_nlp import root
from quranic_nlp import utils
from quranic_nlp import constant
# import dependency_parsing as dp
# import postagger as pt
# import lemmatizer
# import root
# import constant
# import utils

soure = None
ayeh = None

class NLP():

    postagger_model = None
    depparser_model = None
    lemma_model = None
    root_model = None

    Token.set_extension("dep_arc", default=None)
    Token.set_extension("root", default=None)
    Doc.set_extension("sentences", default=None)
    Doc.set_extension("revelation_order", default=None)
    Doc.set_extension("surah", default=None)
    Doc.set_extension("ayah", default=None)
    Doc.set_extension("sim_ayahs", default=None)
    Doc.set_extension("text", default=None)
    Doc.set_extension("translations", default=None)
    Doc.set_extension("hadiths", default=None)

    def __init__(self, lang, pipelines, translation_lang):

        global nlp
        nlp = spacy.blank(lang)
        self.nlp = nlp

        self.dict = {'dep': 'dependancyparser', 'pos': 'postagger', 'root': 'root', 'lem': 'lemmatize'}
        self.pipelines = pipelines.split(',')

        self.nlp.add_pipe('Quran')
        global translationlang
        translationlang = translation_lang

        if ('dep') in pipelines:
            global depparser_model
            depparser_model = dp.load_model()
            self.nlp.add_pipe('dependancyparser')

        if ('pos') in pipelines:
            global postagger_model
            postagger_model = pt.load_model()
            self.nlp.add_pipe('postagger')

        if 'lem' in pipelines:
            global lemma_model
            lemma_model = lemmatizer.load_model()
            self.nlp.add_pipe('lemmatize')

        if 'root' in pipelines:
            global root_model
            root_model = root.load_model()
            self.nlp.add_pipe('root')

    @Language.component('Quran')
    def initQuran(doc):
        try:
            # sent = Doc(nlp.vocab)
            global soure
            global ayeh
            text = doc.text
            soure, ayeh = utils.search_in_quran(text)
            print('soure:', soure, ', ayeh:', ayeh)
            words, spaces = utils.get_words_and_spaces(soure, ayeh)
            # print(words)
            # print(spaces)
            doc._.sentences = Doc(nlp.vocab, words=words, spaces=spaces)
            sent = doc._.sentences
            
            
            sent._.revelation_order = utils.get_revelation_order(soure)
            sent._.surah = utils.get_sourah_name_from_soure_index(soure)
            sent._.ayah = ayeh
            sent._.text = utils.get_text(soure, ayeh)
            sent._.translations = utils.get_translations(translationlang, soure, ayeh)
            sent._.sim_ayahs = utils.get_sim_ayahs(soure, ayeh)
            sent._.hadiths = utils.get_hadiths(soure, ayeh)
            return sent
        except:
            raise Exception('not found:', 'soure=', soure, ',ayeh=', ayeh)


    @Language.component('dependancyparser', assigns=["token.dep"])
    def depparser(doc):

        output = dp.depparser(depparser_model, soure, ayeh)
        if output:
            for d, out in zip(doc, output):
                if 'head' in out:
                    head = out['head']
                    arc = out['arc']
                    rel = out['rel']
                    d.dep_ = rel
                    d._.dep_arc = arc
                    d.head = doc[utils.get_indexes_from_words(soure, ayeh)[head]]

        return doc

    @Language.component('postagger', assigns=["token.pos"])
    def postagger(doc):

        output = pt.postagger(postagger_model, soure, ayeh)
        if output:
            for d, tags in zip(doc, output):
                if 'pos' in tags:
                    d.pos_ = constant.POS_FA_UNI[tags['pos']]

        return doc

    @Language.component('lemmatize', assigns=["token.lemma"])
    def lemmatizer(doc):

        output = lemmatizer.lemma(lemma_model, soure, ayeh)
        if output:        
            for d, tags in zip(doc, output):
                if 'lemma' in tags:
                    d.lemma_ = tags['lemma']
        return doc

    @Language.component('root')
    def rooter(doc):

        output = root.root(root_model, soure, ayeh)
        if output:
            for d, tags in zip(doc, output):
                if 'root' in tags:
                    d._.root = tags['root']
        return doc


class Pipeline():
    def __new__(cls, pipeline, translation_lang=None):
        language = NLP('ar', pipeline, translation_lang)
        nlp = language.nlp
        return nlp


def load_pipline(pipelines):
    language = NLP('ar', pipelines)
    nlp = language.nlp
    return nlp


def to_json(pipelines, doc):
    dict_list = []
    for i, d in enumerate(doc):
        dictionary = {}
        dictionary['id'] = i+1
        dictionary['text'] = d
        if 'root' in pipelines:
            dictionary['root'] = d._.root
        if 'lemma' in pipelines:
            dictionary['lemma'] = d.lemma_
        if 'pos' in pipelines:
            dictionary['pos'] = d.pos_
        if 'dep' in pipelines:
            dictionary['rel'] = d.dep_
            dictionary['arc'] = d._.dep_arc
            dictionary['head'] = d.head
        dict_list.append(dictionary)
    return dict_list
