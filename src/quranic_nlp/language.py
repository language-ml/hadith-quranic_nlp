import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token
import pandas as pd
import numpy as np
import requests
import json
import re
from quranic_nlp import utils
from quranic_nlp import dependency_parsing as dp
from quranic_nlp import postagger as pt
from quranic_nlp import root
from quranic_nlp import lemmatizer
# import utils
# import dependency_parsing as dp
# import postagger as pt
# import root
# import lemmatizer

soure = None
ayeh = None

def findSent(doc):

    qSyntaxSemantics = []
    for i in range(1, 115):
        files = utils.recursive_glob(utils.AYEH_SEMANTIC, f'{i}-*.json')
        files.sort(key=lambda f: int(''. join(filter(str. isdigit, f))))
        qSyntaxSemantics.append(files)
    file = qSyntaxSemantics[soure - 1][ayeh - 1]
    with open(file) as f:
        data = json.load(f)

    nodes = data['Data']['ayeh']['node']['Data']
    nodes = pd.DataFrame(nodes)
    nodes.index = nodes["id"]
    nodes = nodes.sort_index()
    words = nodes['Word'].values
    spaces = np.full(len(words), True)

    for inx, (w, s) in enumerate(zip(words, nodes['xml'].apply(lambda x: x.split('Seq')[1].split('\"')[1] if x != None else None).values)):
        if s != None and int(s) == 2:
            spaces[inx - 1] = False

    global conv
    con = dict()
    for inx, id in enumerate(words):
        con[id] = inx
    conv = con

    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    return doc


class NLP():
    """
    In this class a blank pipeline in created and it is initialized based on our trained models
    possible pipelines: [ dependancyparser ]
    """
    postagger_model = None
    depparser_model = None
    lemma_model = None
    root_model = None

    Token.set_extension("dep_arc", default=None)
    Token.set_extension("root", default=None)
    Doc.set_extension("revelation_order", default=None)
    Doc.set_extension("surah", default=None)
    Doc.set_extension("ayah", default=None)
    Doc.set_extension("sentences", getter=findSent)
    Doc.set_extension("sim_ayahs", default=None)
    Doc.set_extension("translations", default=None)

    def __init__(self, lang, pipelines, translation_lang):

        global nlp
        nlp = spacy.blank(lang)
        self.nlp = nlp

        self.dict = {'dep': 'dependancyparser',
                     'pos': 'postagger', 'root': 'root', 'lem': 'lemmatize'}
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
        global soure
        global ayeh
        sent = Doc(nlp.vocab)
        text = doc.text
        if '#' in text:
            if not bool(re.search('[ا-ی]', text)):
                soure, ayeh = doc.text.split('#')
                soure, ayeh = int(soure), int(ayeh)
                sent = doc._.sentences
            else:
                soure_name, ayeh = doc.text.split('#')
                ayeh = int(ayeh)
                soure_name = soure_name.strip()
                if not str(soure_name).startswith('ال ') and str(soure_name).startswith('ال'):
                    soure_name = soure_name[2:]
                rep = requests.post('https://hadith.ai/preprocessing/',
                                    json={"query": soure_name, "dediac": 'true'})
                if rep.ok:
                    out = rep.json()['output']
                    for inx, output in enumerate(utils.AYEH_INDEX):
                        if out in output: 
                            soure = inx + 1
                            sent = doc._.sentences
                    
        else:
            rep = requests.post(
                'https://hadith.ai/quranic_extraction/', json={"query": text, 'min_tok': 3, 'min_char': 3})
            
            if rep.ok:
                out = rep.json()['output']['quran_id']
                if out:
                    soure, ayeh = out[0][0].split('##')
                    soure, ayeh = int(soure), int(ayeh) - 1
                    sent = doc._.sentences
        if soure:
            df = pd.read_csv(utils.QURAN_ORDER)
            df.index = df['index']
            sent._.revelation_order = df.loc[soure]['order_name']
            sent._.surah = df.loc[soure]['soure']
            sent._.ayah = ayeh
            sent._.translations = utils.get_translations(translationlang, soure, ayeh)
            sent._.sim_ayahs = utils.get_sim_ayahs(soure, ayeh)
        return sent

    @Language.component('dependancyparser', assigns=["token.dep"])
    def depparser(doc):

        output = dp.depparser(depparser_model, soure, ayeh)
        for d, out in zip(doc, output):
            if 'head' in out:
                head = out['head']
                arc = out['arc']
                rel = out['rel']
                d.dep_ = rel
                d._.dep_arc = arc
                d.head = doc[conv[head]]

        return doc

    @Language.component('postagger', assigns=["token.pos"])
    def postagger(doc):

        output = pt.postagger(postagger_model, soure, ayeh)
        for d, tags in zip(doc, output):
            if 'pos' in tags:
                d.pos_ = utils.POS_FA_UNI[tags['pos']]

        return doc

    @Language.component('lemmatize', assigns=["token.lemma"])
    def lemmatizer(doc):

        output = lemmatizer.lemma(postagger_model, soure, ayeh)
        for d, tags in zip(doc, output):
            if 'lemma' in tags:
                d.lemma_ = tags['lemma']
        return doc

    @Language.component('root')
    def lemmatizer(doc):

        output = root.root(postagger_model, soure, ayeh)
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
