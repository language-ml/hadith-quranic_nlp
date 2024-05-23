import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import requests
import fnmatch
import json

import os
import re

from quranic_nlp import constant
# import constant

def get_sim_ayahs(soure, ayeh):
    output = []
    
    with open(constant.SIMILARITY_AYAT, encoding="utf-8") as file:
        sims = file.readlines()
        for sim in sims:
            ayeha = sim.split('\t')
            so = int(ayeha[0][:-3])
            ay = int(ayeha[0][-3:])
            if soure == so and ay == ayeh:
                for aye in ayeha[1:]:
                    temp , cost = aye.split(':')
                    output.append(str(int(temp[:-3])) + '#' + str(int(temp[-3:])))
    return output

def get_text(soure, ayeh):
    tree = ET.parse(constant.QURAN_XML)
    root = tree.getroot()

    # Search for elements with a specific attribute value
    for elem in root.iter('sura'):
        if elem.attrib.get('index') == str(soure):
            for e in elem.iter('aya'):
                if e.attrib.get('index') == str(ayeh):
                    return e.attrib.get('text')

def get_translations(input, soure, ayeh):
    if not input:
        return ''
    if '#' in input:
        lang, index = input.split('#')
        name = constant.TRANSLATION[lang][int(index)].split()[0]
        filename = os.path.join(constant.TRANSLATE_QURAN, lang, name+'.txt')
        with open(filename, encoding="utf-8") as file:
            txt = file.read()
        tempAyeh = ayeh
        if soure == 1:
            tempAyeh += 1
        start = re.search(f"{soure}\|{tempAyeh}\|", txt)
        end = re.search(f"{soure}\|{tempAyeh+1}\|", txt)
        
        if end != None:
            return txt[start.end():end.start()]
        else:
            end2 = re.search(f"{soure+1}\|{1}\|", txt)
            if end2 != None:
                return txt[start.end():end2.start()]
            else:
                return txt[start.end():].split('\n')[0]
    else:
        txt_traslation = []
        for names in constant.TRANSLATION[lang]:
            name = names.split()[0]
            filename = os.path.join(constant.TRANSLATE_QURAN, lang, name+'.txt')
            with open(filename, encoding="utf-8") as file:
                txt = file.read()
            start = re.search(f"{soure}\|{ayeh+1}\|", txt)
            end = re.search(f"{soure}\|{ayeh+2}\|", txt)
            if end != None:
                txt_traslation.append(txt[start.end():end.start()])
            else:
                txt_traslation.append(txt[start.end:])
        return txt_traslation


def print_all_translations():
    for key, value in constant.TRANSLATION.items() :
        for val in value:
            # val = val.split('(')[0].strip()
            val = val.split('(')[1].split(')')[0].strip()
            print (key, val)


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        good_files = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in good_files)
    return results

def get_hadiths(soure, ayeh, filter_number = 10):
    try:
        ids = requests.post('https://hadith.ai/get_hadith_content/get_ayah', json={"suraId": soure, "ayaId": ayeh}).json()
        hadiths = requests.post('https://hadith.ai/get_hadith_content/create_hadith', json={  "hid": ids[:filter_number], "out_type": "json"}).json()
        lst = []
        for i in hadiths:
            had = hadiths[i]
            text = had['narrators'] + '\n' + had['hadith'] + '\n' + had['translation_text']
            lst.append(text) 
        return lst
    except:
        print('Error to get hadiths')
        return None



def search_in_quran(text):
    """
     search in quran:
        - when # not in text => search text in all quran 
        - when # in text => search index ayeh

    Args:
        text (_type_): this is input for search in quran with multi ways

    Returns:
        _type_: return soure and ayeh
    """
    sor = None
    aye = None
    if '#' not in text:        
        rep = requests.post('https://hadith.ai/quranic_extraction/', json={"query": text, 'min_tok': 3, 'min_char': 3})        
        if rep.ok and rep.json()['output']['quran_id']:
            out = rep.json()['output']['quran_id'][0]
            #TODO : add ways -- now i use defaul 0
            sor, aye = out[0].split('##')
            sor, aye = int(sor), int(aye)
        else:
            print(rep)         

    else:
        if not bool(re.search('[ا-ی]', text)):
            sor, aye = text.split('#')
            sor, aye = int(sor), int(aye)
        else:
            soure_name, aye = text.split('#')
            aye = int(aye)
            sor = get_index_soure_from_name_soure(soure_name.strip())
    return sor, aye
    

def get_index_soure_from_name_soure(soure_name):
    if not str(soure_name).startswith('ال ') and str(soure_name).startswith('ال'):
        soure_name = soure_name[2:]
    rep = requests.post('https://hadith.ai/preprocessing/', json={"query": soure_name, "dediac": 'true'})
    if rep.ok:
        soure_name = rep.json()['output']

    soure = None
    for inx, output in enumerate(constant.AYEH_INDEX):
        if soure_name in output: 
            soure = inx + 1
    return soure                

def get_revelation_order(soure):
    df = pd.read_csv(constant.QURAN_ORDER)
    df.index = df['index']
    return df.loc[soure]['order_name']

def get_sourah_name_from_soure_index(soure):
    df = pd.read_csv(constant.QURAN_ORDER)
    df.index = df['index']
    return df.loc[soure]['soure']


def get_words_and_spaces(soure, ayeh):

    qSyntaxSemantics = []
    for i in range(1, 115):
        files = recursive_glob(constant.AYEH_SEMANTIC, f'{i}-*.json')
        if len(files) == 0:
            raise('data not downloaded')
        files.sort(key=lambda f: int(''. join(filter(str. isdigit, f))))
        qSyntaxSemantics.append(files)

    file = qSyntaxSemantics[soure - 1][ayeh - 1]
    
    with open(file, encoding="utf-8") as f:
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
    return words, spaces

def get_indexes_from_words(soure, ayeh):
    words, _ = get_words_and_spaces(soure, ayeh)
    indexes = dict()
    for inx, id in enumerate(words):
        indexes[id] = inx
    return indexes