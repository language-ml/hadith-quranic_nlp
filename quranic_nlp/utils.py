import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import requests
import fnmatch
import json
import os
import re

from quranic_nlp import constant


def get_sim_ayahs(soure, ayeh):
    """Return list of similar verse references (as 'surah#ayah' strings) for the given verse."""
    output = []
    with open(constant.SIMILARITY_AYAT, encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            so = int(parts[0][:-3])
            ay = int(parts[0][-3:])
            if soure == so and ay == ayeh:
                for part in parts[1:]:
                    ref, _ = part.split(':')
                    output.append(f"{int(ref[:-3])}#{int(ref[-3:])}")
    return output


def get_text(soure, ayeh):
    """Return the raw Quranic text for the given verse."""
    tree = ET.parse(constant.QURAN_XML)
    root = tree.getroot()
    for sura in root.iter('sura'):
        if sura.attrib.get('index') == str(soure):
            for aya in sura.iter('aya'):
                if aya.attrib.get('index') == str(ayeh):
                    return aya.attrib.get('text')
    return None


def get_translations(lang_input, soure, ayeh):
    """
    Return translation text for the given verse.

    Parameters
    ----------
    lang_input : str or None
        Either ``'<lang>#<index>'`` (e.g. ``'fa#1'``) for a single translator,
        or ``'<lang>'`` for all translators of that language.
    soure : int
        Surah (chapter) index (1-based).
    ayeh : int
        Verse index (1-based).
    """
    if not lang_input:
        return ''

    if '#' in lang_input:
        lang, index = lang_input.split('#')
        name = constant.TRANSLATION[lang][int(index)].split()[0]
        filepath = os.path.join(constant.TRANSLATE_QURAN, lang, name + '.txt')
        with open(filepath, encoding='utf-8') as f:
            txt = f.read()
        temp_ayeh = ayeh + 1 if soure == 1 else ayeh
        start = re.search(rf"{soure}\|{temp_ayeh}\|", txt)
        end = re.search(rf"{soure}\|{temp_ayeh + 1}\|", txt)
        if start is None:
            return ''
        if end is not None:
            return txt[start.end():end.start()]
        end2 = re.search(rf"{soure + 1}\|1\|", txt)
        if end2 is not None:
            return txt[start.end():end2.start()]
        return txt[start.end():].split('\n')[0]
    else:
        translations = []
        for name_entry in constant.TRANSLATION[lang_input]:
            name = name_entry.split()[0]
            filepath = os.path.join(constant.TRANSLATE_QURAN, lang_input, name + '.txt')
            with open(filepath, encoding='utf-8') as f:
                txt = f.read()
            start = re.search(rf"{soure}\|{ayeh + 1}\|", txt)
            end = re.search(rf"{soure}\|{ayeh + 2}\|", txt)
            if start is None:
                translations.append('')
            elif end is not None:
                translations.append(txt[start.end():end.start()])
            else:
                translations.append(txt[start.end():].split('\n')[0])
        return translations


def print_all_translations():
    """Print all available translation languages and translator names."""
    for lang, entries in constant.TRANSLATION.items():
        for entry in entries:
            if '(' in entry and ')' in entry:
                translator = entry.split('(')[1].split(')')[0].strip()
            else:
                translator = entry.strip()
            print(lang, translator)


def recursive_glob(treeroot, pattern):
    """Recursively find files matching *pattern* under *treeroot*."""
    results = []
    for base, _, files in os.walk(treeroot):
        for fname in fnmatch.filter(files, pattern):
            results.append(os.path.join(base, fname))
    return results


def get_hadiths(soure, ayeh, filter_number=10):
    """
    Fetch related hadiths for the given verse from the hadith.ai API.

    Returns a list of hadith strings, or ``None`` on network error.
    """
    try:
        ids_resp = requests.post(
            'https://hadith.ai/get_hadith_content/get_ayah',
            json={'suraId': soure, 'ayaId': ayeh},
            timeout=10,
        )
        ids_resp.raise_for_status()
        ids = ids_resp.json()

        hadiths_resp = requests.post(
            'https://hadith.ai/get_hadith_content/create_hadith',
            json={'hid': ids[:filter_number], 'out_type': 'json'},
            timeout=10,
        )
        hadiths_resp.raise_for_status()
        hadiths = hadiths_resp.json()

        result = []
        for hadith in hadiths.values():
            text = (
                hadith.get('narrators', '') + '\n'
                + hadith.get('hadith', '') + '\n'
                + hadith.get('translation_text', '')
            )
            result.append(text)
        return result
    except Exception as exc:
        print(f'Could not fetch hadiths: {exc}')
        return None


def search_in_quran(text):
    """
    Resolve a verse reference to (surah_index, ayah_index).

    Accepted input formats:

    1. ``'<surah_number>#<ayah_number>'`` — e.g. ``'1#1'``
    2. ``'<surah_name>#<ayah_number>'``   — e.g. ``'حمد#1'``
    3. Free Arabic text to search        — e.g. ``'رب العالمین'``

    The last two formats require internet access.

    Returns
    -------
    tuple[int, int]
        ``(surah_index, ayah_index)``
    """
    if '#' not in text:
        resp = requests.post(
            'https://quranic-api.qcri.org/qextract_ayah',
            headers={'accept': 'application/json', 'Content-Type': 'application/json'},
            json={
                'query': text,
                'target_verses': '',
                'min_token_num': 2,
                'min_char_len_prop': 50,
                'consecutive_verses_priority': False,
                'custom_bert_token_threshold': 0,
                'use_lm': True,
                'inexact_match': True,
                'use_rule_based': True,
                'detailed_output': True,
            },
            timeout=30,
        )
        if resp.ok:
            output = resp.json().get('output', {})
            quran_ids = (
                output.get('regex_qe', {}).get('quran_id') or
                output.get('inexact_match', {}).get('quran_id')
            )
            if quran_ids and quran_ids[0]:
                soure, ayeh = quran_ids[0][0].split('##')
                return int(soure), int(ayeh)
        raise ValueError(f'Verse not found for query: {text!r}')

    if not bool(re.search('[ا-ی]', text)):
        soure, ayeh = text.split('#')
        return int(soure), int(ayeh)

    soure_name, ayeh_str = text.split('#')
    ayeh = int(ayeh_str)
    soure = get_index_soure_from_name_soure(soure_name.strip())
    return soure, ayeh


def get_index_soure_from_name_soure(soure_name):
    """Return the surah index (1-based) for a given surah name."""
    if not soure_name.startswith('ال ') and soure_name.startswith('ال'):
        soure_name = soure_name[2:]
    try:
        resp = requests.post(
            'https://hadith.ai/preprocessing/',
            json={'query': soure_name, 'dediac': 'true'},
            timeout=10,
        )
        if resp.ok:
            soure_name = resp.json()['output']
    except Exception:
        pass

    for idx, names in enumerate(constant.AYEH_INDEX):
        if soure_name in names:
            return idx + 1
    raise ValueError(f'Surah name not found: {soure_name!r}')


def get_revelation_order(soure):
    """Return the revelation order for the given surah index."""
    df = pd.read_csv(constant.QURAN_ORDER)
    df.index = df['index']
    return df.loc[soure]['order_name']


def get_sourah_name_from_soure_index(soure):
    """Return the surah name for the given surah index."""
    df = pd.read_csv(constant.QURAN_ORDER)
    df.index = df['index']
    return df.loc[soure]['soure']


def get_words_and_spaces(soure, ayeh):
    """Return (words, spaces) arrays for the given verse from the semantic JSON data."""
    qSyntaxSemantics = []
    for i in range(1, 115):
        files = recursive_glob(constant.AYEH_SEMANTIC, f'{i}-*.json')
        if not files:
            raise FileNotFoundError(
                f'Data not found for surah {i}. '
                'Run `quranic_data` to download required data files.'
            )
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        qSyntaxSemantics.append(files)

    with open(qSyntaxSemantics[soure - 1][ayeh - 1], encoding='utf-8') as f:
        data = json.load(f)

    nodes = pd.DataFrame(data['Data']['ayeh']['node']['Data'])
    nodes.index = nodes['id']
    nodes = nodes.sort_index()

    words = nodes['Word'].values
    spaces = np.ones(len(words), dtype=bool)
    for idx, xml_val in enumerate(
        nodes['xml'].apply(
            lambda x: x.split('Seq')[1].split('"')[1] if x is not None else None
        ).values
    ):
        if xml_val is not None and int(xml_val) == 2:
            spaces[idx - 1] = False
    return words, spaces


def get_indexes_from_words(soure, ayeh):
    """Return a dict mapping word text to its index in the verse."""
    words, _ = get_words_and_spaces(soure, ayeh)
    return {word: idx for idx, word in enumerate(words)}
