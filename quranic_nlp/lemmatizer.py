import pandas as pd
from quranic_nlp import utils
# import utils


def load_model():
    morphologhy = pd.read_csv(utils.MORPHOLOGY)
    morphologhy = morphologhy.fillna('')
    return morphologhy


def lemma(model, soure, ayeh):
    if soure == None:
        return None
    
    data = model[(model['soure'] == soure - 1) & (model['ayeh'] == ayeh - 1)]
    # gb_soure = model.groupby('soure')
    # gb_soure = [gb_soure.get_group(x) for x in gb_soure.groups]
    # df = gb_soure[soure - 1]

    # gb = df.groupby('ayeh')
    # gb = [gb.get_group(x) for x in gb.groups]
    # data = gb[ayeh - 1]

    output = []
    for lemma in data['Lemma']:
        out = dict()
        try:
            if lemma:
                out['lemma'] = lemma
            else:
                out['lemma'] = ''
            output.append(out)
        except:
            output.append(out)

    return output
