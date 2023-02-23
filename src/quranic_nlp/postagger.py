import pandas as pd
from quranic_nlp import utils
# import utils


def load_model():
    syntax_data = utils.recursive_glob(utils.AYEH_SYNTAX, '*.xlsx')
    syntax_data.sort()
    return syntax_data


def postagger(model, soure, ayeh):

    file = model[soure - 1]
    df = pd.read_excel(file)
    gb = df.groupby('Ayah')
    gb = [gb.get_group(x) for x in gb.groups]
    data = gb[ayeh - 1]

    data.index = data['id']

    output = []
    for id in data['id'].values:
        out = dict()
        out['pos'] = data.loc[id]['data'].split('Pos')[1].split('\"')[1]
        output.append(out)
    return output
