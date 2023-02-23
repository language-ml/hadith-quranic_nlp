import pandas as pd
import json
from quranic_nlp import utils
# import utils


def load_model():
    qSyntaxSemantics = []
    for i in range(1, 115):
        files = utils.recursive_glob(utils.AYEH_SEMANTIC, f'{i}-*.json')
        files.sort(key=lambda f: int(''. join(filter(str. isdigit, f))))
        # datas = []
        # for file in files:
        #     with open(file) as f:
        #         data = json.load(f)
        # datas.append(data)
        qSyntaxSemantics.append(files)
    return qSyntaxSemantics


def depparser(model, soure, ayeh):

    file = model[soure - 1][ayeh - 1]
    with open(file) as f:
        data = json.load(f)

    realations = data['Data']['relationName']['Data']
    realations = pd.DataFrame(realations)
    realations.index = realations["Id"]

    edges = data['Data']['ayeh']['edge']['Data']
    edges = pd.DataFrame(edges)
    if len(edges) == 0:
        return []
    edges.index = edges["to"]

    nodes = data['Data']['ayeh']['node']['Data']
    nodes = pd.DataFrame(nodes)
    nodes.index = nodes["id"]
    nodes = nodes.sort_index()

    output = []

    for id in nodes['id'].values:
        out = dict()
        if id in edges['to']:

            if (type(edges.loc[id]) == pd.DataFrame):
                for inx, edge in edges.loc[id].iterrows():
                    out['rel'] = realations.loc[edge['relationId']]['name']
                    out['head'] = nodes.loc[edge['from']]['Word']
                    out['arc'] = edge['arrow']

            else:
                out['rel'] = realations.loc[edges.loc[id]['relationId']]['name']
                out['head'] = nodes.loc[edges.loc[id]['from']]['Word']
                out['arc'] = edges.loc[id]['arrow']

        output.append(out)
    return output
