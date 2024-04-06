import pandas as pd
import json
import os
# from quranic_nlp import utils
import utils


def load_model():
    syntax_data = utils.recursive_glob(utils.AYEH_SYNTAX, '*.xlsx')
    syntax_data.sort()
    return syntax_data


def postagger(model, soure, ayeh):
    if soure == None:
        return None
    
    ## in the name of god 
    if soure == 1 and ayeh == 1:
        with open(os.path.join(utils.AYEH_SEMANTIC, '1-1.json'), encoding="utf-8") as f:
            data = json.load(f)
        data = data['Data']['ayeh']['node']['Data']
        output = []
        
        for i in data:
            out = dict()
            d = i['xml']
            out['pos'] = d.split('Pos')[1].split('\"')[1]
            output.append(out)
        return output
    
    
    file = model[soure - 1]
    df = pd.read_excel(file)

    # gb = df.groupby('Ayah')
    # gb = [gb.get_group(x) for x in gb.groups]
       
    if soure == 1:
        data = df[model['ayeh'] == ayeh - 2]
    #     data = gb[ayeh - 2]
    else:
        data = df[model['ayeh'] == ayeh - 2]
    #     data = gb[ayeh - 1]

    data.index = data['id']
    
    output = []
    for id in data['id'].values:
        out = dict()
        out['pos'] = data.loc[id]['data'].split('Pos')[1].split('\"')[1]
        output.append(out)
    return output
