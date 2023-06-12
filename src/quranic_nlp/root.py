import pandas as pd
import json
import os
from quranic_nlp import utils
# import utils

def load_model():
    morphologhy = pd.read_csv(utils.MORPHOLOGY)
    morphologhy = morphologhy.fillna('')
    return morphologhy

def root(model, soure, ayeh):
    if soure == None:
        return None
    
    gb_soure = model.groupby('soure')
    gb_soure = [gb_soure.get_group(x) for x in gb_soure.groups]
    df = gb_soure[soure - 1]

    gb = df.groupby('ayeh')
    gb = [gb.get_group(x) for x in gb.groups]
    data = gb[ayeh - 1]

    output = []
    for root in data['Root']:
        out = dict()
        try:
            if root:
                out['root'] = root
            else:
                out['root'] = ''
            output.append(out)
        except:
            output.append(out)

    return output
