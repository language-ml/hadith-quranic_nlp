import pandas as pd
from quranic_nlp import constant


def load_model():
    """Load and return the morphology CSV as a DataFrame."""
    morphology = pd.read_csv(constant.MORPHOLOGY)
    return morphology.fillna('')


def root(model, soure, ayeh):
    """Return per-token root dicts for the given verse."""
    if soure is None:
        return None
    data = model[(model['soure'] == soure) & (model['ayeh'] == ayeh)]
    output = []
    for root_value in data['Root']:
        try:
            output.append({'root': root_value if root_value else ''})
        except Exception:
            output.append({})
    return output
