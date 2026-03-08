import pandas as pd
from quranic_nlp import constant


def load_model():
    """Load and return the morphology CSV as a DataFrame."""
    morphology = pd.read_csv(constant.MORPHOLOGY)
    return morphology.fillna('')


def lemma(model, soure, ayeh):
    """Return per-token lemma dicts for the given verse."""
    if soure is None:
        return None
    data = model[(model['soure'] == soure - 1) & (model['ayeh'] == ayeh - 1)]
    output = []
    for lem_value in data['Lemma']:
        try:
            output.append({'lemma': lem_value if lem_value else ''})
        except Exception:
            output.append({})
    return output
