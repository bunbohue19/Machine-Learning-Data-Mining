from pathlib import Path
import pickle

from .ensemble import *

def __load_model(filename: str) -> any:
    path = Path(__file__).parent / filename
    try:
        with path.open('rb') as file:
            obj = pickle.load(file)
            return obj
    except Exception as e:
        print(f"Error occurred while loading the object: {str(e)}")
        return None

def load_decision_tree():
    return __load_model('./decisiontree.pkl')

def load_bagging():
    return __load_model('./bagging.pkl')

def load_random_forest():
    return __load_model('./randomforest.pkl')
