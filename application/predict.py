import gc

import numpy as np

from preprocessing import preprocess_inputs
import models


def predict_fn(*input_args) -> str:
    X = np.array([preprocess_inputs([*input_args])])
    model = input_args[-1]
    model_obj = None
    match model:
        case 'Decision Tree':
            model_obj = models.load_decision_tree()
        case 'Random Forest':
            model_obj = models.load_random_forest()
        case 'Bagging':
            model_obj = models.load_bagging()
    outcome: float = model_obj.predict(X)[0][0]
    result = ''
    match outcome:
        case -1.0:
            result = 'Dropout'
        case 1.0:
            result = 'Graduated'
        # Should be unreachable here
        case _:
            result = 'Enrolled'

    # Force garbage collect, not sure if it really helps
    del model_obj
    gc.collect()

    return result
