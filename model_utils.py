import pickle

def save_model(model, save_path):
    '''
    model - A model object
    save_path - path to save  the model
    '''
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(load_path):
    '''
    load_path -- path to load the model
    '''
    with open(load_path, 'rb') as f:
        model = pickle.load(f)
    return model