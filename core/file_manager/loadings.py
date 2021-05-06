import json
import pickle


def load_json(json_path):
    with open(json_path, 'r') as j:
        json_dict = json.load(j)
        j.close()

    return json_dict


def pickle_load(filepath):
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
        f.close()

    return obj
