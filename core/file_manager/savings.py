import json
import pickle


def save_json(json_dict, filepath):
    with open(filepath, 'w') as j:
        json.dump(json_dict, j, indent=4)
        j.close()


def pickle_save(object, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(object, f)
        f.close()