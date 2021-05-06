import os


def exists(path):
    return os.path.exists(path)


def ensure_folder(folder):
    if not exists(folder):
        os.makedirs(folder)
        return True

    return False
