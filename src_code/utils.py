import os
import dill
import pickle
import dataclasses as dataclass

def save_object(file_path, obj):
    """
    To save the required object
    """
    
    dir_path = os.path.dirname(file_path)

    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

def load_object(file_path):
    """
    To load in the required object 
    """
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)

"""
@dataclass 
class Classes:
    def __init__(self, classes):
        self.classes = classes
        self.count = len(classes)
"""
