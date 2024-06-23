import os
import pandas as pd

from src_code.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, input):

        #Object paths
        model_path = os.path.join('saved_obj', "model.pkl")
        processor_path = os.path.join('saved_obj', "preprocessor.pkl")

        #Load in the objects
        model = load_object(model_path)
        processor = load_object(processor_path)

        #Process data and predict 
        processed_input = processor.fit_transform(input)
        pred = model.predict(processed_input)

        return pred 
