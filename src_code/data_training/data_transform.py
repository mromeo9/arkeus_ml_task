import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

@dataclass
class DataTransform:
    def __init__(self):
        pass


    def data_transform(self):
        """
        This function is used for transforming the data to be used for training and testing
        """

        df = pd.read_csv('data\data.csv') #Import the data

        X = df.drop(columns=['Categories'], axis=1) #Split the inputs from the ouputs 
        Y = df['Categories']

        #Apply preprocessing step
        y_processor = LabelEncoder()
        y = y_processor.fit_transform(Y)

        x_processor = StandardScaler()
        x = x_processor.fit_transform(X)

        #Split the data into training and testing 
        X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=9)

        train_array = np.c_[X_train, y_train]
        test_array = np.c_[X_test, y_test]
        return train_array, test_array

if __name__ == "__main__":
    dt = DataTransform()
    train, test = dt.data_transform()
    print(train)