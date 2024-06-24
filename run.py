from src_code.data_training.data_transform import DataTransform
from src_code.data_training.model_training import ModelTrainer

if __name__ == "__main__":

    # Import and transform the data into the test train split 
    dt = DataTransform()
    train, test = dt.data_transform()

    #Train and output the evluation metrics
    model = ModelTrainer()
    model.model_trainer(train_arr=train, test_arr=test)