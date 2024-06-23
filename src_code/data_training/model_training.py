import os
from dataclasses import dataclass

from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score
from sklearn.svm import SVC

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("saved_models", "model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_cofig = ModelTrainerConfig()
    
    def model_trainer(self, train_arr, test_arr):
        """
        This is for initiating the training of the model
        """

        #Split the training and test 
        X_train, y_train, X_test, y_test = (
            train_arr[:,:-1],
            train_arr[:,-1:].reshape(-1),
            test_arr[:,:-1],
            test_arr[:,-1:].reshape(-1)
        )


        #Initiate the model 
        model = SVC()
        model.fit(X_train, y_train)

        #Predict the training and test set 
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        f1_train, accuracy_train, precision_train, recall_train = self.evaluation(y_train, y_train_pred)
        f1_test, accuracy_test, precision_test, recall_test = self.evaluation(y_test, y_test_pred)

        print('Model performance for Training set')
        print("- F1 score: {:.4f}".format(f1_train))
        print("- Accuracy: {:.4f}".format(accuracy_train))
        print("- Precision: {:.4f}".format(precision_train))
        print("- Recall: {:.4f}".format(recall_train))

        print('----------------------------------')
    
        print('Model performance for Test set')
        print("- F1 score: {:.4f}".format(f1_test))
        print("- Accuracy: {:.4f}".format(accuracy_test))
        print("- Precision: {:.4f}".format(precision_test))
        print("- Recall: {:.4f}".format(recall_test))

        print('='*35)
        print('\n')

    def evaluation(self,y_true, y_pred):
        """
        This function is for evaluating the model

        """
        f1 = f1_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        return f1, accuracy, precision, recall
