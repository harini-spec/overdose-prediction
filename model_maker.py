import random
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import os.path
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC



CSV_FILE_PATH       = 'drug_consumption.csv'
MODEL_PICKLE_PATH   = 'model_reg.pkl'
# VECTOR_PICKLE_PATH  = 'count_vect_5.pkl'

class MLSingleton:

    __instance = None

    def __init__(self):
        """ Virtually private constructor. """
        if MLSingleton.__instance != None:
            raise Exception("This class is a singleton!")

        self.model_name         = MODEL_PICKLE_PATH

        try:
            # Load models from pickle
            self.model              = pickle.load(open(self.model_name, 'rb'))

        except FileNotFoundError as file_err:
            print('Error while loading pickles : ', file_err)

            # Train model
            self.train_model()

        # print('Loaded model and vectorizer')

        MLSingleton.__instance = self

    def train_model(self, force = False):

        if(not force):

            print(f'{MODEL_PICKLE_PATH} available? : ', os.path.isfile(MODEL_PICKLE_PATH))
            if(os.path.isfile(MODEL_PICKLE_PATH)):
                print('Skipped as the file is already there')
                return

        print('Training fresh model and dumping pickle')

        df = pd.read_csv(CSV_FILE_PATH)
        copy_df = df.copy()      

        columns = ['Alcohol','Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack',
           'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms','Nicotine', 'Semer', 'VSA']

        cp = ['User_Alcohol','User_Amphet', 'User_Amyl', 'User_Benzos', 'User_Caff', 'User_Cannabis', 'User_Choc', 'User_Coke', 'User_Crack',
           'User_Ecstasy', 'User_Heroin', 'User_Ketamine', 'User_Legalh', 'User_LSD', 'User_Meth', 'User_Mushrooms','User_Nicotine', 'User_Semer', 'User_VSA']

        for column in columns:
            le = LabelEncoder()
            copy_df[column] = le.fit_transform(copy_df[column])

        for i in range(len(columns)):
            copy_df.loc[((copy_df[columns[i]]==0) | (copy_df[columns[i]]==1)),cp[i]] = 'Non-user'
            copy_df.loc[((copy_df[columns[i]]==2) | (copy_df[columns[i]]==3) | (copy_df[columns[i]]==4) | (copy_df[columns[i]]==5) | (copy_df[columns[i]]==6)),cp[i]] = 'User'

        for column in copy_df.columns:
            le = LabelEncoder()
            copy_df[column] = le.fit_transform(copy_df[column])

        feature_col_names = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore',
       'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS']
        predicted_class_names = ['User_Benzos']

        X = copy_df[feature_col_names].values
        y = copy_df[predicted_class_names].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        # vectorizer = CountVectorizer()
        # X = vectorizer.fit_transform(ex)

        svm         =       SVC(kernel="linear", C=1,random_state=0)
        clf         =       svm.fit(X_train, y_train.ravel())
        # pipe_lr.predict([ex1])
        
        # local_model                      = linear_model.LogisticRegression()
        # clf                              = local_model.fit(x_train,y_train)

        pickle.dump(clf, open(MODEL_PICKLE_PATH, 'wb'))
        print('Dumped model into ', MODEL_PICKLE_PATH)

        # as fresh model created apply them
        self.model              = pickle.load(open(self.model_name, 'rb'))

    @staticmethod 
    def getInstance():
        """ Static access method. """
        if MLSingleton.__instance == None:
            MLSingleton()

        return MLSingleton.__instance

    def classify_single_predict(self, ex):

        predicted_value = self.model.predict(ex)

        return predicted_value


def predict_usage(data):

    listObj = [data]

    MLSingletonObj = MLSingleton.getInstance()
    user = MLSingletonObj.classify_single_predict(listObj)

    # print(emotion)

    return user

def startpy():

    print(predict_usage([2,1,4,0,2,41,9,3,30,26,3,1]))
    
    # s = MLSingleton.getInstance()
    # # train the model
    # # s.train_model(force=True)
    # ex = ["Rape is a serious crime and people have to take a careful note on it", "Movie contains lots of gore scenes.", "He need justice because it is completely unfair to him", "Movie contains lots of gore scenes. I particularly did not feel good about it", "Ted Bundy was a nasty guy. He needs to be punished"]
    # # ex = ["Rape is a serious crime and people have to take a careful note on it"]
    # print(s.classify_single_predict(ex))


if __name__ == "__main__":
    startpy()