import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def splitData(X,y,test=0.3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=12)
    print('X_train.shape = ', X_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('X_test.shape = ', X_test.shape)
    print('y_test.shape = ', y_test.shape)
    return X_train, X_test, y_train, y_test

def readCsv(file,header=None,sep=','):
    return pd.read_csv(file,header=header, sep=sep)

def descrpitDf(df):
    print('describe:', df.describe().transpose())
    print('head:', df.head())
    print('dtypes:', df.dtypes)
    print('columns:', df.columns)
    print('shape:', df.shape)
    #print('Class labels', np.unique(df['Class label']))
    
def getGeoData(file):
    df = readCsv(file)
    #descrpitDf(df)
    return df


'''
def preProcessData(df):
    nrow, ncol = df.shape
    le = LabelEncoder()
    #print(df.columns)
    for i in range(ncol):
        if df.dtypes[i] == object:
            column = df.columns[i]
            #print(i,column)
            le.fit(df[column])
            le_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            #print(column, le_mapping)
            df[column] = df[column].map(le_mapping)

    return df
'''

def main():
    pass

if __name__ == "__main__":
    main()
