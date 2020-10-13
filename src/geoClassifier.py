#unicode Python3 Steven
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score

from genDataSet import writeToCsv
from getDataSet import getGeoData,splitData,descrpitDf
from modelClassifierCreate import *
from visualClustering import visualClusterResult

def pcaData(data,N=100):
    print('pca before, data shape=',data.shape)
    fit = PCA(n_components=N).fit(data)
    print('n_components_:',fit.n_components_)
    print('n_features_:',fit.n_features_)
    print('n_samples_:',fit.n_samples_)
    print('explained_variance_ratio_:',fit.explained_variance_ratio_)
    print('singular_values_:',fit.singular_values_)
    
    data = fit.transform(data)
    print('pca after, data shape=',data.shape)
    return data

def preprocessingData(data):
    scaler = MinMaxScaler()# StandardScaler() #
    scaler.fit(data)
    data = scaler.transform(data)
    #print('\n',data[:5])    
    #print('scaler=',data[:5])
    return data
    
def preDataSet_GSE114783():
    file = r'.\data\GSE114783\GSE114783.csv'
    df = getGeoData(file).T
    descrpitDf(df)
    
    df.columns = df.iloc[0]
    df = df[1:]

    print('Before:\n', df.head())
    print('Class labels:', np.unique(df['Type']))
    class_mapping = {
        'healthy control':              1,
        'chronic hepatitis B':          2,
        'hepatitis B virus carrier':    3,
        'liver cirrhosis':              4,
        'hepatocellular carcinoma':     5,
    }
    df['Type'] = df['Type'].map(class_mapping)
    print('After:\n', df.head())
    
    if 0:
        N=500
        name = 'GSE114783ok_'+str(N)+'.txt'
        writeToCsv(df.iloc[:, -1*N:], r'.\data\GSE114783' + '\\' + name, sep=' ', index=None, header=None)

    X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
    
    X = pcaData(X,N=30)
    X = preprocessingData(X) #scaler
    return splitData(X, y, test=0.2)

def preDataSet_GSE25097():
    file = r'.\data\GSE25097\GSE25097.csv'
    df = getGeoData(file).T
    descrpitDf(df)
    
    df.columns = df.iloc[0]
    df = df[1:]

    print('Before:\n', df.head())
    print('Class labels:', np.unique(df['Type']))
    class_mapping = {
        'healthy':          1,
        'cirrhotic':        2,
        'non_tumor':        3,
        'tumor':            4,
    }
    df['Type'] = df['Type'].map(class_mapping)
    print('After:\n', df.head())
    
    if 0:
        N=500
        name = 'GSE114783ok_'+str(N)+'.txt'
        writeToCsv(df.iloc[:, -1*N:], r'.\data\GSE114783' + '\\' + name, sep=' ', index=None, header=None)

    X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
    
    X = pcaData(X,N=30) #PCA
    X = preprocessingData(X)
    return splitData(X, y, test=0.2)
    
def createModels():
    models = []
    
    models.append(createDecisionTree(3))
    models.append(createMLPClassifier())
    models.append(createRandomForestClf())
    #models.append(createLogisticRegression())
    models.append(createRidgeClassifier())
    models.append(createSGDClassifier())
    models.append(createSVM_svc())
    #models.append(createSVM_NuSVC())
    models.append(createSVM_LinearSVC())
    models.append(createKNeighborsClassifier())
    # models.append(createRadiusNeighborsClassifier())
    # models.append(createNearestCentroid())
    # models.append(createGaussianProcessClassifier())
    # models.append(createGaussianNB())
    # models.append(createMultinomialNB())
    # models.append(createComplementNB())
    # models.append(createBernoulliNB())
    # #models.append(createCategoricalNB())
    # models.append(createLabelPropagation())
    
    #print(models)
    return models

def train(): 
    X_train, X_test, y_train, y_test = preDataSet_GSE25097() #preDataSet_GSE114783()

    models = createModels()

    print('\n----------------training start--------------')
    for model,modelName in models:     
        try:  
            t = time()
            model.fit(X_train, np.ravel(y_train, order='C'))
            
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)
            tt = round(time()-t, 4)
            
            acc_train = accuracy_score(y_train, pred_train)
            acc_test = accuracy_score(y_test, pred_test)
            print(modelName, ", run in %.2fs" % (tt), ", Accuracy on Train:", round(acc_train,6),", Accuracy on Test:", round(acc_test,6))
        except:
            print(modelName,', failed!')
            continue
        
        #visualization
        visualClusterResult(X_train,pred_train,modelName)
        #break    
    
    print('----------------training end--------------\n')
    # print("\n run in %.2fs" % (tt))
    # #plotCLusteringResult(rawData,model.labels_)
    
'''
----------------training start----GSE114783----------
DecisionTreeClassifier , run in 0.43s , Accuracy score: 0.454545
MLPClassifier , run in 0.36s , Accuracy score: 0.272727
RandomForestClassifier , run in 0.21s , Accuracy score: 0.545455
RidgeClassifier , run in 0.21s , Accuracy score: 0.454545
SGDClassifier , run in 0.27s , Accuracy score: 0.454545
SVC pipline , run in 0.40s , Accuracy score: 0.454545
LinearSVC pipline , run in 2.10s , Accuracy score: 0.363636
KNeighborsClassifier , run in 0.24s , Accuracy score: 0.272727
RadiusNeighborsClassifier , failed!
NearestCentroid , run in 0.20s , Accuracy score: 0.363636
GaussianProcessClassifier , run in 0.42s , Accuracy score: 0.272727
GaussianNB , run in 0.23s , Accuracy score: 0.454545
MultinomialNB , run in 0.21s , Accuracy score: 0.363636
ComplementNB , run in 0.21s , Accuracy score: 0.454545
BernoulliNB , run in 0.22s , Accuracy score: 0.272727
CategoricalNB , failed!
D:\Python36\lib\site-packages\sklearn\semi_supervised\_label_propagation.py:205: RuntimeWarning: invalid value encountered in true_divide
  probabilities /= normalizer
LabelPropagation , run in 0.21s , Accuracy score: 0.0
----------------training end--------------

----------------training start-----GSE25097--pca=30-------
DecisionTreeClassifier , run in 0.01s , Accuracy on Train: 0.997753 , Accuracy on Test: 0.928571
MLPClassifier , run in 0.07s , Accuracy on Train: 0.449438 , Accuracy on Test: 0.383929
RandomForestClassifier , run in 0.02s , Accuracy on Train: 0.995506 , Accuracy on Test: 0.910714
RidgeClassifier , run in 0.00s , Accuracy on Train: 0.964045 , Accuracy on Test: 0.955357
SGDClassifier , run in 0.01s , Accuracy on Train: 0.970787 , Accuracy on Test: 0.946429
SVC , run in 0.02s , Accuracy on Train: 0.907865 , Accuracy on Test: 0.901786
LinearSVC , run in 0.02s , Accuracy on Train: 0.977528 , Accuracy on Test: 0.982143
KNeighborsClassifier , run in 0.03s , Accuracy on Train: 0.959551 , Accuracy on Test: 0.857143
NearestCentroid , run in 0.00s , Accuracy on Train: 0.934831 , Accuracy on Test: 0.9375
GaussianProcessClassifier , run in 0.50s , Accuracy on Train: 0.948315 , Accuracy on Test: 0.946429
GaussianNB , run in 0.00s , Accuracy on Train: 0.934831 , Accuracy on Test: 0.919643
MultinomialNB , run in 0.00s , Accuracy on Train: 0.898876 , Accuracy on Test: 0.883929
ComplementNB , run in 0.00s , Accuracy on Train: 0.889888 , Accuracy on Test: 0.892857
BernoulliNB , run in 0.00s , Accuracy on Train: 0.485393 , Accuracy on Test: 0.383929
LabelPropagation , run in 0.01s , Accuracy on Train: 1.0 , Accuracy on Test: 0.892857
----------------training end--------------

----------------training start-------GSE25097 no pca-------
DecisionTreeClassifier , run in 10.65s , Accuracy on Train: 0.997753 , Accuracy on Test: 0.928571
MLPClassifier , run in 2.61s , Accuracy on Train: 0.469663 , Accuracy on Test: 0.526786
RandomForestClassifier , run in 0.47s , Accuracy on Train: 0.997753 , Accuracy on Test: 0.946429
RidgeClassifier , run in 0.53s , Accuracy on Train: 1.0 , Accuracy on Test: 0.964286
SGDClassifier , run in 2.35s , Accuracy on Train: 1.0 , Accuracy on Test: 0.973214
SVC , run in 21.52s , Accuracy on Train: 0.878652 , Accuracy on Test: 0.892857
LinearSVC , run in 13.75s , Accuracy on Train: 1.0 , Accuracy on Test: 0.982143
KNeighborsClassifier , run in 19.16s , Accuracy on Train: 0.907865 , Accuracy on Test: 0.821429
NearestCentroid , run in 0.20s , Accuracy on Train: 0.914607 , Accuracy on Test: 0.919643
GaussianProcessClassifier , run in 85.02s , Accuracy on Train: 1.0 , Accuracy on Test: 0.526786
GaussianNB , run in 1.25s , Accuracy on Train: 0.959551 , Accuracy on Test: 0.928571
MultinomialNB , run in 0.10s , Accuracy on Train: 0.892135 , Accuracy on Test: 0.910714
ComplementNB , run in 0.10s , Accuracy on Train: 0.910112 , Accuracy on Test: 0.910714
BernoulliNB , run in 0.41s , Accuracy on Train: 0.676404 , Accuracy on Test: 0.383929
D:\Python36\lib\site-packages\sklearn\semi_supervised\_label_propagation.py:205: RuntimeWarning: invalid value encountered in true_divide
  probabilities /= normalizer
LabelPropagation , run in 0.55s , Accuracy on Train: 1.0 , Accuracy on Test: 0.026786
----------------training end--------------
'''

def main():
    train()
    
if __name__ == "__main__":
    main()
