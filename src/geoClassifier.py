#unicode Python3 Steven
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sn

from genDataSet import writeToCsv
from getDataSet import getGeoData,splitData,descrpitDf
from modelClassifierCreate import *
from visualClustering import visualClusterResult
from featureSelection import *

def pcaData(data, y, N=100):
    print('pca before, data shape=',data.shape)
    N = min(N,data.shape[0])
    fit = PCA(n_components=N).fit(data)
    print('n_components_:',fit.n_components_)
    print('n_features_:',fit.n_features_)
    print('n_samples_:',fit.n_samples_)
    print('explained_variance_ratio_:',fit.explained_variance_ratio_)
    print('singular_values_:',fit.singular_values_)
    
    data = fit.transform(data)
    print('pca after, data shape=',data.shape)
    return data,y

def preprocessingData(data):
    scaler = MinMaxScaler()# StandardScaler() #
    scaler.fit(data)
    data = scaler.transform(data)
    #print('\n',data[:5])    
    #print('scaler=',data[:5])
    return data
    
def featureSelect(X,y,N=30):
    return FeatureExtractChiSquare(X,y,N=N)
    #return FeatureExtract_RFE(X,y,N=N)
    # return FeatureExtract_ETC(X,y,N=N)
    #return pcaData(X,y,N=N)

def preDataSet_GSE114783():
    file = r'..\data\GSE114783\GSE114783.csv'
    df = getGeoData(file).T
    descrpitDf(df)
    
    #df.columns = df.iloc[0]
    #df = df[1:]
    #df.rename(columns=df.iloc[0], inplace = True)
    #df.drop(df.index[0], inplace = True)
    #eaders = df.iloc[0]
    #df  = pd.DataFrame(df.values[1:], columns=headers)
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0)).reset_index(drop=True)
    df.columns.name = None

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

    return df
    # X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
    
    # X, y = pcaData(X,y,N=30)
    # X = preprocessingData(X) #scaler
    # return splitData(X, y, test=0.2)

def statisticData(labels,df):
    def autolabel(ax,rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height().round(2)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
    stat = {}
    for i in labels:
        da = df[df['Type'] == i]
        print(i, da.shape)
        stat[i] = [da.shape[0]]
       
    print('stat=',stat) 
    dfStat = pd.DataFrame.from_dict(stat)
    print('dfStat=\n',dfStat)
    '''start plot'''
    fontsize = 8
    title = 'GSE25097 sample statistics'
    # ax = dfStat.plot(kind='bar',y=None)
    # ax.set_title(title,fontsize=fontsize)
    # ax.legend(fontsize=fontsize)
    # plt.setp(ax.get_xticklabels(), rotation=30, ha="right",fontsize=fontsize)
    # plt.setp(ax.get_yticklabels(),fontsize=fontsize)
    # #plt.subplots_adjust(left=0.30, bottom=None, right=0.98, top=None, wspace=None, hspace=None)   
    # #plt.savefig(str(i+1)+'.png')
    # plt.show()
    plt.rcParams.update({'font.size': fontsize})
    ax = plt.subplot(1,1,1)
    ax.set_title('',fontsize=fontsize)
    rect = ax.bar(dfStat.columns, dfStat.iloc[0,:])
    autolabel(ax,rect)
    plt.ylabel('Numbers')
    plt.show()
    
def filterData(labels, df, selectDict):
    dataDict = {}
    for i in labels:
        dataDict[i] = df[df['Type'] == i]
    
    # for i in  dataDict:
    #     print(i, dataDict[i].shape)
    
    selRes = pd.DataFrame()
    for key,value in selectDict.items():
        #print(key,value)
        sel = dataDict[key][:value]
        selRes = pd.concat([selRes,sel])
    print('selRes.shape=',selRes.shape)
    #print(selRes)
    return selRes
    
def preDataSet_GSE25097(filter=False):
    file = r'..\data\GSE25097\GSE25097.csv'
    df = getGeoData(file).T
    descrpitDf(df)

    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0)).reset_index(drop=True)
    df.columns.name = None

    print('Before:\n', df.head())
    labels = np.unique(df['Type'])
    
    #statisticData(labels,df)
    if filter:
        selectDict={ 'healthy':6, 'cirrhotic':6, 'non_tumor':6, 'tumor':6}
        df = filterData(labels,df,selectDict)
    
    print('Class labels:',labels)
    class_mapping = {
        'healthy':          1,  #6
        'cirrhotic':        2,  #40
        'non_tumor':        3,  #243
        'tumor':            4,  #268
    }
    df['Type'] = df['Type'].map(class_mapping)
    print('After:\n', df.head())
    
    if 0:
        N=500
        name = 'GSE114783ok_'+str(N)+'.txt'
        writeToCsv(df.iloc[:, -1*N:], r'..\data\GSE114783' + '\\' + name, sep=' ', index=None, header=None)

    return df
    
def createModels():
    models = []
    
    models.append(createDecisionTree(3))
    #models.append(createMLPClassifier())
    models.append(createRandomForestClf())
    #models.append(createLogisticRegression())
    #models.append(createRidgeClassifier())
    #models.append(createSGDClassifier())
    #models.append(createSVM_svc())
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

def confusionMatrix(clf,predictions, targets):
    def plotConfusionMatrix(matrix, classes=['1','2','3','4']):
        plt.rcParams['savefig.dpi'] = 300
        df_cm = pd.DataFrame(matrix, index=classes, columns=classes)
        #fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
        
        plt.figure(figsize=(5, 4))
        sn.set(font_scale=1)  # for label font size
        #sn.set_style('whitegrid')
        #sn.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
        cmap = sn.cubehelix_palette(light=1, as_cmap=True) #plt.cm.Greys plt.cm.Blues
        sn.heatmap(df_cm, annot=True, linecolor='k', linewidths='.2', cmap=cmap, fmt='.20g', annot_kws={"size": 12})  # font size 
        plt.show()
    
    print("*"*60, confusionMatrix.__name__)
    results = confusion_matrix(targets, predictions)
    print('Confusion Matrix :')
    print(clf.classes_)
    print(results)
    print('Accuracy Score :', accuracy_score(targets, predictions))
    #print('Report : ')
    #print(classification_report(targets, predictions))
    plotConfusionMatrix(results)


def train(): 
    if 1:
        df = preDataSet_GSE25097() #preDataSet_GSE114783()
        
        X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
        #X = pcaData(X,N=30) #PCA
        X,y,_ = featureSelect(X,y,N=30)
        X = preprocessingData(X) #scaler
    else:
        N=850
        X = np.random.random((N,10))
        y = np.random.randint(1,5, size=N)
    
    X_train, X_test, y_train, y_test = splitData(X, y, test=0.2)

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
        
        confusionMatrix(model, model.predict(X), y)
        print('\n')
        #visualization
        #visualClusterResult(X_train,pred_train,modelName)
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
