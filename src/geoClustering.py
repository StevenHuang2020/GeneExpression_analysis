#unicode Python3 Steven
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.neighbors.nearest_centroid import NearestCentroid

from modelCreate import createKMeans,createAgglomerate,createDBSCAN
from modelCreate import calculateSSE,calculateDaviesBouldin
from visualClustering import visualClusterResult,plotSilhouetteValues
from genPetData import getCsv
from plotApplication import plotModelCSM,plotCLusteringResult

def getModelMeasureByModel(data,model):
    return getModelMeasure(data,model.labels_)

def getModelMeasure(data,labels):
    sse,dbValue,csm = 0,0,0     
    #k = len(np.unique(labels))
    k = len(list(set(labels)))
    if k>1:
        #print(data.shape,model.labels_)
        csm = silhouette_score(data, labels, metric='euclidean')
        clf = NearestCentroid()
        clf.fit(data, labels)
        print('centroids=',clf.centroids_)
        sse = calculateSSE(data,labels,clf.centroids_)
        dbValue = calculateDaviesBouldin(data,labels)
    
    sse = round(sse,4)
    csm = round(csm,4)
    dbValue = round(dbValue,4)
    print('SSE=', sse,'DB=',dbValue,'CSM=',csm,'clusters=',k)    
    #print("Silhouette Coefficient: %0.3f" % csm)
    #print('clusters=',k)
    return sse,dbValue,csm,k    

def descriptData(data): #after preprocessing
    print('data.shape=',data.shape)
    s = 0
    for i in range(data.shape[1]):
        column = data[:,[i]]
        min = np.min(column)
        max = np.max(column)
        dis = np.abs(min-max)
        print("column:",i,'min,max,dis=',min,max,dis)
        s += dis
    print('s=',s)

def preprocessingData(data):
    scaler = MinMaxScaler()# StandardScaler() #
    scaler.fit(data)
    data = scaler.transform(data)
    #print('\n',data[:5])    
    #print('scaler=',data[:5])
    descriptData(data)
    return data

def Models():
    models = []
    models.append(('KMeans',createKMeans))
    models.append(('Agglomerative',createAgglomerate))
    return models

def trainModel(dataName,data,N=11): 
    data = preprocessingData(data)
    
    df = pd.DataFrame()
    columns=['Algorithm', 'K', 'tt(s)', 'SSE','DB','CSM']
    modelName='KMeans'
    for i in range(2,N,1):  #2 N 1   
        if 0:
            modelName='KMeans'       
            model = createKMeans(i)
        else:
            modelName='Agglomerative'       
            model = createAgglomerate(i)
        
        t = time()
        model.fit(data)
        
        tt = round(time()-t, 4)
        print("\nmodel:%s iter i=%d run in %.2fs" % (modelName,i,tt))
        sse,dbValue,csm,k = getModelMeasure(data,model.labels_)
            
        #dbName = dataName + str(data.shape)
        line = pd.DataFrame([[modelName, k, tt,sse,dbValue,csm]], columns=columns)
        df = df.append(line,ignore_index=True)
        #visualClusterResult(data,model.labels_,k,modelName+'_K_'+str(k))
        #plotSilhouetteValues(dataName,modelName,k, data, model.labels_)
        #print('cluster_labels=',np.unique(model.labels_))

    #print(df)
    df.to_csv(r'./db/' + modelName + '_result.csv',index=True)
    
    #plotModelCSM(modelName,df)
    #index,bestK = getBestkFromCSM(dataName,modelName,df)
    #bestLine = df.iloc[index,:]
    #print('bestLine=',index,'bestK=',bestK,'df=\n',bestLine)
    #return bestLine
    
def getBestkFromCSM(datasetName,modelName,df):
    print('df=\n',df)
    x = df.loc[:,['K']].values #K
    y = df.loc[:,['CSM']].values #CSM
 
    index = np.argmax(y)
    bestK = x[index][0]
    print('index,bestK=',index,bestK)
    return index,bestK
    
def prepareDataSet():
    if 0:
        df = getCsv('./db/petIotRecordsAll.csv')
        df = df.loc[:,['latitude','longitude']]
        #df = df[:100]
    else:
        df = getCsv('./db/statistic_result.csv')
        df = df.loc[:,['latitude_center','longitude_center']]
    
    return df

#cluster: 0 latitude: -36.8542531 longitude 174.76641334
#cluster: 1 latitude: -36.85011364 longitude 174.76240119
#cluster: 2 latitude: -36.84837064 longitude 174.74868692

def getCulteredCentroid(rawData, labels):
    cluster_labels = np.unique(labels)
    print('cluster_labels=',cluster_labels)
    # print(rawData.shape)
    # print(rawData[:5])
    # print(labels.shape)
    # print(labels)
    centroids = []
    for i in cluster_labels:
        lines = np.where(labels == i)
        print(i,len(lines[0].flatten()))
        #print(lines[0])
        data = rawData.iloc[lines[0],:]
        #print(data)
        if 0:
            latitudeCenter = round(np.mean(data['latitude']),8)
            longitudeCenter = round(np.mean(data['longitude']),8)
        else:
            latitudeCenter = round(np.mean(data['latitude_center']),8)
            longitudeCenter = round(np.mean(data['longitude_center']),8)
            
        print('cluster:',i,'latitude:', latitudeCenter,'longitude', longitudeCenter)
        centroids.append([latitudeCenter,longitudeCenter])
    return centroids
      
#centroids= [[-36.8542531, 174.76641334], [-36.85011364, 174.76240119], [-36.84837064, 174.74868692]]  
def train(rawData): 
    data = preprocessingData(rawData)
    
    model = createKMeans(3)
    t = time()
    model.fit(data)
    
    tt = round(time()-t, 4)
    print("\n run in %.2fs" % (tt))
    sse,dbValue,csm,k = getModelMeasure(data,model.labels_)
        
    #visualClusterResult(data,model.labels_,k,'KMeans_K_'+str(k))
    #plotSilhouetteValues('','KMeans',k, data, model.labels_)
    #print('cluster_labels=',np.unique(model.labels_))
    centroids = getCulteredCentroid(rawData,model.labels_)
    print('centroids=',centroids)
    
    plotCLusteringResult(rawData,model.labels_)
    
def main():
    rawData = prepareDataSet()
    #trainModel('petGPSLocation',rawData)
    train(rawData)
    
if __name__ == "__main__":
    main()
