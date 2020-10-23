from bioinfokit import analys, visuz
from geoClassifier import preDataSet_GSE25097,preDataSet_GSE114783
import pandas as pd
import matplotlib.pyplot as plt
from featureSelection import *

def heatMap(df):
    #cmap='RdYlGn' cmap="seismic"
    plt.rcParams.update({'font.size': 8})
    plt.rcParams['savefig.dpi'] = 300
    figSize=(5,8)
    font = (6,6) #x font, y font
    visuz.gene_exp.hmap(df=df, dim=figSize, show=True, tickfont=font) #clus:False not show heirachical
    #visuz.gene_exp.hmap(df=df, dim=figSize, show=True, tickfont=font, rowclus=False,colclus=False)
    visuz.gene_exp.hmap(df=df, dim=figSize, show=True, tickfont=font, rowclus=False)
    visuz.gene_exp.hmap(df=df, dim=figSize, show=True, tickfont=font, colclus=False)
    
def getHeatmapData():
    df = preDataSet_GSE25097(True) #preDataSet_GSE114783() 
    print('1:\n',df.head())
    
    # new_header = df.iloc[0] #grab the first row for the header
    # df = df[1:] #take the data less the header row
    # df.columns = new_header
    # print('2:\n',df.head())
    #df = df.iloc[:,1:]
    df.set_index('GSM_ID', inplace=True)
    print('df.columns=',df.columns)
    print('df.index=',df.index)
       
    #df  = df.iloc[:5,:10]
    #print('3:\n',df.head())
    
    df = df.T
    print('4:\n',df.head())
    print('df.columns=',df.columns)
    print('df.index=',df.index)
    #df = df.set_index(df.columns[0])
   
    # df.columns = df.iloc[0]
    # df = df[1:]

    print('5:\n',df.head())
    print(df.shape)
    df.columns.name='GSM_ID'
    df.to_csv('heatmapData_filterAll_24.csv')
    return df
    

def filterData(labels, df, selectDict):
    dataDict = {}
    for i in labels:
        dataDict[i] = df[df['Type'] == i]
    
    for i in  dataDict:
        print(i, dataDict[i].shape)
    
    selRes = pd.DataFrame()
    print('-----------select to compare------------')
    for key,value in selectDict.items():
        if value == 0:
            continue
        sel = dataDict[int(key)][:value]
        selRes = pd.concat([selRes,sel])
        #print('sel=\n',sel)
        selIdList = sel.index.tolist()
        print('type: ', i, 'selection Id: ', selIdList, len(selIdList))
    #print('selRes.shape=',selRes.shape)
    #print(selRes)
    print('-----------select final-----------------')
    return selRes

def getFromFile(file):
    df = pd.read_csv(file)
    #print(df.head())
    df = df.set_index(df.columns[0])
    print(df.head())
    
    '''start to filter samples'''
    '''
    class_mapping = {
        'healthy':          1,  #6
        'cirrhotic':        2,  #40
        'non_tumor':        3,  #243
        'tumor':            4,  #268
    }
    '''
    if 1: 
        labels = np.unique(df['Type'])
        #modify here to select number of differ type of samples 
        selectDict={ '1':6, '2':10, '3':10, '4':10}
        df = filterData(labels,df,selectDict)
    
    if 0:
        df.index.name='GSM_ID'
        df.columns.name='Gene'
        df = df.drop(['Type'], axis=1)
    else:
        #tranpose
        df = df.T
        #df.columns = df.iloc[0]
        #df = df.reindex(df.index.drop(0)).reset_index(drop=True)
        #df.columns.name = None
        print('df.columns=',df.columns)
        print('df.index=',df.index)
        df.index.name='Gene'
        df.columns.name='GSM_ID'
        df = df.drop(['Type'])
    print(df.head())
    return df

def getFilterAll():
    #return getFromFile('heatmapData_filterAll_24.csv')
    file = r'..\data\GSE25097\GSE25097_select30.csv' #GSE25097.csv
    df = getFromFile(file)
    return df

def featureSelction():
    # df = preDataSet_GSE114783()
    # dst = r'..\data\GSE114783\GSE114783_select30.csv'
    # select = 30
    
    df = preDataSet_GSE25097()
    dst = r'..\data\GSE25097\GSE25097_select15.csv'
    select = 15
    
    #print(df.head())
    print('df.shape=',df.shape)
    df = df.set_index(df.columns[0])
    print(df.head())

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    
    #X,y = FeatureExtractChiSquare(X,y,N=select)
    X,y,support = FeatureExtract_RFE(X,y,N=select)

    print('support.shape=', support.shape)
    support = np.append(support,[True])
    print('support.shape=', support.shape)
    
    #print(X[:5])
    #print(y[:5])
    #df = df[:,:-1] #remove type column
    df = df.iloc[:,support] #
    print('df.shape=',df.shape)
    print(df.head())
    df.to_csv(dst)
    
def main():
    #return featureSelction()
    if 0:
        df = getHeatmapData()
    else:
        df = getFilterAll()
        heatMap(df)
    
if __name__=='__main__':
    main()
            