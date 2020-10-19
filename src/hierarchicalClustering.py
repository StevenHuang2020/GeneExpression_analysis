import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

from geoClassifier import *

def plot_dendrogram(model, labels, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)
    print('linkage_matrix.shape=',linkage_matrix.shape)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, labels=labels, **kwargs)

def plotHierarchical(data,labels,title='Hierarchical Clustering Dendrogram'):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    
    model = model.fit(data)
    plt.title(title)
    # plot the top three levels of the dendrogram
    #labels=[['a','a','b','b'],['c','c','d','d']]
    plot_dendrogram(model,labels,  leaf_rotation=90., leaf_font_size=8, ) #truncate_mode='level', p=3, show_contracted=True
    #plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.ylabel('distance')
    plt.show()
    
def main():
    if 0:
        iris = load_iris()
        X = iris.data
        X = X[:30]
        print(X.shape,type(X))
        print(X)
        labels = ['id_'+str(i) for i in range(X.shape[0])]
    elif 0:
        X = np.array([[1,2,3],
                    [1,3,2],
                    [10,12,14],
                    ]) #[20,12,11,14,10], [21,22,14,23,22], [100,130,120,120,200]
    else:
        df = preDataSet_GSE25097()
    
        if 1:
            front=12
            X, y = df.iloc[:front, 1:-1].values, df.iloc[:front, -1].values
            labels = df.iloc[:front,0].values
        else:
            X, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
            labels = df.iloc[:,0].values
            
        X = pcaData(X,N=30) #PCA
        X = preprocessingData(X) #scaler

        
    print('labels=',labels)
    plotHierarchical(X,labels)
    
    
if __name__=='__main__':
    main()
    