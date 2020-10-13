import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
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
    dendrogram(linkage_matrix, **kwargs)

def plotHierarchical(data):
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    
    model = model.fit(data)
    plt.title('Hierarchical Clustering Dendrogram')
    # plot the top three levels of the dendrogram
    #labels=[['a','a','b','b'],['c','c','d','d']]
    plot_dendrogram(model, truncate_mode='level', p=3,show_contracted=True)
    plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.show()
    
def main():
    if 0:
        iris = load_iris()
        X = iris.data
        print(X.shape,type(X))
        print(X)
    else:
        X = np.array([[1,2,3],
                    [1,3,2],
                    [10,12,14],
                    ]) #[20,12,11,14,10], [21,22,14,23,22], [100,130,120,120,200]
    
    plotHierarchical(X)
    
    
if __name__=='__main__':
    main()
    