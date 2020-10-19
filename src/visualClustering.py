import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from manifoldLearning import manifoldLearningTransform,plotManifold

def visualClusterResult(data, labels, title, show=False):
    #print('data.shape=',data.shape,'labels.shape=',labels.shape)
    if 0:
        method='PCA'
        Y = PCA(n_components=2).fit_transform(data)
    elif 1:
        #Y = tsne(data, 2, 10, 50,max_iter=500)
        Y = TSNE(n_components=2, verbose=0, perplexity=50, n_iter=1000).fit_transform(data)#scikit-learn
        method='TSNE'
    else:
        name = 'LLE'
        name = 'LTSA'
        name = 'Hessian LLE'
        name = 'Modified LLE'
        name = 'Isomap'
        name = 'MDS'
        name = 'SE'
        name = 't-SNE'
        
        if name == 'Hessian LLE' or name == 'Modified LLE' or name == 'LTSA':
            Y = manifoldLearningTransform(data,name=name,eigen_solver='dense')
        else:
            Y = manifoldLearningTransform(data,name=name)
        
        method = 'manifold' + '_' + name
        
    title = title + '_' + method
    plt.clf()
    plt.title(title)
    
    #print('Y.shape=',Y.shape)
    if 0:
        markers=['+','o','*','x','s','v','<','>']
        for i, c in enumerate(np.unique(labels)):
            plt.scatter(Y[:,0][labels==c], Y[:,1][labels==c], c=labels[labels==c], marker=markers[i])
    else:
        plt.scatter(Y[:, 0], Y[:, 1], s=20, c=labels) #marker='s',marker='+' edgecolor='black'
        
    plt.savefig('../images/'+title+'.png')
    if show:
        plt.show()

def test():
    plt.title('test')
    markers=['+','o','*','x','s','v','<','>']
    plt.scatter(10,10, s=20, c='r', marker=markers[7]) #marker='s',edgecolor='black'
    plt.show()
        
def main():
    test()

if __name__=='__main__':
    main()