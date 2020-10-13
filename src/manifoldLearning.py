from collections import OrderedDict
from functools import partial
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold, datasets


def test():
    n_points = 1000
    X, color = datasets.make_s_curve(n_points, random_state=0)
    n_neighbors = 10
    n_components = 2

    # Create figure
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle("Manifold Learning with %i points, %i neighbors"
                % (1000, n_neighbors), fontsize=14)

    # Add 3d scatter plot
    ax = fig.add_subplot(251, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    ax.view_init(4, -72)

    # Set-up manifold methods
    LLE = partial(manifold.LocallyLinearEmbedding, n_neighbors, n_components, eigen_solver='auto')

    methods = OrderedDict()
    methods['LLE'] = LLE(method='standard')
    methods['LTSA'] = LLE(method='ltsa')
    methods['Hessian LLE'] = LLE(method='hessian')
    methods['Modified LLE'] = LLE(method='modified')
    methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components,
                                            n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
                                    random_state=0)

    # Plot results
    for i, (label, method) in enumerate(methods.items()):
        t0 = time()
        Y = method.fit_transform(X)
        t1 = time()
        print("%s: %.2g sec" % (label, t1 - t0))
        ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
        ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')

    plt.show()

def manifoldLearningTransform(X,n_components=2, n_neighbors=10,name='SE',eigen_solver='auto'):
    LLE = partial(manifold.LocallyLinearEmbedding, n_neighbors, n_components, eigen_solver=eigen_solver)

    methods = OrderedDict()
    methods['LLE'] = LLE(method='standard')
    methods['LTSA'] = LLE(method='ltsa')
    methods['Hessian LLE'] = LLE(method='hessian')
    methods['Modified LLE'] = LLE(method='modified')
    methods['Isomap'] = manifold.Isomap(n_neighbors, n_components)
    methods['MDS'] = manifold.MDS(n_components, max_iter=100, n_init=1)
    methods['SE'] = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    
    for i, (label, method) in enumerate(methods.items()):
        #print(label,method)
        if name == label:
            return method.fit_transform(X)
    assert(0)
            
def plotManifold(X,labels,n_components=2, n_neighbors=10):
    names=['LLE','LTSA','Hessian LLE','Modified LLE','Isomap','MDS','SE','t-SNE']
    for name in names:
        t0 = time()
        if name == 'Hessian LLE' or name == 'Modified LLE' or name == 'LTSA':
            Y = manifoldLearningTransform(X,name=name,eigen_solver='dense')
        else:
            Y = manifoldLearningTransform(X,name=name)
            
        t1 = time()
        print("%s: %.2g sec" % (name, t1 - t0))
        ax = plt.subplot(1,1,1)
        ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=plt.cm.Spectral)
        ax.set_title("%s (%.2g sec)" % (name, t1 - t0))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis('tight')
        plt.show()
    
def main():
    #test()
    pass
    
if __name__=='__main__':
    main()
    