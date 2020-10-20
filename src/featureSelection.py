import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
from time import time

def FeatureExtractChiSquare(X,Y, N=3):
    print("*"*50,FeatureExtractChiSquare.__name__)
    t = time()
    test = SelectKBest(score_func=chi2,k=N)
    fit = test.fit(X, Y)
    #print('X.shape = ',X.shape)
    #print('Y.shape = ', Y.shape)

    np.set_printoptions(precision=3)
    print('scores = ', fit.scores_)

    features = fit.transform(X)
    tt = round(time()-t, 4)
    print("run in %.2fs" % (tt), ' after chi square,features.shape = ',features.shape)
    return features,Y

def FeatureExtract_RFE(X,Y,N=3):
    print("*" * 50, FeatureExtract_RFE.__name__)
    t = time()
    #estimator = SVR(kernel="linear")
    estimator = SVC(kernel="linear", C=1)
    rfe = RFE(estimator, n_features_to_select=N, step=1)

    fit = rfe.fit(X, Y)
    features = fit.transform(X)
    
    tt = round(time()-t, 4)
    # print(fit.classes_)
    print("Num Features: %d" % (fit.n_features_))
    print("Selected Features: %s" % (fit.support_))
    print("Feature Ranking: %s" % (fit.ranking_))
    #print(fit.support_)
    print("run in %.2fs" % (tt), ' after FRE features.shape = ',features.shape)
    return features, Y

def FeatureExtract_PCA(X,Y,N=3):
    print("*" * 50, FeatureExtract_PCA.__name__)
    t = time()
    rfe = PCA(n_components=N)
    fit = rfe.fit(X, Y)
    features = fit.transform(X)
    tt = round(time()-t, 4)
    print("Explained Variance: %s" % (fit.explained_variance_))
    print("Explained Variance ratio: %s" % (fit.explained_variance_ratio_))
    print('components_=',fit.components_)
    print('n_components_=',fit.n_components_)
    print('n_samples_=',fit.n_samples_)
    print('n_features_=',fit.n_features_)
    print('singular_values_=',fit.singular_values_)
    
    print("run in %.2fs" % (tt), ' after PCA features.shape = ', features.shape)
    return features, Y

def FeatureExtract_ETC(X,Y):
    print("*" * 50, FeatureExtract_ETC.__name__)
    t = time()
    model = ExtraTreesClassifier(max_depth=3, min_samples_leaf=2)
    fit = model.fit(X, Y)
    print('Feature importance:', model.feature_importances_)
    t = SelectFromModel(fit, prefit=True)  # extra step required as we are using an ensemble classifier here
    features = t.transform(X)
    tt = round(time()-t, 4)
    print("run in %.2fs" % (tt), ' after ETC features.shape = ', features.shape)
    return features, Y

def FeatureExtract_VT(X,Y,N=3):
    print("*" * 50, FeatureExtract_VT.__name__)
    t = time()
    vt = VarianceThreshold()
    features = vt.fit_transform(X)
    tt = round(time()-t, 4)
    
    print("run in %.2fs" % (tt), ' after VT features.shape = ', features.shape)
    return features, Y
