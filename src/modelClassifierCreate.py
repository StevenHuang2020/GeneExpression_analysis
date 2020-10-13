#Steven 
#classifier model design
from time import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import make_pipeline
from sklearn.semi_supervised import LabelPropagation

def createDecisionTree(min_samples_split=2):
    return DecisionTreeClassifier(criterion='entropy', min_samples_split=min_samples_split), 'DecisionTreeClassifier'

def createMLPClassifier(lr=0.1):
    return MLPClassifier(hidden_layer_sizes=(5,5), random_state=1, learning_rate_init=lr, max_iter=150), 'MLPClassifier'

def createRandomForestClf():
    return RandomForestClassifier(n_estimators=10,random_state=42), 'RandomForestClassifier'

def createLogisticRegression():
    return LogisticRegression(), 'LogisticRegression'

def createRidgeClassifier():
    return RidgeClassifier(),'RidgeClassifier'

def createSGDClassifier():
    return SGDClassifier(),'SGDClassifier'

def createSVM_svc():
    #return make_pipeline(StandardScaler(), svm.SVC(gamma='auto')),'SVC pipline'
    return svm.SVC(gamma='auto'), 'SVC'

def createSVM_NuSVC():
    #return make_pipeline(StandardScaler(), svm.NuSVC()),'NuSVC pipline'
    return svm.NuSVC(), 'NuSVC'

def createSVM_LinearSVC():
    #return make_pipeline(StandardScaler(), svm.LinearSVC(random_state=0, tol=1e-5)),'LinearSVC pipline'
    return svm.LinearSVC(random_state=0, tol=1e-5), 'LinearSVC'

def createKNeighborsClassifier(k=3):
    return KNeighborsClassifier(n_neighbors=k),'KNeighborsClassifier'

def createRadiusNeighborsClassifier():
    return RadiusNeighborsClassifier(), 'RadiusNeighborsClassifier'

def createNearestCentroid():
    return NearestCentroid(),'NearestCentroid'

def createGaussianProcessClassifier():
    return GaussianProcessClassifier(),'GaussianProcessClassifier'

def createGaussianNB():
    return GaussianNB(),'GaussianNB'

def createMultinomialNB():
    return MultinomialNB(),'MultinomialNB'

def createComplementNB():
    return ComplementNB(),'ComplementNB'  

def createBernoulliNB():
    return BernoulliNB(),'BernoulliNB'

def createCategoricalNB():
    return CategoricalNB(),'CategoricalNB'   

def createLabelPropagation():
    return LabelPropagation(),'LabelPropagation'

