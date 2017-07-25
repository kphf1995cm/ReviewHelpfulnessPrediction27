# encoding utf-8
from sklearn import datasets
from sklearn import svm
iris=datasets.load_iris()
digits=datasets.load_digits()
print iris
print digits

clf=svm.SVC(gamma=0.001,C=100)
clf.fit(digits.data[:-1],digits.target[:-1])
print  svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
  gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False)





