from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('tfidf', TfidfTransformer()),
                ('chi2', SelectKBest(chi2, k=1000)),
                      ('nb', MultinomialNB())])
classif = SklearnClassifier(pipeline)
print classif
classif = SklearnClassifier(LinearSVC())
