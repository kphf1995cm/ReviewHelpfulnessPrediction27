#! /usr/bin/env python2.7
#coding=utf-8

"""
Use positive and negative review set as corpus to train a sentiment classifier.
This module use labeled positive and negative reviews as training set, then use nltk scikit-learn api to do classification task.
Aim to train a classifier automatically identifiy review's positive or negative sentiment, and use the probability as review helpfulness feature.

"""

import textProcessing as tp
import pickle
import itertools
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

import sklearn
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import accuracy_score


# 1. Load positive and negative review data

posNegDir = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet'
pos_review = tp.seg_fil_senti_excel(posNegDir + '/pos_review.xlsx', 1, 1, 'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
neg_review = tp.seg_fil_senti_excel(posNegDir + '/neg_review.xlsx', 1, 1, 'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
# pos_review = tp.seg_fil_senti_excel("D:/code/sentiment_test/pos_review.xlsx", 1, 1)
# neg_review = tp.seg_fil_senti_excel("D:/code/sentiment_test/neg_review.xlsx", 1, 1)

pos = pos_review
neg = neg_review

"""
# Cut positive review to make it the same number of nagtive review (optional)

shuffle(pos_review)
size = int(len(pos_review)/2 - 18)

pos = pos_review[:size]
neg = neg_review

"""


# 2. Feature extraction function
# 2.1 Use all words as features
def bag_of_words(words):
    return dict([(word, True) for word in words])


# 2.2 Use bigrams as features (use chi square chose top 200 bigrams)
def bigrams(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(bigrams)


# 2.3 Use words and bigrams as features (use chi square chose top 200 bigrams)
def bigram_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


# 2.4 Use chi_sq to find most informative features of the review
# 2.4.1 First we should compute words or bigrams information score
def create_word_scores():
    posNegDir = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet'
    posdata = tp.seg_fil_senti_excel(posNegDir + '/pos_review.xlsx', 1, 1,
                                        'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    negdata = tp.seg_fil_senti_excel(posNegDir + '/neg_review.xlsx', 1, 1,
                                        'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

def create_bigram_scores():
    posNegDir = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet'
    posdata = tp.seg_fil_senti_excel(posNegDir + '/pos_review.xlsx', 1, 1,
                                     'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    negdata = tp.seg_fil_senti_excel(posNegDir + '/neg_review.xlsx', 1, 1,
                                     'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 8000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 8000)

    pos = posBigrams
    neg = negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# Combine words and bigrams and compute words and bigrams information scores
def create_word_bigram_scores():
    posNegDir = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet'
    posdata = tp.seg_fil_senti_excel(posNegDir + '/pos_review.xlsx', 1, 1,
                                     'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    negdata = tp.seg_fil_senti_excel(posNegDir + '/neg_review.xlsx', 1, 1,
                                     'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_finder = BigramCollocationFinder.from_words(posWords)
    bigram_finder = BigramCollocationFinder.from_words(negWords)
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

    pos = posWords + posBigrams
    neg = negWords + negBigrams

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    for word in pos:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in neg:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores

# Choose word_scores extaction methods
#word_scores = create_word_scores()
# word_scores = create_bigram_scores()
word_scores = create_word_bigram_scores()


# 2.4.2 Second we should extact the most informative words or bigrams based on the information score
# number 为特征选取的维度
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

def sort_word_score(word_scores):
    words=sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)
    return words

# 2.4.3 Third we could use the most informative words and bigrams as machine learning features
# Use chi_sq to find most informative words of the review
def best_word_features(words,best_words):
    return dict([(word, True) for word in words if word in best_words])

# Use chi_sq to find most informative bigrams of the review
def best_word_features_bi(words,best_words):
    return dict([(word, True) for word in nltk.bigrams(words) if word in best_words])

# Use chi_sq to find most informative words and bigrams of the review
def best_word_features_com(words,best_words):
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3



# 3. Transform review to features by setting labels to words in review
# 提取积极评论里的特征
def pos_features(feature_extraction_method,best_words):
    posFeatures = []
    for i in pos:
        posWords = [feature_extraction_method(i,best_words),'pos']
        posFeatures.append(posWords)
    return posFeatures
# 提取消极评论里的特征
def neg_features(feature_extraction_method,best_words):
    negFeatures = []
    for j in neg:
        negWords = [feature_extraction_method(j,best_words),'neg']
        negFeatures.append(negWords)
    return negFeatures

# 构建训练集和测试集 选取单词+二元词形式
def get_trainset_testset_testtag(dimension):
    word_scores = create_word_bigram_scores() # 计算 词+二元词 信息得分
    best_words=find_best_words(word_scores,dimension) #排序 挑前dimension个
    posFeatures = pos_features(best_word_features_com,best_words) #提取积极文本里面的数据
    negFeatures = neg_features(best_word_features_com,best_words) #提取消极文本里面的数据
    shuffle(posFeatures)  # 将序列的所有元素随机排列
    shuffle(negFeatures)
    size_pos = int(len(pos_review) * 0.75)
    size_neg = int(len(neg_review) * 0.75)

    train_set = posFeatures[:size_pos] + negFeatures[:size_neg]
    test_set = posFeatures[size_pos:] + negFeatures[size_neg:]

    test, tag_test = zip(*test_set)  # 将特征和分类结果分离开
    return train_set,test,tag_test

# 构建训练集 选取 单词+二元词形式
# 将所有数据作为训练数据
def get_trainset(dimension):
    word_scores = create_word_bigram_scores() # 计算 词+二元词 信息得分
    best_words=find_best_words(word_scores,dimension) #排序 挑前dimension个
    posFeatures = pos_features(best_word_features_com,best_words) #提取积极文本里面的数据
    negFeatures = neg_features(best_word_features_com,best_words) #提取消极文本里面的数据
    shuffle(posFeatures)  # 将序列的所有元素随机排列
    shuffle(negFeatures)
    size_pos = int(len(pos_review))
    size_neg = int(len(neg_review))
    train_set = posFeatures[:size_pos] + negFeatures[:size_neg]
    return train_set

best_words = find_best_words(word_scores, 1500) # Set dimension and initiallize most informative words

# posFeatures = pos_features(bigrams)
# negFeatures = neg_features(bigrams)

# posFeatures = pos_features(bigram_words)
# negFeatures = neg_features(bigram_words)

posFeatures = pos_features(best_word_features,best_words)
negFeatures = neg_features(best_word_features,best_words)

'''
test posFeatures:
for x in posFeatures:
    for k,v in x[0].iteritems():
        print k,v,
    print x[1]
for x in negFeatures:
    for k,v in x[0].iteritems():
        print k,v,
    print x[1]
output:

[{不能:True,手机:True,力:True,机:True,后悔:True,比较:True,软件:True},neg]
'''
# posFeatures = pos_features(best_word_features_com)
# negFeatures = neg_features(best_word_features_com)



# 4. Train classifier and examing classify accuracy
# Make the feature set ramdon
shuffle(posFeatures) # 将序列的所有元素随机排列
shuffle(negFeatures)

# 75% of data items used as training set (in fact, it have a better way by using cross validation function)
size_pos = int(len(pos_review) * 0.75)
size_neg = int(len(neg_review) * 0.75)

train_set = posFeatures[:size_pos] + negFeatures[:size_neg]
test_set = posFeatures[size_pos:] + negFeatures[size_neg:]

test, tag_test = zip(*test_set) # 将特征和分类结果分离开

def clf_score(classifier):
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)
    predict = classifier.batch_classify(test)
    return accuracy_score(tag_test, predict)

print 'BernoulliNB`s accuracy is %f' %clf_score(BernoulliNB())
#print 'GaussianNB`s accuracy is %f' %clf_score(GaussianNB())
print 'MultinomiaNB`s accuracy is %f' %clf_score(MultinomialNB())
print 'LogisticRegression`s accuracy is %f' %clf_score(LogisticRegression())
print 'SVC`s accuracy is %f' %clf_score(SVC(gamma=0.001, C=100., kernel='linear'))
print 'LinearSVC`s accuracy is %f' %clf_score(LinearSVC())
print 'NuSVC`s accuracy is %f' %clf_score(NuSVC())



# 5. After finding the best classifier, then check different dimension classification accuracy
def score(classifier,train_set,test,tag_test):
    classifier = SklearnClassifier(classifier)
    classifier.train(train_set)

    pred = classifier.batch_classify(test)
    return accuracy_score(tag_test, pred)

def get_best_classfier_and_dimention():
    bestClassfier = ''
    bestDimention = '0'
    curAccuracy = 0.0
    dimention = ['500', '1000', '1500', '2000', '2500', '3000']
    word_scores = create_word_bigram_scores()  # 创建单个词 二元词 字典序列{word1:score1,word2:score2,}
    # for w,s in word_scores.iteritems():
    #     print w,s,
    # print ''
    sort_word=sort_word_score(word_scores)
    for w,s in sort_word:
        print w,s,
    print ''
    for d in dimention:
        best_words = find_best_words(word_scores, int(d))  # 找到上述字典序列里面得分最高的 d 个词[word1,word2,]
        for x in best_words:
            print x,
        print ''
        posFeatures = pos_features(best_word_features_com,best_words)  # 得到[[{word1:true,word2:true},'pos'],[]]
        negFeatures = neg_features(best_word_features_com,best_words)  # 得到[[{word1:true,word2:true},'neg'],[]]

        # Make the feature set ramdon
        shuffle(posFeatures)
        shuffle(negFeatures)
        # 75% of features used as training set (in fact, it have a better way by using cross validation function)
        size_pos = int(len(pos_review) * 0.75)
        size_neg = int(len(neg_review) * 0.75)

        trainset = posFeatures[:size_pos] + negFeatures[:size_neg]
        testset = posFeatures[size_pos:] + negFeatures[size_neg:]

        test, tag_test = zip(*testset)
        BernoulliNBScore=score(BernoulliNB(),trainset,test,tag_test)
        MultinomialNBScore=score(MultinomialNB(),trainset,test,tag_test)
        LogisticRegressionScore=score(LogisticRegression(),trainset,test,tag_test)
        SVCScore=score(SVC(),trainset,test,tag_test)
        LinearSVCScore=score(LinearSVC(),trainset,test,tag_test)
        NuSVCScore=score(NuSVC(),trainset,test,tag_test)
        if BernoulliNBScore>curAccuracy:
            curAccuracy=BernoulliNBScore
            bestClassfier='BernoulliNB()'
            bestDimention=d
        if MultinomialNBScore>curAccuracy:
            curAccuracy=MultinomialNBScore
            bestClassfier='MultinomialNB()'
            bestDimention=d
        if LogisticRegressionScore>curAccuracy:
            curAccuracy=LogisticRegressionScore
            bestClassfier='LogisticRegression()'
            bestDimention=d
        if SVCScore>curAccuracy:
            curAccuracy=SVCScore
            bestClassfier='SVC()'
            bestDimention=d
        if LinearSVCScore>curAccuracy:
            curAccuracy=LinearSVCScore
            bestClassfier='LinearSVC()'
            bestDimention=d
        if NuSVCScore>curAccuracy:
            curAccuracy=NuSVCScore
            bestClassfier='NuSVC()'
            bestDimention=d
        classifierNameList=['BernoulliNB()'.decode('utf-8'),'MultinomialNB()'.decode('utf-8'),'LogisticRegression()'.decode('utf-8'),'SVC()'.decode('utf-8'),'LinearSVC()'.decode('utf-8'),'NuSVC()'.decode('utf-8')]
        classifierAccList=[str(BernoulliNBScore).decode('utf-8'),str(MultinomialNBScore).decode('utf-8'),str(LogisticRegressionScore).decode('utf-8'),str(SVCScore).decode('utf-8'),str(LinearSVCScore).decode('utf-8'),str(NuSVCScore).decode('utf-8')]
        f = open('D:/ReviewHelpfulnessPrediction\BuildedClassifier/' + 'classifierDimenAcc.txt', 'a')
        for pos in range(len(classifierAccList)):
            f.write(classifierNameList[pos]+'\t'+str(d).decode('utf-8')+'\t'+classifierAccList[pos]+'\n')
        f.close()


        print 'BernoulliNB`s accuracy is %f' % BernoulliNBScore
        print 'MultinomiaNB`s accuracy is %f' % MultinomialNBScore
        print 'LogisticRegression`s accuracy is %f' % LogisticRegressionScore
        print 'SVC`s accuracy is %f' % SVCScore
        print 'LinearSVC`s accuracy is %f' % LinearSVCScore
        print 'NuSVC`s accuracy is %f' % NuSVCScore
        print

    return bestClassfier,bestDimention,curAccuracy

bestClassfier,bestDimention,bestAccuracy=get_best_classfier_and_dimention()
print bestClassfier,bestDimention,bestAccuracy
def storeClassifierDimenAcc(classifier,dimen,acc):
    f=open('D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+'bestClassifierDimenAcc.txt','w')
    f.write(classifier+'\t'+dimen+'\t'+acc);
    f.close()

#storeClassifierDimenAcc(bestClassfier.decode('utf-8'),bestDimention.decode('utf-8'),str(bestAccuracy).decode('utf-8'))
#storeClassifierDimenAcc('BernoulliNB()'.decode('utf-8'),'1500'.decode('utf-8'),'0.940298507463'.decode('utf-8'))



# bestClassfier=''
# bestDimention='0'
# curAccuracy=0.0
# dimention = ['500','1000','1500','2000','2500','3000']
#
# for d in dimention:
#     word_scores = create_word_bigram_scores() #创建单个词 二元词 字典序列{word1:score1,word2:score2,}
#     best_words = find_best_words(word_scores, int(d)) # 找到上述字典序列里面得分最高的 d 个词[word1,word2,]
#
#     posFeatures = pos_features(best_word_features_com) # 得到[[{word1:true,word2:true},'pos'],[]]
#     negFeatures = neg_features(best_word_features_com) # 得到[[{word1:true,word2:true},'neg'],[]]
#
#     # Make the feature set ramdon
#     shuffle(posFeatures)
#     shuffle(negFeatures)
#
#     # 75% of features used as training set (in fact, it have a better way by using cross validation function)
#     size_pos = int(len(pos_review) * 0.75)
#     size_neg = int(len(neg_review) * 0.75)
#
#     trainset = posFeatures[:size_pos] + negFeatures[:size_neg]
#     testset = posFeatures[size_pos:] + negFeatures[size_neg:]
#
#     test, tag_test = zip(*testset)
#
#     print 'BernoulliNB`s accuracy is %f' %score(BernoulliNB())
#     print 'MultinomiaNB`s accuracy is %f' %score(MultinomialNB())
#     print 'LogisticRegression`s accuracy is %f' %score(LogisticRegression())
#     print 'SVC`s accuracy is %f' %score(SVC())
#     print 'LinearSVC`s accuracy is %f' %score(LinearSVC())
#     print 'NuSVC`s accuracy is %f' %score(NuSVC())
#     print



# 6. Store the best classifier under best dimension
def store_classifier(clf, trainset, filepath):
    classifier = SklearnClassifier(clf)
    classifier.train(trainset)
    # use pickle to store classifier
    pickle.dump(classifier, open(filepath,'w'))

# MultinomialNB() 1500 0.940298507463
#BernoulliNB() 1500 0.940298507463
# 存储性能最佳的分类器
trainSet=get_trainset(1500) #将所有数据作为训练数据
store_classifier(BernoulliNB(),trainSet,'D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+'BernoulliNB.pkl')

