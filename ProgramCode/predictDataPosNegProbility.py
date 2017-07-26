#! /usr/bin/env python2.7
#coding=utf-8

"""
Use a stored sentiment classifier to identifiy review positive and negative probability.
"""

import textProcessing as tp
import pickle
import itertools
import chardet
import sklearn
import numpy
import scipy
import time
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

#import sklearn


# 1. Load data
reviewDataSetPath='D:/ReviewHelpfulnessPrediction\ReviewSet/HTC_Z710t_review_2013.6.5.xlsx'
sentiment_review = tp.seg_fil_senti_excel(reviewDataSetPath, 1, 4,'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')


# 2. Feature extraction method
# 计算单个词 和二元词得分
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

# number 为特征选取的维度
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

# Initiallize word's information score and extracting most informative words
#word_scores = create_word_bigram_scores()
#best_words = find_best_words(word_scores, 1500) # Be aware of the dimentions

def best_word_features(words,best_words):
    return dict([(word, True) for word in words if word in best_words])

# Use chi_sq to find most informative words and bigrams of the review
def best_word_features_com(words,best_words):
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3

# 3. Function that making the reviews to a feature set
# 数据集应处理成这种形式：[[明天,阳光],[],[],[],]
def extract_features(dataset,best_words):
    feat = []
    for i in dataset:
        feat.append(best_word_features_com(i,best_words))
    return feat

# 4. Load classifier
#  装载分类器，得到SklearnClassifier
word_scores = create_word_bigram_scores() #计算词语信息得分
best_words = find_best_words(word_scores, 1500) # Be aware of the dimentions 选取前1500个信息得分高的词语
def predictDataSentimentPro(oriDataPath,preResStorePath):
    start=time.clock()
    reviewDataSetPath = 'D:/ReviewHelpfulnessPrediction\ReviewSet/HTC_Z710t_review_2013.6.5.xlsx'
    #reviewDataSetPath='D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet/pos_review.xlsx'
    review = tp.get_excel_data(reviewDataSetPath, 1, 4, "data")
    sentiment_review = tp.seg_fil_senti_excel(reviewDataSetPath, 1, 4, 'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    # for x in sentiment_review:
    #     for y in x:
    #         print y,
    #     print ''
    #word_scores = create_word_bigram_scores()
    # for w,s in word_scores.iteritems():
    #     print w,s,
    # print ''
    #best_words = find_best_words(word_scores, 1500) # Be aware of the dimentions
    # for x in best_words:
    #     print x,
    classifierPath = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature/sentiment_classifier.pkl'
    #classifierPath='D:/ReviewHelpfulnessPrediction\BuildedClassifier/BernoulliNB.pkl'
    clf = pickle.load(open(classifierPath))
    review_feature=extract_features(sentiment_review,best_words)
    pred = clf.batch_prob_classify(review_feature)
    p_file = open(preResStorePath, 'w')
    reviewCount = 0
    for i in pred:
        reviewCount += 1
        p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
    p_file.close()
    p_file = open(oriDataPath, 'w')
    for d in review:
        p_file.write(d.encode('utf-8')+'\n')
    p_file.close()
    p_file = open('D:/ReviewHelpfulnessPrediction\ReviewDataFeature/HTCOriDataFea.txt', 'w')
    for d in review_feature:
        for w,b,in d.iteritems():
            p_file.write(w.encode('utf-8') + ' '+str(b)+'\t')
        p_file.write('\n')
    p_file.close()
    end=time.clock()
    return reviewCount,end-start
preResStorePath='D:/ReviewHelpfulnessPrediction\ReviewDataFeature/HTCPosNegProb.txt'
oriDataPath='D:/ReviewHelpfulnessPrediction\ReviewDataFeature/HTCOriData.txt'
recordNum,runningTime=predictDataSentimentPro(oriDataPath,preResStorePath)
print recordNum,runningTime

