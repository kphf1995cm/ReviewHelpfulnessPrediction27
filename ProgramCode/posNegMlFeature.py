#! /usr/bin/env python2.7
#coding=utf-8

"""
Use a stored sentiment classifier to identifiy review positive and negative probability.
This module aim to extract review sentiment probability as review helpfulness features.

"""

import textProcessing as tp
import pickle
import itertools
import sklearn
import numpy
import scipy
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

#import sklearn


# 1. Load data
reviewDataSetPath='D:/ReviewHelpfulnessPrediction\ReviewSet/HTC_Z710t_review_2013.6.5.xlsx'
review = tp.get_excel_data(reviewDataSetPath, 1, 4, "data")
sentiment_review = tp.seg_fil_senti_excel(reviewDataSetPath, 1, 4,'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')


# 2. Feature extraction method
# Used for transform review to features, so it can calculate sentiment probability by classifier
# 计算整个语料里面每个词和双词搭配的信息量
# 以单个词语和出现频率为前5000双词作为特征
# return :
# 返回每个词以及得分
'''
return :
第五 1.64131573422
当是 4.8096346704
(u'\u624b\u52a8', u'\u5bfc\u5165') 0.831674969506
(u'\u4e4b\u8bcd', u'\u55b7') 0.831674969506
test code:
word_scores=create_words_bigrams_scores()
for word,score in word_scores.iteritems():
    print word,score
'''
# 可参考http://blog.csdn.net/chenglansky/article/details/31371033 里面解释相当详细
def create_words_bigrams_scores():
    posNegDir='D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet'
    posdata = tp.seg_fil_senti_excel(posNegDir+'/pos_review.xlsx', 1, 1,'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    # 获取积极数据
    # 去掉了一些停顿词，做了分词处理
    # input sense 3.0 很棒，真机其实不错。sense 3.0 确实很漂亮，4.3寸 16:9的屏幕很霸气也清晰，整体运行很流畅。现在软件兼容的问
    # output sense 3.0 很棒 真机 其实 不错 sense 3.0 确实 很漂亮 4.3 寸 16 9 屏幕 很 霸气 清晰 整体 运行 很 流畅 现在 软件 兼容 问题 几乎
    # for x in posdata[1]:
    #     print x,
    negdata = tp.seg_fil_senti_excel(posNegDir+'/neg_review.xlsx', 1, 1,'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    # for x in posdata[0]:
    #     print x
    
    posWords = list(itertools.chain(*posdata)) # 把多维数组解链成一维数组
    # print len(posWords)
    # for x in posWords:
    #     print x,
    negWords = list(itertools.chain(*negdata))
    # print len(negWords)


    # 把文本变成双词搭配的形式
    bigram_finder = BigramCollocationFinder.from_words(posWords)
    # 使用卡方统计方法，选择排名前5000的双词 5000为自设置的一个阈值
    posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    # for x in posBigrams:
    #     for w in x:
    #         print w,
    #     print ''
    # print len(posBigrams)
    bigram_finder = BigramCollocationFinder.from_words(negWords)

    #posBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    negBigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    # for x in negBigrams:
    #     for w in x:
    #         print w,
    #     print ''
    # print len(negBigrams)
    # 把所有词和双词搭配一起作为特征
    pos = posWords + posBigrams
    # for x in pos:
    #     print x
    neg = negWords + negBigrams
    # 进行特征选择
    word_fd = FreqDist() # 统计所有词词频

    cond_word_fd = ConditionalFreqDist() # 统计积极文本中词频和消极文本中词频
    for word in pos:
        #word_fd.inc(word)
        word_fd[word]+=1
        #cond_word_fd['pos'].inc(word)
        cond_word_fd['pos'][word]+=1
    for word in neg:
        #word_fd.inc(word)
        word_fd[word] += 1
        #cond_word_fd['neg'].inc(word)
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N() #获取积极词频数量
    neg_word_count = cond_word_fd['neg'].N() #获取消极词频数量
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        #print word,freq
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        # 函数怎么计算的 不知道
        # 计算积极词的卡方统计量，这里也可以计算互信息等其它统计量
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        # 计算消极词的卡方统计量，这里也可以计算互信息等其它统计量
        word_scores[word] = pos_score + neg_score
        # 一个词的信息量等于积极卡方统计量加上消极卡方统计量

    return word_scores

# 根据每个词信息量进行倒序排序，选择排名靠前的信息量的词，也即选择前number个
# 特征选择 降维 number为特征的维度 可不断调节
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda(w,s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words

# Initiallize word's information score and extracting most informative words
word_scores = create_words_bigrams_scores()
best_words = find_best_words(word_scores, 2000) # Be aware of the dimentions
#print len(best_words)
# for x in best_words:
#     print x,

# 把选出的这些词作为特征（这就是选择了信息量丰富的特征）
'''
test code:
for x in sentiment_review[1]:
    print x,
print ''
bwf=best_word_features(sentiment_review[1])
for x,b in bwf.iteritems():
    print x,b,
'''
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


# 3. Function that making the reviews to a feature set
# 数据集应处理成这种形式：[[明天,阳光],[],[],[],]
'''
test code:
feat=extract_features(sentiment_review)
for x in feat:
    for w,b in x.iteritems():
        print w,
    print ''
# 4. Load classifier
'''
def extract_features(dataset):
    feat = []
    for i in dataset:
        feat.append(best_word_features(i))
    return feat

# 4. Load classifier
#  装载分类器，得到SklearnClassifier
clf = pickle.load(open('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature/sentiment_classifier.pkl'))


# test_features=[{'about':1},{'that':0},{'this':0}]
# clf.prob_classify_many(test_features)

# Testing single review
#pred = clf.batch_classify(extract_features(sentiment_review[:2])) # An object contian positive and negative probabiliy
pred = clf.batch_prob_classify(extract_features(sentiment_review)) # An object contian positive and negative probabiliy
#pred = clf.prob_classify_many(extract_features(sentiment_review)) # An object contian positive and negative probabiliy

pred2 = []
for i in pred:
    pred2.append([i.prob('pos'), i.prob('neg')])

for r in review:
    print r
    print "pos probability score: %f" %pred2[review.index(r)][0]
    print "neg probability score: %f" %pred2[review.index(r)][1]
    print

    
# 5. Store review sentiment probabilty socre as review helpfulness features
def store_sentiment_prob_feature(sentiment_dataset, storepath):
    pred = clf.batch_prob_classify(extract_features(sentiment_dataset))
    p_file = open(storepath, 'w')
    reviewCount=0
    for i in pred:
        reviewCount+=1
        p_file.write(str(i.prob('pos')) + ' ' + str(i.prob('neg')) + '\n')
    p_file.close()
    return reviewCount
storeProPath='D:/ReviewHelpfulnessPrediction\ReviewDataFeature/HTC_SenProbFea.txt'
recordNum=store_sentiment_prob_feature(sentiment_review,storeProPath)
print recordNum

