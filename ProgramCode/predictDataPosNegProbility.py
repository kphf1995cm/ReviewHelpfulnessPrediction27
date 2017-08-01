#! /usr/bin/env python2.7
#coding=utf-8

"""
Use a stored sentiment classifier to identifiy review positive and negative probability.
"""

import textProcessing as tp
import pickle
import itertools
import numpy as np
import time
import chardet
import xlwt
from random import shuffle

import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

'''几点注意说明'''
'''
如果人工标注的数据（训练数据）发生更改，需要更改create_word_bigram_scores()函数里面的posdata，negdata来重新计算词语信息得分
需要以要预测数据所在的路劲作为参数
'''

'''1 导入要预测的数据，并将数据做分词以及去停用词处理，得到[[word1,word2,],[],]'''
#reviewDataSetPath='D:/ReviewHelpfulnessPrediction\ReviewSet/HTC_Z710t_review_2013.6.5.xlsx'
#sentiment_review = tp.seg_fil_senti_excel(reviewDataSetPath, 1, 4,'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')


'''计算 单个词语以及二元词语的信息增量得分'''
'''注意 需要导入带标签的积极以及消极评论语料库(如果训练数据发生修改的话，里面的相应参数需要修改)'''
def create_word_bigram_scores():
    posNegDir = 'D:/ReviewHelpfulnessPrediction\LabelReviewData'
    posdata = tp.seg_fil_senti_excel(posNegDir + '/posNegLabelData.xls', 1, 1,
                                     'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    negdata = tp.seg_fil_senti_excel(posNegDir + '/posNegLabelData.xls', 2, 1,
                                     'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')

    posWords = list(itertools.chain(*posdata))
    negWords = list(itertools.chain(*negdata))

    bigram_pos_finder = BigramCollocationFinder.from_words(posWords)
    posBigrams = bigram_pos_finder.nbest(BigramAssocMeasures.chi_sq, 5000)
    bigram_neg_finder = BigramCollocationFinder.from_words(negWords)
    negBigrams = bigram_neg_finder.nbest(BigramAssocMeasures.chi_sq, 5000)

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

'''挑选信息量大的前number个词语作为分类特征'''
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


'''2 特征提取，提取每句话里面的特征'''
'''两种方式 单词 单词+二元词'''
def best_word_features(words,best_words):
    return dict([(word, True) for word in words if word in best_words])
def best_word_features_com(words,best_words):
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3


''' 提取语句列表的特征'''
'''数据集应处理成这种形式：[[明天,天气],[],[],[],]'''
'''采用单词+二元词方式'''
def extract_features(dataset,best_words):
    feat = []
    for i in dataset:
        feat.append(best_word_features_com(i,best_words))
    return feat

'''3 分类预测'''
'''读取最佳分类器 最佳分类维度'''
def read_best_classifier_dimension():
    f = open('D:/ReviewHelpfulnessPrediction\BuildedClassifier/bestClassifierDimenAcc.txt')
    clf_dim_acc=f.readline()
    data=clf_dim_acc.split('\t')
    best_classifier=data[0]
    best_dimension=data[1]
    return best_classifier,best_dimension

start=time.clock()
best_classifier,best_dimension=read_best_classifier_dimension()
word_scores = create_word_bigram_scores() #计算词语信息得分
best_words = find_best_words(word_scores, int(best_dimension)) # 选取前best_dimension个信息得分高的词语作为特征 best_dimension根据最佳分类器的最佳维度来设定
end=time.clock()
print 'feature extract time:',end-start
'''输出类标签 分类概率 原始数据 原始数据特征 调试过程中采用'''
def predictDataSentimentPro(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,sheetNum,colNum,desDir):
    reviewDataSetPath=reviewDataSetDir+'/'+reviewDataSetName+reviewDataSetFileType
    oriDataPath=desDir+'/'+reviewDataSetName+'OriData.txt'
    oriDataFeaPath = desDir + '/' + reviewDataSetName + 'OriFea.txt'
    preResStorePath=desDir+'/'+reviewDataSetName+'ClassPro.txt'
    preTagStorePath=desDir+'/'+reviewDataSetName+'ClassTag.txt'
    start=time.clock()
    #reviewDataSetPath = 'D:/ReviewHelpfulnessPrediction\ReviewSet/HTC_Z710t_review_2013.6.5.xlsx'
    #reviewDataSetPath='D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet/pos_review.xlsx'
    review = tp.get_excel_data(reviewDataSetPath, sheetNum, colNum, "data")# 读取待分类数据
    #将待分类数据进行分词以及去停用词处理
    sentiment_review = tp.seg_fil_senti_excel(reviewDataSetPath, sheetNum, colNum, 'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    #提取待分类数据特征
    review_feature = extract_features(sentiment_review, best_words)
    #classifierPath = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature/sentiment_classifier.pkl'
    classifierPath='D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+best_classifier+'.pkl'
    #装载分类器
    clf = pickle.load(open(classifierPath))
    #分类之预测数据类标签
    data_tag=clf.batch_classify(review_feature)
    p_file = open(preTagStorePath, 'w')
    for i in data_tag:
        p_file.write(str(i)+ '\n')
    p_file.close()
    #分类之预测数据积极、消极可能性
    pred = clf.batch_prob_classify(review_feature)
    # 记录分类结果 积极可能性 消极可能性
    p_file = open(preResStorePath, 'w')
    reviewCount = 0
    for i in pred:
        reviewCount += 1
        p_file.write(str(i.prob('pos')) + '\t' + str(i.prob('neg')) + '\n')
    p_file.close()
    # 记录原始数据
    p_file = open(oriDataPath, 'w')
    for d in review:
        p_file.write(d.encode('utf-8')+'\n')
    p_file.close()
    p_file = open(oriDataFeaPath, 'w')
    # 记录原始数据特征提取结果
    for d in review_feature:
        for w,b,in d.iteritems():
            p_file.write(w.encode('utf-8') + ' '+str(b)+'\t')
        p_file.write('\n')
    p_file.close()
    end=time.clock()
    return reviewCount,end-start

'''只输出类标签 分类概率 实际应用中采用'''
def predDataSentPro(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,sheetNum,colNum,desDir):
    reviewDataSetPath=reviewDataSetDir+'/'+reviewDataSetName+reviewDataSetFileType
    preResStorePath=desDir+'/'+reviewDataSetName+'ClassPro.txt'
    preTagStorePath=desDir+'/'+reviewDataSetName+'ClassTag.txt'
    start=time.clock()
    sentiment_review = tp.seg_fil_senti_excel(reviewDataSetPath, sheetNum, colNum, 'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    #提取待分类数据特征
    review_feature = extract_features(sentiment_review, best_words)
    #classifierPath = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature/sentiment_classifier.pkl'
    classifierPath='D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+best_classifier+'.pkl'
    #装载分类器
    clf = pickle.load(open(classifierPath))
    #分类之预测数据类标签
    data_tag=clf.batch_classify(review_feature)
    p_file = open(preTagStorePath, 'w')
    for i in data_tag:
        p_file.write(str(i)+ '\n')
    p_file.close()
    #分类之预测数据积极、消极可能性
    pred = clf.batch_prob_classify(review_feature)
    # 记录分类结果 积极可能性 消极可能性
    p_file = open(preResStorePath, 'w')
    reviewCount = 0
    for i in pred:
        reviewCount += 1
        p_file.write(str(i.prob('pos')) + '\t' + str(i.prob('neg')) + '\n')
    p_file.close()
    end=time.clock()
    return reviewCount,end-start

def predTxtDataSentPro(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,desDir):
    reviewDataSetPath = reviewDataSetDir + '/' + reviewDataSetName + reviewDataSetFileType
    oriDataPath = desDir + '/' + reviewDataSetName + 'OriData.txt'
    oriDataFeaPath = desDir + '/' + reviewDataSetName + 'OriFea.txt'
    preResStorePath = desDir + '/' + reviewDataSetName + 'ClassPro.txt'
    preTagStorePath = desDir + '/' + reviewDataSetName + 'ClassTag.txt'
    start = time.clock()
    # reviewDataSetPath = 'D:/ReviewHelpfulnessPrediction\ReviewSet/HTC_Z710t_review_2013.6.5.xlsx'
    # reviewDataSetPath='D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet/pos_review.xlsx'
    review = tp.get_txt_data(reviewDataSetPath, "lines")  # 读取待分类数据
    # 将待分类数据进行分词以及去停用词处理
    sentiment_review = tp.seg_fil_txt(reviewDataSetPath,'lines')
    # 提取待分类数据特征
    review_feature = extract_features(sentiment_review, best_words)
    # classifierPath = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature/sentiment_classifier.pkl'
    classifierPath = 'D:/ReviewHelpfulnessPrediction\BuildedClassifier/' + best_classifier + '.pkl'
    # 装载分类器
    clf = pickle.load(open(classifierPath))
    # 分类之预测数据类标签
    data_tag = clf.batch_classify(review_feature)
    p_file = open(preTagStorePath, 'w')
    for i in data_tag:
        p_file.write(str(i) + '\n')
    p_file.close()
    # 分类之预测数据积极、消极可能性
    pred = clf.batch_prob_classify(review_feature)
    # 记录分类结果 积极可能性 消极可能性
    p_file = open(preResStorePath, 'w')
    reviewCount = 0
    for i in pred:
        reviewCount += 1
        p_file.write(str(i.prob('pos')) + '\t' + str(i.prob('neg')) + '\n')
    p_file.close()
    # 记录原始数据
    p_file = open(oriDataPath, 'w')
    for d in review:
        p_file.write(d.encode('utf-8') + '\n')
    p_file.close()
    p_file = open(oriDataFeaPath, 'w')
    # 记录原始数据特征提取结果
    for d in review_feature:
        for w, b, in d.iteritems():
            p_file.write(w.encode('utf-8') + ' ' + str(b) + '\t')
        p_file.write('\n')
    p_file.close()
    end = time.clock()
    return reviewCount, end - start

# reviewDataSetDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
# reviewDataSetName='FiltnewoutOriData'
# reviewDataSetFileType='.txt'
# desDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
# recordNum,runningTime=predTxtDataSentPro(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,desDir)
# print 'handle sentences num:',recordNum,' running time:',runningTime

'''输出类标签 分类概率 原始数据 原始数据特征 将结果保存在excel文件中'''
def predictDataSentTagProToExcel(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,sheetNum,colNum,desDir):
    reviewDataSetPath=reviewDataSetDir+'/'+reviewDataSetName+reviewDataSetFileType
    preDataResPath=desDir+'/'+reviewDataSetName+'RawDataTagProFea.xls'
    start=time.clock()
    review = tp.get_excel_data(reviewDataSetPath, sheetNum, colNum, "data")# 读取待分类数据
    #将待分类数据进行分词以及去停用词处理
    sentiment_review = tp.seg_fil_senti_excel(reviewDataSetPath, sheetNum, colNum, 'D:/ReviewHelpfulnessPrediction/PreprocessingModule/sentiment_stopword.txt')
    #提取待分类数据特征
    review_feature = extract_features(sentiment_review, best_words)
    #classifierPath = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature/sentiment_classifier.pkl'
    classifierPath='D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+best_classifier+'.pkl'
    #装载分类器
    clf = pickle.load(open(classifierPath))
    dataItemCount=len(sentiment_review)
    #分类之预测数据类标签
    data_tag=clf.batch_classify(review_feature)
    #分类之预测数据积极、消极可能性
    res_pro = clf.batch_prob_classify(review_feature)
    # 记录分类结果 积极可能性 消极可能性
    # 记录原始数据
    # 记录原始数据特征提取结果
    # for d in review_feature:
    #     for w,b,in d.iteritems():
    #         p_file.write(w.encode('utf-8') + ' '+str(b)+'\t')
    #     p_file.write('\n')
    # p_file.close()
    preResFile=xlwt.Workbook(encoding='utf-8')
    preResSheet=preResFile.add_sheet('RawDataTagProFea')
    for rowPos in range(dataItemCount):
        preResSheet.write(rowPos,0,review[rowPos])#原始数据
        preResSheet.write(rowPos,1,data_tag[rowPos])#类标签
        preResSheet.write(rowPos,2,str(res_pro[rowPos].prob('pos')))#积极概率
        preResSheet.write(rowPos, 3, str(res_pro[rowPos].prob('neg')))#消极概率
        feature=''
        #feature='_'.join(review_feature[rowPos].keys())
       # print type(review_feature[rowPos].keys()),
        # 特征里面可能出现二元词的情况
        for x in review_feature[rowPos].keys():
            if type(x) is not nltk.types.TupleType:
                feature+=x
            else:
                feature+='_'.join(x)
            feature+=' '
        preResSheet.write(rowPos, 4, feature)#特征
    preResFile.save(preDataResPath)
    end=time.clock()
    return dataItemCount,end-start

reviewDataSetDir='D:/ReviewHelpfulnessPrediction\LabelReviewData'
reviewDataSetName='pdd_label_data'
reviewDataSetFileType='.xls'
desDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
recordNum,runningTime=predictDataSentTagProToExcel(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,1,1,desDir)
print 'handle sentences num:',recordNum,' classify time:',runningTime

