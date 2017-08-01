#! /usr/bin/env python2.7
#coding=utf-8

import textProcessing as tp
import time
import numpy as np
import xlwt
import xlrd
import pickle
import itertools
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

'''预测未标记数据的类别 大致过程如下：'''
'''1  装载数据，数据预处理（分词及去停用词)'''
'''2  提取特征(程度词性个数特征、句子个数及词语数量特征、基于词典的情感得分特征、积极消极可能性特征)'''
'''3  装载分类器 分类预测'''
'''4  保存结果，并绘制情感波动曲线'''

'''特征提取模块的函数'''

'''a 提取形容词、副词、动词数量特征'''
'''返回 形容词 副词 动词 特征列表[[adjNum,advNum,vNum],[],],其中参数rawData为原始数据列表（未经分词处理）'''
'''在处理弹幕数据时，时间性能大致1s可以处理1000条数据(词性标注比较耗时 看看可否优化（tp.postagger(review, 'list')）)'''
def count_adj_adv_v(rawData):
    begin=time.clock()
    adj_adv_num = []
    a = 0
    d = 0
    v = 0
    for review in rawData:
        pos = tp.postagger(review, 'list')
        for i in pos:
            if i[1] == 'a':
                a += 1
            elif i[1] == 'd':
                d += 1
            elif i[1] == 'v':
                v += 1
        adj_adv_num.append((a, d, v))
        a = 0
        d = 0
        v = 0
    end=time.clock()
    print 'extract adj_adv_v_num feature time is:',end-begin,'handle data item num is:',len(rawData)
    return adj_adv_num


'''b 提取句子、词语数量、句子平均词语数特征'''
'''返回句子、词语数量、句子平均词语数特征列表,其中参数rawData为原始数据列表（未经分词处理）'''
'''在处理弹幕数据时，时间性能大致0.1s可以处理1000条数据'''
def word_sent_count(rawData):
    begin=time.clock()
    word_sent_count = []
    for review in rawData:
        sents = tp.cut_sentence_2(review)# 切割成句子
        words = tp.segmentation(review,'list')#切割成词语
        sent_num = len(sents)
        word_num = len(words)
        sent_word = float(word_num)/float(sent_num)  # review length = word number/sentence number 也即每个句子平均含有词语数量
        word_sent_count.append([word_num, sent_num, sent_word])
    end=time.clock()
    print 'extract word_sent_count feature time is:', end - begin, 'handle data item num is:', len(rawData)
    return word_sent_count


'''
c 提取积极、消极得分，平均得分，标准偏差特征
  模块目标是提取一条评论的 positive/negative score, average score and standard deviation features (all 6 features)
  情感分析依赖于情感词典
'''
'''导入情感词典 情感词典、程度词字典作为全局变量'''
posdict = tp.get_txt_data("D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\PositiveAndNegativeDictionary/posdict.txt","lines")
negdict = tp.get_txt_data("D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\PositiveAndNegativeDictionary/negdict.txt","lines")
'''导入形容词、副词、否定词等程度词字典'''
mostdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/most.txt', 'lines')
verydict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/very.txt', 'lines')
moredict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/more.txt', 'lines')
ishdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/ish.txt', 'lines')
insufficientdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/insufficiently.txt', 'lines')
inversedict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/inverse.txt', 'lines')
'''匹配程度词并设置权重'''
'''parm：word  当前情感词的前面词语 sentiment_value 当前情感词的情感值'''
def match(word, sentiment_value):
	if word in mostdict:
		sentiment_value *= 2.0
	elif word in verydict:
	    sentiment_value *= 1.5
	elif word in moredict:
	    sentiment_value *= 1.25
	elif word in ishdict:
	    sentiment_value *= 0.5
	elif word in insufficientdict:
	    sentiment_value *= 0.25
	elif word in inversedict:
	    sentiment_value *= -1
	return sentiment_value
'''将得分正数化 Example: [5, -2] →  [7, 0]; [-4, 8] →  [0, 12]'''
def transform_to_positive_num(poscount, negcount):
    pos_count = 0
    neg_count = 0
    if poscount < 0 and negcount >= 0:
        neg_count = negcount - poscount #bug
        pos_count = 0
    elif negcount < 0 and poscount >= 0:
        pos_count = poscount - negcount
        neg_count = 0
    elif poscount < 0 and negcount < 0:
        neg_count = -poscount
        pos_count = -negcount
    else:
        pos_count = poscount
        neg_count = negcount
    return [pos_count, neg_count]
'''计算全部评论的情感得分列表'''
'''返回 [[[pos,neg],[pos,neg],],[],]'''
def sentence_sentiment_score(rawData):
    cuted_review = []
    for cell in rawData:
        cuted_review.append(tp.cut_sentence_2(cell))
    all_review_count = []
    for review in cuted_review:
        single_review_count = []
        for sent in review:
            seg_sent = tp.segmentation(sent, 'list')
            i = 0  # word position counter
            a = 0  # sentiment word position
            poscount = 0  # count a pos word
            negcount = 0
            for word in seg_sent:
                if word in posdict:
                    poscount += 1
                    for w in seg_sent[a:i]:
                        poscount = match(w, poscount)
                    a = i + 1

                elif word in negdict:
                    negcount += 1
                    for w in seg_sent[a:i]:
                        negcount = match(w, negcount)
                    a = i + 1

                elif word == '！'.decode('utf8') or word == '!'.decode('utf8'):
                    for w2 in seg_sent[::-1]:
                        if w2 in posdict:
                            poscount += 2
                            break
                        elif w2 in negdict:
                            negcount += 2
                            break
                i += 1

            single_review_count.append(transform_to_positive_num(poscount, negcount))  # [[s1_score], [s2_score], ...]
        all_review_count.append(
            single_review_count)  # [[[s11_score], [s12_score], ...], [[s21_score], [s22_score], ...], ...]
    return all_review_count
'''计算全部评论的特征列表  参数格式为[[[pos,neg],[pos,neg],],[],]'''
'''返回[[Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg],[],]'''
def all_review_sentiment_score(senti_score_list):
    score = []
    for review in senti_score_list:
        score_array = np.array(review)
        Pos = np.sum(score_array[:,0])
        Neg = np.sum(score_array[:,1])
        AvgPos = np.mean(score_array[:,0])
        AvgNeg = np.mean(score_array[:,1])
        StdPos = np.std(score_array[:,0])
        StdNeg = np.std(score_array[:,1])
        score.append([Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg])
    return score
'''c 提取语句情感得分特征'''
'''返回 positive/negative score, average score and standard deviation features (all 6 features),其中参数rawData为原始数据列表（未经分词处理）'''
'''在处理弹幕数据时，时间性能大致1.5s可以处理1000条数据'''
def get_sent_score_fea(rawData):
    begin=time.clock()
    sent_score_fea=all_review_sentiment_score(sentence_sentiment_score(rawData))
    end=time.clock()
    print 'extract sent_score feature time is:', end - begin, 'handle data item num is:', len(rawData)
    return sent_score_fea


'''d 提取积极消极可能性特征'''
'''注意点：如果人工标注的数据（训练数据）发生更改，需要更改create_word_bigram_scores()函数里面的posdata，negdata来重新计算词语信息得分'''

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
'''提取每句话里面的特征 特征形式：单词+二元词'''
def best_word_features_com(words,best_words):
    d1 = dict([(word, True) for word in words if word in best_words])
    d2 = dict([(word, True) for word in nltk.bigrams(words) if word in best_words])
    d3 = dict(d1, **d2)
    return d3
''' 提取语句列表的特征'''
'''数据集(已做过分词以及去除停用词处理)形式：[[明天,天气],[],[],[],]'''
'''采用单词+二元词方式'''
def extract_features(segFiltData,best_words):
    feat = []
    for i in segFiltData:
        feat.append(best_word_features_com(i,best_words))
    return feat
'''读取最佳分类器 最佳分类维度 存储在D:/ReviewHelpfulnessPrediction\BuildedClassifier/bestClassifierDimenAcc.txt路径下'''
def read_best_classifier_dimension():
    f = open('D:/ReviewHelpfulnessPrediction\BuildedClassifier/bestClassifierDimenAcc.txt')
    clf_dim_acc=f.readline()
    data=clf_dim_acc.split('\t')
    best_classifier=data[0]
    best_dimension=data[1]
    return best_classifier,best_dimension
'''获得最佳信息得分词 最佳分类器 作为全局变量'''
start=time.clock()
best_classifier,best_dimension=read_best_classifier_dimension()
word_scores = create_word_bigram_scores() #计算词语信息得分
best_words = find_best_words(word_scores, int(best_dimension)) # 选取前best_dimension个信息得分高的词语作为特征 best_dimension根据最佳分类器的最佳维度来设定
end=time.clock()
print 'get best_words time:',end-start
'''获得数据积极消极可能性特征'''
'''  参数：segFiltData(已做过分词以及去除停用词处理)'''
'''返回数据形式：[[posPro,negPro],[]]'''
'''在处理弹幕数据时，时间性能大致0.05s可以处理1000条数据'''
def get_pos_neg_pro_fea(segFiltData):
    begin=time.clock()
    #提取待分类数据特征
    review_feature = extract_features(segFiltData, best_words)
    classifierPath='D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+best_classifier+'.pkl'
    #装载分类器
    clf = pickle.load(open(classifierPath))
    #分类之预测数据积极、消极可能性
    pred = clf.batch_prob_classify(review_feature)
    # 记录分类结果 积极可能性 消极可能性
    posNegPro=[]
    for i in range(len(pred)):
        posNegPro.append([pred[i].prob('pos'),pred[i].prob('neg')])
    end=time.clock()
    print 'extract pro_neg_pro feature time is:', end - begin, 'handle data item num is:', len(segFiltData)
    return posNegPro


'''装载数据,提取特征'''
'''原始数据保存格式为excel  第sheetNum第colNum列的每一行代表一项数据 '''
'''rawDataPath 为数据目录'''
'''在处理弹幕数据时，时间性能大致3s可以处理1000条数据'''
def loadRawDataExtractFea(rawDataPath,sheetNum,colNum):
    begin=time.clock()
    '''获取原始数据列表'''
    unlabedRawData = tp.get_excel_data(rawDataPath, sheetNum, colNum, 'data')
    '''获取经分词及去停用词处理后的数据列表'''
    unlabedSegFiltData = tp.seg_fil_excel(rawDataPath, sheetNum, colNum)
    word_adj_adv_v_num_fea=count_adj_adv_v(unlabedRawData)
    word_sent_fea=word_sent_count(unlabedRawData)
    word_sent_score_fea=get_sent_score_fea(unlabedRawData)
    word_pos_neg_pro_fea=get_pos_neg_pro_fea(unlabedSegFiltData)
    dataItemNum=len(unlabedRawData)
    dataItemFea=[]
    for i in range(dataItemNum):
        singleFea=[]
        for x in word_adj_adv_v_num_fea[i]:
            singleFea.append(x)
        for x in word_sent_fea[i]:
            singleFea.append(x)
        for x in word_sent_score_fea[i]:
            singleFea.append(x)
        for x in word_pos_neg_pro_fea[i]:
            singleFea.append(x)
        dataItemFea.append(singleFea)
    end=time.clock()
    print 'load data and extract all feature time is:', end - begin, 'handle data item num is:', dataItemNum
    return dataItemFea

rawDataPath='D:/ReviewHelpfulnessPrediction\LabelReviewData/pdd_label_data.xls'
sheetNum=1
colNum=1
data_all_fea=loadRawDataExtractFea(rawDataPath,sheetNum,colNum)
for x in data_all_fea:
    for y in x:
        print y,
    print ''









