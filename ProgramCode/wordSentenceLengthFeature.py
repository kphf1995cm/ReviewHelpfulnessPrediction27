#! /usr/bin/env python2.7
#coding=utf-8

"""
Counting review's word number, sentence number and review length
This module aim to extract review's word number and sentence number and review length features.

"""

import textProcessing as tp


# Function counting review word number, sentence number and review length
def word_sent_count(dataset):
    word_sent_count = []
    for review in dataset:
        sents = tp.cut_sentence_2(review)# 切割成句子
        words = tp.segmentation(review,'list')#切割成词语
        sent_num = len(sents)
        word_num = len(words)
        sent_word = float(word_num)/float(sent_num)  # review length = word number/sentence number 也即每个句子平均含有词语数量
        word_sent_count.append([word_num, sent_num, sent_word])
    return word_sent_count


# Store features
def store_word_sent_num_features(filepath, sheetnum, colnum, storepath):
    data = tp.get_excel_data(filepath, sheetnum, colnum, 'data')
    word_sent_num = word_sent_count(data) # Need initiallized

    f = open(storepath,'w')
    reviewNum=0
    for i in word_sent_num:
        f.write(str(i[0])+' '+str(i[1])+' '+str(i[2])+'\n')
        reviewNum+=1
    f.close()
    return reviewNum

reviewDataSetPath='D:/ReviewHelpfulnessPrediction\ReviewSet/HTC_Z710t_review_2013.6.5.xlsx'
storeAdjPath='D:/ReviewHelpfulnessPrediction\ReviewDataFeature/HTC_WordSenLenFea.txt'
recordNum=store_word_sent_num_features(reviewDataSetPath,1,4,storeAdjPath)
print recordNum
