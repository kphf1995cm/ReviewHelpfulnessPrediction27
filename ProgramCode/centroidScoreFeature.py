#! /usr/bin/env python2.7
#coding=utf-8

"""
Compute review centroid score by combinating every word's tfidf score.
This module use filtered review data in a txt file and gensim tf-idf model to extract this review feature.

"""
'''TF_IDF理解 参考链接 http://blog.csdn.net/zrc199021/article/details/53728499'''
'''TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。'''
'''
   TF_IDF越大，反映词语越重要，越能反映主题
   词频 TF= 在某一个文档中词条w出现的次数/该文档中词条总数
   逆向文件频率 IDF=log（语料库中文档总数/(包含词条w的文档数+1)）
   TF_IDF=TF*IDF
'''
import textProcessing as tp
import logging
import time
from gensim import corpora, models, similarities


"""
1. Create a txt file with seg and filtered reviews
input: An excel file with product reviews
    手机很好，很喜欢。
    三防出色，操作系统垃圾！
    Defy用过3年感受。。。
    刚买很兴奋。当时还流行，机还很贵
output: A txt file store filtered reviews, every line is a review
    手机 很 好 很 喜欢 
    三防 出色 操作系统 垃圾 
    Defy 用过 3 年 感受 
    刚买 很 兴奋 当时 还 流行 机 还 很 贵
"""
 
def store_seg_fil_result(filepath, sheetnum, colnum, storepath):
    # Read excel file of review and segmention and filter stopwords
    seg_fil_result = tp.seg_fil_excel(filepath, sheetnum, colnum)
 
    # Store filtered reviews
    fil_file = open(storepath, 'w')
    for sent in seg_fil_result:
        for word in sent:
            fil_file.write(word.encode('utf8')+' ')
        fil_file.write('\n')
    fil_file.close()
    return len(seg_fil_result)

"""
input: A txt file store filtered reviews as corpus
        手机 很 好 很 喜欢 
        三防 出色 操作系统 垃圾 
        Defy 用过 3 年 感受 
        刚买 很 兴奋 当时 还 流行 机 还 很 贵
output: A list of tfidf total score of every review (store in a txt file)
"""
def centroid(datapath, storepath):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Read review data from txt file
    class MyCorpus(object):
        def __iter__(self):
            for line in open(datapath):
                yield line.split()

    # Change review data to gensim corpus format
    Corp = MyCorpus()
    # for x in Corp:
    #     for y in x:
    #         print y,
    #     print ''
    # print ''
    dictionary = corpora.Dictionary(Corp)# 根据字典将文档中的词转换为对应的数字
    # print type(dictionary)
    # for x in dictionary:
    #     print type(x),x,
    # print ''
    corpus = [dictionary.doc2bow(text) for text in Corp] #统计文档中每个词的个数
    # for x in corpus:
    #     print type(x),x,
    # print ''

    # Make the corpus become a tf-idf model
    tfidf = models.TfidfModel(corpus)

    # Compute every word's tf-idf score
    corpus_tfidf = tfidf[corpus]
    # for x in corpus_tfidf:
    #     print x,
    # print ''

    # Compute review centroid score by combinating every word's tf-idf score
    centroid = 0
    review_centroid = []
    for doc in corpus_tfidf:
        for token in doc:
            centroid += token[1]
        review_centroid.append(centroid)
        centroid = 0
    # for x in review_centroid:
    #     print x,
    # print ''

    # Store review centroid score into a txt file
    centroid_file = open(storepath, 'w')
    for i in review_centroid:
        centroid_file.write(str(i)+'\n')
    centroid_file.close()

def store_centroid_score(dataSetDir,dataSetName,dataSetFileType,sheetNum,colNum,dstDir):
    start=time.clock()
    filepath = dataSetDir + '/' + dataSetName + dataSetFileType
    temprespath=dstDir + '/' + dataSetName + 'CentroidTempRes.txt'
    reviewNum=store_seg_fil_result(filepath,sheetNum,colNum,temprespath)
    storepath = dstDir + '/' + dataSetName + 'CentroidScoreFea.txt'
    centroid(temprespath,storepath)
    end=time.clock()
    return reviewNum,end-start

# reviewDataSetDir='D:/ReviewHelpfulnessPrediction\ReviewSet'
# reviewDataSetName='HTC_Z710t_review_2013.6.5'
# reviewDataSetFileType='.xlsx'
# desDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
# recordNum,runningTime=store_centroid_score(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,1,5,desDir)
# print 'handle sentences num:',recordNum,' running time:',runningTime
