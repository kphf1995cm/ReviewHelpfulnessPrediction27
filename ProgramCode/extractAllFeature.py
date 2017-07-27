#! /usr/bin/env python2.7
#coding=utf-8

import adjAdvVFeature
import centroidScoreFeature
import predictDataPosNegProbility
import wordSentenceLengthFeature
import posNegSentiDictFeature
import numpy as np

def getLabelDataFeature():
    reviewDataSetDir = 'D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\MachineLearningFeature\SenimentReviewSet'
    reviewDataSetName = ['pos_review','neg_review']
    classify_tag=['pos','neg']
    reviewDataSetFileType = '.xlsx'
    desDir = 'D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
    sheetNum=1
    sheetColNum=1
    # 存储积极评论里的特征
    feature_txt_name = ['AdjAdvVFea.txt', 'CentroidTempRes.txt', 'ClassPro.txt', 'WordSentNumFea.txt','SentiDictFea.txt']
    all_data_tag_feature=[]
    for pos in range(len(classify_tag)):
        adjAdvVFeature.store_adj_adv_v_num_feature(reviewDataSetDir, reviewDataSetName[pos], reviewDataSetFileType, sheetNum,sheetColNum, desDir)
        centroidScoreFeature.store_centroid_score(reviewDataSetDir, reviewDataSetName[pos], reviewDataSetFileType, sheetNum,sheetColNum, desDir)
        predictDataPosNegProbility.predictDataSentimentPro(reviewDataSetDir, reviewDataSetName[pos], reviewDataSetFileType,sheetNum, sheetColNum, desDir)
        wordSentenceLengthFeature.store_word_sent_num_features(reviewDataSetDir, reviewDataSetName[pos],reviewDataSetFileType, sheetNum, sheetColNum, desDir)
        posNegSentiDictFeature.read_review_set_and_store_score(reviewDataSetDir, reviewDataSetName[pos],reviewDataSetFileType, sheetNum, sheetColNum, desDir)
        src_feature=[]
        for x in feature_txt_name:
            feature_txt_path=desDir+'/'+reviewDataSetName[pos]+x
            f=open(feature_txt_path)
            data=f.readlines()
            print data
            src_feature.append(np.loadtxt(f))
            f.close()
        rowNum=len(src_feature)
        colNum=len(src_feature[0])
        for colPos in range(colNum):
            single_row_tag_fea=[]
            single_row_tag_fea.append(classify_tag[pos])
            for rowPos in rowNum:
                for x in src_feature[rowPos][colPos]:
                    single_row_tag_fea.append(x)
            all_data_tag_feature.append(single_row_tag_fea)

    for x in all_data_tag_feature:
        for y in x:
            print y,
        print ''

getLabelDataFeature()

