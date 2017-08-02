#! /usr/bin/env python2.7
#coding=utf-8

import adjAdvVFeature
import centroidScoreFeature
import predictDataPosNegProbility
import wordSentenceLengthFeature
import posNegSentiDictFeature
import numpy as np
import time

'''提取所有的特征 将其保存在D:/ReviewHelpfulnessPrediction\ReviewDataFeature/AllFeatureLabelData.txt'''
'''如果训练数据（人工标注数据）发生更改，需要修改相应的数据所在的目录,即以下几项需要更改
    reviewDataSetDir = 'D:/ReviewHelpfulnessPrediction\LabelReviewData'
    reviewDataSetName = ['posNegLabelData','posNegLabelData']
    classify_tag=['1','0']# pos 为 1，neg 为 0
    reviewDataSetFileType = '.xls'
   注意：predictDataPosNegProbility这个模块里面的训练数据也需要更改，以修正best_words
'''
def getLabelDataFeature():
    begin=time.clock()
    reviewDataSetDir = 'D:/ReviewHelpfulnessPrediction\LabelReviewData'
    reviewDataSetName = ['posNegLabelData','posNegLabelData']
    classify_tag=['1','0']# pos 为 1，neg 为 0
    reviewDataSetFileType = '.xls'
    desDir = 'D:\ReviewHelpfulnessPrediction\ReviewDataFeature'
    sheetNum=1
    sheetColNum=1
    # 存储积极评论里的特征
    #feature_txt_name = ['AdjAdvVFea.txt', 'CentroidScoreFea.txt', 'ClassPro.txt', 'WordSentNumFea.txt','SentiDictFea.txt']
    feature_txt_name = ['AdjAdvVFea.txt', 'ClassPro.txt', 'WordSentNumFea.txt',
                        'SentiDictFea.txt']
    all_data_tag_feature=[]
    for pos in range(len(classify_tag)):
        sheetNum=pos+1
        adjAdvVFeature.store_adj_adv_v_num_feature(reviewDataSetDir, reviewDataSetName[pos], reviewDataSetFileType, sheetNum,sheetColNum, desDir)
        centroidScoreFeature.store_centroid_score(reviewDataSetDir, reviewDataSetName[pos], reviewDataSetFileType, sheetNum,sheetColNum, desDir)
        predictDataPosNegProbility.predictDataSentimentPro(reviewDataSetDir, reviewDataSetName[pos], reviewDataSetFileType,sheetNum, sheetColNum, desDir)
        wordSentenceLengthFeature.store_word_sent_num_features(reviewDataSetDir, reviewDataSetName[pos],reviewDataSetFileType, sheetNum, sheetColNum, desDir)
        posNegSentiDictFeature.read_review_set_and_store_score(reviewDataSetDir, reviewDataSetName[pos],reviewDataSetFileType, sheetNum, sheetColNum, desDir)
        src_feature=[]
        for x in feature_txt_name:
            feature_txt_path=desDir+'/'+reviewDataSetName[pos]+x
            print feature_txt_path
            f=open(feature_txt_path)
            txt_data=np.loadtxt(f,delimiter='\t')
            src_feature.append(txt_data)
            f.close()
        rowNum=len(src_feature)
        colNum=len(src_feature[0])
        print rowNum,colNum
        for colPos in range(colNum):
            single_row_tag_fea=[]
            single_row_tag_fea.append(classify_tag[pos])
            for rowPos in range(rowNum):
                for arrayPos in range(src_feature[rowPos][colPos].size):
                    single_row_tag_fea.append(src_feature[rowPos][colPos][arrayPos])
            all_data_tag_feature.append(single_row_tag_fea)

    allDataFeaturePath='D:/ReviewHelpfulnessPrediction\ReviewDataFeature/AllFeatureLabelData.txt'
    f=open(allDataFeaturePath,'w')
    for x in all_data_tag_feature:
        for y in x:
            f.write(str(y)+'\t')
        f.write('\n')
    f.close()
    end=time.clock()
    return colNum,end-begin
recordNum,runningTime=getLabelDataFeature()
print recordNum,runningTime

