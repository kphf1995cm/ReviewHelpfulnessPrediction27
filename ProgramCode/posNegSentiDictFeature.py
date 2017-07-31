#! /usr/bin/env python2.7
#coding=utf-8

"""
Compute a review's positive and negative score, their average score and standard deviation.
This module aim to extract review positive/negative score, average score and standard deviation features (all 6 features).
Sentiment analysis based on sentiment dictionary.
"""
'''
计算一条评论 积极、消极得分，平均得分，标准偏差
模块目标是提取一条评论的 positive/negative score, average score and standard deviation features (all 6 features)
情感分析依赖于情感词典
'''

import textProcessing as tp
import numpy as np
import time
import xlwt
import xlrd

'''1 导入情感词典以及数据集'''
'''导入情感词典'''
posdict = tp.get_txt_data("D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\PositiveAndNegativeDictionary/posdict.txt","lines")
negdict = tp.get_txt_data("D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\PositiveAndNegativeDictionary/negdict.txt","lines")

'''导入形容词、副词、否定词等程度词字典'''
mostdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/most.txt', 'lines')
verydict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/very.txt', 'lines')
moredict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/more.txt', 'lines')
ishdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/ish.txt', 'lines')
insufficientdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/insufficiently.txt', 'lines')
inversedict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/inverse.txt', 'lines')

'''导入数据集'''
review = tp.get_excel_data("D:/ReviewHelpfulnessPrediction/ReviewSet/HTC_Z710t_review_2013.6.5.xlsx", 1,4, "data")

'''2 基于字典的情感分析 基本功能'''

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


'''3 计算评论的情感特征'''

'''计算单条评论的情感特征，单条评论可能还有多个句子 score_list=[[pos1,neg1],[pos2,neg2],]'''
'''返回 [Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg]'''
def sumup_sentence_sentiment_score(score_list):
	score_array = np.array(score_list) # Change list to a numpy array
	Pos = np.sum(score_array[:,0]) # Compute positive score
	Neg = np.sum(score_array[:,1])
	AvgPos = np.mean(score_array[:,0]) # Compute review positive average score, average score = score/sentence number
	AvgNeg = np.mean(score_array[:,1])
	StdPos = np.std(score_array[:,0]) # Compute review positive standard deviation score
	StdNeg = np.std(score_array[:,1])
	return [Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg]

# 代码有问题,它是按照情感词个数来计算的，它更强调的是位于后面的情感词
# 计算单条评论情感得分
# input：除了电池不给力 都很好
# output:[1.5, 0.0, 0.75, 0.0, 0.75, 0.0]
# test code:print(single_review_sentiment_score(review[1]))
'''计算单条评论的得分列表 '''
'''返回 [Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg]'''
def single_review_sentiment_score(review):
	single_review_senti_score = []
	cuted_review = tp.cut_sentence_2(review)# 将评论切割成句子

	for sent in cuted_review:
		seg_sent = tp.segmentation(sent, 'list')# 将句子做分词处理
		i = 0 # word position counter
		s = 0 # sentiment word position
		poscount = 0 # count a positive word
		negcount = 0 # count a negative word

		for word in seg_sent:
		    if word in posdict:
		        poscount += 1
		        for w in seg_sent[s:i]:
		           poscount = match(w, poscount)
		        s = i + 1 # a是什么

		    elif word in negdict:
		        negcount += 1
		        for w in seg_sent[s:i]:
		        	negcount = match(w, negcount)
		        s = i + 1 # a是什么

		    # Match "!" in the review, every "!" has a weight of +2 ！强调句子情感
		    elif word == "！".decode('utf8') or word == "!".decode('utf8'):
		        for w2 in seg_sent[::-1]:
		            if w2 in posdict:
		            	poscount += 2
		            	break
		            elif w2 in negdict:
		                negcount += 2
		                break                    
		    i += 1

		single_review_senti_score.append(transform_to_positive_num(poscount, negcount))
		#print(sumup_sentence_sentiment_score(single_review_senti_score))
	review_sentiment_score = sumup_sentence_sentiment_score(single_review_senti_score)

	return review_sentiment_score


'''计算全部评论的情感得分列表'''
'''返回 [[[pos,neg],[pos,neg],],[],]'''
def sentence_sentiment_score(dataset):
    cuted_review = []
    for cell in dataset:
        cuted_review.append(tp.cut_sentence_2(cell))

    all_review_count = []
    for review in cuted_review:
        single_review_count=[]
        for sent in review:
            seg_sent = tp.segmentation(sent, 'list')
            i = 0 #word position counter
            a = 0 #sentiment word position
            poscount = 0 #count a pos word
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
                
            single_review_count.append(transform_to_positive_num(poscount, negcount)) #[[s1_score], [s2_score], ...]
        all_review_count.append(single_review_count) # [[[s11_score], [s12_score], ...], [[s21_score], [s22_score], ...], ...]

    return all_review_count

'''计算全部评论的特征列表'''
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


'''4 存储情感字典特征'''

'''parm@review_set; 评论列表，每条评论可以含有多个句子 '''
def store_sentiment_dictionary_score(review_set, storepath):
	score_list=sentence_sentiment_score(review_set)
	sentiment_score = all_review_sentiment_score(score_list)

	f = open(storepath,'w')
	reviewCount=0
	for i in sentiment_score:
		f.write(str(i[0])+'\t'+str(i[1])+'\t'+str(i[2])+'\t'+str(i[3])+'\t'+str(i[4])+'\t'+str(i[5])+'\n')
		reviewCount+=1
	f.close()
	return reviewCount
'''
function: read review data set and store score data
test code:
read_review_set_and_store_score("D:/ReviewHelpfulnessPrediction/ReviewSet/HTC_Z710t_review_2013.6.5.xlsx", 1,4,'D:/ReviewHelpfulnessPrediction\ReviewSetScore/HTC.txt')
'''
def read_review_set_and_store_score(dataSetDir,dataSetName,dataSetFileType,sheetNum,colNum,dstDir):
	start=time.clock()
	dataSetPath=dataSetDir+'/'+dataSetName+dataSetFileType
	dstPath=dstDir+'/'+dataSetName+'SentiDictFea.txt'
	review = tp.get_excel_data(dataSetPath, sheetNum, colNum, "data")
	# for x in review:
	# 	print x
	res=store_sentiment_dictionary_score(review,dstPath)
	end=time.clock()
	return res,end-start

# reviewDataSetDir='D:/ReviewHelpfulnessPrediction\ReviewSet'
# reviewDataSetName='HTC_Z710t_review_2013.6.5'
# reviewDataSetFileType='.xlsx'
# desDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
# recordNum,runningTime=read_review_set_and_store_score(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,1,4,desDir)
# print 'handle sentences num:',recordNum,' running time:',runningTime

def read_txt_review_set_and_store_score(dataSetDir,dataSetName,dataSetFileType,dstDir):
	start=time.clock()
	dataSetPath=dataSetDir+'/'+dataSetName+dataSetFileType
	dstPath=dstDir+'/'+dataSetName+'SentiDictFea.txt'
	review = tp.get_txt_data(dataSetPath,"lines")
	# for x in review:
	# 	print x
	res=store_sentiment_dictionary_score(review,dstPath)
	end=time.clock()
	return res,end-start

# reviewDataSetDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
# reviewDataSetName='FiltnewoutOriData'
# reviewDataSetFileType='.txt'
# desDir='D:/ReviewHelpfulnessPrediction\ReviewDataFeature'
# recordNum,runningTime=read_txt_review_set_and_store_score(reviewDataSetDir,reviewDataSetName,reviewDataSetFileType,desDir)
# print 'handle sentences num:',recordNum,' running time:',runningTime

'''过滤器 过滤掉不含主观情感的语句 客观语句'''
'''构建情感词典 这里只简单地分为积极和消极'''
sentiment_dict=posdict+negdict
'''判断单条评论是否为具备情感倾向语句 如果评论里有一个词位于情感词典中，则可认为该句具备情感倾向'''
def is_single_review_sentiment(review):
	cuted_review = tp.cut_sentence_2(review)# 将评论切割成句子
	for sent in cuted_review:
		seg_sent = tp.segmentation(sent, 'list')# 将句子做分词处理
		for word in seg_sent:
		    if word in  sentiment_dict:
		        return True
	return False

def filt_objective_sentence(srcpath,para,dstpath):
	begin=time.clock()
	raw_data=tp.get_txt_data(srcpath,para)
	f = open(dstpath, 'w')
	count=0
	for x in raw_data:
		if is_single_review_sentiment(x)==True:
			f.write(x.encode('utf-8') + '\n')
			count+=1
	f.close()
	end=time.clock()
	return end-begin,count

'''删除重复评论'''
'''将评论保存在excel文件中 格式为 评论：出现次数'''
def remove_duplicate_comment(srcpath,para,excelpath):
	begin = time.clock()
	raw_data = tp.get_txt_data(srcpath, para)
	review_diff_set={}
	pre_count=len(raw_data)
	cur_count=0
	for x in raw_data:
		if review_diff_set.has_key(x)==False:
			review_diff_set[x]=1
			cur_count+=1
		else:
			review_diff_set[x]+=1
	excel_file = xlwt.Workbook(encoding='utf-8')
	sheet_name = 'label_data'
	sheet_pos = 1
	excel_sheet = excel_file.add_sheet(sheet_name + str(sheet_pos))
	row_pos = 0
	excel_sheet.write(row_pos, 0, 'review_data')
	excel_sheet.write(row_pos, 1, 'review_count')
	row_pos += 1
	for w,c in review_diff_set.iteritems():
		if row_pos == 65536:
			sheet_pos += 1
			excel_sheet = excel_file.add_sheet(sheet_name + str(sheet_pos))
			row_pos = 0
			excel_sheet.write(row_pos, 0, 'review_data')
			excel_sheet.write(row_pos, 1, 'review_count')
			row_pos = 1
			excel_sheet.write(row_pos, 0, w)
			excel_sheet.write(row_pos, 1, str(c))
			row_pos += 1
		excel_sheet.write(row_pos, 0, w)
		excel_sheet.write(row_pos, 1, str(c))
		row_pos += 1
	excel_file.save(excelpath)

	end=time.clock()
	return pre_count,cur_count,end-begin
# pre_count,cur_count,running_time=remove_duplicate_comment('D:/ReviewHelpfulnessPrediction\ReviewDataFeature/FiltnewoutOriData.txt','lines','D:/ReviewHelpfulnessPrediction\LabelReviewData/label_review_count_data.xls')
# print pre_count,cur_count,running_time




#print filt_objective_sentence('D:/ReviewHelpfulnessPrediction\ReviewDataFeature/newoutOriData.txt','lines','D:/ReviewHelpfulnessPrediction\ReviewDataFeature/FiltnewoutOriData.txt')

'''检查标记数据 看看是否出现格式错误，如出现，显示出现错误的行数,并返回正确标记的数据'''
'''参数 labelRowNum为已标记的行数量'''
'''将标记数据按照主客观 积消极 鉴黄 分类存储在labelDataDir目录下'''


def judge_label_data(labelDataPath, labelRowNum, labelDataDir):
	table = xlrd.open_workbook(labelDataPath)
	sheet = table.sheets()[0]
	errorRow = []  # 错误行
	subjectiveSubDataItem = []  # 主观数据项
	subjectiveObjDataItem = []  # 客观数据项
	sentimentPosDataItem = []  # 积极数据项
	sentimentNegDataItem = []  # 消极数据项
	eroticEroDataItem = []  # 鉴黄
	eroticNorDataItem = []
	srcDataColPos = 0
	subjectiveColPos = 2
	sentimentColPos = 3
	eroticColPos = 4
	excelData = []
	for rowPos in range(1, labelRowNum):
		excelData.append(sheet.row_values(rowPos))
	for rowPos in range(0, labelRowNum - 1):
		if excelData[rowPos][subjectiveColPos] == 1:
			if excelData[rowPos][sentimentColPos] == 0:
				sentimentNegDataItem.append(excelData[rowPos][srcDataColPos])
			elif excelData[rowPos][sentimentColPos] == 1:
				sentimentPosDataItem.append(excelData[rowPos][srcDataColPos])
			else:
				errorRow.append([rowPos + 2, 'sentiment_tendency value error'])
			subjectiveSubDataItem.append(excelData[rowPos][srcDataColPos])
		elif excelData[rowPos][subjectiveColPos] == 0:
			subjectiveObjDataItem.append(excelData[rowPos][srcDataColPos])
		else:
			errorRow.append([rowPos + 2, 'is_subjective value error'])
		if excelData[rowPos][eroticColPos] == 1:
			eroticEroDataItem.append(excelData[rowPos][srcDataColPos])
		elif excelData[rowPos][eroticColPos] == 0:
			eroticNorDataItem.append(excelData[rowPos][srcDataColPos])
		else:
			errorRow.append([rowPos + 2, 'is_erotic value error'])
	for x in errorRow:
		print x
	print 'subjective and objective num:', len(subjectiveSubDataItem), len(subjectiveObjDataItem)
	print 'postive and negtive num:', len(sentimentPosDataItem), len(sentimentNegDataItem)
	print 'erotic and normal num:', len(eroticEroDataItem), len(eroticNorDataItem)
	colPos = 0
	'''存储主客观标注的数据'''
	subObjFile = xlwt.Workbook(encoding='utf-8')
	subjectiveSheet = subObjFile.add_sheet('subjective_data')
	for rowPos in range(len(subjectiveSubDataItem)):
		subjectiveSheet.write(rowPos, colPos, subjectiveSubDataItem[rowPos])
	objectiveSheet = subObjFile.add_sheet('objective_data')
	for rowPos in range(len(subjectiveObjDataItem)):
		objectiveSheet.write(rowPos, colPos, subjectiveObjDataItem[rowPos])
	subObjFile.save(labelDataDir + '/' + 'subObjLabelData.xls')

	'''存储积消极标注的数据'''
	posNegFile = xlwt.Workbook(encoding='utf-8')
	postiveSheet = posNegFile.add_sheet('postive_data')
	for rowPos in range(len(sentimentPosDataItem)):
		postiveSheet.write(rowPos, colPos, sentimentPosDataItem[rowPos])
	negtiveSheet = posNegFile.add_sheet('negtive_data')
	for rowPos in range(len(sentimentNegDataItem)):
		negtiveSheet.write(rowPos, colPos, sentimentNegDataItem[rowPos])
	posNegFile.save(labelDataDir + '/' + 'posNegLabelData.xls')

	'''存储鉴黄标注数据'''
	eroNorFile=xlwt.Workbook(encoding='utf-8')
	eroticSheet=eroNorFile.add_sheet('erotic_data')
	for rowPos in range(len(eroticEroDataItem)):
		eroticSheet.write(rowPos,colPos,eroticEroDataItem[rowPos])
	normalSheet=eroNorFile.add_sheet('normal_data')
	for rowPos in range(len(eroticNorDataItem)):
		normalSheet.write(rowPos,colPos,eroticNorDataItem[rowPos])
	eroNorFile.save(labelDataDir + '/' + 'eroNorLabelData.xls')


judge_label_data('D:/ReviewHelpfulnessPrediction\LabelReviewData/label_review_count_data.xls', 1200,
				 'D:/ReviewHelpfulnessPrediction\LabelReviewData')






