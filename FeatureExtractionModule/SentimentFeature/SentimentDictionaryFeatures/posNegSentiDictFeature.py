#! /usr/bin/env python2.7
#coding=utf-8

"""
Compute a review's positive and negative score, their average score and standard deviation.
This module aim to extract review positive/negative score, average score and standard deviation features (all 6 features).
Sentiment analysis based on sentiment dictionary.

"""
import PreprocessingModule.textProcessing as tp
#import textProcessing as np
import numpy as np

# 1. Load dictionary and dataset
# Load sentiment dictionary
posdict = tp.get_txt_data("D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\PositiveAndNegativeDictionary/posdict.txt","lines")
negdict = tp.get_txt_data("D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\PositiveAndNegativeDictionary/negdict.txt","lines")

# Load AdverbsOfDegreeDictionary
mostdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/most.txt', 'lines')
verydict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/very.txt', 'lines')
moredict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/more.txt', 'lines')
ishdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/ish.txt', 'lines')
insufficientdict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/insufficiently.txt', 'lines')
inversedict = tp.get_txt_data('D:/ReviewHelpfulnessPrediction\FeatureExtractionModule\SentimentFeature\SentimentDictionaryFeatures\SentimentDictionary\AdverbsOfDegreeDictionary/inverse.txt', 'lines')

# Load dataset
#review = tp.get_excel_data("D:/ReviewHelpfulnessPrediction/ReviewSet/HTC_Z710t_review_2013.6.5.xlsx", 1,4, "data")

# 2. Sentiment dictionary analysis basic function
# Function of matching adverbs of degree and set weights
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

# Function of transforming negative score to positive score
# Example: [5, -2] →  [7, 0]; [-4, 8] →  [0, 12]
# 可能有bug
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


# 3.1 Single review's positive and negative score
# Function of calculating review's every sentence sentiment score
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
def single_review_sentiment_score(review):
	single_review_senti_score = []
	cuted_review = tp.cut_sentence_2(review)

	for sent in cuted_review:
		seg_sent = tp.segmentation(sent, 'list')
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

		    # Match "!" in the review, every "!" has a weight of +2
		    elif word == "！" or word == "!":
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

# Testing
#print(single_review_sentiment_score(review[1]))


# 3.2 All review dataset's sentiment score
'''
test code:
score_list=sentence_sentiment_score(review)
for x in score_list:
	print(x)
'''
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

                elif word == '！' or word == '!':
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

# Compute a all review's sentiment score
'''
function:Compute a all review's sentiment score
test code:
for i in all_review_sentiment_score(sentence_sentiment_score(review)):
	print(i)
'''
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


# 4. Store sentiment dictionary features
def store_sentiment_dictionary_score(review_set, storepath):
	sentiment_score = all_review_sentiment_score(sentence_sentiment_score(review_set))

	f = open(storepath,'w')
	for i in sentiment_score:
	    f.write(str(i[0])+'\t'+str(i[1])+'\t'+str(i[2])+'\t'+str(i[3])+'\t'+str(i[4])+'\t'+str(i[5])+'\n')
	f.close()

'''
function: read review data set and store score data
test code:
read_review_set_and_store_score("D:/ReviewHelpfulnessPrediction/ReviewSet/HTC_Z710t_review_2013.6.5.xlsx", 1,4,'D:/ReviewHelpfulnessPrediction\ReviewSetScore/HTC.txt')
'''
def read_review_set_and_store_score(dataSetPath,sheetNum,colNum,scoreStorePath):
	review = tp.get_excel_data(dataSetPath, sheetNum, colNum, "data")
	store_sentiment_dictionary_score(review,scoreStorePath)

read_review_set_and_store_score("D:/ReviewHelpfulnessPrediction/ReviewSet/HTC_Z710t_review_2013.6.5.xlsx", 1,4,'D:/ReviewHelpfulnessPrediction\ReviewSetScore/HTC.txt')
