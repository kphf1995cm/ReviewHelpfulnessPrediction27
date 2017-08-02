#! /usr/bin/env python2.7
#coding=utf-8

"""
Use scikit-learn to test different classifier's review helpfulness prediction performance, and test different feature subset's prediction performance
This module is the last part of review helpfulness prediction research.
"""

import numpy as np
import xlwt
from random import shuffle
import pickle

from nltk import SklearnClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import cross_validation
from sklearn.metrics import f1_score, precision_score, recall_score
import time

'''注意：如果训练数据发生更改，需要更改data = read_data("D:/ReviewHelpfulnessPrediction\ReviewDataFeature/AllFeatureLabelData.txt")'''
def read_data(datapath):
	f = open(datapath)
	f.readline()
	data = np.loadtxt(f)
	return data
'''读取训练数据'''
data = read_data("D:/ReviewHelpfulnessPrediction\ReviewDataFeature/AllFeatureLabelData.txt")
# 2. Feature subset
# linguistic = data[:, 4:10]
# informative = np.hstack((data[:, 1:4], data[:, 20:21]))
# difference = data[:, 10:12]
# sentiment = data[:, 12:20]

# IDS = np.hstack((data[:, 1:4], data[:, 10:21]))
# LIS = np.hstack((data[:, 1:10], data[:, 12:21]))
# LDS = data[:, 4:20]
# LID = np.hstack((data[:, 1:12], data[:, 20:21]))

# LI = np.hstack((data[:, 1:10], data[:, 20:21]))
# LD = data[:, 4:12]
# LS = np.hstack((data[:, 4:10], data[:, 12:20]))
# ID = np.hstack((data[:, 1:4], data[:, 10:12], data[:, 20:21]))
# IS = np.hstack((data[:, 1:4], data[:, 12:21]))
# DS = data[:, 10:20]

# L1 = data[:, 4:7]
# L2 = data[:, 7:10]
# S1 = data[:, 12:14]
# S2 = data[:, 14:16]
# S3 = data[:, 16:18]
# S4 = data[:, 18:20]

# Sentiment feature subset
# S12 = data[:, 12:16]
# S13 = np.hstack((data[:, 12:14], data[:, 16:18]))
# S14 = np.hstack((data[:, 12:14], data[:, 18:20]))
# S23 = data[:, 14:18]
# S24 = np.hstack((data[:, 14:16], data[:, 18:20]))
# S34 = data[:, 16:20]

# SP = np.hstack((data[:, 12:13], data[:, 14:15], data[:, 16:17], data[:, 18:19]))
# SN = np.hstack((data[:, 13:14], data[:, 15:16], data[:, 17:18], data[:, 19:20]))


# 3. Load classifier
# 3.1 Classifier for binary classifiy
'''
clf = svm.SVC(gamma=0.001, C=100.)
clf = svm.SVR() # have bug
clf = LogisticRegression(penalty='l1', tol=0.01)
clf = tree.DecisionTreeClassifier()
clf = GaussianNB()
clf = BernoulliNB()
clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=1, random_state=0) # have bug
'''
#
# # 3.2 Classifier for mulit classify
# # clf = OneVsOneClassifier(svm.SVC(gamma=0.001, C=100.))
# # clf = OneVsOneClassifier(svm.SVR())
# # clf = OneVsRestClassifier(LogisticRegression(penalty='l1', tol=0.01))
#
#

# 4. Cross validate classifier's accuracy

# 5. Cross validate for all metrics, include precision, recall and f1 measure (macro, micro)
'''
功能：获取分类器的分类效果
参数：分类特征 类标签 分类器
'''
def metric_evaluation(feature, target,clf):
	k_fold = cross_validation.KFold(len(feature), n_folds=10) # 10-fold cross validation
	metric = []
	for train, test in k_fold:
		target_pred = clf.fit(feature[train], target[train]).predict(feature[test]) # 训练分类器并预测测试集类标签
		p = precision_score(target[test], target_pred,average='macro')
		r = recall_score(target[test], target_pred,average='macro')
		f1_macro = f1_score(target[test], target_pred, average='macro')
		f1_micro = f1_score(target[test], target_pred, average='micro')
		metric.append([p,r,f1_macro,f1_micro])

	metric_array = np.array(metric)
	presionScore=np.mean(metric_array[:, 0])  # Precision score
	recallScore=np.mean(metric_array[:, 1])  # Recall score
	f1MacroScore=np.mean(metric_array[:, 2])  # F1-macro score
	f1MicroScore=np.mean(metric_array[:, 3])  # F1-micro score
	return [str(clf),presionScore,recallScore,f1MacroScore,f1MicroScore]

def get_f1_score_evaluation(feature, target,clf):
	k_fold = cross_validation.KFold(len(feature), n_folds=10)  # 10-fold cross validation
	metric = []
	for train, test in k_fold:
		target_pred = clf.fit(feature[train], target[train]).predict(feature[test])  # 训练分类器并预测测试集类标签
		f1Score = f1_score(target[test], target_pred, average='macro')
		metric.append(f1Score)
	metric_array = np.array(metric)
	f1MacroScore = np.mean(metric_array[0:])  # F1-macro score
	return f1MacroScore

def get_precision_evaluation(feature, target,clf):
	k_fold = cross_validation.KFold(len(feature), n_folds=10)  # 10-fold cross validation
	metric = []
	for train, test in k_fold:
		target_pred = clf.fit(feature[train], target[train]).predict(feature[test])  # 训练分类器并预测测试集类标签
		preciosnScore = precision_score(target[test], target_pred, average='macro')#按照实际值权重求取各个类别精度平均值
		metric.append(preciosnScore)
	metric_array = np.array(metric)
	Score = np.mean(metric_array[0:])  # F1-macro score
	return Score

def get_neg_precision_evaluation(feature, target,clf):
	k_fold = cross_validation.KFold(len(feature), n_folds=10)  # 10-fold cross validation
	metric = []
	for train, test in k_fold:
		target_pred = clf.fit(feature[train], target[train]).predict(feature[test])  # 训练分类器并预测测试集类标签
		preciosnScore = precision_score(target[test], target_pred, average=None)[0] #获取消极类精度
		#preciosnScore = precision_score(target[test], target_pred) # 获取积极类精度
		metric.append(preciosnScore)
	metric_array = np.array(metric)
	Score = np.mean(metric_array[0:])  # F1-macro score
	return Score
'''获得最佳分类器名字以及精度'''
def get_best_classifier():
	begin = time.clock()
	# classifyList=[svm.SVC(gamma=0.001, C=100.),svm.SVR(),LogisticRegression(penalty='l1', tol=0.01),tree.DecisionTreeClassifier(),GaussianNB(),BernoulliNB(),RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=1, random_state=0)]
	classifyList = [svm.SVC(gamma=0.001, C=100.), LogisticRegression(penalty='l1', tol=0.01),
					tree.DecisionTreeClassifier(), GaussianNB(), BernoulliNB()]
	# 读取txt数据 每一行为 类标签 特征 的形式
	dataNum = len(data)
	shuffle(data)  # Make data ramdon
	print type(data),type(data[0])
	helpfulness_target = data[:, 0]  # 取类标签，第一列作为类标签
	helpfulness_feature = data[:, 1:]  # 取分类特征，其余列
	bestClassifier=''
	bestScore=0.0
	for classifier in classifyList:
		curScore=get_neg_precision_evaluation(helpfulness_feature, helpfulness_target, classifier)
		if bestScore<curScore:
			bestScore=curScore
			bestClassifier=classifier
	end = time.clock()
	print 'data item num:',dataNum,'running time:',end-begin
	return bestClassifier,bestScore
'''存储最佳分类器名字以及精度'''
def store_classifier_acc_name(classifier,acc):
    f=open('D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+'bestClassifierAcc.txt','w')
    f.write(classifier+'\t'+acc+'\n');
    f.close()
'''存储最佳分类器'''
'''feature:[[f1,f2,f3,],[],[],]'''
'''target[c1,c2,]'''
def store_classifier(clf, filepath):
	shuffle(data)  # Make data ramdon
	target = data[:, 0]  # First column of the dataset is review helpfulness label 第一列作为类标签
	feature = data[:, 1:]  # The rest of the dataset is review helpfulness features 其余列作为分类特征
	clf.fit(feature, target)
	pickle.dump(clf, open(filepath, 'w'))
'''
功能：存储各个分类器的分类效果 精度 召回率 f1-macro f1-micro
参数：存储路径
执行过程：
读取txt数据（每一行为 类标签 特征 的形式）
数据随机化
取类标签 取分类特征 
获取分类器的分类效果

'''
def store_classify_metric(storePath):
	begin=time.clock()
	#classifyList=[svm.SVC(gamma=0.001, C=100.),svm.SVR(),LogisticRegression(penalty='l1', tol=0.01),tree.DecisionTreeClassifier(),GaussianNB(),BernoulliNB(),RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=1, random_state=0)]
	classifyList = [svm.SVC(gamma=0.001, C=100.), LogisticRegression(penalty='l1', tol=0.01),
					tree.DecisionTreeClassifier(), GaussianNB(), BernoulliNB()]
	# 读取txt数据 每一行为 类标签 特征 的形式
	data = read_data("D:/ReviewHelpfulnessPrediction\ReviewDataFeature/AllFeatureLabelData.txt")
	dataNum=len(data)
	shuffle(data)  # Make data ramdon
	helpfulness_target = data[:, 0] #取类标签，第一列作为类标签
	helpfulness_feature = data[:, 1:] #取分类特征，其余列
	rowHeader=['classifier','precision','recall','f1macro','f1micro']
	classifyMetric=[]
	classifyMetric.append(rowHeader)
	for classifier in classifyList:
		classifyMetric.append(metric_evaluation(helpfulness_feature,helpfulness_target,classifier))
	workbook=xlwt.Workbook()
	sheet=workbook.add_sheet('classifier result')
	for rowPos in range(len(classifyMetric)):
		for colPos in range(len(classifyMetric[rowPos])):
			sheet.write(rowPos,colPos,classifyMetric[rowPos][colPos])
	workbook.save(storePath)
	end=time.clock()
	return dataNum,end-begin
# rowHeader=['classifier','precision','recall','f1macro','f1micro']
# workbook=xlwt.Workbook()
# sheet=workbook.add_sheet('classifier result')
# for pos in range(len(rowHeader)):
# 	sheet.write(0,pos,rowHeader[pos])
# workbook.save('test.xls')
#Testing
#metric_evaluation(helpfulness_feature, helpfulness_target,clf
bestClassifier,bestScore=get_best_classifier()
print bestClassifier,bestScore
store_classifier_acc_name(str(bestClassifier).decode('utf-8'),str(bestScore).decode('utf-8'))
store_classifier(bestClassifier,'D:/ReviewHelpfulnessPrediction\BuildedClassifier/'+str(bestClassifier)[0:15]+'.pkl')

