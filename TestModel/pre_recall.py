#! /usr/bin/env python2.7
#coding=utf-8

from sklearn.metrics import precision_score, recall_score, f1_score


''''''

y_true = [0, 1,0, 0, 1, 1]
y_pred = [0, 0, 0, 0, 1, 1]
print precision_score(y_true, y_pred, average='macro')
print precision_score(y_true, y_pred, average='micro')# micro good 相对于预测数据来说
print precision_score(y_true, y_pred, average='weighted')
print precision_score(y_true, y_pred, average=None)
print precision_score(y_true, y_pred)

print recall_score(y_true, y_pred, average='macro')
print recall_score(y_true, y_pred, average='micro')
print recall_score(y_true, y_pred, average='weighted')
print recall_score(y_true, y_pred, average=None)
print recall_score(y_true, y_pred)

print f1_score(y_true, y_pred, average='macro')
print f1_score(y_true, y_pred, average='micro')
print f1_score(y_true, y_pred, average='weighted')
print f1_score(y_true, y_pred, average=None)
print f1_score(y_true, y_pred)