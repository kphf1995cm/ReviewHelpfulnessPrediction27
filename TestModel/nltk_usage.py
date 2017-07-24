# coding=utf-8
import chardet
import jieba
import nltk
import chardet
import codecs
import re
from nltk.collocations import *

# sstr='中文处理'
# print chardet.detect(sstr)
# print sstr

train_corpus = "数据库坏了,测试数据库,用户支付表,支付金额,支付用户,测试数据库,用户支付表,支付金额,支付用户,用户支付"
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

finder = BigramCollocationFinder.from_words(jieba.cut(train_corpus))
finder.apply_word_filter(lambda w: w.lower() in [',', '.', '，', '。'])
word_best=finder.nbest(bigram_measures.pmi, 10)
print word_best
for x in word_best:
    for y in x:
        print y,
    print ' '

finder = TrigramCollocationFinder.from_words(jieba.cut(train_corpus))
finder.apply_word_filter(lambda w: w.lower() in [',', '.', '，', '。'])
finder.nbest(trigram_measures.pmi, 10)
