# coding=utf-8
import codecs
import re
sstr='中文处理'
print sstr
filename = u'中文.txt'
f = codecs.open(filename, mode='r+', encoding='utf-8')
try:
    key = raw_input(u'输入查找汉字Pycharm：')
except:
    prompt = '输入查找汉字'.decode('utf-8').encode('gbk')
    key = raw_input(prompt)
try:
    u_key = key.decode('utf-8')
except:
    u_key = key.decode('gbk')
text = '一句新句子\n'.decode('utf-8')
print 'text', text, type(text)
print 'u_key', u_key, type(u_key)
newline = 0
for line in f:
    print line,
    if line == text:
        newline += 1
    if re.match(u'.*%s' % u_key, line) is not None:
        print u'在“' + line.strip() + u'”中找到' + u_key
print u'一共' + str(newline) + u'句“一句新句子”'
f.write(u'一句新句子')
f.write('\n')
f.close()

