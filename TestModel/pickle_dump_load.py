#! /usr/local/env python
# -*- coding=utf-8 -*-

if __name__ == "__main__":
    import cPickle

    # 序列化到文件
    obj = 123, "abcdedf", ["ac", 123], {"key": "value", "key1": "value1"}
    print obj
    # 输出：(123, 'abcdedf', ['ac', 123], {'key1': 'value1', 'key': 'value'})
    # r 读写权限 r b 读写到二进制文件
    f = open(r"a.txt", "w ")
    cPickle.dump(obj, f)
    f.close()
    f = open(r"a.txt")
    print cPickle.load(f)
    # 输出：(123, 'abcdedf', ['ac', 123], {'key1': 'value1', 'key': 'value'})

    # 序列化到内存（字符串格式保存），然后对象可以以任何方式处理如通过网络传输
    obj1 = cPickle.dumps(obj)
    print type(obj1)
    # 输出：<type 'str'>
    print obj1
    # 输出：python专用的存储格式
    obj2 = cPickle.loads(obj1)
    print type(obj2)
    # 输出：<type 'tuple'>
    print obj2
    # 输出：(123, 'abcdedf', ['ac', 123], {'key1': 'value1', 'key': 'value'})