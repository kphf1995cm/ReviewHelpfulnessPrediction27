import numpy

b=numpy.loadtxt('b.txt',delimiter=' ')
print b
print type(b),b.ndim,b.size
for x in b:
    print x.ndim,x.size
    for y in x:
        print y,
    print ''

