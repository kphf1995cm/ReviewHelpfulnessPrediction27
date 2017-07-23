import numpy as np
import math
# a=[[1,2,3],[4,5,6],[7,8,9]]
# print(type(a))
# print(a[0][1:2])
# print(a)
# b=np.array(a)
# print(type(b))
# print(b)
#
# print(np.sum(b[:,1]))
# print(np.mean(b[:,1]))
# print(np.std(b[:,1]))

def calstd(x):
	sum=0
	for i in x:
		sum+=i
	ave=sum/len(x)
	std=0
	for i in x:
		std+=(ave-i)*(ave-i)
	return math.sqrt(std)

print(calstd([2,5,8]))

a=[1,2,3,4]
print(a[::])


