__author__ = 'yxz'
#Knn采用测量不同特征值之间距离的方法进行分类
#Knn优点：精确度高，对异常值不敏感，无数据输入假定
#   缺点：计算复杂度高，空间复杂度高
#适用数据范围：数值型和标称型
#Knn工作原理：
#   1.训练集中每一个样本集都有标签
#   2.输入没有标签的测试机之后，将新的数据集样本每个特征与训练集进行比较，最后选择距离最小的前k个

import numpy as np
import operator

#创建数据集和标签
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]]) #坐标
    labels=['A','A','B','B']                            #标签
    return group,labels

#距离计算：欧式距离
#inX:用于分类的输入向量；dataSet：训练样本 ；lables：标签向量
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0] #numpy中shape函数标识shape[0]--矩阵的行  shape[1]--矩阵的列
    #tile扩充矩阵，将inX数组行上重复dataSetSize次，列上重复1次以达到和dataSet一样的维度
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    #axis=0 按行压缩，即每个列的不同行相加
    #axis=1 按列压缩，即每个行的不同列相加
    sqlDistances=sqDiffMat.sum(axis=1)
    distances=sqlDistances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlable=labels[sortedDistIndicies[i]]
        classCount[voteIlable]=classCount.get(voteIlable,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]