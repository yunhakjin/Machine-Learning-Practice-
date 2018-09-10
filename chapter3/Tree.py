__author__ = 'yxz'
#决策树的一个重要的任务是为了理解数据中蕴含的知识信息，因此决策树可以使用不熟悉的数据集合，并从中
#提取出一系列规则，这些机器根据数据集创建规则的过程，就是机器学习的过程

#优点：计算复杂度不高，输出结果易于理解，对中间值的缺失不敏感，可以处理不相关特征数据
#缺点：可能会产生过度匹配问题
#适用数据类型：数值型和标称型

#创建决策树
#createBranch():
#  检测数据集中的每个子项是否属于同一分类
#  If so return 类标签
#  Else
#    寻找划分数据集的最好特征
#    划分数据集
#    创建分支节点
#       for 每个划分的子集
#           调用函数createBranch()并增加返回结果到分支节点中
#    return 分支节点

#在划分数据集之前之后休尼希发生的变化称为信息增益，指导如何计算信息增益，我们就可以计算，每个特征值划分数据集获得的信息增益，获得信息增益
#最高的特征就是最好的选择    信息增益/熵：定义为信息的期望值

from math import log
import numpy as np
import operator

def createData():
    dataSet=[
        [1,1,'maybe'],
        [1,1,'yes'],
        [1,0,'no'],
        [0,1,'no'],
        [0,1,'no']
    ]
    labels=['no surfacing','flippers']
    return dataSet,labels

#计算香浓熵
def calShannonEnt(dataSet):
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    num=len(dataSet)
    for key in labelCounts:
        prob=float(labelCounts[key])/num
        shannonEnt-=prob*log(prob,2)
    return shannonEnt

#划分数据集，输入函数的参数：待划分的数据集；划分数据集的特征；需要返回的特征的值
def splitDataSet(dataSet,axis,value):
    returnDataSet=[]
    for data in dataSet:
        if data[axis]==value:
            returnSet=data[:axis]
            returnSet.extend(data[axis+1:])
            returnDataSet.append(returnSet)
    return returnDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1   #保证数据集每一行的最后一列都是当前行的标签
    baseEntropy=calShannonEnt(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    totalLen=float(len(dataSet))
    for i in range(numFeatures):
        featList=[item[i] for item in dataSet]
        uniqueVals=set(featList)
        newEnt=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/totalLen
            newEnt+=prob*calShannonEnt(subDataSet)
        infoGain=baseEntropy-newEnt
        if infoGain>bestInfoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature

#工具函数
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#构建决策树（利用递归的方式）
#递归结束条件：程序遍历完所有划分数据集的属性，或者每个分支下的所有实例都具有相同的分类
#则得到一个叶子节点或者终止块。
def createTree(dataSet,labels):
    classList=[item[-1] for item in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]             #类别完全相同则停止继续划分
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues=[item[bestFeat] for item in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
