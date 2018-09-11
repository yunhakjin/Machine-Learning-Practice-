__author__ = 'yxz'
#基于朴素贝叶斯的分类
#基于概率，将对象分类到概率较大的那一个类别中
import numpy as np
import math
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

#创建一个包含所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet=set()
    for document in dataSet:
        for value in document:
            vocabSet.add(value)
    return list(vocabSet)

#输入参数为词汇表以及某个文档，输出为文档向量，向量的每一个元素为0/1，分别标识词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList,inputSet):
    returnnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnnVec[vocabList.index(word)]=1
        else:
            print("the word: %s is not in my vocabulary!"%(word))
    return returnnVec

#朴素贝叶斯词袋模型：
#如果一个词在文档中出现不止以此，这可能意味着包含该词是否出现在文档中所不能表达的某种信息，这种方法被称为“词袋模型”
def bagOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return returnVec

#朴素贝叶斯分类器训练函数
#算法：
#计算每个类别中的文档数目
#对每篇训练文档：
#   对每个类别：
#     如果词条出现在文档中->增加该词条的计数值
#     增加所有词条的计数值
#对每个类别：
#   对每个词条：
#     将该词条的数目初一总词条数目得到条件概率
#返回每个类别的条件概率
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/float(numTrainDocs)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=np.log(p1Num/p1Denom)
    p0Vect=np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+np.log(pClass1)
    p0=sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"分类为",classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry=['stupid','garbage']
    thisDoc=np.array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,"分类为",classifyNB(thisDoc,p0V,p1V,pAb))