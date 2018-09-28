__author__ = 'yxz'
#逻辑回归主要思想： 根据现有的数据对分类边界线建立回归公式，以此进行分类
#逻辑回归优点：计算代价不高，已于实现和理解
#       缺点：容易欠拟合，分类精度可能不高
#       适用数据类型：数值型和标称型数据
#逻辑回归可以被看作是一种概率估计
#Sigmoid函数： 1/（1+e(-z)）
import numpy as np
import matplotlib.pyplot as plt
import random

def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    file=open(filename)
    for line in file:
        line=line.strip().split()
        #分别对应X0，X1，X2
        dataMat.append([1.0,float(line[0]),float(line[1])])
        labelMat.append(int(line[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

#梯度上升方法 alpha是步长 maxCycles是迭代次数 tranpose转置
def granAscent(dataMatIn,classLabels):
    dataMatrix=np.mat(dataMatIn)
    labelMat=np.mat(classLabels).transpose()
    m,n=np.shape(dataMatrix)
    alpha=0.001
    maxCycles=500
    weights=np.ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        errors=(labelMat-h)
        weights=weights+alpha*np.transpose(dataMatrix)*errors
    return weights

def stocGradAscent0(dataMatrix,classLabels):
    m,n=np.shape(dataMatrix)
    alpha=0.01
    weights=np.ones(n)
    for i in range(m):
        h=sigmoid(sum(dataMatrix[i]*weights))
        error=classLabels[i]-h
        weights=weights+alpha*error*dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for j in range(numIter):
        dataIndex=range(m)
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(random.uniform(0,len(dataIndex)))
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            errors=classLabels[randIndex]-h
            weights=weights+alpha*errors*dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


#画出决策边界
def plotBestFit(weights,filename):
    dataMat,labelMat=loadDataSet(filename)
    dataArr=np.array(dataMat)
    n=np.shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    x=x.reshape(1,60)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
