__author__ = 'yxz'
import numpy as np
import matplotlib
import chapter2.knn as knn
import matplotlib.pyplot as plt
#数据样本【每年获得的飞行常客里程数，玩视频游戏所消耗时间百分比，每周消费的冰淇淋公升数】
#分类：不喜欢的人，魅力一般的人，极具魅力的人
def file2Mat(filename):
    file=open(filename)
    arrayLines=file.readlines()
    arrayLinesLen=len(arrayLines)
    returnMat=np.zeros((arrayLinesLen,3))
    classLables=[]
    index=0
    for line in arrayLines:
        line=line.strip()  #截取所有的回车字符
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLables.append(listFromLine[-1])
        index+=1
    return returnMat,classLables

#数值归一化  newValue=(oldValue-minValue)/(maxValue-minValue)
def autoNorm(dataSet):
    minValue=dataSet.min(0)
    maxValue=dataSet.max(0)
    ranges=maxValue-minValue
    m=dataSet.shape[0]
    normDataSet=dataSet-np.tile(minValue,(m,1))
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minValue

def show(filename):
    mat,lables=file2Mat(filename)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(mat[:,1],mat[:,0],15.0*np.array(lables),15.0*np.array(lables))
    plt.show()

def datingTest(file):
    mat,lables=file2Mat(file)
    normMat,ranges,minValue=autoNorm(mat)
    row=normMat.shape[0]
    errorCount=0.0
    numTestVec=int(row*0.1) #取10%作为测试集
    for i in range(numTestVec):
        classifyResult=knn.classify0(normMat[i,:],normMat[numTestVec:row,:],lables[numTestVec:row],3)
        print("the classifier came back with %s, the real answer is %s"%(classifyResult,lables[i]))
        if(classifyResult!=lables[i]):
            errorCount+=1.0
    print("the total error rate is %f"%(errorCount/float(numTestVec)))
