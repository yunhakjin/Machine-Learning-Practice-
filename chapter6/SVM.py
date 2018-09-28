__author__ = 'yxz'
import random
# 支持向量机：
#   优点：泛化错误率低，计算开销不大，结果易解释
#   缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅仅适用于处理二类问题
#   适用数据类型：数值型和标称型数据
# 支持向量：离分隔超平面最近的那些点。目标：最大化支持向量到分隔面的距离（用到最优化的方法），也就是令分得的几个类分的足够开

#SMO算法（序列最小优化）
#   目标：求出一系列alpha和b，一旦求出了这些alpha，就很容易计算出权重向量w，并得到分隔超平面
#   工作原理：每次循环中选择两个alpha进行优化处理。一旦找到一对合适的alpha，那么就增大其中一个同时减少另一个。
#           这里所谓的“合适”就是指两个alpha必须要符合一定的条件，条件之一就是这两个alpha必须要在间隔边界之外，
#           而其第二个条件则是这两个alpha还没有进行过间化处理或者不在边界上
def loadDataSet(filename):
    dataMat=[]
    labelMat=[]
    file=open(filename)
    for line in file.readlines():
        line=line.strip().split('\t')
        dataMat.append([float(line[0]),float(line[1])])
        labelMat.append(float(line[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while i==j:
        j=int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj>H:
        aj=H
    elif L>aj:
        aj=L
    return aj

#简化版本的SMO算法思想
#  创建一个alpha向量并将其初始化位0的向量
#    当迭代次数小于最大迭代次数时（外循环）：
#      对数据集中的每个数据向量（内循环）：
#        如果该数据向量可以被优化：
#           随机选择另外一个数据向量
#           同时优化这两个向量
#           如果两个向量都不能被优化，推出内循环
#      如果所有向量都没有被优化，增加迭代数目，继续下一次循环