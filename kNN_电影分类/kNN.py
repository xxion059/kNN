from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
kNN算法
Parameters:
    inX - 用于分类的数据(测试集)
    dataSet - 用于训练的数据(训练集)
    labels - 分类标签
    k - kNN算法参数，选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果
'''
def classify0(inX, dataSet, labels, k):    
    dataSetSize = dataSet.shape[0]                  #numpy函数shape[0]返回dataSet的行数  ‘如果是shape[1]就是列’  
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #将inX重复成dataSetSize行1列的新矩阵,'做差'
    sqDiffMat = diffMat ** 2                        #二维特征相减后,'平方'
    sqDistances = sqDiffMat.sum(axis = 1)           #sum()所有元素相加，sum(0)列相加，sum(1)行相加,'平方和'
    distances = sqDistances ** 0.5                  #'开方'，计算出距离
    sortedDistIndices = distances.argsort()          #返回distances中元素从小到大排序后的索引值
    
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    #python3中使用items()代替python2中的iteritems()
    #key = operator.itemgetter(0)是根据字典的键进行排序
    #key = operator.itemgetter(1)是根据字典的值进行排序
    #reverse降序排序字典
    return sortedClassCount[0][0]