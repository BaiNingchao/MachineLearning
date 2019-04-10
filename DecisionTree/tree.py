# coding:utf-8

from math import log
from decimal import Decimal
from collections import Counter
import operator
import treePlotter



'''创建数据集'''
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels



'''计算数据集的香农熵(信息期望值):熵越高表示混合数据越多，度量数据集无序程度'''
def calcShannonEnt(dataSet):
    # -----------计算香农熵实现方式一----
    # numEntries = len(dataSet) # 计算数据集中实例总数
    # labelCounts = {} # 创建字典，计算分类标签label出现的次数
    # for featVec in dataSet:
    #     currentLabel = featVec[-1] # 记录当前实例的标签
    #     if currentLabel not in labelCounts.keys():# 为所有可能的分类创建字典
    #         labelCounts[currentLabel] = 0
    #     labelCounts[currentLabel] += 1
    #     # print(featVec, labelCounts) # 打印特征向量和字典的键值对

    # # 对于label标签的占比，求出label标签的香农熵
    # shannonEnt = 0.0
    # for key in labelCounts:
    #     prob = float(labelCounts[key])/numEntries # 计算类别出现的概率。
    #     shannonEnt -= prob * log(prob, 2) # 计算香农熵，以 2 为底求对数
    # # print(Decimal(shannonEnt).quantize(Decimal('0.00000')))
    # return shannonEnt


    # --------计算香农熵实现方式二------
    # 需要对 list 中的大量计数时,可以直接使用Counter,不用新建字典来计数
    label_count = Counter(data[-1] for data in dataSet) # 统计标签出现的次数
    probs = [p[1] / len(dataSet) for p in label_count.items()] # 计算概率
    shannonEnt = sum([-p * log(p, 2) for p in probs]) # 计算香农熵
    # print(Decimal(shannonEnt).quantize(Decimal('0.00000')))
    return shannonEnt



'''划分数据集:按照特征划分'''
def splitDataSet(dataSet, index, value):
    # -----------切分数据集实现方式一-------
    # retDataSet = []
    # for featVec in dataSet:
    #     if featVec[index] == value:# 判断index列的值是否为value
    #         reducedFeatVec = featVec[:index] # [:index]表示取前index个特征
    #         reducedFeatVec.extend(featVec[index+1:]) # 取接下来的数据
    #         retDataSet.append(reducedFeatVec)
    # print(retDataSet)
    # return retDataSet


    # -----------切分数据集实现方式二-----------
    retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == index and v == value]
    # print(retDataSet)
    return retDataSet



'''
选择最好的数据集划分方式：特征选择，划分数据集、计算最好的划分数据集特征
注意：一是数据集列表元素具备相同数据长度，二是最后一列是标签列
'''
def chooseBestFeatureToSplit(dataSet):
    # # -----------选择最优特征实现方式一------------
    numFeatures = len(dataSet[0]) - 1 # 特征总个数, 最后一列是标签
    baseEntropy = calcShannonEnt(dataSet) # 计算数据集的信息熵
    bestInfoGain, bestFeature = 0.0, -1 # 最优的信息增益值, 和最优的Featurn编号
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet] # 获取各实例第i+1个特征
        uniqueVals = set(featList) # 获取去重后的集合
        newEntropy = 0.0  # 创建一个新的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 比较所有特征中的信息增益，返回最好特征划分的索引值。
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # print(bestFeature)
    return bestFeature


    # -----------选择最优特征实现方式二--------------
    # base_entropy = calcShannonEnt(dataSet) # 计算初始香农熵
    # best_info_gain = 0
    # best_feature = -1
    # # 遍历每一个特征
    # for i in range(len(dataSet[0]) - 1):
    #     # 对当前特征进行统计
    #     feature_count = Counter([data[i] for data in dataSet])
    #     # 计算分割后的香农熵
    #     new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) for feature in feature_count.items())
    #     # 更新值
    #     info_gain = base_entropy - new_entropy
    #     # print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
    #     if info_gain > best_info_gain:
    #         best_info_gain = info_gain
    #         best_feature = i
    # # print(best_feature)
    # return best_feature



'''多数表决方法决定叶子节点的分类：选择出现次数最多的一个结果'''
def majorityCnt(classList):
    # -----------多数表决实现的方式一--------------
    # classCount = {}   # 标签字典，用于统计类别频率
    # for vote in classList: # classList标签的列表集合
    #     if vote not in classCount.keys():
    #         classCount[vote] = 0
    #     classCount[vote] += 1
    # # 取出结果（yes/no），即出现次数最多的结果
    # sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print('sortedClassCount:', sortedClassCount)
    # return sortedClassCount[0][0]


    # -----------多数表决实现的方式二-----------------
    major_label = Counter(classList).most_common(1)[0]
    print('sortedClassCount:', major_label[0])
    return major_label[0]



'''创建决策树'''
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果数据集的最后一列的第一个值出现的次数=整个集合的数量，也就说只有一个类别，就只直接返回结果就行
    # 第一个停止条件：所有的类标签完全相同，则直接返回该类标签。
    # count() 函数是统计括号中的值在list中出现的次数
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果数据集只有1列，那么最初出现label次数最多的一类，作为结果
    # 第二个停止条件：使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组。
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优的列，得到最优列对应的label含义
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 获取label的名称
    bestFeatLabel = labels[bestFeat]
    # 初始化myTree
    myTree = {bestFeatLabel: {}}
    # 所以这行代码导致函数外的同名变量被删除了元素，造成例句无法执行，提示'no surfacing' is not in list
    # del(labels[bestFeat])
    # 取出最优列，然后它的branch做分类
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # print('myTree', value, myTree)
    print(myTree)
    return myTree



'''用决策树分类函数'''
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0] # 获取tree的根节点对于的key值
    secondDict = inputTree[firstStr]  # 通过key得到根节点对应的value
    # 判断根节点名称获取根节点在label中的先后顺序，这样就知道输入的testVec怎么开始对照树来做分类
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    print(classLabel)
    return classLabel



'''使用pickle模块存储决策树'''
def storeTree(inputTree, filename):
    import pickle
    # -------------- 第一种方法 --------------
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

    # -------------- 第二种方法 --------------
    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)


'''决策树判断是否是鱼'''
def fishTest():
    # 1.创建数据和结果标签
    myDat, labels = createDataSet()

    # 计算label分类标签的香农熵
    calcShannonEnt(myDat)

    # 求第0列 为 1/0的列的数据集【排除第0列】
    print('1---', splitDataSet(myDat, 0, 1))
    print('0---', splitDataSet(myDat, 0, 0))

    # 计算最好的信息增益的列
    print(chooseBestFeatureToSplit(myDat))

    import copy
    myTree = createTree(myDat, copy.deepcopy(labels))
    print(myTree)
    # [1, 1]表示要取的分支上的节点位置，对应的结果值
    print(classify(myTree, labels, [1, 1]))

    # 获得树的高度
    print(get_tree_height(myTree))

    # 画图可视化展现
    treePlotter.createPlot(myTree)



'''预测隐形眼镜的测试代码'''
def ContactLensesTest():
    # 加载隐形眼镜相关的 文本文件 数据
    fr = open('lenses.txt')
    # 解析数据，获得 features 数据
    lenses = [inst.strip().split('    ') for inst in fr.readlines()]
    # 得到数据的对应的 Labels
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 使用上面的创建决策树的代码，构造预测隐形眼镜的决策树
    lensesTree = createTree(lenses, lensesLabels)
    print(lensesTree)
    # 画图可视化展现
    treePlotter.createPlot(lensesTree)




'''递归获得决策树的高度'''
def get_tree_height(tree):
    if not isinstance(tree, dict):
        return 1

    child_trees =list(tree.values())[0].values()

    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height

    return max_height + 1




if __name__ == '__main__':
    # # 1 打印数据集和标签
    dataset,label=createDataSet()
    # print(dataset)
    # print(label)

    # 2 计算数据集的熵
    # calcShannonEnt(dataset)

    #3 划分数据集
    # splitDataSet(dataset,0,1)

    # 4 选择最好的数据集划分方式
    # chooseBestFeatureToSplit(dataset)

    # 5多数表决方法决定叶子节点的分类
    # classList = [example[-1] for example in dataset]
    # majorityCnt(classList)

    # 6创建决策树
    # createTree(dataset, label)

    # 7 用决策树分类函数
    # myTree = treePlotter.retrieveTree(0)
    # # print(myTree)
    # classify(myTree,label,[1,0])

    # 8 使用pickle模块存储决策树
    # storeTree(myTree,'classifierStorage.txt')
    # readData = grabTree('classifierStorage.txt')
    # print(readData)

    # 9 决策树判断是否是鱼
    # fishTest()

    # 10 预测隐形眼镜类型
    ContactLensesTest()