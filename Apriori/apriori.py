#!/usr/bin/python
# coding: utf8

'''一步步轻松教你学关联规则算法
   url：https://bainingchao.github.io/
'''

from numpy import *


'''加载数据集'''
def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]



'''创建集合C1即对dataSet去重排序'''
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    # frozenset表示冻结的set 集合，元素无改变把它当字典的 key 来使用
    return C1
    # return map(frozenset, C1)



''' 计算候选数据集CK在数据集D中的支持度，返回大于最小支持度的数据'''
def scanD(D,Ck,minSupport):
    # ssCnt 临时存放所有候选项集和频率.
    ssCnt = {}
    for tid in D:
        # print('1:',tid)
        for can in map(frozenset,Ck):      #每个候选项集can
            # print('2:',can.issubset(tid),can,tid)
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] +=1

    numItems = float(len(D)) # 所有项集数目
    # 满足最小支持度的频繁项集
    retList  = []
    # 满足最小支持度的频繁项集和频率
    supportData = {}

    for key in ssCnt:
        support = ssCnt[key]/numItems   #除以总的记录条数，即为其支持度
        if support >= minSupport:
            retList.insert(0,key)       #超过最小支持度的项集，将其记录下来。
        supportData[key] = support
    return retList, supportData




''' Apriori算法：输入频繁项集列表Lk，输出所有可能的候选项集 Ck'''
def aprioriGen(Lk, k):
    retList = [] # 满足条件的频繁项集
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            # print '-----i=', i, k-2, Lk, Lk[i], list(Lk[i])[: k-2]
            # print '-----j=', j, k-2, Lk, Lk[j], list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList



'''找出数据集中支持度不小于最小支持度的候选项集以及它们的支持度即频繁项集。
算法思想：首先构建集合C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度。满足最小支持度要求的项集构成集合L1。然后L1 中的元素相互组合成C2，C2再进一步过滤变成L2，以此类推，直到C_n的长度为0时结束，即可找出所有频繁项集的支持度。
返回：L 频繁项集的全集
      supportData 所有元素和支持度的全集
'''
def apriori(dataSet, minSupport=0.5):
    # C1即对dataSet去重排序，然后转换所有的元素为frozenset
    C1 = createC1(dataSet)
    # 对每一行进行 set 转换，然后存放到集合中
    D = list(map(set, dataSet))
    # 计算候选数据集C1在数据集D中的支持度，并返回支持度大于minSupport 的数据
    L1, supportData = scanD(D, C1, minSupport)
    # L 加了一层 list, L一共 2 层 list
    L = [L1];k = 2
    # 判断L第k-2项的数据长度是否>0即频繁项集第一项。第一次执行时 L 为 [[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]]。L[k-2]=L[0]=[frozenset([1]), frozenset([3]), frozenset([2]), frozenset([5])]，最后面 k += 1
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k) # 例如: 以 {0},{1},{2} 为输入且 k = 2 则输出 {0,1}, {0,2}, {1,2}. 以 {0,1},{0,2},{1,2} 为输入且 k = 3 则输出 {0,1,2}

        # 返回候选数据集CK在数据集D中的支持度大于最小支持度的数据
        Lk, supK = scanD(D, Ck, minSupport)
        # 保存所有候选项集的支持度，如果字典没有就追加元素，如果有就更新元素
        supportData.update(supK)
        if len(Lk) == 0:
            break
        # Lk 表示满足频繁子项的集合，L 元素在增加，例如:
        # l=[[set(1), set(2), set(3)]]
        # l=[[set(1), set(2), set(3)], [set(1, 2), set(2, 3)]]
        L.append(Lk)
        k += 1
    return L, supportData



'''生成关联规则
    Args:
        L 频繁项集列表
        supportData 频繁项集支持度的字典
        minConf 最小置信度
    Returns:
        bigRuleList 可信度规则列表（关于 (A->B+置信度) 3个字段的组合）
'''
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            # 组合总的元素并遍历子元素，转化为 frozenset集合存放到 list 列表中
            H1 = [frozenset([item]) for item in freqSet]
            # print(H1)
            # 2 个的组合else, 2 个以上的组合 if
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList



'''计算可信度（confidence）
Args:
    freqSet 频繁项集中的元素，例如: frozenset([1, 3])
    H 频繁项集中的元素的集合，例如: [frozenset([1]), frozenset([3])]
    supportData 所有元素的支持度的字典
    brl 关联规则列表的空数组
    minConf 最小可信度
Returns:
    prunedH 记录 可信度大于阈值的集合
'''
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 记录可信度大于最小可信度（minConf）的集合
    prunedH = []
    for conseq in H: # 假设 freqSet = frozenset([1, 3]), H = [frozenset([1]), frozenset([3])]，那么现在需要求出 frozenset([1]) -> frozenset([3]) 的可信度和 frozenset([3]) -> frozenset([1]) 的可信度
        conf = supportData[freqSet]/supportData[freqSet-conseq] # 支持度定义: a -> b = support(a | b) / support(a). 假设  freqSet = frozenset([1, 3]), conseq = [frozenset([1])]，那么 frozenset([1]) 至 frozenset([3]) 的可信度为 = support(a | b) / support(a) = supportData[freqSet]/supportData[freqSet-conseq] = supportData[frozenset([1, 3])] / supportData[frozenset([1])]
        if conf >= minConf:
            # 只要买了 freqSet-conseq 集合，一定会买 conseq 集合（freqSet-conseq 集合和 conseq集合 是全集）
            print (freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH



"""递归计算频繁项集的规则
    Args:
        freqSet 频繁项集中的元素，例如: frozenset([2, 3, 5])
        H 频繁项集中的元素的集合，例如: [frozenset([2]), frozenset([3]), frozenset([5])]
        supportData 所有元素的支持度的字典
        brl 关联规则列表的数组
        minConf 最小可信度
"""
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    # H[0] 是 freqSet 的元素组合的第一个元素，并且 H 中所有元素的长度都一样，长度由 aprioriGen(H, m+1) 这里的 m + 1 来控制
    # 该函数递归时，H[0] 的长度从 1 开始增长 1 2 3 ...
    # 假设 freqSet = frozenset([2, 3, 5]), H = [frozenset([2]), frozenset([3]), frozenset([5])]
    # 那么 m = len(H[0]) 的递归的值依次为 1 2
    # 在 m = 2 时, 跳出该递归。假设再递归一次，那么 H[0] = frozenset([2, 3, 5])，freqSet = frozenset([2, 3, 5]) ，没必要再计算 freqSet 与 H[0] 的关联规则了。
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        # 生成 m+1 个长度的所有可能的 H 中的组合，假设 H = [frozenset([2]), frozenset([3]), frozenset([5])]
        # 第一次递归调用时生成 [frozenset([2, 3]), frozenset([2, 5]), frozenset([3, 5])]
        # 第二次 。。。没有第二次，递归条件判断时已经退出了
        Hmp1 = aprioriGen(H, m+1)
        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # print ('Hmp1=', Hmp1)
        # print ('len(Hmp1)=', len(Hmp1), 'len(freqSet)=', len(freqSet))
        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if (len(Hmp1) > 1):
            # print '----------------------', Hmp1
            # print len(freqSet),  len(Hmp1[0]) + 1
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


'''测试频繁项集生产'''
def testApriori():
    # 加载测试数据集
    dataSet = loadDataSet()
    print ('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.7)
    print ('L(0.7): ', L1)
    print ('supportData(0.7): ', supportData1)

    print ('->->->->->->->->->->->->->->->->->->->->->->->->->->->->')

    # Apriori 算法生成频繁项集以及它们的支持度
    L2, supportData2 = apriori(dataSet, minSupport=0.5)
    print ('L(0.5): ', L2)
    print ('supportData(0.5): ', supportData2)



def testGenerateRules():
    # 加载测试数据集
    dataSet = loadDataSet()
    print ('dataSet: ', dataSet)

    # Apriori 算法生成频繁项集以及它们的支持度
    L1, supportData1 = apriori(dataSet, minSupport=0.5)
    print ('L(0.7): ', L1)
    print ('supportData(0.7): ', supportData1)

    # 生成关联规则
    rules = generateRules(L1, supportData1, minConf=0.5)
    print ('rules: ', rules)



if __name__ == "__main__":
    # 1 加载数据集
    # dataSet = loadDataSet()
    # print(dataSet)

    # 2 计算候选数据集CK在数据集D中的支持度
    # C1 = createC1(dataSet)  # 创建集合C1
    # print(C1)
    # L1, support = scanD(dataSet, C1,0.5)
    # print('满足最小支持度的频繁项集是：\n',L1,'\n频繁项集的支持度\n',support)

    #3 创建频繁项集
    # retList = aprioriGen([L1][0],2)
    # print(retList)

    # L, supportData = apriori(dataSet,0.5)
    # print(L,'\n',supportData,'\n\n')

    # 4 生成频繁项集
    # rules = generateRules(L,supportData,minConf=0.7)
    # print(rules)

    # 5 测试关联和生成规则
    # testApriori()
    # testGenerateRules()


    # 6 项目案例：发现毒蘑菇的相似特性
    # 得到全集的数据
    dataSet = [line.split() for line in open("./mushroom.dat").readlines()]
    L, supportData = apriori(dataSet, minSupport=0.4)
    # 2表示毒蘑菇，1表示可食用的蘑菇
    # 找出关于2的频繁子项出来，就知道如果是毒蘑菇，那么出现频繁的也可能是毒蘑菇
    for item in L[2]:
        if item.intersection('2'):
            print (item)

    # for item in L[3]:
    #     if item.intersection('2'):
    #         print (item)

