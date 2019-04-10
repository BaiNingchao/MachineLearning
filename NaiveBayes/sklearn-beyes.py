# coding:utf8

import re,time
from decimal import Decimal
import numpy as np
from numpy import *
from sklearn.naive_bayes import GaussianNB # 高斯朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB # 多项式贝叶斯
from sklearn.naive_bayes import BernoulliNB # 伯努利朴素贝叶斯
# 结巴分词
import jieba,itertools
import jieba.posseg as pseg


class Global(object):
    def __init__(self, arg):
        super(Global, self).__init__()
        self.arg = arg


'''-----------分词去停用词,创建数据集----------------'''
myVocabList = [] # 设置词汇表的全局变量

'''创建数据集和类标签'''
def loadDataSet():
    docList = [];classList = []  # 文档列表、类别列表、文本特征
    dirlist = ['C3-Art','C4-Literature','C5-Education','C6-Philosophy','C7-History']
    for j in range(5):
        for i in range(1, 11): # 总共10个文档
            # 切分，解析数据，并归类为 1 类别
            wordList = textParse(open('./fudan/%s/%d.txt' % (dirlist[j],i),encoding='UTF-8').read())
            docList.append(wordList)
            classList.append(j)
            # print(i,'\t','./fudan/%s/%d.txt' % (dirlist[j],i),'\t',j)
    # print(len(docList),len(classList),len(fullText))
    global myVocabList
    myVocabList = createVocabList(docList)  # 创建单词集合
    return docList,classList,myVocabList



''' 利用jieba对文本进行分词，返回切词后的list '''
def textParse(str_doc):
    # 正则过滤掉特殊符号、标点、英文、数字等。
    r1 = '[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    str_doc=re.sub(r1, '', str_doc)

    # 创建停用词列表
    stwlist = set([line.strip() for line in open('./stopwords.txt', 'r', encoding='utf-8').readlines()])
    sent_list = str_doc.split('\n')
    # word_2dlist = [rm_tokens(jieba.cut(part), stwlist) for part in sent_list]  # 分词并去停用词
    word_2dlist = [rm_tokens([word+"/"+flag+" " for word, flag in pseg.cut(part) if flag in ['n','v','a','ns','nr','nt']], stwlist) for part in sent_list] # 带词性分词并去停用词
    word_list = list(itertools.chain(*word_2dlist)) # 合并列表
    return word_list



''' 去掉一些停用词、数字、特殊符号 '''
def rm_tokens(words, stwlist):
    words_list = list(words)
    for i in range(words_list.__len__())[::-1]:
        word = words_list[i]
        if word in stwlist:  # 去除停用词
            words_list.pop(i)
        elif len(word) == 1:  # 去除单个字符
            words_list.pop(i)
        elif word == " ":  # 去除空字符
            words_list.pop(i)
    return words_list




# 本地存储数据集和标签
def storedata():
    # 3. 计算单词是否出现并创建数据矩阵
    # trainMat =[[0,1,2,3],[2,3,1,5],[0,1,4,2]] # 训练集
    # classList = [0,1,2] #类标签
    docList,classList,myVocabList = loadDataSet()
    # 计算单词是否出现并创建数据矩阵
    trainMat = []
    for postinDoc in docList:
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))
    res = ""
    for i in range(len(trainMat)):
        res +=' '.join([str(x) for x in trainMat[i]])+' '+str(classList[i])+'\n'
    # print(res[:-1]) # 删除最后一个换行符
    with open('./word-bag.txt','w') as fw:
        fw.write(res[:-1])
    with open('./wordset.txt','w') as fw:
        fw.write(' '.join([str(v) for v in myVocabList]))


# 读取本地数据集和标签
def grabdata():
    f = open('./word-bag.txt') # 读取本地文件
    arrayLines = f.readlines() # 行向量
    tzsize = len(arrayLines[0].split(' '))-1 # 列向量，特征个数减1即数据集
    returnMat = zeros((len(arrayLines),tzsize))    # 0矩阵数据集
    classLabelVactor = []                     # 标签集，特征最后一列

    index = 0
    for line in arrayLines: # 逐行读取
        listFromLine = line.strip().split(' ')    # 分析数据，空格处理
        # print(listFromLine)
        returnMat[index,:] = listFromLine[0:tzsize] # 数据集
        classLabelVactor.append(int(listFromLine[-1])) # 类别标签集
        index +=1
    # print(returnMat,classLabelVactor)
    myVocabList=writewordset()
    return returnMat,classLabelVactor,myVocabList

def writewordset():
    f1 = open('./wordset.txt')
    myVocabList =f1.readline().split(' ')
    for w in myVocabList:
        if w=='':
            myVocabList.remove(w)
    return myVocabList


'''获取所有文档单词的集合'''
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 操作符 | 用于求两个集合的并集
    # print(len(vocabSet),len(set(vocabSet)))
    return list(vocabSet)



'''文档词袋模型，创建矩阵数据'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec



'''-----------朴素贝叶斯分类模型训练----------------'''

'''高斯朴素贝叶斯'''
def MyGaussianNB(trainMat='',Classlabels='',testDoc=''):
    # -----sklearn GaussianNB Dome-------
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    # Y = np.array([1, 1, 1, 2, 2, 2])
    # clf = GaussianNB()

    # clf.fit(X, Y)
    # print(clf.predict([[-0.8, -1]]))

    # # clf_pf.partial_fit(X, Y, np.unique(Y))
    # # print(clf_pf.predict([[-0.8, -1]]))

    # -----sklearn GaussianNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 高斯分布
    clf = GaussianNB()
    clf.fit(X, Y)
    # 测试预测结果
    index = clf.predict(testDoc) # 返回索引
    reslist = ['Art','Literature','Education','Philosophy','History']
    print(reslist[index[0]])



'''多项朴素贝叶斯'''
def MyMultinomialNB(trainMat='',Classlabels='',testDoc=''):
    #-----sklearn MultinomialNB_多项朴素贝叶斯 Dome-------
    # X = np.random.randint(5, size=(6, 100))
    # y = np.array([1, 2, 3, 4, 5, 6])

    # clf = MultinomialNB()
    # clf.fit(X, y)
    # print(clf.predict(X[2:3]))

    # -----sklearn MultinomialNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 多项朴素贝叶斯
    clf = MultinomialNB()
    clf.fit(X, Y)
    # 测试预测结果
    index = clf.predict(testDoc) # 返回索引
    reslist = ['Art','Literature','Education','Philosophy','History']
    print(reslist[index[0]])


'''伯努利朴素贝叶斯'''
def MyBernoulliNB(trainMat='',Classlabels='',testDoc=''):
    #-----sklearn BernoulliNB_伯努利朴素贝叶斯 Dome-------
    X = np.random.randint(2, size=(6, 100))
    Y = np.array([1, 2, 3, 4, 4, 5])
    clf = BernoulliNB()
    clf.fit(X, Y)
    print(clf.predict(X[2:3]))

    # -----sklearn BernoulliNB-------
    # 训练数据
    X = np.array(trainMat)
    Y = np.array(Classlabels)
    # 多项朴素贝叶斯
    clf = BernoulliNB()
    clf.fit(X, Y)
    # 测试预测结果
    index = clf.predict(testDoc) # 返回索引
    reslist = ['Art','Literature','Education','Philosophy','History']
    print(reslist[index[0]])



'''-----------朴素贝叶斯分类应用测试----------------'''





# 测试分类效果
# def testingNB(dataSet,Classlabels,testDoc):
#     1. 加载数据集
#     dataSet,Classlabels = grabdata()
#     # 2. 创建单词集合
#     myVocabList = createVocabList(dataSet)
#     # 3. 计算单词是否出现并创建数据矩阵
#     trainMat = []
#     for postinDoc in dataSet:
#         trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))
#     4测试数据
#     testEntry = textParse(open('./fudan/test/C6-2.txt',encoding='UTF-8').read())
#     testDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry))
#     5 测试预测结果
#     MyGaussianNB(trainMat,Classlabels,testDoc)
#     MyMultinomialNB(trainMat,Classlabels,testDoc)
#     MyBernoulliNB(trainMat,Classlabels,testDoc)


def testingNB():
    # 加载数据集和单词集合
    trainMat,Classlabels,myVocabList = grabdata() # 读取训练结果
    # 测试数据
    testEntry = textParse(open('./fudan/test/C6-2.txt',encoding='UTF-8').read())
    testDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry)) # 测试数据
    # 测试预测结果
    MyGaussianNB(trainMat,Classlabels,testDoc)
    MyMultinomialNB(trainMat,Classlabels,testDoc)
    MyBernoulliNB(trainMat,Classlabels,testDoc)





if __name__ =='__main__':
    t1 = time.time()
    # 加载数据集和标签
    # dataset,labels,myVocabList = loadDataSet()
    # print(len(dataset),len(labels))

    # 1 高斯朴素贝叶斯
    # MyGaussianNB()

    # 2 多项朴素贝叶斯
    # MyMultinomialNB()

    # 3 伯努利朴素贝叶斯
    # MyBernoulliNB()

    # 4 数据集本地存储与读取
    # storedata()
    # grabdata()

    # # 测试方法
    # trainMat,Classlabels,myVocabList = grabdata() # 读取训练结果
    # testEntry = textParse(open('./fudan/test/C6-2.txt',encoding='UTF-8').read())
    # # myVocabList = createVocabList(loadDataSet()[0])
    # testDoc = np.array(bagOfWords2VecMN(myVocabList, testEntry)) # 测试数据
    # MyGaussianNB(trainMat,Classlabels,testDoc)

    testingNB()

    t2= time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n")
