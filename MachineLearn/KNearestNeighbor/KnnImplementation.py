import csv
import math
import random
import operator

'''
Description:python调用机器学习库scikit-learn的K临近算法，实现花瓣分类
Author:Bai Ningchao
DateTime:2017年1月3日15:07:25
Blog URL:http://www.cnblogs.com/baiboy/
Document：http://scikit-learn.org/stable/modules/neighbors.html
'''

'加载数据集,split划分数据集为训练集和测试集'
def loadDataset(filename,split,trainingSet=[],testSet=[]):
    with open(filename,'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        # print(dataset[:20])
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
                # print(dataset[x])
            else:
                testSet.append(dataset[x])
                # print('-->',dataset[x])

'计算距离'
def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x]-instance2[x]),2)
    return math.sqrt(distance)

'返回最近的K个label'
def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
        return neighbors

'根据投票进行分类划分'
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(),key=operator.itemgetter(1),reverse=True)
    return sortedVotes[0][0]

'预测结果，算出精确度'
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet)))*100.0




def main():
    trainingSet = []
    testSet = []
    split = 0.7 # 2/3训练集，1/3测试集
    loadDataset(r'../datafile/irisdata.txt',split,trainingSet,testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
  #generate predictions
    predictions = []
    k =5
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print ('>predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    print ('predictions: ' + repr(predictions))
    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == '__main__':
    main()