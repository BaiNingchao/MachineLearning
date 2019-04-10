# coding: utf8
'''
Description:Logistic Regression
Author: Bai Ningchao
Created on 2018-9-4
Blog: http://www.cnblogs.com/baiboy/
'''

from numpy import *
import matplotlib.pyplot as plt



# -------------逻辑回归梯度上升优化算法，训练回归系数-------------------------



'''加载数据集和类标签'''
def loadDataSet(file_name):
    # dataMat为原始数据， labelMat为原始数据的标签
    dataMat,labelMat = [],[]
    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        if len(lineArr) == 1:
            continue    # 这里如果就一个空的元素，则跳过本次循环
        # 为了方便计算，我们将每一行的开头添加一个 1.0 作为 X0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


'''加载数据集和类标签2'''
def loadDataSet2(file_name):
    frTrain = open(file_name)
    trainingSet,trainingLabels = [],[]
    for line in frTrain.readlines():
        currLine = line.strip().split(',')
        # print(len(currLine))
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[len(currLine)-1]))
    return trainingSet,trainingLabels


''' sigmoid跳跃函数 '''
def sigmoid(ZVar):
    # gradAscent1 使用下面代码
    # if ZVar>=0:
    #     return 1.0/(1+exp(-ZVar))
    # else:
    #     return exp(ZVar)/(1+exp(ZVar))

    # gradAscent,gradAscent0 使用下面代码
    return 1.0 / (1 + exp(-ZVar))
    # Tanh是Sigmoid的变形，与 sigmoid 不同的是，tanh 是0均值的。因此，实际应用中，tanh 会比 sigmoid 更好。
    # return 2 * 1.0/(1+exp(-2*ZVar)) - 1



''' 正常的梯度上升法，得到的最佳回归系数 '''
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)  # 转换为 NumPy 矩阵
    # 转化为矩阵[[0,1,0,1,0,1.....]]，并转制[[0],[1],[0].....]
    # transpose() 行列转置函数
    # 将行向量转化为列向量   =>  矩阵的转置
    labelMat = mat(classLabels).transpose()  # 首先将数组转换为 NumPy 矩阵，然后再将行向量转置为列向量
    # m->数据量，样本数 n->特征数
    m, n = shape(dataMatrix) # 矩阵的行数和列数
    # print(m,n)
    alpha = 0.001  # alpha代表向目标移动的步长
    maxCycles = 500 # 迭代次数
    weights = ones((n, 1)) # 代表回归系数,ones((n,1)) 长度和特征数相同矩阵全是1
    for k in range(maxCycles):
        # n*3   *  3*1  = n*1
        h = sigmoid(dataMatrix * weights)  # 矩阵乘法
        # labelMat是实际值
        error = (labelMat - h)  # 向量相减
        # 0.001* (3*m)*(m*1) 表示在每一个列上的一个误差情况，最后得出 x1,x2,xn的系数的偏移量
        weights = weights + alpha * dataMatrix.transpose() * error  # 矩阵乘法，最后得到回归系数
    return array(weights)
# 思考？步长和迭代次数的初始值



''' 随机梯度上升'''
# 梯度上升与随机梯度上升的区别？梯度下降在每次更新数据集时都需要遍历整个数据集，计算复杂都较高；随机梯度下降一次只用一个样本点来更新回归系数
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)  # 初始化长度为n的数组，元素全部为 1
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求 f(x)的值，f(x)=a1*x1+b2*x2+..+nn*xn
        h = sigmoid(sum(dataMatrix[i] * weights))
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        # 0.01*(1*1)*(1*n)
        weights = array(weights) + alpha * error * array(mat(dataMatrix[i]))
    return array(weights.transpose())



''' 改进版的随机梯度上升，使用随机的一个样本来更新回归系数'''
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # 创建与列数相同的矩阵的系数矩阵
    # 随机梯度, 循环150,观察是否收敛
    for j in range(numIter):
        dataIndex = list(range(m)) # [0, 1, 2 .. m-1]
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.0001 # alpha随着迭代不断减小非0
            # random.uniform(x, y) 随机生成下一个实数，它在[x,y]范围内
            Index = int(random.uniform(0, len(dataIndex)))
            # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn
            h = sigmoid(sum(dataMatrix[dataIndex[Index]] * weights))
            error = classLabels[dataIndex[Index]] - h
            weights = weights + alpha * error *array(mat(dataMatrix[dataIndex[Index]]))
            del (dataIndex[Index])
    # print(weights.transpose())
    return weights.transpose()



# -------------分析数据可视化决策边界-------------------------

''' 数据可视化展示 '''
def plotBestFit(dataArr, labelMat, weights):
    n = shape(dataArr)[0]
    xcord1,xcord2,ycord1,ycord2 = [],[],[],[]
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    """
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2
    """
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()



'''数据集决策可视化'''
def simpleTest(file_name):
    # 1.收集并准备数据
    dataMat, labelMat = loadDataSet(file_name)
    # 2.训练模型，  f(x)=a1*x1+b2*x2+..+nn*xn中 (a1,b2, .., nn).T的矩阵值
    dataArr = array(dataMat)
    weights = stocGradAscent1(dataArr, labelMat)
    # 数据可视化
    plotBestFit(dataArr, labelMat, weights)




# -----------Logistic回归和梯度上升算法来预测患有疝病的死亡问题---------------

'''测试Logistic算法分类'''
def testClassier():
    # 使用改进后的随机梯度上升算法 求得在此数据集上的最佳回归系数 trainWeights
    file_name = './HorseColicTraining.txt'
    trainingSet,trainingLabels = loadDataSet2(file_name)
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # 根据特征向量预测结果
    teststr = '2.000000,1.000000,38.300000,40.000000,24.000000,1.000000,1.000000,3.000000,1.000000,3.000000,3.000000,1.000000,0.000000,0.000000,0.000000,1.000000,1.000000,33.000000,6.700000,0.000000,0.000000'
    currLine = teststr.strip().split(',')
    lineArr = []
    for i in range(len(currLine)):
        lineArr.append(float(currLine[i]))
    res = classifyVector(array(lineArr), trainWeights)
    # 打印预测结果
    reslut = ['死亡','存活']
    print('预测结果是：',int(res))



'''分类函数，根据回归系数和特征向量来计算 Sigmoid的值,大于0.5函数返回1，否则返回0'''
def classifyVector(featuresV, weights):
    prob = sigmoid(sum(featuresV * weights))
    print(prob)
    if prob > 0.9: return 1.0
    else: return 0.0



'''打开测试集和训练集,并对数据进行格式化处理'''
def colicTest():
    file_name = './HorseColicTraining.txt'
    trainingSet,trainingLabels = loadDataSet2(file_name)
    # 使用改进后的随机梯度上升算法 求得在此数据集上的最佳回归系数 trainWeights
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    frTest = open('./HorseColicTest.txt')
    errorCount = 0 ; numTestVec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split(',')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(
                currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("逻辑回归算法测试集的错误率为: %f" % errorRate)
    return errorRate


# 调用 colicTest() 10次并求结果的平均值
def multiTest():
    numTests = 10;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("迭代 %d 次后的平均错误率是: %f" % (numTests, errorSum / float(numTests)))



# -----------------------主函数--------------------------------------------

if __name__ == "__main__":
    # 1 加载数据集和类标签
    # file_name = r'./test.txt'
    # dataMat, labelMat=loadDataSet(file_name)
    # print(dataMat,labelMat)

    # file_name = './HorseColicTraining.txt'
    # dataMat, labelMat=loadDataSet2(file_name)

    # 2.1 梯度上升计算回归系数
    # weights = gradAscent(dataMat,labelMat)
    # print(weights)

    # 2.2 随机梯度上升计算回归系数
    # weightsV = stocGradAscent0(dataMat,labelMat)
    # print(weightsV)

    # 2.3 随机梯度上升计算回归系数
    # weightsVal = stocGradAscent1(dataMat,labelMat)
    # print(weightsVal)

    # 3 数据分析可视化决策边界
    # plotBestFit(array(dataMat),labelMat,weightsVal)

    # 4 数据集决策可视化
    # simpleTest(file_name)


    # 5测试Logistic算法分类'''
    # testClassier()

    # 6 查看错误率
    colicTest()

    # 7 10次平均错误率
    # multiTest()


