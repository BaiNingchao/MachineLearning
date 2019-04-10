#!/usr/bin/python
# coding: utf-8


from numpy import linalg as la
from numpy import *

'''一步步教你轻松学奇异值分解SVD降维算法
Blog：https://bainingchao.github.io/
Date：2018年10月11日11:03:42
'''


'''原矩阵'''
def loadExData():
    return [[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]


'''示例矩阵'''
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


'''利用SVD提高推荐效果，菜肴矩阵'''
def loadExData3():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
           [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
           [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]



'''基于欧氏距离相似度计算，假定inA和inB 都是列向量
相似度=1/(1+距离),相似度介于0-1之间
norm：范式计算，默认是2范数，即:sqrt(a^2+b^2+...)
'''
def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))


'''皮尔逊相关系数
范围[-1, 1]，归一化后[0, 1]即0.5 + 0.5 *
相对于欧式距离，对具体量级（五星三星都一样）不敏感皮尔逊相关系数
'''
def pearsSim(inA, inB):
    # 检查是否存在3个或更多的点不存在，该函数返回1.0，此时两个向量完全相关。
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


'''计算余弦相似度
如果夹角为90度相似度为0；两个向量的方向相同，相似度为1.0
余弦取值-1到1之间，归一化到0与1之间即：相似度=0.5 + 0.5*cosθ
余弦相似度cosθ=(A*B/|A|*|B|)
'''
def cosSim(inA, inB):
    num = float(inA.T*inB) # 矩阵相乘
    denom = la.norm(inA)*la.norm(inB) # 默认是2范数
    return 0.5 + 0.5*(num/denom)



'''基于物品相似度的推荐引擎
descripte：计算某用户未评分物品中，以对该物品和其他物品评分的用户的物品相似度，然后进行综合评分
dataMat         训练数据集
user            用户编号
simMeas         相似度计算方法
item            未评分的物品编号
Returns: ratSimTotal/simTotal  评分（0～5之间的值）
'''
def standEst(dataMat, user, simMeas, item):
    # 得到数据集中的物品数目
    n = shape(dataMat)[1]
    # 初始化两个评分值
    simTotal = 0.0 ; ratSimTotal = 0.0
    # 遍历行中的每个物品（对用户评过分的物品遍历，并与其他物品进行比较）
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0: # 终止循环
            continue
        # 寻找两个都评级的物品,变量overLap 给出两个物品中已被评分的元素索引ID
        # logical_and 计算x1和x2元素的真值。
        # print(dataMat[:, item].T.A, ':',dataMat[:, j].T.A )
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 如果相似度为0，则两着没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        # 如果存在重合的物品，则基于这些重合物重新计算相似度。
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity  用户相似度，   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal/simTotal



'''分析 Sigma 的长度取值
根据自己的业务情况，就行处理，设置对应的 Singma 次数
通常保留矩阵 80% ～ 90% 的能量，就可以得到重要的特征并取出噪声。
'''
def analyse_data(Sigma, loopNum=20):
    # 总方差的集合（总能量值）
    Sig2 = Sigma**2
    SigmaSum = sum(Sig2)
    for i in range(loopNum):
        SigmaI = sum(Sig2[:i+1])
        print('主成分：%s, 方差占比：%s%%' % (format(i+1, '2.0f'), format(SigmaI/SigmaSum*100, '.2f')))


'''基于SVD的评分估计
Args:
    dataMat         训练数据集
    user            用户编号
    simMeas         相似度计算方法
    item            未评分的物品编号
Returns:
    ratSimTotal/simTotal     评分（0～5之间的值）
'''
def svdEst(dataMat, user, simMeas, item):
    # 物品数目
    n = shape(dataMat)[1]
    # 对数据集进行SVD分解
    simTotal = 0.0 ;  ratSimTotal = 0.0
    # 奇异值分解,只利用90%能量值的奇异值，奇异值以NumPy数组形式保存
    U, Sigma, VT = la.svd(dataMat)
    # 分析 Sigma 的长度取值
    # analyse_data(Sigma, 20)

    # 如果要进行矩阵运算，就必须要用这些奇异值构建出一个对角矩阵
    Sig4 = mat(eye(4) * Sigma[: 4]) # eye对角矩阵
    # 利用U矩阵将物品转换到低维空间中，构建转换后的物品(物品+4个主要的特征)
    xformedItems = dataMat.T * U[:, :4] * Sig4.I # I 逆矩阵
    # print('dataMat', shape(dataMat))
    # print('U[:, :4]', shape(U[:, :4]))
    # print('Sig4.I', shape(Sig4.I))
    # print('VT[:4, :]', shape(VT[:4, :]))
    # print('xformedItems', shape(xformedItems))

    # 对于给定的用户，for循环在用户对应行的元素上进行遍历
    # 和standEst()函数的for循环一样，这里相似度计算在低维空间下进行的。
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        # 相似度的计算方法也会作为一个参数传递给该函数
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        # for 循环中加入了一条print语句，以便了解相似度计算的进展情况。如果觉得累赘，可以去掉
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 对相似度不断累加求和
        simTotal += similarity
        # 对相似度及对应评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        # 计算估计评分
        return ratSimTotal/simTotal



'''recommend函数推荐引擎，默认调用standEst函数，产生最高的N个推荐结果
Args:
    dataMat         训练数据集
    user            用户编号
    simMeas         相似度计算方法
    estMethod       使用的推荐算法
Returns:  返回最终 N 个推荐结果
'''
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找未评级的物品,对给定的用户建立一个未评分的物品列表
    unratedItems = nonzero(dataMat[user, :].A == 0)[1] # .A: 矩阵转数组
    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return 'you rated everything'
    # 物品的编号和评分值
    itemScores = []
    # 在未评分物品上进行循环
    for item in unratedItems:
        # 获取 item 该物品的评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    # 按照评分得分 进行逆排序，获取前N个未评级物品进行推荐
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]



'''图像压缩函数'''
def imgLoadData(filename):
    myl = []
    for line in open(filename).readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    # 矩阵调入后，就可以在屏幕上输出该矩阵
    myMat = mat(myl)
    return myMat


'''打印矩阵
由于矩阵保护了浮点数，因此定义浅色和深色，遍历所有矩阵元素，当元素大于阀值时打印1，否则打印0
'''
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1)
            else:
                print(0)
        print('')


'''实现图像压缩，允许基于任意给定的奇异值数目来重构图像
Args:
    numSV       Sigma长度
    thresh      判断的阈值
'''
def imgCompress(numSV=3, thresh=0.8):
    # 构建一个列表
    myMat = imgLoadData('./0_5.txt')

    print("****original matrix****")
    # 对原始图像进行SVD分解并重构图像e
    printMat(myMat, thresh)

    # 通过Sigma 重新构成SigRecom来实现
    # Sigma是一个对角矩阵，因此需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
    U, Sigma, VT = la.svd(myMat)
    # SigRecon = mat(zeros((numSV, numSV)))
    # for k in range(numSV):
    #     SigRecon[k, k] = Sigma[k]

    # 分析插入的 Sigma 长度
    # analyse_data(Sigma, 20)

    SigRecon = mat(eye(numSV) * Sigma[: numSV])
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print("****reconstructed matrix using %d singular values *****" % numSV)
    printMat(reconMat, thresh)




if __name__ == "__main__":
    '''
    # 1 对矩阵进行SVD分解(用python实现SVD)
    Data = loadExData() # m*n =7*5即7行5列
    print('Data:\n', mat(Data))
    U, Sigma, VT = linalg.svd(Data)
    # Sigma前3个数值比后两个值大很多，可以将这两个值去掉
    print('U:\n', U) # 7*7即m*m
    print('Sigma\n', Sigma) # 7*5即m*n
    print('VT:\n', VT)
    print('VTT:\n', VT.T) # 5*5即n*n

    # 2 重构一个3x3的矩阵Sig3
    Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    print('Initial Data:\n',U[:, :3] * Sig3 * VT[:3, :])
    '''

    '''
    # 3 相似度计算
    # 3.1 计算欧氏距离,比较第一列商品A和第四列商品D的相似率
    myMat = mat(loadExData())
    print(myMat)
    print('欧氏距离\n',ecludSim(myMat[:, 0], myMat[:, 2]))
    # 3.2 计算余弦相似度
    print('余弦相似度距离\n',cosSim(myMat[:, 0], myMat[:, 2]))
    # 3.3 计算皮尔逊相关系数
    print('皮尔逊距离\n',pearsSim(myMat[:, 0], myMat[:, 2]))
    '''


    # 4 计算相似度的方法
    myMat = mat(loadExData3())
    # 计算相似度的第一种方式
    # print(recommend(myMat, 1, estMethod=svdEst))
    # 计算相似度的第二种方式
    # print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

    # 默认推荐（菜馆菜肴推荐示例）
    print('菜馆菜肴推荐结果：',recommend(myMat, 2))


    # 5 利用SVD提高推荐效果,分析长度取值
    # U, Sigma, VT = la.svd(mat(loadExData2()))
    # print(Sigma)                 # 计算矩阵的SVD来了解其需要多少维的特征
    # analyse_data(Sigma)

    # 压缩图片
    # imgCompress(2)