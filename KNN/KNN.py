# conding=utf8
from numpy import *  # 科学计算包
import operator      # 运算符包
from numpy import shape  # 查看矩阵或数组的方法
import numpy as np
import visualplot
from BaseClass import *  # 导入基础类库
import jieba,sys,os ,time,re,json,codecs

from scipy.spatial.distance import cdist

path1=os.path.abspath('.')   #表示当前所处的文件夹的绝对路径
path2=os.path.abspath('..')  #表示当前所处的文件夹上一级文件夹的绝对路径



'''KNN创建数据源，返回数据集和标签'''
def create_dataset():
    group = array(random.randint(0,10,size=(20,3))) # 数据集
    labels = ['A','A','B','B','A','A','B','B','A','A','B','B','A','A','B','B'] # 标签
    return group,labels


'''对文件进行格式处理，便于分类器可以理解'''
def file_matrix(filename):
    f = open(filename)
    arrayLines = f.readlines()
    returnMat = zeros((len(arrayLines),3))    # 数据集
    classLabelVactor = []                     # 标签集
    index = 0
    for line in arrayLines:
        listFromLine = line.strip().split('    ')    # 分析数据，空格处理
        returnMat[index,:] = listFromLine[0:3]
        classLabelVactor.append(int(listFromLine[-1]))
        index +=1
    return returnMat,classLabelVactor


'''数值归一化：特征值转化为0-1之间：newValue = (oldValue-min)/(max-min)'''
def norm_dataset(dataset):
    minVals = dataset.min(0)  # 参数0是取得列表中的最小值，而不是行中最小值
    maxVals = dataset.max(0)
    ranges = maxVals - minVals
    normdataset = zeros(shape(dataset)) # 生成原矩阵一样大小的0矩阵

    m = dataset.shape[0]
    # tile:复制同样大小的矩阵
    molecular = dataset - tile(minVals,(m,1))  # 分子： (oldValue-min)
    Denominator = tile(ranges,(m,1))           # 分母：(max-min)
    normdataset = molecular/Denominator     # 归一化结果。

    # print('归一化的数据结果：\n'+str(normdataset))
    return normdataset,ranges,minVals


'''array数据转化json'''
def norm_Json(dataset):
    noredataset = norm_dataset(dataset)[0] # 数据归一化
    number1 = np.around(noredataset[:,1], decimals=4) # 获取数据集第二列
    number2 = np.around(noredataset[:,2], decimals=4) # 获取数据集第三列
    returnMat=zeros((dataset.shape[0],2))             # 二维矩阵
    returnMat[:,0] = number1
    returnMat[:,1] = number2

    file_path = os.path.abspath(r"./datasource/test.json")
    json.dump(returnMat.tolist(), codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


    #
    # strlist = "["
    # for ret in number2:
    #     strlist += str(ret)+'\n'
    # strlist += ']'
    # with open(os.path.abspath(r"./datasource/test.txt"),'w') as f:
    #     f.write(strlist)


''' 构造KNN分类器
    vecX:输入向量，待测数据
    filename: 特征集文件路径
    isnorm:是否进行归一化处理
    k:k值的选择，默认选择3
'''
def knn_classifier(vecX,dataset,labels,isnorm='Y',k=3):
    # 距离计算（方法1）
    if isnorm == 'Y':
        normMat,ranges,minVals = norm_dataset(dataset)     # 对数据进行归一化处理
        normvecX = norm_dataset(vecX)
    else:
        normMat = dataset
        normvecX = vecX

    m = normMat.shape[0]
    # tile方法是在列向量vecX，datasetSize次，行向量vecX1次叠加
    diffMat = tile(normvecX,(m,1)) - normMat
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)   # axis=0 是列相加,axis=1是行相加
    distances = sqDistances**0.5
    # print('vecX向量到数据集各点距离：\n'+str(distances))

    sortedDistIndexs = distances.argsort(axis=0)  # 距离排序，升序
    # print(sortedDistIndicies)

    classCount = {}   # 统计前k个类别出现频率
    for i in range(k):
        votelabel = labels[sortedDistIndexs[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1 #统计键值
    # 类别频率出现最高的点,itemgetter(0)按照key排序，itemgetter(1)按照value排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(str(vecX)+'KNN的投票决策结果：\n'+str(sortedClassCount[0][0]))
    return sortedClassCount[0][0]


'''测试评估算法模型'''
def test_knn_perfor(filename):
    hoRatio = 0.1
    dataset,label = file_matrix(filename)              # 获取训练数据和标签
    normMat,ranges,minVals = norm_dataset(dataset)     # 对数据进行归一化处理
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)                       # 10%的测试数据条数

    errorCount = 0.0                                   # 统计错误分类数
    for i in range(numTestVecs):
        # X=np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]])
        # normMat[i,:]  >>  取数据集前100个做测试集
        # normMat[numTestVecs:m,:]  >>  取数据集100-1000共900个做训练集
        # label[numTestVecs:m]  >>  取900个训练集的标签
        classifierResult = knn_classifier(normMat[i,:],normMat[numTestVecs:m,:],label[numTestVecs:m],3)          # 此处分类器可以替换不同分类模型
        print('分类结果:'+str(classifierResult)+'\t\t准确结果:'+str(label[i]))

        if classifierResult != label[i]:
            errorCount += 1.0
        Global.all_FileNum += 1
    print('总的错误率为：'+str(errorCount/float(numTestVecs))+"\n总测试数据量: "+str(Global.all_FileNum))


'''调用可用算法'''
def show_classifyPerson(filename):
    resultlist = ['不喜欢','还可以','特别喜欢']
    ffMiles = float(input('每年飞行的历程多少公里？\n'))
    percentTats = float(input('玩游戏时间占百分比多少？\n')) # [751,13,0.4][40920 ,8.326976,0.953952]
    iceCream = float(input('每周消费冰淇淋多少公升？\n'))

    dataset,labels = file_matrix(filename) # 数据格式化处理
    inArr = array([ffMiles,percentTats,iceCream])

    classifierResult = knn_classifier(inArr,dataset,labels,3) # 数据归一化并进行分类
    print('预测的约会结果是：'+resultlist[classifierResult-1])




if __name__ == '__main__':
    t1=time.time()

    '''1 KNN模拟数据分类算法'''
    # dataset,labels = create_dataset()
    # print('特征集：\n'+str(dataset))
    # print('标签集：\n'+str(labels))
    # vecs = array([[0,0,0],[0.1,2,3],[1,0.10,0.5]])
    # for vec in vecs:
    #     knn_classifier(vec,dataset,labels,3)

    '''2 KNN针对文件的分类算法'''
    filename = os.path.abspath(r'./datasource/datingTestSet2.txt')
    dataset,labels = file_matrix(filename)
    # knn_classifier([5569,4.875435,0.728658],dataset,labels,3)

    '''3 文件数据图形化分析数据 '''
    dataset,labels = file_matrix(filename)
    noredataset = norm_dataset(dataset)[0]
    title = ['约会数据游戏和饮食散列点','玩游戏所耗时间百分比','每周消耗在冰淇淋的公升数']
    visualplot.analyze_data_plot(noredataset,labels,title)

    '''4 测试评估算法模型'''
    # test_knn_perfor(filename)

    '''5 调用可用算法'''
    # show_classifyPerson(filename)


    t2=time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n")


