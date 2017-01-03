import TrainDTree


csvfileurl=r'../datafile/AllElectronics.csv'  #训练数据
testfileurl=r'../datafile/test.csv'           #测试数据

dummy=TrainDTree.trainDicisionTree(csvfileurl)  #返回训练数据的数值化表示
# RowdummyX=TrainDTree.trainDicisionTree(testfileurl) # 返回测试数据的数值化数据

# TrainDTree.DicisionTree(dummy,RowdummyX[0]) #通过模型训练预测新的数据

