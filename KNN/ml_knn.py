# coding=gbk
import KNN
import visualplot
from numpy import *
import os





'''KNN分类算法'''
def show_knn_dome():
    dataset,labels = KNN.create_dataset()
    KNN.knn_classifier([0,0],dataset,labels,3)


'''图形可视化分析数据'''
def show_visual_dataset(filename,title):
    dataset,labels = KNN.file_matrix(filename)   # 数据格式化处理
    visualplot.analyze_data_plot(dataset,labels,title) # 可视化分析数据







if __name__ == '__main__':
    # 文件路径
    filename = os.path.abspath(r'./datasource/datingTestSet2.txt')

    # KNN分类算法
    # show_knn_dome()

    #图形可视化分析数据
    # title = ['约会数据游戏和饮食散列点','玩游戏所耗时间百分比','每周消耗在冰淇淋的公升数']
    # show_visual_dataset(filename,title)

    # 调用算法
    KNN.show_classifyPerson(filename)


