# coding=gbk
import KNN
import visualplot
from numpy import *
import os





'''KNN�����㷨'''
def show_knn_dome():
    dataset,labels = KNN.create_dataset()
    KNN.knn_classifier([0,0],dataset,labels,3)


'''ͼ�ο��ӻ���������'''
def show_visual_dataset(filename,title):
    dataset,labels = KNN.file_matrix(filename)   # ���ݸ�ʽ������
    visualplot.analyze_data_plot(dataset,labels,title) # ���ӻ���������







if __name__ == '__main__':
    # �ļ�·��
    filename = os.path.abspath(r'./datasource/datingTestSet2.txt')

    # KNN�����㷨
    # show_knn_dome()

    #ͼ�ο��ӻ���������
    # title = ['Լ��������Ϸ����ʳɢ�е�','����Ϸ����ʱ��ٷֱ�','ÿ�������ڱ���ܵĹ�����']
    # show_visual_dataset(filename,title)

    # �����㷨
    KNN.show_classifyPerson(filename)


