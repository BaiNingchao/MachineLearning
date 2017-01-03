from sklearn import neighbors
from sklearn import datasets
'''
Description:python调用机器学习库scikit-learn的K临近算法，实现花瓣分类
Author:Bai Ningchao
DateTime:2017年1月3日15:07:25
Blog URL:http://www.cnblogs.com/baiboy/
Document：http://scikit-learn.org/stable/modules/neighbors.html
'''

knn=neighbors.KNeighborsClassifier()

iris=datasets.load_iris()

# print(iris)
'传入参数，data是花的特征数据，样本数据；target是分类的数据，归为哪类。'
knn.fit(iris.data,iris.target)

'knn预测分类,参数是 萼片和花瓣的长度和宽度'
predictedlabel = knn.predict([[0.1,0.2,0.3,0.4]])

print('分类结果[0:setosa][1:versicolor][2:virginica]：\n',predictedlabel,':setosa' if predictedlabel[0]==0 else 'versicolor or virginica')


