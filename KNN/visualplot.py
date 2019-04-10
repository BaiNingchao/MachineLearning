from numpy import shape
from numpy import *  # 科学计算包
from decimal import Decimal
from datetime import datetime
import numpy as np

import matplotlib,pygal,csv,requests
import matplotlib.pyplot as plt
from pygal.style import LightColorizedStyle as LCS, LightStyle as LS
import os,json
import KNN

#加入中文显示
import  matplotlib.font_manager as fm
# 解决中文乱码，本案例使用宋体字
myfont=fm.FontProperties(fname=r"C:\\Windows\\Fonts\\simsun.ttc")



'''
散列表分析数据：
dataset：数据集
datingLabels：标签集
Title:列表，标题、横坐标标题、纵坐标标题。
'''
def analyze_data_plot(dataset,datingLabels,Title):
    fig = plt.figure()
    # 将画布划分为1行1列1块
    ax = fig.add_subplot(111)
    ax.scatter(dataset[:,0],dataset[:,1],15.0*array(datingLabels),15.0*array(datingLabels))

     # 设置散点图标题和横纵坐标标题
    plt.title(Title[0],fontsize=25,fontname='宋体',fontproperties=myfont)
    plt.xlabel(Title[1],fontsize=15,fontname='宋体',fontproperties=myfont)
    plt.ylabel(Title[2],fontsize=15,fontname='宋体',fontproperties=myfont)

    # 设置刻度标记大小,axis='both'参数影响横纵坐标，labelsize刻度大小
    # plt.tick_params(axis='both',which='major',labelsize=10)

    # 设置每个坐标轴取值范围
    # plt.axis([-1,25,-1,2.0])

    # 截图保存图片
    # plt.savefig('datasets_plot.png',bbox_inches='tight')

    # 显示图形
    plt.show()


'''折线图'''
def line_chart(xvalues,yvalues):
    # 绘制折线图,c颜色设置，alpha透明度
    plt.plot(xvalues,yvalues,linewidth=0.5,alpha=0.5,c='red') # num_squares数据值，linewidth设置线条粗细

    # 设置折线图标题和横纵坐标标题
    plt.title("Python绘制折线图",fontsize=30,fontname='宋体',fontproperties=myfont)
    plt.xlabel('横坐标',fontsize=20,fontname='宋体',fontproperties=myfont)
    plt.ylabel('纵坐标',fontsize=20,fontname='宋体',fontproperties=myfont)

    # 设置刻度标记大小,axis='both'参数影响横纵坐标，labelsize刻度大小
    plt.tick_params(axis='both',labelsize=14)

    # 显示图形
    plt.show()


'''散点图'''
def scatter_chart(xvalues,yvalues):
    # 绘制散点图，s设置点的大小,c数据点的颜色，edgecolors数据点的轮廓
    plt.scatter(xvalues,yvalues,c='green',edgecolors='none',s=5)

    # 设置散点图标题和横纵坐标标题
    plt.title("Python绘制折线图",fontsize=30,fontname='宋体',fontproperties=myfont)
    plt.xlabel('横坐标',fontsize=20,fontname='宋体',fontproperties=myfont)
    plt.ylabel('纵坐标',fontsize=20,fontname='宋体',fontproperties=myfont)

    # 设置刻度标记大小,axis='both'参数影响横纵坐标，labelsize刻度大小
    plt.tick_params(axis='both',which='major',labelsize=10)

    # 设置每个坐标轴取值范围
    # plt.axis([80,100,6400,10000])

    # 显示图形
    plt.show()

    # 自动保存图表,bbox_inches剪除图片空白区
    # plt.savefig('squares_plot.png',bbox_inches='tight')


'''直方图'''
def histogram(xvalues,yvalues):
    # 绘制直方图
    hist = pygal.Bar()

    # 设置散点图标题和横纵坐标标题
    hist.title = '事件频率的直方图'
    hist.x_title = '事件的结果'
    hist.y_title = '事件的频率'

    # 绘制气温图,设置图形大小
    fig = plt.figure(dpi=128,figsize=(10,6))

    # 事件的结果
    hist.x_labels = xvalues

    # 事件的统计频率
    hist.add('事件',yvalues)

    hist.x_label_rotation=45  # x坐标倾斜

    # 保存文件路径
    hist.render_to_file('die_visual.svg')

'''气温趋势图'''
def temper_char():
    fig = plt.figure(dpi=128,figsize=(10,6))
    dates,highs,lows = [],[],[]
    with open(os.path.abspath(r'./datasource/weather07.csv')) as f:
        reader = csv.reader(f)
        header_row = next(reader) # 返回文件第一行
        # enumerate 获取元素的索引及其值
        # for index,column_header in enumerate(header_row):
        #     print(index,column_header)
        for row in reader:
            current_date = datetime.strptime(row[0],"%Y-%m-%d")
            dates.append(current_date)
            highs.append(int(row[1]))
            lows.append((int(row[3])))


    # 接收数据并绘制图形,facecolor填充区域颜色
    plt.plot(dates,highs,c='red',linewidth=4,alpha=0.5)
    plt.plot(dates,lows,c='green',linewidth=4,alpha=0.5)
    plt.fill_between(dates,highs,lows,facecolor='blue',alpha=0.2)

    # 设置散点图标题和横纵坐标标题
    plt.title("日常最高气温，2018年7月",fontsize=24,fontname='宋体',fontproperties=myfont)
    plt.xlabel('横坐标',fontsize=20,fontname='宋体',fontproperties=myfont)
    plt.ylabel('温度',fontsize=20,fontname='宋体',fontproperties=myfont)

    # 绘制斜的日期
    fig.autofmt_xdate()

    # 设置刻度标记大小,axis='both'参数影响横纵坐标，labelsize刻度大小
    plt.tick_params(axis='both',which='major',labelsize=15)

    # 显示图形
    plt.show()

'''Github最受欢迎的星标项目'''
def repos_hist():
    #查看API速率限制
    # url = https://api.github.com/rate_limit
    # 执行github API调用并存储响应
    url = 'https://api.github.com/search/repositories?q=language:python&sort=stars'
    r = requests.get(url)
    print("Status code:",r.status_code) # 状态码200表示成功

    # 将API响应存储在一个变量里面
    response_dict = r.json()
    print("Hithub总的Python仓库数：",response_dict['total_count'])

    # 探索有关仓库的信息
    repo_dicts = response_dict['items']

    names,stars = [],[]
    for repo_dict in repo_dicts:
        names.append(repo_dict['name'])
        stars.append(repo_dict['stargazers_count'])

    # 可视化,x_label_rotation围绕x轴旋转45度，show_legend图例隐藏与否
    my_style = LS(base_style=LCS)

    my_config = pygal.Config()
    my_config.x_label_rotation=45 # 横坐标字体旋转角度
    my_config.show_legend=False
    my_config.title_font_size=24 # 标题大小
    my_config.label_font_size=14 # 副标题大小，纵横坐标数据
    my_config.major_label_font_size = 18 # 主标签大小，纵坐标5000整数倍
    my_config.truncate_label=15  # 项目名称显示前15个字
    my_config.show_y_guides=False # 隐藏水平线
    my_config.width=1200 # 自定义宽度
    # chart = pygal.Bar(style=my_style,x_label_rotation=45,show_legend=False)
    chart = pygal.Bar(my_config,style=my_style)
    chart.title = 'Github最受欢迎的星标项目'
    chart.x_labels = names
    chart.add('星标',stars)
    chart.render_to_file('python_repos.svg')



def Show2dCorpora(corpus):
    nodes = list(corpus)
    ax0 = [x[0][1] for x in nodes] # 绘制各个doc代表的点
    ax1 = [x[1][1] for x in nodes]
    # print(ax0)
    # print(ax1)
    plt.plot(ax0,ax1,'o')
    plt.show()






if __name__ == '__main__':
    filename = os.path.abspath(r'./datasource/datingTestSet2.txt')
    dataset,labels = KNN.file_matrix(filename)

    # xvalues = list(range(1,100)) #校正坐标点，即横坐标值列表
    # yvalues = [x**2 for x in xvalues] # 纵坐标值列表
    # x_result = [1,2,3,4,5,6]
    # y_frequencies = [152,171,175,168,150,179]

    '''1 折线图'''
    # X=np.array([[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]])
    # X[:,0]  >>  [ 0  2  4  6  8 10 12 14]
    # X[:,1]  >>  [ 1  3  5  7  9 11 13 15]
    # X[0,:]  >>  [0 1]
    # X[1,:]  >>  [2 3]
    # X[1:5,:]  >>[[2 3],[4 5],[6 7],[8 9]]

    # line_chart(noredataset[:,1],noredataset[:,2])

    '''2 散点图'''
    # scatter_chart(dataset[:,1],dataset[:,2])

    # Echart显示
    KNN.norm_Json(dataset)

    '''3 直方图'''
    # histogram([float('%.2f' % x) for x in dataset[:10,2]],[float('%.2f' % x) for x in dataset[:10,1]])

    '''4 气温趋势图'''
    # temper_char()

    '''5 Github最受欢迎的星标项目'''
    # repos_hist()
