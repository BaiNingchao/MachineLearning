#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    项目名称：英文论文信息抽取系统
    模块功能：格式转换后对文章进行无监督学习序列标注
    开发时间：2018年6月27日17:02:12
'''
from decimal import Decimal
import time,fnmatch,os,re,sys,shutil
# 公共变量，计数器,错误文件路径，文件大小
from BaseClass import Global  #全局变量
from BaseClass import TraversalFun # 文件遍历处理基类函数


'''
txt格式文件批量章节标注
'''
def LabelBatch(filepath,newdir=''):
    # 文件路径切分为上级路径和文件名
    prapath,filename = os.path.split(filepath)
    if newdir=='':
        newdir = prapath
    else:
        newdir = newdir
    newpath = os.path.join(newdir,filename)
    print(newpath)
    # 对应标注字典
    flagdict ={'a b s t r a c t':'</T>\n<A>\n', 'Abstract':'</T>\n<A>\n', 'ABSTRACT':'</T>\n<A>\n', 'INTRODUCTION':'</A>\n<I>\n', 'Introduction':'</A>\n<I>\n', 'Experimental':'</I>\n<E>\n', 'Experiment':'</I>\n<E>\n', 'EXPERIMENTAL':'</I>\n<E>\n', 'Materials':'</I>\n<E>\n', 'Results':'</E>\n<R>\n', 'RESULTS':'</E>\n<R>\n', 'Discussion':'</E>\n<R>\n', 'Conclusions':'</R>\n<CO>\n', 'CONCLUSIONS':'</R>\n<CO>\n', 'Conclusion':'</R>\n<CO>\n', 'Acknowledgments':'</CO>\n<AC>\n', 'ACKNOWLEDGMENTS':'</CO>\n<AC>\n', 'References':'</AC>\n<RE>\n', 'REFERENCES':'</AC>\n<RE>\n'}

    # 对文本进行处理
    labelpagper = ""    # 标注后的完整文本
    fp = open(filepath,'r', encoding='utf-8')
    lines = fp.readlines()     # 统计文本总行数
    labelpagper += '<T>\n'  # 打上标题开始标签
    flaglabel=0     # 缺失章节标志
    for i in range(len(lines)):
        flagword = str(lines[i]).strip()  # 去除数据前后空格
        lineLen = len(str(lines[i]).strip().split(' ')) # 排除干扰信息，提高运行速度
        if lineLen <= 8:
            # 打上摘要开始标签
            if 'a b s t r a c t' in flagword or 'Abstract' in flagword or 'ABSTRACT' in flagword:
                labelpagper += flagdict['Abstract']
                flaglabel = 1
            # 打上引言的开始标签
            if "Introduction" in flagword or "INTRODUCTION" in flagword:
                labelpagper += flagdict['Introduction']
            # 打上实验的开始标签
            if "Experimental" in flagword or "Experiment" in flagword or flagword=="EXPERIMENTAL" or "Materials and method" in flagword:
                labelpagper += flagdict['Experiment']
            # 打上结果的开始标签
            if "Results" in flagword or "RESULTS" in flagword or "Discussion" in flagword:
                labelpagper += flagdict['Results']
            # 打上结论的开始标签
            if "Conclusions" in flagword or "CONCLUSIONS" in flagword or "Conclusion" in flagword:
                labelpagper += flagdict['Conclusions']
            # 打上感谢的开始标签
            if "Acknowledgements" in flagword or "ACKNOWLEDGMENTS" in flagword:
                labelpagper += flagdict['Acknowledgments']
            # 打上参考文献的开始标签
            if "References" in flagword or "REFERENCES" in flagword:
                labelpagper += flagdict['References']
        if len(lines[i])>6: # 过滤图片公式转化等干扰信息
            labelpagper +=lines[i]
        # 处理部分章节缺失情况
        if newdir.split('/')[-1] == 'CategoryC':
            if i>2 and flaglabel==0:
                labelpagper += flagdict['Abstract']
                flaglabel = 1
        i+=1                            # 读取下一行
    labelpagper += '</RE>\n'

    with open(newpath,'w', encoding='utf-8') as f:
        f.write(labelpagper)
    Global.all_FileNum += 1








if __name__ == '__main__':

    t1=time.time()
    rootDir = r"../Document/EnPapers_batch_txt/" # 默认处理路径
    saveDir = r'../Document/EnPapers_Import/' # 默认保存结果路径
    # 每次生成清除上次的数据
    if not os.path.exists(saveDir):
        TraversalFun.mkdir(saveDir)
    else:
        shutil.rmtree(saveDir)
        TraversalFun.mkdir(saveDir)

    ''' 单个txt论文文本标注 '''
    # print ('单个论文标注:\n')
    # LabelBatch(os.path.join(rootDir,r'CategoryA/A_Paper1.txt'),saveDir)
    # print ('\n恭喜你！论文标注完成。')



    ''' 批量pdf文件转化txt'''
    print ('\n批量生成的文件:')
    # 默认方法参数打印所有文件路径
    tra=TraversalFun(rootDir,LabelBatch,saveDir)
    tra.TraversalDir()
    print ('\n恭喜你！批量文件信息抽取介绍。')


    print ('共处理文档数目：'+str(Global.all_FileNum)+' 个' )
    t2=time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n")
    input()

