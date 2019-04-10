#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
    项目名称：英文论文信息抽取系统
    模块功能：对标注后的英文论文信息抽取
    开发时间：2018年6月26日20:30:42
'''

from decimal import Decimal
import time,fnmatch,os,re,sys,shutil
# 公共变量，计数器,错误文件路径，文件大小
from BaseClass import Global  #全局变量
from BaseClass import TraversalFun # 文件遍历处理基类函数


'''
论文txt格式文件批量信息抽取
'''
def ExtractBatch(filepath,newdir=''):
    TraversalFun.mkdir(newdir) # 检查保存路径是否存在
    # 文件路径切分为上级路径和文件名
    prapath,filename = os.path.split(filepath)
    # 论文题目
    Global.StrTitle+="\n->"+prapath.split('/')[-1]+':'+filename+"论文题目\n"
    # 论文摘要
    Global.StrAbstract+="\n->"+prapath.split('/')[-1]+':'+filename+"论文摘要\n"
    # 论文引言
    Global.StrIntro +="\n->"+prapath.split('/')[-1]+':'+filename+"论文引言\n"
    # 实验方法
    Global.StrExpMeth +="\n->"+prapath.split('/')[-1]+':'+filename+"实验方法\n"
    # 结果讨论
    Global.StrResDis+="\n->"+prapath.split('/')[-1]+':'+filename+"结果讨论\n"
    # 论文结论
    Global.StrConclude+="\n->"+prapath.split('/')[-1]+':'+filename+"论文结论\n"
    # 论文感谢
    Global.StrAcknow+="\n->"+prapath.split('/')[-1]+':'+filename+"论文感谢\n"
    # 参考文献
    Global.StrRef +="\n->"+prapath.split('/')[-1]+':'+filename+"参考文献\n"

    begin =0    # 游标上标
    end =0      # 游标下标
    # 文本中下的标签
    flaglist =['<T>\n','</T>\n','<A>\n','</A>\n','<I>\n','</I>\n','<E>\n','</E>\n','<R>\n','</R>\n','<CO>\n','</CO>\n','<AC>\n','</AC>\n','<RE>\n','</RE>\n']
    # 对文本进行处理
    fp = open(filepath,'r', encoding='utf-8')
    lines = fp.readlines()     # 统计文本总行数
    for j in range(len(flaglist)):
        for i in range(len(lines)):
            if lines[i] == flaglist[j]:  # 如果遇到标记，记录游标加一
                begin = i+1
            if lines[i] == flaglist[j+1]: # 遇到结束标记，记录游标
                end = i
            while begin < end:  # 提取标记中的内容
                if flaglist[j] == '<T>\n':
                    Global.StrTitle += str(lines[begin])        # 论文题目
                elif flaglist[j] == '<A>\n':
                    Global.StrAbstract += str(lines[begin])     # 论文摘要
                elif flaglist[j] == '<I>\n':
                    Global.StrIntro +=str(lines[begin])        # 论文引言
                elif flaglist[j] == '<E>\n':
                    Global.StrExpMeth +=str(lines[begin])      # 实验方法
                elif flaglist[j] == '<R>\n':
                    Global.StrResDis += str(lines[begin])       # 结果讨论
                elif flaglist[j] == '<CO>\n':
                    Global.StrConclude += str(lines[begin])     # 论文结论
                elif flaglist[j] == '<AC>\n':
                    Global.StrAcknow += str(lines[begin])       # 论文感谢
                elif flaglist[j] == '<RE>\n':
                    Global.StrRef += str(lines[begin])          # 参考文献
                # else:
                #     return
                begin = begin+1 # 游标向下走动

        if j <= len(flaglist)-3:
            j = j+2
        else:
            break
    Global.all_FileNum += 1


'''
论文抽取数据的批量保存
'''
def SaveBatchExtract(newdir=''):
    isExists=os.path.exists(newdir)
    if not isExists:
        TraversalFun.mkdir(newdir)
    else:
        shutil.rmtree(newdir)
        TraversalFun.mkdir(newdir)
    namelist=['title','abstract','intro','conthe','expmeth','resdis','conclude','acknow','ref']
    new_txt_name=[] # 新的文件名列表
    newpath=[] # 新的文件名路径
    for z in range(len(namelist)):
        new_txt_name.append(namelist[z]+'.txt')
        if new_txt_name[z] ==None:
            return
        newpath.append(os.path.join(newdir,new_txt_name[z])) # 遍历每部分章节的文件
        print(str(z)+":"+newpath[z])
        '''保存各章节内容'''
        with open(newpath[z],'w', encoding='utf-8') as f:
            if z==0:
                f.write(Global.StrTitle)
            elif z == 1:
                f.write(Global.StrAbstract)
            elif z == 2:
                f.write(Global.StrIntro)
            elif z == 3:
                f.write(Global.StrConThe)
            elif z == 4:
                f.write(Global.StrExpMeth)
            elif z == 5:
                f.write(Global.StrResDis)
            elif z == 6:
                f.write(Global.StrConclude)
            elif z == 7:
                f.write(Global.StrAcknow)
            elif z == 8:
                f.write(Global.StrRef)
            else:
                break



if __name__ == '__main__':

    t1=time.time()
    rootDir = r"../Document/EnPapers_Import/" # 默认处理路径
    saveDir = r'../Document/EnPapers_Export/' # 默认保存结果路径

    ''' 单个pdf文件转化为txt '''
    # print ('单个抽取的文件:\n')
    # ExtractBatch(os.path.join(rootDir,r'A_Paper1.txt'),saveDir)
    # SaveBatchExtract(saveDir)
    # print ('\n恭喜你！单文件信息抽取介绍。')

    ''' 批量pdf文件转化txt'''
    print ('\n批量生成的文件:')
    # 默认方法参数打印所有文件路径
    tra=TraversalFun(rootDir,ExtractBatch,saveDir)
    tra.TraversalDir()
    SaveBatchExtract(saveDir)  #保存批量文件
    print ('\n恭喜你！批量文件信息抽取介绍。')

    print ('共处理文档数目：'+str(Global.all_FileNum)+' 个' )
    t2=time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n")
    input()

