'''
    项目名称：英文论文信息抽取系统
    模块功能：pdf格式英文论文批量和单文件转化txt格式
    开发时间：2018年5月28日10:35:12
'''

# pdfminer库的地址 https://pypi.python.org/pypi/pdfminer3k
# 下载后，用cmd执行命令 setup.py install
# http://www.cnblogs.com/baiboy/
from pdfminer.pdfparser import PDFParser,PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTTextBoxHorizontal,LAParams
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed

from decimal import Decimal
import time,fnmatch,os,re,sys
# 公共变量，计数器,错误文件路径，文件大小
from BaseClass import Global  #全局变量
from BaseClass import TraversalFun # 文件遍历处理基类函数

import shutil,os
# 清除警告
import logging
logging.Logger.propagate = False
logging.getLogger().setLevel(logging.ERROR)


'''
pdf格式文件批量转化为txt格式文件
'''
def BatchPdfToTxt(filepath,newdir=''):
    # 文件路径切分为上级路径和文件名
    prapath,filename = os.path.split(filepath)
    if newdir=='':
        newdir = prapath
    else:
        newdir = newdir
    new_txt_name=TraversalFun.TranType(filename,"pdf2txt")
    if new_txt_name ==None:
        return
    TraversalFun.mkdir(newdir) # 创建目录

    try:
        newpath = os.path.join(newdir,new_txt_name)
        print ("格式转换后保留路径："+newpath)

        # 对文本进行处理
        fp = open(filepath, 'rb')  # 以二进制读模式打开
        praser = PDFParser(fp)  #用文件对象来创建一个pdf文档分析器
        doc = PDFDocument()  # 创建一个PDF文档
        praser.set_document(doc)  # 连接分析器 与文档对象
        doc.set_parser(praser)
        doc.initialize()  # 提供初始化密码，如果没有密码 就创建一个空的字符串

        # 检测文档是否提供txt转换，不提供就忽略
        if not doc.is_extractable:
            Global.error_file_list.append(filepath)
            return

        rsrcmgr = PDFResourceManager() # 创建PDf 资源管理器 来管理共享资源
        laparams = LAParams() # 创建一个PDF设备对象
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)  # 创建一个PDF解释器对象
        pdfStr = ""    # 定义存放结果的字符串
        # 循环遍历列表，每次处理一个page的内容
        for page in doc.get_pages(): # doc.get_pages() 获取page列表
            interpreter.process_page(page)
            layout = device.get_result()  # 接受该页面的LTPage对象
            # 这里layout是一个LTPage对象 里面存放着 这个page解析出的各种对象 一般包括LTTextBox, LTFigure, LTImage, LTTextBoxHorizontal 等等 想要获取文本就获得对象的text属性，
            for x in layout:
                if (isinstance(x, LTTextBoxHorizontal)):
                    pdfStr = pdfStr + x.get_text()
        #保存这些内容
        with open(newpath,'wb') as f:
            f.write(pdfStr.encode())

        # 限制文件列表,默认限制5K以下
        filesize = os.path.getsize(newpath)
        # print(filesize)
        # print(Global.limit_file_size)
        if filesize < Global.limit_file_size :
            Global.limit_file_list.append(newpath+"\t\t"+ str(Decimal(filesize/1024).quantize(Decimal('0.00'))) +"KB")
            os.remove(newpath)
        else :
            Global.all_FileNum+=1
    except Exception as e:
        Global.error_file_list.append(filepath)
        return
    finally:
        pass


'''
文件批量转化日志
'''
def filelogs(path):
    prapath,filename = os.path.split(path)
    # 创建日志目录
    dirpath = prapath+r"/"+filename+"_logs"
    TraversalFun.mkdir(dirpath)
    # 错误文件路径
    errorpath = dirpath+r"/errorlogs.txt"
    # 限制文件路径
    limitpath = dirpath+r"/limitlogs.txt"
    # 错误文件日志写入
    TraversalFun.writeFile('\n'.join(Global.error_file_list),errorpath)
    # # 限制文件日志写入
    TraversalFun.writeFile('\n'.join(Global.limit_file_list),limitpath)



'''
格式转换功能：通过调用TraversalFun类的封装遍历单个文件，传方法参数对单个文
              件不同功能处理
TraversalFun：
            1 rootDir，待处理目录路径
            2 deffun，可选方法参数，对文件具体操作。默认打印遍历文件路径
TraversalDir方法：
            可选参数，定义导出目录头标志，默认：Totxt-，即Totxt-原目录名称。
'''
if __name__ == '__main__':

    t1=time.time()
    # print(r'请输入文档的根目录（路径用\或\\均可）:')
    # rootDir = input() # 根目录文件路径,支持自定义目录路径
    rootDir = r"../Document/EnPapers" # 默认处理路径
    # rootDir = r'C:\Users\Administrator\Desktop\序列标注'
    # saveDir1 = r'../Document/EnPapers_single/'
    saveDir1 = r'../Document/EnPapers_batch_txt/'
    # 每次生成清除上次的数据
    if not os.path.exists(saveDir1):
        TraversalFun.mkdir(saveDir1)
    else:
        shutil.rmtree(saveDir1)
        TraversalFun.mkdir(saveDir1)

    ''' 单个pdf文件转化为txt '''
    # print ('单个生成的文件:\n')
    # BatchPdfToTxt(os.path.join(rootDir,r'CategoryA/newfile.pdf'),saveDir1)

    ''' 批量pdf文件转化txt'''
    print ('\n批量生成的文件:')
    tra=TraversalFun(rootDir,BatchPdfToTxt) # 默认方法参数打印所有文件路径
    tra.TraversalDir()



    # ''' 写入日志文件 '''
    filelogs(rootDir)



    print ('共处理文档数目：'+str(Global.all_FileNum+len(Global.error_file_list)+len(Global.limit_file_list))+' 个,其中:\n \
        1) 筛选文件(可用)'+str(Global.all_FileNum)+'个.\n \
        2) 错误文件(不能识别)'+ str(len(Global.error_file_list)) +'个.\n \
        3) 限制文件(<0.1K)'+ str(len(Global.limit_file_list))+'个.' )
    t2=time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n")
    input()

