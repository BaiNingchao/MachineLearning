'''
    项目名称：英文论文信息抽取系统
    模块功能：公共方法类
    开发时间：2018年4月18日17:35:12
'''

# -*- coding: utf-8 -*-
from win32com import client as wc
import os,fnmatch,time,sys
import shutil
from decimal import Decimal
import sys
sys.path


'''
功能描述：遍历文件夹根路径，对子文件单独处理
'''
class TraversalFun():
    def __init__(self,rootDir,deffun=None,newDir=""):
        self.rootDir = rootDir # 待处理文件夹根路径
        self.deffun = deffun   # 方法参数处理单文件，自定义函数。
        self.newDir = newDir   # 新目录路径


    '''
    功能描述：遍历根文件夹,对文件遍历并进行操作
    '''
    def TraversalDir(self,defpara='batch_txt'): # defpara:自定义目录头（可选）
        # 文件路径切分为上级路径和文件名('F:\\kjxm\\kjt', '1.txt')
        prapath,filename = os.path.split(self.rootDir)
        newDir =""
        if self.newDir=="":
            newDir = os.path.abspath(os.path.join(prapath,filename+"_"+defpara))
        else:
            newDir = self.newDir
        # 该目下创建一个新目录newdir，用来放转化后的txt文本
        print("保存目录路径："+newDir)
        if not os.path.exists(newDir):
            os.mkdir(newDir)
        if Global.debug:
            print(newDir)
        try:
            # 递归遍历word文件并将其转化txt文件
            TraversalFun.AllFiles(self,self.rootDir,newDir)
        except Exception as e:
            raise e
        finally:
            pass


    '''
    功能名称：递归遍历所有文件，并自动转化txt文件，存储在指定路径下。
    '''
    def AllFiles(self,rootDir,newDir=''):
        for lists in os.listdir(rootDir):
            # 待处理文件夹名字集合
            path = os.path.join(rootDir, lists)
            print("*"*5+"正在遍历目录：:"+path+"*"*5)
            # 核心算法，对文件具体操作
            if os.path.isfile(path):
                self.deffun(path,newDir) # 通过方法参数实现
            if os.path.isdir(path):
                newpath = os.path.join(newDir, lists)
                if not os.path.exists(newpath):
                    os.mkdir(newpath)
                if Global.debug:
                    print(newpath)
                # 递归遍历文件
                TraversalFun.AllFiles(self,path,newpath)



    '''
    功能名称：检查文件类型，并转化目标类型，返回新的文件名。
    '''
    @staticmethod
    #filename，待处理文件。typename，指定文本处理格式。
    def TranType(filename,typename):
        new_name = ""    # 返回新的文件名，如：申报书.txt
        if typename == "pdf2txt" : # pdf2txt，pdf文件转化为txt文件
            #如果不是pdf文件：继续
            if not fnmatch.fnmatch(filename, '*.pdf') and not fnmatch.fnmatch(filename, '*.PDF') :
                return
            #如果是pdf临时文件：继续
            if fnmatch.fnmatch(filename, '~$*'):
                return
            #得到一个新的文件名,把原文件名的后缀改成txt
            if fnmatch.fnmatch(filename, '*.pdf'):
                new_name = filename[:-4]+'.txt'
            else:
                return
        return new_name


    '''
    功能名称：文件的写操作。
    '''
    @staticmethod
    # strs：需要写入的字符串内容。filepath: 指定保存路径。
    def writeFile(strs,filepath):
        f = open(filepath,"w",encoding="utf-8")
        f.write(strs)
        f.close()


    '''
    功能名称：文件的读操作
    '''
    @staticmethod
    def readFile(filepath):
        isfile = os.path.exists(filepath)
        readstr = ""
        if isfile:
            with open(filepath,"r",encoding="utf-8") as f:
                readstr += f.read()
        else:
            return
        return readstr


    @staticmethod
    def AllRead(filepath):
        isfile = os.path.exists(filepath)
        readstr = ""
        if isfile:
            with open(filepath,"r",encoding="utf-8") as f:
                readstr = f.read()
        else:
            return
        return readstr


    '''
    功能名称：创建目录
    '''
    @staticmethod
    def mkdir(dirpath):
        # 判断路径是否存在
        isExists=os.path.exists(dirpath)
        # 判断结果
        if not isExists:
            os.makedirs(dirpath)
            print(dirpath+' 创建成功')
        else:
            pass





'''
功能名称：全局变量
'''
class Global(object):
    try:
        all_FileNum = 0 # 处理文件数目总数
        debug = 0 # 错误信息标记
        error_file_list = [] # pdf处理中错误文件路径列表
        limit_file_list = [] # 限制文件列表
        limit_file_size = 0.1 * 1024 #默认文件大小限制0.1k
        feature_word = ['a','d','j','l','m','n','nr','ns','nt','nz','v','vn','eng','w','sc']    # 停用词表
        stopwords ={}.fromkeys([line.strip() for line in open('../Document/StopWord/EN_stopwords.txt','r',encoding='utf-8')]) # 停用词表
        all_Strs = "" # 全局字符串


        '''单独提取各章节内容'''
        StrTitle = ""       # 论文题目
        StrAbstract = ""    # 论文摘要
        StrIntro = ""       # 论文引言
        StrConThe = ""      # 概念原理
        StrExpMeth = ""     # 实验方法
        StrResDis = ""      # 结果讨论
        StrConclude = ""    # 论文结论
        StrAcknow = ""      # 论文感谢
        StrRef = ""         # 参考文献


    except Exception as e:
        raise e
    finally:
        pass

'''
功能名称：测试方法，打印目录文件
'''
def TestMethod(path,newpath=''):
    if os.path.isfile(newpath):
        print("this is file name:"+newpath)
    if os.path.isdir(newpath):
        print("this is dir name:"+newpath)
    else:
        pass


if __name__ == '__main__':
    t1=time.time()
    # 根目录文件路径
    # http://www.cnblogs.com/baiboy/
    rootDir = r"E:\test"
    # 批量处理格式文件,1个必选参数目录路径rootDir，1个可选方法参数deffun
    tra=TraversalFun(rootDir,TestMethod) # 默认方法参数打印所有文件路径
    # 1个可选参数，定义导出目录的头标志，默认：Totxt-。即Totxt-原目录名称
    tra.TraversalDir()

    t2=time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n")
    input()
