#!/usr/bin/env python
# coding:utf-8

# from win32com import client as wc
import os,fnmatch,time,sys,shutil,re
import jieba,itertools
import jieba.posseg as pseg
from decimal import Decimal


'''
功能描述：遍历目录，对子文件单独处理
参数描述：
        1 rootdir：待处理的目录路径
        2 deffun： 方法参数，默认为空
        3 savepath: 保存路径
'''
class TraversalFun():

    def __init__(self,rootdir,deffun=None,savedir=""):
        self.rootdir = rootdir # 目录路径
        self.deffun = deffun   # 参数方法
        self.savedir = savedir # 保存路径


    ''' 遍历目录文件'''
    def TraversalDir(self,defpar='newpath'):
        try:
            # 支持默认和自定义保存目录
            newdir = TraversalFun.creat_savepath(self,defpar)
            # 递归遍历word文件并将其转化txt文件
            TraversalFun.AllFiles(self,self.rootdir,newdir)
        except Exception as e:
            raise e


    '''支持默认和自定义保存目录'''
    # @staticmethod
    def creat_savepath(self,defpar):
        # 文件路径切分为上级路径和文件名('F:\\kjxm\\kjt', '1.txt')
        prapath,filename = os.path.split(self.rootdir)
        newdir = ""
        if self.savedir=="":
            newdir = os.path.abspath(os.path.join(prapath,filename+"_"+defpar))
        else:
            newdir = self.savedir
        print("保存目录路径：\n"+newdir)
        if not os.path.exists(newdir):
            os.mkdir(newdir)
        return newdir


    '''递归遍历所有文件，并提供具体文件操作功能。'''
    def AllFiles(self,rootdir,newdir=''):
        # 返回指定目录包含的文件或文件夹的名字的列表
        for lists in os.listdir(rootdir):
            # 待处理文件夹名字集合
            path = os.path.join(rootdir, lists)
            # 核心算法，对文件具体操作
            if os.path.isfile(path):
                self.deffun(path,newdir) # 具体方法实现功能
                # TraversalFun.filelogs(rootdir)  # 日志文件
            # 递归遍历文件目录
            if os.path.isdir(path):
                newpath = os.path.join(newdir, lists)
                if not os.path.exists(newpath):
                    os.mkdir(newpath)
                TraversalFun.AllFiles(self,path,newpath)
                # yield path,newpath # 迭代


    ''' 通过指定关键字操作，检查文件类型并转化目标类型'''
    def TranType(filename,typename):
        # print("本方法支持文件类型处理格式：pdf2txt，代表pdf转化为txt；word2txt，代表word转化txt；word2pdf，代表word转化pdf。")
        # 新的文件名称
        new_name = ""
        if typename == "pdf2txt" :
            #如果不是pdf文件，或者是pdf临时文件退出
            if not fnmatch.fnmatch(filename, '*.pdf') or not fnmatch.fnmatch(filename, '*.PDF') or fnmatch.fnmatch(filename, '~$*'):
                return
            # 如果是pdf文件，修改文件名
            if fnmatch.fnmatch(filename, '*.pdf') or fnmatch.fnmatch(filename, '*.PDF'):
                new_name = filename[:-4]+'.txt' # 截取".pdf"之前的文件名
        if typename == "word2txt" :
            #如果是word文件：
            if fnmatch.fnmatch(filename, '*.doc') :
                new_name = filename[:-4]+'.txt'
                print(new_name)
            if fnmatch.fnmatch(filename, '*.docx'):
                new_name = filename[:-5]+'.txt'
            # 如果不是word文件，或者是word临时文件退出
            else:
                return
        if typename == "word2pdf" :
            #如果是word文件：
            if fnmatch.fnmatch(filename, '*.doc'):
                new_name = filename[:-4]+'.pdf'
            if fnmatch.fnmatch(filename, '*.docx'):
                new_name = filename[:-5]+'.pdf'
            #如果不是word文件：继续
            else:
                return
        return new_name


    '''记录文件处理日志'''
    def filelogs(rootdir):
        prapath,filename = os.path.split(rootdir)
        # 创建日志目录
        dirpath = prapath+r"/"+filename+"_logs"
        TraversalFun.mkdir(dirpath)
        # 错误文件路径
        errorpath = dirpath+r"/errorlogs.txt"
        # 限制文件路径
        limitpath = dirpath+r"/limitlogs.txt"
        # 错误文件日志写入
        TraversalFun.writeFile(errorpath,'\n'.join(Global.error_file_list))
        # # 限制文件日志写入
        TraversalFun.writeFile(limitpath,'\n'.join(Global.limit_file_list))


    '''清空目录文件'''
    def cleardir(dirpath):
        if not os.path.exists(dirpath):
            TraversalFun.mkdir(dirpath)
        else:
            shutil.rmtree(dirpath)
            TraversalFun.mkdir(dirpath)


    ''' 文件的写操作'''
    def writeFile(filepath,strs): #encoding="utf-8"
        with open(filepath,'w') as f:
            f.write(strs)


    ''' 文件的读操作'''
    def readFile(filepath):
        isfile = os.path.exists(filepath)
        readstr = ""
        if isfile:
            with open(filepath,"r") as f:
                readstr = f.read()
        else:
            return
        return readstr


    ''' 创建目录 '''
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
功能描述：提供全局变量类
作    者：白宁超
时    间：2017年10月24日15:07:38
'''
class Global(object):
    try:
        all_FileNum = 0      # 处理文件数目总数
        debug = 0            # 错误信息标记
        error_file_list = [] # pdf处理中错误文件路径列表
        limit_file_list = [] # 限制文件列表
        limit_file_size = 1 * 1024  # 设置默认文件限制1K
        all_Strs = ""        # 全局字符串
        all_lists = []       # 全局列表变量
        classflag = 1 # 文本分类的类标
    except Exception as e:
        raise e



'''-----------4. 特征词选取并转换成文本向量----------------'''
class CHSegWords():

    def __init__(self,filepath='',str_doc=''):
        self.filepath = filepath # 分词路径
        self.str_doc = str_doc

    # 创建停用词列表
    def createstoplist():
        # 停用词列表
        stwlist =set([line.strip() for line in open(os.path.abspath(r'./datasource/NLPIR_stopwords.txt'),'r',encoding='utf-8').readlines()])
        # stopwords ={}.fromkeys([line.strip() for line in open(os.path.abspath(r'./datasource/NLPIR_stopwords.txt'),'r',encoding='utf-8')]) # 停用词字典
        return stwlist


    # 替换特殊字符，如\u3000
    def rm_char(text):
        text = re.sub('\u3000', '', text)
        return text


    # 去掉一些停用词、数字、特殊符号
    def rm_tokens(words, stwlist):
        words_list = list(words)
        for i in range(words_list.__len__())[::-1]:
            word = words_list[i]
            if word in stwlist:  # 去除停用词
                words_list.pop(i)
            elif word.isdigit():  # 去除数字
                words_list.pop(i)
            elif len(word) == 1:  # 去除单个字符
                words_list.pop(i)
            elif word == " ":  # 去除空字符
                words_list.pop(i)
        return words_list


    # 利用jieba对文本进行分词，返回切词后的list
    def seg_doc(self):
        if self.str_doc!='': # 处理自定义字符串
            sent_list = self.str_doc.split('\n')  # 读取文件按行拆分
        else: # 处理文本文件
            # sent_list = TraversalFun.readFile(self.filepath).split('\n')  # 读取文件按行拆分
            sent_list = TraversalFun.readFile(self.filepath).split('\n')
        sent_list = map(CHSegWords.rm_char, sent_list)  # 去掉一些字符，例如\u3000
        stwlist = CHSegWords.createstoplist()
        word_2dlist = [CHSegWords.rm_tokens(jieba.cut(part, cut_all=False), stwlist) for part in sent_list]  # 分词并去停用词
        word_list = list(itertools.chain(*word_2dlist))
        # word_list_str = ",".join(word_list)
        # print("word_list_str:", word_list_str)
        return word_list


    # 利用jieba对文本进行分词，返回切词后的list
    def seg_pseg_doc(self):
        if self.str_doc!='': # 处理自定义字符串
            sent_list = self.str_doc.split('\n')  # 读取文件按行拆分
        else: # 处理文本文件
            sent_list = TraversalFun.readFile(self.filepath)# 读取文件按行拆分

        stwlist = CHSegWords.createstoplist()
        word_2dlist= pseg.cut(sent_list) # 分词
        word_list = []
        for key in word_2dlist:
            if key.word not in stwlist and len(key.word)!=1:# 停用词处理
                # dictwords.append(key.word+"/"+key.flag)
                word_list.append(key.word)

        return word_list



'''-----------1. 高效的读取文本文件-------------------'''
# 用生成器读取多个文件夹
class GeneratorReadFolders(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):  # 迭代器
        for file in os.listdir(self.par_path):
            file_abspath = os.path.join(self.par_path, file)
            if os.path.isdir(file_abspath):  # if file is a folder
                yield file_abspath  # use generator



# 用生成器读取多个文件
class GeneratorReadFiles(object):
    def __init__(self, par_path):
        self.par_path = par_path

    def __iter__(self):  # 迭代器
        folders = GeneratorReadFolders(self.par_path)
        for folder in folders:              # level directory
            # print("folder:", folder)
            catg = folder.split(os.sep)[-1]
            for file in os.listdir(folder):     # secondary directory
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    this_file = open(file_path, 'rb')  # 以rb读取方式文件 更快
                    content = this_file.read()
                    yield catg, file, content  # use generator
                    this_file.close()



'''-----------2. 处理文本的HTML标签、特殊符号（如微博文本）----------------'''

# 过滤HTML中的标签
# @param htmlstr HTML字符串.
def filter_tags(htmlstr):
    # 把script标签中的内容全部清除 added by candymoon
    rex = r'<script .*?>.*?</script>'
    dr = re.compile(rex, re.S)
    htmlstr = dr.sub('', htmlstr)
    # 先过滤CDATA
    re_cdata = re.compile('//<!CDATA\[[ >]∗ //\] > ', re.I)  # 匹配CDATA
    re_script = re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>', re.I)
    # Script
    re_style = re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>', re.I)
    # style
    re_br = re.compile('<br\s*?/?>')
    # 处理换行
    re_h = re.compile('</?\w+[^>]*>')
    # HTML标签
    re_comment = re.compile('<!--[^>]*-->')
    # HTML注释
    s = re_cdata.sub('', htmlstr)
    # 去掉CDATA
    s = re_script.sub('', s)  # 去掉SCRIPT
    s = re_style.sub('', s)
    # 去掉style
    s = re_br.sub('', s)
    # 将br转换为换行
    s = re_h.sub('', s)  # 去掉HTML 标签
    s = re_comment.sub('', s)
    # 去掉HTML注释
    re_comment1 = re.compile('<!--[^.*?]-->')
    s = re_comment1.sub('', s)
    # 去掉多余的空行
    blank_line = re.compile('\n+')
    s = blank_line.sub('', s)

    blank_line_l = re.compile('\n')
    s = blank_line_l.sub('', s)

    blank_kon = re.compile('\t')
    s = blank_kon.sub('', s)

    blank_one = re.compile('\r\n')
    s = blank_one.sub('', s)

    blank_two = re.compile('\r')
    s = blank_two.sub('', s)

    blank_three = re.compile(' ')
    s = blank_three.sub('', s)

    http_link = re.compile(r'(http://.+.html)')
    s = http_link.sub('', s)

    s = replaceCharEntity(s)  # 替换实体
    return s

# 替换常用HTML字符实体.
# 使用正常的字符替换HTML中特殊的字符实体.
# 你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
# @param htmlstr HTML字符串.
def replaceCharEntity(htmlstr):
    CHAR_ENTITIES = {'nbsp': ' ', '160': ' ',
                     'lt': '<', '60': '<',
                     'gt': '>', '62': '>',
                     'amp': '&', '38': '&',
                     'quot': '"''"', '34': '"', }

    re_charEntity = re.compile(r'&#?(?P<name>\w+);')
    sz = re_charEntity.search(htmlstr)
    while sz:
        entity = sz.group()  # entity全称，如>
        key = sz.group('name')  # 去除&;后entity,如>为gt
        try:
            htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
        except KeyError:
            # 以空串代替
            htmlstr = re_charEntity.sub('', htmlstr, 1)
            sz = re_charEntity.search(htmlstr)
    return htmlstr


# 清洗HTML标签文本
def extract(html_content, min_size = 10):
    try:
        html_content = ' '.join(html_content.split()) # 去掉多余的空格
        html_content = filter_tags(html_content)
        html_content = replaceCharEntity(html_content)
        html_content = ' '.join(html_content.split())  # 去掉多余的空格
        # print(len(html_content), html_content)
        zhPattern = re.compile(u'[\u4e00-\u9fa5]+')  # 判断是否包含中文
        match = zhPattern.search(html_content)
        if not match:
            return None
        if len(html_content) < min_size:
            return None
        return html_content
    except Exception as e:
        print(e)
        return None

#  微博数据清洗
def weibo_clear(weibo):
    rex = r'回复@\S+:'  # 格式：回复//@...:
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'//@\S+:'  # 格式：//@...:
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'@\S+ ?'  # 格式：@...空格
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'@\S+:?'  # 格式：@...:空格
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'#\S+#'  # 格式：#...#
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'null'  # 格式：null
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'【(.*?)】'  # 格式：【\S+】
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'我参与了发起的投票'  # 格式：我参与了...发起的投票
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'我分享了的文章'  # 格式：我分享了...的文章
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'我投给了“”这个选项。你也快来表态吧~'  # 格式：我投给了“...”这个选项。你也快来表态吧~
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'我发起了一个投票.*?网页链接'  # 格式：我发起了一个投票...网页链接
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'发表了博文'  # 格式：发表了博文
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'发表了一篇转载博文'  # 格式：发表了一篇转载博文
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r' http://\S+'  # 格式：http://
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'http://.*? '  # 格式：http://
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'查看详情：'  # 格式：查看详情：
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'转发微博 '  # 格式：转发微博
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'分享视频 '  # 格式：分享视频
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r' 网页链接'  # 格式： 网页链接
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'（记者.*?）'  # 格式：（记者 ...）
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'（图源见水印）'  # 格式：（图源见水印）
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'...展开全文c'  # 格式：...展开全文c
    dr = re.compile(rex, re.S)
    weibo = dr.sub('', weibo)

    rex = r'\[.*?\]'  # 格式：[...]
    dr = re.compile(rex, re.S)
    weibo = dr.sub(' ', weibo)

    rex = r'\|.*\... .*?\...'  # 格式：|xxx... xxx...
    dr = re.compile(rex, re.S)
    weibo = dr.sub(' ', weibo)

    rex = r'\|.*?\...'  # 格式：|xxx...
    dr = re.compile(rex, re.S)
    weibo = dr.sub(' ', weibo)

    return weibo




def TestMethod(filepath,newpath):
    if os.path.isfile(filepath) :
        print("this is file name:"+filepath)
    else:
        pass


if __name__ == '__main__':
    t1=time.time()

    # 根目录文件路径
    # rootDir = os.path.abspath(r"./fudan")
    # tra=TraversalFun(rootDir,TestMethod) # 默认方法参数打印所有文件路径
    # tra.TraversalDir()                   # 遍历文件并进行相关操作

    '''2 测试分词'''
    filepaths = os.path.abspath(r"./fudan/C3-Art/C3-Art0002.txt")
    chseg = CHSegWords(filepaths)
    print(chseg.seg_doc())



    t2=time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n")
    input()