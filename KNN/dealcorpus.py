
import os,sys,time
import jieba,nltk,logging
import jieba.posseg as pseg
import visualplot
import numpy as np
from decimal import Decimal
from BaseClass import *  #自定义基础类库
from BaseClass import TraversalFun
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


'''jieba中文分词'''
def ch_seg_words(filepath,newpath):
    # 文件路径切分为上级路径和文件名
    prapath,filename = os.path.split(filepath)
    newpath = os.path.join(newpath,filename)
    # print ("切词后保留路径："+newpath)
    try:
        # 对文本进行切词处理并剔除停用词等脏数据
        word_list=CHSegWords(filepath).seg_doc() # 391,19.9s
        # word_list=CHSegWords(filepath).seg_pseg_doc() # 391,158s
        #保存这些内容
        # Global.all_Strs +=' '.join(word_list)+' '+str(os.path.split(prapath)[-1])+'\n'
        # word_list.append(os.path.split(prapath)[-1])
        Global.all_lists.append(word_list)
        TraversalFun.writeFile(newpath,str(' '.join(word_list)))
        Global.all_FileNum+=1
        return word_list
    except Exception as e:
        Global.error_file_list.append(filepath)
        print(e)
        return



# 利用nltk进行词频特征统计
def nltk_wf_feature(word_list=None):
    # word_list = ['我', '爱', '成都']    # word_list参数样例数据
    freq_dist = nltk.FreqDist(word_list)

    # print(freq_dist.values()) # 统计次数

    # freq_list = []
    # num_words = len(freq_dist.values())
    # for i in range(num_words):
    #     freq_list.append([list(freq_dist.keys())[i],list(freq_dist.values())[i]])
    # freqArr = np.array(freq_list).T
    # print(str(freq_list))

    # 打印统计的词频
    for key in freq_dist.keys():
        print(key, freq_dist.get(key))



# 利用sklearn计算tfidf值特征
def sklearn_tfidf_feature(corpus=None):
    # corpus参数样例数据如下：
    # corpus = ["我 来到 成都 春熙路",
    #           "今天 在 宽窄巷子 耍 了 一天 ",
    #           "成都 整体 来说 还是 挺 安逸 的",
    #           "成都 的 美食 真 巴适 惨 了"]
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for遍历某一类文本下的词语权重
        print(u"-------这里输出第", i, u"类文本的词语tf-idf权重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])


'''-----------5. 自定义规则提取特征词----------------'''
# 针对不同业务场景（评论情感判断），可以自定义特征抽取规则（完全freestyle，不一定非得用算法来提取，只要有效果就ok）
def extract_feature_words():
    text = '成都的美食真的太多了，美味又便宜，简直就是吃货的天堂啊！'
    user_pos_list = ['a', 'ad', 'an', 'n']  # 用户自定义特征词性列表
    for word, pos in pseg.cut(text):
        if pos in user_pos_list:
            print(word, pos)



# 利用gensim包进行特征词提取（推荐使用）
def gensim_feature(corpus=None):

    # corpus参数样例数据如下：
    # corpus = [["我", "来到", "成都", "春熙路"],
    #           ["今天", "在", "宽窄巷子", "耍", "了", "一天"],
    #           ["成都", "整体", "来说", "还是", "挺", "安逸", "的"],
    #           ["成都", "的", "美食", "真", "巴适", "惨", "了"]]
    # dictionary = corpora.Dictionary(corpus)  # 构建语料词典

    # # 收集停用词和仅出现一次的词的id
    # stop_ids = [dictionary.token2id[stopword] for stopword in user_stop_word_list if stopword in dictionary.token2id]
    # once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
    # dictionary.filter_tokens(stop_ids + once_ids) # 删除停用词和仅出现一次的词
    # dictionary.compactify()  # 消除id序列在删除词后产生的不连续的缺口
    # dictionary.save('mycorpus.dict')  # 把字典保存起来，方便以后使用

    # 统计词频特征
    # dfs = dictionary.dfs  # 词频词典
    # for key_id, c in dfs.items():
    #     print(dictionary[key_id], c)

    # 统计元组词频
    # token2id = dictionary.token2id
    # dfs = dictionary.dfs
    # token_info = {}
    # for word in token2id:
    #     token_info[word] = dict(
    #         word = word,
    #         id = token2id[word],
    #         freq = dfs[token2id[word]]
    #     )
    # token_items = token_info.values()
    # token_items = sorted(token_items, key = lambda x:x['id']) #根据id排序
    # print('The info of dictionary: ')
    # print(token_items)
    # print('--------------------------')

    # 转换成doc_bow
    # doc_bow_corpus = [dictionary.doc2bow(doc_cut) for doc_cut in corpus]

    # 生成tfidf特征
    # tfidf_model = models.TfidfModel(dictionary=dictionary)  # 生成tfidf模型
    # tfidf_corpus = [tfidf_model[doc_bow] for doc_bow in doc_bow_corpus]  # 将每doc_bow转换成对应的tfidf_doc向量
    # TraversalFun.writeFile(os.path.abspath(r"./R0.txt"),str(tfidf_corpus))
    # print(tfidf_corpus[0])
    # vec = [(0, 2), (4, 1)]
    # print(tfidf_model[vec])

    # 生成lsi特征（潜在语义索引）
    # lsi_model = models.LsiModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=5)  # 生成lsi model
    # # 生成corpus of lsi
    # lsi_corpus = [lsi_model[tfidf_doc] for tfidf_doc in tfidf_corpus]  # 转换成lsi向量

    # 生成lda特征(主题模型)
    # lda_model = models.LdaModel(corpus=tfidf_corpus, id2word=dictionary, num_topics=100)  # 生成lda model
    # # 生成corpus of lsi
    # lda_corpus = [lda_model[tfidf_doc] for tfidf_doc in tfidf_corpus]  # 转换成lda向量

    # 图形化展示
    # visualplot.Show2dCorpora(lda_corpus)

    # 生成随机映射（Random Projections，RP, 优点：减小空维度、CPU和内存都很友好）
    # rp_model = models.RpModel(tfidf_corpus, num_topics=4)
    # rp_corpus = [rp_model[tfidf_doc] for tfidf_doc in tfidf_corpus]  # 转换成随机映射tfidf向量

    # 分层狄利克雷过程（Hierarchical Dirichlet Process，HDP ,一种无参数贝叶斯方法）
    # hdp_model = models.HdpModel(doc_bow_corpus, id2word=dictionary)
    # hdp_corpus = [hdp_model[doc_bow] for doc_bow in doc_bow_corpus]  # 转换成HDP向量

    # 文档向量和词向量 (Doc2Vec and Word2Vec)
    tld_list = []
    for ind, line_list in enumerate(corpus):
        tld_list.append(TaggedDocument(line_list, tags=[str(ind)]))

    # size: 词向量特征的大小
    # window： 词向量训练时参考的最大词长度。类似n-gram
    # min_count: 允许最小词的个数。词频低于5的词，不计算词向量
    # workers： 采用并行的个数
    # d2v_model = Doc2Vec(tld_list, min_count=5, window=3, size=1000,workers=4, sample=1e-3, negative=5,iter=5)
    d2v_model = Doc2Vec(tld_list, min_count=10, window=3, vector_size=1000,sample=1e-3, workers=4,epochs=5)
    # 由于Doc2vec的训练过程也可以同时训练Word2vec，所以可以直接获取两个模型，全部保存起来：
    # model.save(save_model_d2v_file_path)
    # d2v_model.save_word2vec_format(os.path.abspath(r"./R1.txt"), binary=True)
    # model.save_word2vec_format(save_model_w2v_file_path, binary=True)

    # 将文本转换成向量矩阵
    arr_list = []
    docvecs = d2v_model.docvecs
    for num in range(len(tld_list)):
        arr_list.append(docvecs[num].tolist())
    corpus =  TraversalFun.writeFile(os.path.abspath(r"./R0.txt"),str(arr_list))
    return arr_list
    # docvecs_matrix = np.asarray(docvecs)
    # print(docvecs_matrix.shape)






if __name__ == '__main__':
    t1=time.time()

    '''1 中文分词单文件测试'''
    # filepaths = os.path.abspath(r"./fudan/C3-Art/C3-Art0002.txt")
    # word_list=ch_seg_words(filepaths,os.path.abspath(r"./"))
    # TraversalFun.writeFile(os.path.abspath(r"./allwords.txt"),str(Global.all_lists)) # 写入处理后的数据

    '''2 中文分词批量文件测试'''
    rootDir = os.path.abspath(r"./fudan1/")
    tra=TraversalFun(rootDir,ch_seg_words) # 默认方法参数打印所有文件路径
    tra.TraversalDir()                   # 遍历文件并进行相关操作
    TraversalFun.writeFile(os.path.abspath(r"./allwords.txt"),str(Global.all_lists))


    '''3 词频特征统计'''
    # nltk_wf_feature(word_list)

    '''4 计算tfidf值特征'''
    # corpus = [' '.join(word_list),"测试 结果 你猜"]
    # sklearn_tfidf_feature(corpus)

    '''5 利用gensim包进行特征词提取'''
    # corpus =  TraversalFun.readFile(os.path.abspath(r"./allwords.txt"))
    # corpuslist = corpus.split('],')
    # gensim_feature(corpuslist)

    '''自定义规则提取特征词'''
    # extract_feature_words()

    t2=time.time()
    totalTime=Decimal(str(t2-t1)).quantize(Decimal('0.0000'))
    print("耗时："+str(totalTime)+" s"+"\n处理文件："+str(Global.all_FileNum)+"个。")
