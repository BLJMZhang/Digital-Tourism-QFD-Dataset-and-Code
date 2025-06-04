import jieba


# 加载停止词
def load_stopwords(files):
    stopwords = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            stopwords.update([line.strip() for line in f])
    return stopwords


# 预处理函数
def preprocess(comments):
    stopwords_files = [
        r'D:\PyCharm\Py_Projects\stopwords\baidu_stopwords.txt',
        r'D:\PyCharm\Py_Projects\stopwords\cn_stopwords.txt',
        r'D:\PyCharm\Py_Projects\stopwords\hit_stopwords.txt',
        r'D:\PyCharm\Py_Projects\stopwords\scu_stopwords.txt'
    ]

    # 加载所有停止词
    stopwords = load_stopwords(stopwords_files)

    # 预处理：分词并去除停止词
    processed_texts = []
    for comment in comments:
        # 使用jieba分词
        words = jieba.cut(comment)
        # 去掉停止词并只保留长度大于1的词
        words = [word for word in words if word not in stopwords and len(word) > 1]
        processed_texts.append(words)

    return processed_texts
