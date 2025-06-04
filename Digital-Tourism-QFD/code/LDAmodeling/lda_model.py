from gensim import corpora
from gensim.models.ldamodel import LdaModel

def create_lda_model(texts, num_topics):
    """
    创建LDA模型
    :param texts: 分词后的评论内容列表
    :param num_topics: 主题数量
    :return: 训练后的LDA模型, 字典, 语料库
    """
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    return lda_model, dictionary, corpus
