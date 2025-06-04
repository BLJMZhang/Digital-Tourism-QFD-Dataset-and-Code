from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel


def find_optimal_num_topics(dictionary, corpus, texts, start, limit, step):
    """
    通过网格搜索找到最佳的主题数量，计算每个模型的一致性得分和困惑度
    :param dictionary: 词典
    :param corpus: 语料库
    :param texts: 分词后的评论内容列表
    :param start: 起始主题数量
    :param limit: 最大主题数量
    :param step: 主题数量的步长
    :return: 各主题数量下的一致性得分和困惑度的列表
    """
    results = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

        # 计算一致性得分
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()

        # 计算困惑度
        perplexity = model.log_perplexity(corpus)

        results.append((num_topics, coherence_score, perplexity))
        print(f"主题数量: {num_topics}, 一致性得分: {coherence_score}, 困惑度: {perplexity}")

    return results
