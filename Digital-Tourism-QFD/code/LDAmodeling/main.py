from data_loader import load_data
from preprocess import preprocess
from gensim import corpora
from lda_model import create_lda_model
from save_results import save_topics
from grid_search import find_optimal_num_topics

def main():
    # 步骤1: 加载数据
    file_path = r'D:\博士\论文\论文1数字化需求分析\原始数据\海洋博物馆\评论.xlsx'
    comments = load_data(file_path)

    # 步骤2: 预处理
    texts = preprocess(comments)
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # 步骤3: 网格搜索确定最佳主题数量
    start, limit, step = 5, 20, 1  # 从5个主题到20个主题，每次增加1个主题
    results = find_optimal_num_topics(dictionary, corpus, texts, start, limit, step)

    # 根据一致性得分和困惑度选择最佳主题数量
    best_num_topics = max(results, key=lambda x: (x[1], -x[2]))[0]  # 优先选择一致性得分高和困惑度低的主题数量
    print(f"最佳主题数量为: {best_num_topics}")

    # 步骤4: 应用最佳主题数量的LDA模型
    lda_model, dictionary, corpus = create_lda_model(texts, best_num_topics)

    # 步骤5: 输出并保存结果
    output_file = 'lda_topics.txt'
    save_topics(lda_model, num_words=10, output_file=output_file)

if __name__ == '__main__':
    main()
