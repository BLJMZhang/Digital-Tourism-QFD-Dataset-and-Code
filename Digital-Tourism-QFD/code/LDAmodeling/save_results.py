def save_topics(lda_model, num_words, output_file):
    """
    将LDA模型的主题结果保存到文本文件中
    :param lda_model: LDA模型
    :param num_words: 每个主题的关键词数量
    :param output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, topic in lda_model.print_topics(num_words=num_words):
            f.write(f"主题 {idx+1}: {topic}\n")
