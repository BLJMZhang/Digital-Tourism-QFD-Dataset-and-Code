import pandas as pd

def load_data(file_path):
    """
    加载Excel文件并提取“评论”列
    :param file_path: Excel文件的路径
    :return: 评论内容的列表
    """
    df = pd.read_excel(file_path)
    return df['评论'].dropna().tolist()
