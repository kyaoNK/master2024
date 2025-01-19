from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def calculate_text_similarities(texts):
    # TF-IDFベクトル化器の初期化
    vectorizer = TfidfVectorizer(
        max_features=5000,  # 使用する特徴量（単語）の最大数
        min_df=2,          # 最小文書頻度（これより少ない出現回数の単語は無視）
        max_df=0.95,       # 最大文書頻度（これより多い出現回数の単語は無視）
        token_pattern=r'(?u)\b\w+\b'  # 単語のトークン化パターン
    )
    
    # テキストをTF-IDFベクトルに変換
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # コサイン類似度の計算
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # 結果をDataFrameに変換（見やすくするため）
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[f"テキスト{i+1}" for i in range(len(texts))],
        columns=[f"テキスト{i+1}" for i in range(len(texts))]
    )
    
    return similarity_matrix, similarity_df