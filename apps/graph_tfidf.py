from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import MeCab as mecab
import ipadic
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import itertools
from concurrent.futures import ProcessPoolExecutor

class CustomTokenizer:
    def __init__(self, terms_dict=None):
        self.terms_dict = {}
        if terms_dict:
            for term_group in terms_dict:
                variants = term_group.split('|')
                variants = sorted(variants, key=len, reverse=True)
                for variant in variants:
                    self.terms_dict[variant] = variants[0]
        self.tagger = mecab.Tagger(ipadic.MECAB_ARGS)
    
    def __call__(self, text):
        processed_text = text
        for term in sorted(self.terms_dict.keys(), key=len, reverse=True):
            if term in processed_text:
                processed_text = processed_text.replace(term, f" {self.terms_dict[term]} ")
        node = self.tagger.parseToNode(processed_text)
        tokens = []
        while node:
            surface = node.surface
            if surface in self.terms_dict:
                tokens.append(self.terms_dict[surface])
            elif node.feature.split(',')[0] in ['名詞', '動詞', '形容詞']:
                tokens.append(surface)
            node = node.next
        return tokens

def filter_edges_by_weight(G, threshold):
    G_filtered = G.copy()
    edges_to_remove = [(u, v) for u, v, data in G_filtered.edges(data=True) if data['weight'] <= threshold]
    G_filtered.remove_edges_from(edges_to_remove)
    return G_filtered

def graph_ave_tfidf(materials, technical_terms):
    G = nx.Graph()
    
    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))
        
    nodes = list(materials.keys())
    
    vectorizer = JapaneseTfidfVectorizer(technical_terms)
    
    total = len(nodes)
    with tqdm(total=total * (total-1) // 2) as pbar:
        for i, u in enumerate(nodes):
            for w in nodes[i+1:]:
                text1 = materials.get(u).get("text")
                text2 = materials.get(w).get("text")
                tfidf_matrix = vectorizer.fit_transform([text1, text2])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                G.add_edge(u, w, weight=similarity)
                pbar.update(1)

    print("\n最初に構築したグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")

    # 孤立ノードの除去
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    print(f"\n孤立したノード: {isolated_nodes}")
    print("\n削除後のグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")
        
    # 重みの統計情報
    if G.number_of_edges() > 0:
        total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
        average_weight = total_weight / G.number_of_edges()
        print(f"平均重み: {average_weight}")
    else:
        average_weight = 0.0
        
    # 重みの分布をプロット
    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    plt.figure(figsize=(10,6))
    plt.hist(weights, bins=100, edgecolor='black')
    plt.xlabel('重み')
    plt.ylabel('頻度')
    plt.title('エッジの重み分布')
    plt.savefig("out/distribution_of_weights.png")
    plt.close()
        
    # 平均重みで枝刈り
    threshold = average_weight
    G_ave = filter_edges_by_weight(G, threshold)
    isolated_nodes = list(nx.isolates(G_ave))
    G_ave.remove_nodes_from(isolated_nodes)
    
    print(f"\n削除したノード: {isolated_nodes}")
    print("\n平均以下削除後のグラフの基本情報:")
    print(f"ノード数: {G_ave.number_of_nodes()}")
    print(f"エッジ数: {G_ave.number_of_edges()}")

    # グラフの可視化
    plt.figure(figsize=(10,6))
    nx.draw(G_ave, with_labels=True) 
    plt.savefig("out/tfidf_ave_graph.png")
    nx.write_gml(G_ave, "out/tfidf_ave_graph.gml")
    plt.close()
    
    return G_ave

def test_graph_ave_tfidf():
    # テスト用のデータ作成
    test_materials = {
        "doc1": {"text": "人工知能 機械学習 データ分析"},
        "doc2": {"text": "機械学習 深層学習 ニューラルネットワーク"},
        "doc3": {"text": "データベース SQL 機械学習"},
        "doc4": {"text": "自然言語処理 形態素解析"}
    }
    
    # テスト用の専門用語リスト
    test_technical_terms = [
        "人工知能",
        "機械学習",
        "データ分析",
        "深層学習",
        "ニューラルネットワーク",
        "データベース",
        "SQL",
        "自然言語処理",
        "形態素解析"
    ]
    
    try:
        # グラフ作成
        G = graph_ave_tfidf(test_materials, test_technical_terms)
        
        # 基本的なチェック
        assert isinstance(G, nx.Graph), "返り値がnx.Graphではありません"
        assert G.number_of_nodes() > 0, "ノードが存在しません"
        assert G.number_of_edges() > 0, "エッジが存在しません"
        
        # エッジの重みのチェック
        for u, v, data in G.edges(data=True):
            assert 'weight' in data, "エッジに重みが設定されていません"
            assert 0 <= data['weight'] <= 1, "重みが0-1の範囲外です"
            
        # 孤立ノードがないことを確認
        assert len(list(nx.isolates(G))) == 0, "孤立ノードが存在します"
        
        print("全てのテストが通過しました！")
        
        # グラフの情報を表示
        print("\nグラフ情報:")
        print(f"ノード数: {G.number_of_nodes()}")
        print(f"エッジ数: {G.number_of_edges()}")
        print("\nエッジの重み:")
        for u, v, data in G.edges(data=True):
            print(f"{u} - {v}: {data['weight']:.3f}")
            
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    # test_graph_ave_tfidf()
    documents = [
        "ETCシステムとICカードによる料金収受について説明します。",  # 文書1
        "自動料金収受システムの導入により、支払いが簡単になりました。",  # 文書2
        "ICカードと集積回路カードの違いについて解説します。"  # 文書3
    ]

    # 専門用語の定義
    specialized_terms = [
        'ETC|ETCシステム|自動料金収受システム',
        'IC|ICカード|集積回路カード'
    ]

    # トークナイザーとベクトライザーの設定
    tokenizer = CustomTokenizer(terms_dict=specialized_terms)
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        min_df=1,  # 1回以上出現する単語を対象
        max_df=0.95  # 95%以上の文書に出現する単語は除外
    )

    # 文書のベクトル化
    X = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    tf_idf_array = X.toarray()

    print("\n各文書のTF-IDF値:")
    for doc_idx, doc in enumerate(documents):
        print(f"\n文書{doc_idx + 1}: {doc}")
        for term_idx, term in enumerate(feature_names):
            if tf_idf_array[doc_idx][term_idx] > 0:
                print(f"  {term}: {tf_idf_array[doc_idx][term_idx]:.4f}")
                
    # デバッグのためのコードを追加
    print("\n専門用語辞書:")
    print(tokenizer.terms_dict)

    print("\nトークン化の結果:")
    for doc in documents:
        print(f"\n文書: {doc}")
        print(f"トークン: {tokenizer(doc)}")

    print("\n特徴量名:")