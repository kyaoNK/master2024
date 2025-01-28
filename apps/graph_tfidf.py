import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import MeCab
import itertools

def tokenize_japanese(text):
    """
    IPAdicを使用した形態素解析により、ストップワードを除去して単語のリストを返す
    """
    import ipadic
    tagger = MeCab.Tagger(ipadic.MECAB_ARGS)
    
    # ストップワードとして扱う品詞（IPA辞書の品詞体系に基づく）
    stop_pos = {'助詞', '助動詞', '記号', '接続詞', '感動詞', '特殊', '非自立', '接頭詞', '副詞化', '形容詞化'}
    
    nodes = []
    node = tagger.parseToNode(text)
    
    while node:
        # 品詞情報の取得
        pos_info = node.feature.split(',')
        pos = pos_info[0]  # 品詞の大分類
        pos_detail = pos_info[1] if len(pos_info) > 1 else ''  # 品詞の詳細分類
        
        # ストップワードでない場合のみ追加
        # 表層形が意味のある文字列で、かつ品詞が対象外でない場合
        if (pos not in stop_pos and 
            pos_detail not in stop_pos and 
            node.surface.strip()):
            nodes.append(node.surface)
        node = node.next
        
    return nodes

class JapaneseTfidfVectorizer(TfidfVectorizer):
    """
    日本語対応のTF-IDFベクトライザー
    """
    def __init__(
        self,
        max_features=None,
        min_df=1,
        max_df=1.0,
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    ):
        super().__init__(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            norm=norm,
            use_idf=use_idf,
            smooth_idf=smooth_idf,
            sublinear_tf=sublinear_tf,
            tokenizer=tokenize_japanese,
            token_pattern=None  # トークナイザーを使用するため無効化
        )

def compute_tfidf_vectors(materials):
    """
    文書群からTF-IDFベクトルを計算する
    """
    documents = [data.get("text", "") for data in materials.values()]
    vectorizer = JapaneseTfidfVectorizer(
        max_features=5000,
        min_df=2,  # 最低文書頻度
        max_df=0.95  # 最大文書頻度（割合）
    )
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # 重要な単語の表示（デバッグ用）
    feature_names = vectorizer.get_feature_names_out()
    print("\n最も重要な特徴語（上位20件）:")
    tfidf_sum = np.array(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = tfidf_sum.argsort()[-20:][::-1]
    for idx in top_indices:
        print(f"{feature_names[idx]}: {tfidf_sum[idx]:.4f}")
    
    return tfidf_matrix, vectorizer

def cosine_similarity_from_tfidf(vec1, vec2):
    """
    2つのTF-IDFベクトル間のコサイン類似度を計算する
    """
    similarity = cosine_similarity(vec1, vec2)
    return float(similarity[0, 0])

def compute_edge_tfidf(args):
    """
    2つのノード間のエッジの重みを計算する
    """
    u, w, vec1, vec2 = args
    weight = cosine_similarity_from_tfidf(vec1, vec2)
    return (u, w, weight)

def filter_edges_by_weight(G, threshold):
    """
    指定した閾値以上の重みを持つエッジのみを残す
    """
    filtered_G = nx.Graph()
    for node, attr in G.nodes(data=True):
        filtered_G.add_node(node, **attr)
        
    for u, v, data in G.edges(data=True):
        if data['weight'] >= threshold:
            filtered_G.add_edge(u, v, **data)

    return filtered_G

def filter_edges_by_percentile(G, lower_percentile=5, upper_percentile=95):
    """
    指定したパーセンタイル範囲内の重みを持つエッジのみを残す
    """
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    lower_bound = np.percentile(weights, lower_percentile)
    upper_bound = np.percentile(weights, upper_percentile)
    
    filtered_G = nx.Graph()
    
    for node, data in G.nodes(data=True):
        filtered_G.add_node(node, **data)
        
    for u, v, data in G.edges(data=True):
        if lower_bound <= data['weight'] <= upper_bound:
            filtered_G.add_edge(u, v, **data)
    
    return filtered_G

def graph_ave_tfidf_parallel(materials):
    """
    TF-IDFベクトルとコサイン類似度を使用してグラフを構築する
    """
    G = nx.Graph()

    # ノードの追加
    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))

    # TF-IDFベクトルの計算
    tfidf_matrix, vectorizer = compute_tfidf_vectors(materials)
    
    nodes = list(materials.keys())
    edge_args = []
    
    # エッジの引数を準備
    for i, u in enumerate(nodes):
        for w in nodes[i+1:]:
            vec1 = tfidf_matrix[list(materials.keys()).index(u)]
            vec2 = tfidf_matrix[list(materials.keys()).index(w)]
            edge_args.append((u, w, vec1, vec2))

    # 並列処理でエッジの重みを計算
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_edge_tfidf, edge_args), total=len(edge_args)):
            if result:
                u, w, weight = result
                G.add_edge(u, w, weight=weight)

    print("\n最初に構築したグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")

    # 孤立ノードの削除
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    print(f"\n孤立したノード: {isolated_nodes}")
    print("\n削除後のグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")

    # 重みの分布を表示
    if G.number_of_edges() > 0:
        total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
        average_weight = total_weight / G.number_of_edges()
        print(f"平均重み: {average_weight}")
    else:
        average_weight = 0.0

    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    plt.figure(figsize=(10,6))
    plt.hist(weights, bins=100, edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.title('Distribution of Weights')
    plt.savefig("out/tfidf_edge_weights_distribution.png", format="png")
    plt.close()

    # 平均以下の重みを持つエッジを削除
    threshold = average_weight
    G_ave = filter_edges_by_weight(G, threshold)
    isolated_nodes = list(nx.isolates(G_ave))
    G_ave.remove_nodes_from(isolated_nodes)

    print(f"\n削除したノード: {isolated_nodes}")
    print("\n平均以下削除後のグラフの基本情報:")
    print(f"ノード数: {G_ave.number_of_nodes()}")
    print(f"エッジ数: {G_ave.number_of_edges()}")

    plt.figure(figsize=(10,6))
    nx.draw(G_ave, with_labels=True)
    plt.savefig("out/tfidf_ave_graph.png")
    nx.write_gml(G_ave, "out/tfidf_ave_graph.gml")
    plt.close()

    return G_ave

def graph_remove_outliers_tfidf_parallel(materials):
    """
    外れ値を除去したTF-IDFベースのグラフを構築する
    """
    G = nx.Graph()
    
    # ノードの追加
    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))

    # TF-IDFベクトルの計算
    tfidf_matrix, vectorizer = compute_tfidf_vectors(materials)
    
    nodes = list(materials.keys())
    edge_args = []
    
    # エッジの引数を準備
    for i, u in enumerate(nodes):
        for w in nodes[i+1:]:
            vec1 = tfidf_matrix[list(materials.keys()).index(u)]
            vec2 = tfidf_matrix[list(materials.keys()).index(w)]
            edge_args.append((u, w, vec1, vec2))

    # 並列処理でエッジの重みを計算
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_edge_tfidf, edge_args), total=len(edge_args)):
            if result:
                u, w, weight = result
                G.add_edge(u, w, weight=weight)

    print("\n最初に構築したグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    print(f"エッジの重みの範囲: {min(weights):.3f} - {max(weights):.3f}")

    # パーセンタイルによるフィルタリング
    G_filtered = filter_edges_by_percentile(G)
    filtered_weights = [data['weight'] for u, v, data in G_filtered.edges(data=True)]
    print(f"フィルタリング後の重みの範囲: {min(filtered_weights):.3f} - {max(filtered_weights):.3f}")

    isolated_nodes = list(nx.isolates(G_filtered))
    G_filtered.remove_nodes_from(isolated_nodes)

    plt.figure(figsize=(10,6))
    nx.draw(G_filtered, with_labels=True)
    plt.savefig("out/tfidf_remove_outliers_graph.png")
    nx.write_gml(G_filtered, "out/tfidf_remove_outliers_graph.gml")
    
    plt.close()

    return G_filtered