import matplotlib.pyplot as plt
from openai import OpenAI
import networkx as nx
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os
import json
from typing import List, Dict, Tuple
import math

CACHE_PATH = 'embedding_cache.json'
MAX_TOKENS_PER_BATCH = 8000  # OpenAIの制限に基づく
TOKENS_PER_CHAR = 0.4  # 日本語テキストの概算トークン数（文字数×0.4）

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)

embedding_cache = load_cache()

def estimate_tokens(text: str) -> int:
    """テキストのトークン数を概算する"""
    return math.ceil(len(text) * TOKENS_PER_CHAR)

def create_optimal_batches(texts: List[str]) -> List[List[str]]:
    print("\nバッチ作成の詳細:")
    print(f"処理するテキスト総数: {len(texts)}")
    texts_with_tokens = [(text, estimate_tokens(text)) for text in texts]
    texts_with_tokens.sort(key=lambda x: x[1], reverse=True)  # 長い順にソート
    
    batches = []
    current_batch = []
    current_tokens = 0
    
    for text, tokens in texts_with_tokens:
        if tokens > MAX_TOKENS_PER_BATCH:
            # 単体で制限を超える場合は個別に処理
            print(f"警告: テキスト（長さ: {len(text)}文字, 推定トークン数: {tokens}）が制限を超えるため個別処理")
            batches.append([text])
            continue
            
        if current_tokens + tokens <= MAX_TOKENS_PER_BATCH:
            current_batch.append(text)
            current_tokens += tokens
        else:
            if current_batch:
                batches.append(current_batch)
            current_batch = [text]
            current_tokens = tokens
    
    if current_batch:
        batches.append(current_batch)
        
    print(f"\n作成されたバッチの詳細:")
    for i, batch in enumerate(batches):
        total_tokens = sum(estimate_tokens(text) for text in batch)
        print(f"バッチ {i+1}: {len(batch)}テキスト, 推定トークン数: {total_tokens}")
    
    return batches

def get_embeddings_batch(texts: List[str]) -> Dict[str, np.ndarray]:
    print(f"\nバッチ処理実行:")
    print(f"バッチサイズ: {len(texts)}テキスト")
    global embedding_cache
    
    # キャッシュされていないテキストを特定
    uncached_texts = []
    uncached_indices = []
    text_hashes = []
    
    for i, text in enumerate(texts):
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        text_hashes.append(text_hash)
        if text_hash not in embedding_cache:
            uncached_texts.append(text)
            uncached_indices.append(i)
    
    # 新しいembeddingを取得
    if uncached_texts:
        print(f"キャッシュミス: {len(uncached_texts)}テキストのembeddingを取得")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=uncached_texts
        )
        
        # キャッシュを更新
        for i, embedding_data in enumerate(response.data):
            text_hash = text_hashes[uncached_indices[i]]
            embedding_cache[text_hash] = embedding_data.embedding
        
        save_cache(embedding_cache)
    
    # 結果を構築
    result = {}
    for text, text_hash in zip(texts, text_hashes):
        result[text] = np.array(embedding_cache[text_hash])
    
    return result

def process_embeddings(materials: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """全テキストのembeddingを効率的に取得"""
    texts = [data["text"] for data in materials.values()]
    batches = create_optimal_batches(texts)
    
    all_embeddings = {}
    for batch in tqdm(batches, desc="Processing embeddings"):
        batch_embeddings = get_embeddings_batch(batch)
        all_embeddings.update(batch_embeddings)
    
    return all_embeddings

def compute_similarities(embeddings: Dict[str, np.ndarray], materials: Dict[str, Dict]) -> List[Tuple]:
    """embeddingから類似度を計算"""
    similarities = []
    nodes = list(materials.keys())
    
    for i, u in enumerate(nodes):
        for w in nodes[i+1:]:
            text_u = materials[u]["text"]
            text_w = materials[w]["text"]
            embedding_u = embeddings[text_u]
            embedding_w = embeddings[text_w]
            
            similarity = cosine_similarity([embedding_u], [embedding_w])[0, 0]
            similarities.append((u, w, similarity))
    
    return similarities

def plot_edge_weights(G: nx.Graph):
    """エッジの重みを棒グラフで可視化"""
    weights = [d['weight'] for (u, v, d) in G.edges(data=True)]
    
    plt.figure(figsize=(10, 6))
    plt.hist(weights, bins=10, edgecolor='black')
    plt.title('Distribution of Edge Weights')
    plt.xlabel('Edge Weight')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig("out/embedding_edge_weights_distribution.png")
    plt.close()
    
    return weights

def prune_graph_by_mean(G: nx.Graph, weights: List[float]):
    """平均重み以下のエッジを削除し、孤立ノードを除去"""
    mean_weight = np.mean(weights)
    print(f"\nエッジの重みの統計:")
    print(f"平均: {mean_weight:.3f}")
    print(f"最小: {min(weights):.3f}")
    print(f"最大: {max(weights):.3f}")
    
    # 平均以下のエッジを削除
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] <= mean_weight]
    G.remove_edges_from(edges_to_remove)
    print(f"\n平均重み（{mean_weight:.3f}）以下のエッジを{len(edges_to_remove)}個削除")
    
    # 孤立ノードの削除
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"孤立ノードを{len(isolated_nodes)}個削除")
    
    return G

def prune_graph_by_percentile(G: nx.Graph, weights: List[float], lower_percentile: float = 5, upper_percentile: float = 95):
    """指定したパーセンタイルの範囲外のエッジを削除"""
    lower_threshold = np.percentile(weights, lower_percentile)
    upper_threshold = np.percentile(weights, upper_percentile)
    
    print(f"\nエッジの重みの統計 (パーセンタイルベース):")
    print(f"{lower_percentile}パーセンタイル: {lower_threshold:.3f}")
    print(f"{upper_percentile}パーセンタイル: {upper_threshold:.3f}")
    print(f"最小: {min(weights):.3f}")
    print(f"最大: {max(weights):.3f}")
    
    # 閾値外のエッジを削除
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                      if d['weight'] <= lower_threshold or d['weight'] >= upper_threshold]
    G.remove_edges_from(edges_to_remove)
    print(f"\n閾値外のエッジを{len(edges_to_remove)}個削除")
    
    # 孤立ノードの削除
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    print(f"孤立ノードを{len(isolated_nodes)}個削除")
    
    return G

def graph_ave_openai_embedding(materials: Dict[str, Dict]) -> nx.Graph:
    """メイン処理"""
    G = nx.Graph()
    
    # ノードの追加
    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))
    
    # embeddingの取得
    embeddings = process_embeddings(materials)
    
    # 類似度の計算と
    similarities = compute_similarities(embeddings, materials)
    
    # エッジの追加
    for u, w, weight in similarities:
        G.add_edge(u, w, weight=weight)
    
    print(f"\n初期グラフ:")
    print(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
    
    # エッジの重みの分布を可視化
    weights = plot_edge_weights(G)
    
    # グラフの枝刈り
    G = prune_graph_by_mean(G, weights)
    
    print(f"\n枝刈り後のグラフ:")
    print(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
    
    # 可視化と保存
    plt.figure(figsize=(10,6))
    nx.draw(G, with_labels=True)
    plt.savefig("out/embedding_ave_graph.png")
    nx.write_gml(G, "out/embedding_ave_graph.gml")
    plt.close()
    
    return G


def graph_remove_outliers_openai_embedding(materials: Dict[str, Dict]) -> nx.Graph:
    """メイン処理"""
    G = nx.Graph()
    
    # ノードの追加
    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))
    
    # embeddingの取得
    embeddings = process_embeddings(materials)
    
    # 類似度の計算と
    similarities = compute_similarities(embeddings, materials)
    
    # エッジの追加
    for u, w, weight in similarities:
        G.add_edge(u, w, weight=weight)
    
    print(f"\n初期グラフ:")
    print(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
    
    # エッジの重みの分布を可視化
    weights = plot_edge_weights(G)
    
    # グラフの枝刈り
    G = prune_graph_by_percentile(G, weights)
    
    print(f"\n枝刈り後のグラフ:")
    print(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")
    
    # 可視化と保存
    plt.figure(figsize=(10,6))
    nx.draw(G, with_labels=True)
    plt.savefig("out/embedding_remove_outliers_graph.png")
    nx.write_gml(G, "out/embedding_remove_outliers_graph.gml")
    plt.close()
    
    return G

if __name__ == "__main__":
    test_materials = {
        'doc1': {'text': '象は鼻が長い動物です。'},
        'doc2': {'text': '象の鼻は長く、様々な用途に使われます。'},
        'doc3': {'text': '象は大きな体を持つ哺乳類です。'},
        'doc4': {'text': '象は体重が4000から6000kgにもなる大型の陸上動物です。成獣のメスで2500から3500kg、オスで4000から6000kgになります。'},
        'doc5': {'text': '''象は非常に知能が高く、道具を使うことができます。
                         鼻を使って木の枝を折り、ハエを追い払ったり、
                         体を掻いたりすることができます。
                         また、水浴びの際には、鼻を使って水を吸い上げ、
                         背中に水をかけることもできます。'''}
    }
    
    G = graph_ave_openai_embedding(test_materials)
    
    assert G.number_of_nodes() >= 2, "グラフは最低2つのノードが必要です"
    assert G.number_of_edges() >= 1, "グラフは最低1つのエッジが必要です"
    
    print("\nグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")
    print(G.nodes(data=True))