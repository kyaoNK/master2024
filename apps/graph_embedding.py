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

CACHE_PATH = 'embedding_cache.json'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'r') as f:
        embedding_cache = json.load(f)

else:
    embedding_cache = {}

def get_openai_embedding(text):
    global embedding_cache

    text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

    if text_hash in embedding_cache:
        return np.array(embedding_cache[text_hash])

    response = client.embeddings.create(model="text-embedding-3-small",
    input=[text])

    embedding = np.array(response.data[0].embedding)

    embedding_cache[text_hash] = embedding.tolist()

    with open(CACHE_PATH, "w") as f:
        json.dump(embedding_cache, f)

    return embedding

def compute_edge_cosine(args):
    u, w, text_u, text_w = args
    try:
        embedding_u = get_openai_embedding(text_u)
        embedding_w = get_openai_embedding(text_w)

        similarity = cosine_similarity([embedding_u], [embedding_w])[0, 0]
        return u, w, similarity
    except Exception as e:
        print(f'Error processing edge {u}-{w}: {e}')
        return None

def graph_openai_embedding(materials):
    G = nx.Graph()

    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))

    nodes = list(materials.keys())

    edge_args = [(u, w, G.nodes[u]["text"], G.nodes[w]["text"]) for i, u in enumerate(nodes) for w in nodes[i+1:]]

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_edge_cosine, edge_args), total=len(edge_args)):
            if result:
                u, w, weight = result
                G.add_edge(u, w, weight=weight)

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    print(f"ノード数: {G.number_of_nodes()}, エッジ数: {G.number_of_edges()}")

    plt.figure(figsize=(10,6))
    nx.draw(G, with_labels=True)
    plt.savefig("out/embedding_graph.png")
    nx.write_gml(G, "out/embedding_graph.gml")
    plt.close()

    return G

if __name__=="__main__":
    test_materials = {
       'doc1': {'text': '象は鼻が長い動物です。'},
       'doc2': {'text': '象の鼻は長く、様々な用途に使われます。'},  # 類似文書
       'doc3': {'text': '象は大きな体を持つ哺乳類です。'}
   }

    # グラフの生成をテスト
    G = graph_openai_embedding(test_materials)

    # 基本的な検証
    assert G.number_of_nodes() >= 2, "グラフは最低2つのノードが必要です"
    assert G.number_of_edges() >= 1, "グラフは最低1つのエッジが必要です"
    
    print("\nグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")
    print(G.nodes(data=True))