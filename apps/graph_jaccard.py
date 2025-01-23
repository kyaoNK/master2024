import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools

def included_technical_term(text, technical_terms):
    included_techinal_terms = list()
    if not text:
        return included_techinal_terms

    text = text.lower()
    for term_group in technical_terms:
        variants = term_group.split('|')
        for term in variants:
            term = term.lower()    
            try:
                if term in text:
                    included_techinal_terms.append(term_group)
                    break
            except TypeError as e:
                print(f"Error processing term '{term}': {e}")
                continue
        
    return included_techinal_terms

def jaccard_coefficient_term(text1, text2, technical_terms):
    
    if not hasattr(jaccard_coefficient_term, 'cache'):
        jaccard_coefficient_term.cache = {}
        
    cache_key = (text1, text2) if text1 < text2 else (text2, text1)
    
    if cache_key in jaccard_coefficient_term.cache:
        return jaccard_coefficient_term.cache[cache_key]
    
    result = calculate_jaccard(text1, text2, technical_terms)
    jaccard_coefficient_term.cache[cache_key] = result
    return result

def calculate_jaccard(text1, text2, technical_terms):
    terms1 = included_technical_term(text1, technical_terms)
    terms2 = included_technical_term(text2, technical_terms)

    s1 = set(terms1)
    s2 = set(terms2)
    
    try:
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))
    except ZeroDivisionError:
        return 0.0

def filter_edges_by_weight(G, threshold):
    filtered_G = nx.Graph()
    for node, attr in G.nodes(data=True):
        filtered_G.add_node(node, **attr)
        
    for u, v, data in G.edges(data=True):
        if data['weight'] >= threshold:
            filtered_G.add_edge(u, v, **data)

    return filtered_G
    
def compute_edge(args):
    u, w, text1, text2, technical_terms = args
    weight = jaccard_coefficient_term(text1, text2, technical_terms)
    return (u, w, weight)
    
def graph_ave_jaccard_parallel(materials, technical_terms):
    G = nx.Graph()

    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))

    nodes = list(materials.keys())

    edge_args = [(u, w, materials[u]["text"], materials[w]["text"], technical_terms) for i, u in enumerate(nodes) for w in nodes[i+1:]]

    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_edge, edge_args), total=len(edge_args)):
            if result:
                u, w, weight = result
                G.add_edge(u, w, weight=weight)
                    
    print("\n最初に構築したグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    print(f"\n孤立したノード: {isolated_nodes}")
    print("\n削除後のグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")
        
    # 平均重みの計算と表示
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
    plt.savefig("out/distribution_of_weights.png", format="png")
    plt.close()
        
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
    plt.savefig("out/jaccard_ave_graph.png")
    nx.write_gml(G_ave, "out/jaccard_ave_graph.gml")
    plt.close()
    
    return G_ave
    
def graph_remove_outliers_jaccard_parallel(materials, technical_terms):
    
    def filter_edges_by_percentile(G, lower_percentile=5, upper_percentile=95):
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
    
    G = nx.Graph()
    for node_id, data in materials.items():
        G.add_node(node_id, label=node_id, text=data.get("text"))
        
    nodes = list(materials.keys())
    
    edge_args = [(u, w, materials[u]["text"], materials[w]["text"], technical_terms) for i, u in enumerate(nodes) for w in nodes[i+1:]]
    
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(compute_edge, edge_args), total=len(edge_args)):
            if result:
                u, w, weight = result
                G.add_edge(u, w, weight=weight)

    print("\n最初に構築したグラフの基本情報:")
    print(f"ノード数: {G.number_of_nodes()}")
    print(f"エッジ数: {G.number_of_edges()}")

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    # ----- debug ----- #
    # print(f"\n孤立したノード: {isolated_nodes}")
    # print("\n削除後のグラフの基本情報:")
    # print(f"ノード数: {G.number_of_nodes()}")
    # print(f"エッジ数: {G.number_of_edges()}")
    # ----- debug ----- #
    
    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    print(f"エッジの重みの範囲: {min(weights):.3f} - {max(weights):.3f}")
    
    G_filtered = filter_edges_by_percentile(G)
    filtered_weights = [data['weight'] for u, v, data in G_filtered.edges(data=True)]
    print(f"フィルタリング後の重みの範囲: {min(filtered_weights):.3f} - {max(filtered_weights):.3f}")
    
    isolated_nodes = list(nx.isolates(G_filtered))
    G_filtered.remove_nodes_from(isolated_nodes)
    
    # ----- debug ----- #
    # print(f"\n削除したノード: {isolated_nodes}")
    # print("\n上下5%削除後のグラフの基本情報:")
    # print(f"ノード数: {G.number_of_nodes()}")
    # print(f"エッジ数: {G.number_of_edges()}")
    # ----- debug ----- #
    
    plt.figure(figsize=(10,6))
    nx.draw(G_filtered, with_labels=True)
    plt.savefig("out/jaccard_remove_outliers_graph.png")
    nx.write_gml(G_filtered, "out/jaccard_remove_outliers_graph.gml")
    plt.close()
    
    return G_filtered

if __name__=="__main__":
    text = "ETCシステムを利用して..."
    terms = ["ETC|ETCシステム|自動料金収受システム"]
    result = included_technical_term(text, terms)
    print(result)
    # 結果: ['ETC|ETCシステム|自動料金収受システム']
    
    
    
    terms1 = ["ETC|ETCシステム|自動料金収受システム", "トップダウンテスト", "トリプルメディア"]
    terms2 = ["ETC|ETCシステム|自動料金収受システム"]
    
    s1 = set(terms1)
    s2 = set(terms2)
    
    try:
        print(float(len(s1.intersection(s2)) / len(s1.union(s2))))
    except ZeroDivisionError:
        print("ゼロで割っている")