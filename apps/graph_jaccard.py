import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import numpy as np

def included_techinal_term(text, technical_terms):
    included_techinal_terms = list()
    if not text:
        return included_techinal_terms

    text = text.lower()

    for term in technical_terms:
        term = term.lower()
        try:
            if term in text:
                included_techinal_terms.append(term)
        except TypeError as e:
            print(f"Error processing term '{term}': {e}")
            continue
        
    return included_techinal_terms

def jaccard_coefficient_term(text1, text2, technical_terms):
    
    terms1 = included_techinal_term(text1, technical_terms)
    terms2 = included_techinal_term(text2, technical_terms)

    s1 = set(terms1)
    s2 = set(terms2)
    try:
        return float(len(s1.intersection(s2)) / len(s1.union(s2)))
    except ZeroDivisionError:
        return 0.0

def filter_edges_by_weight(G, threshold):
    filtered_G = nx.Graph()
    filtered_G.add_nodes_from(G.nodes())
    
    for u, v, data in G.edges(data=True):
        if data['weight'] >= threshold:
            filtered_G.add_edge(u, v, **data)

    return filtered_G
    
def graph_ave_jaccard(materials, technical_terms):
    G = nx.Graph()
    for i, (text_id) in enumerate(materials.keys()):
        G.add_node(i, label=str(text_id))
    
    nodes = list(materials.keys())
    total = len(nodes)
    with tqdm(total=total * (total-1) // 2) as pbar:
        for i, u in enumerate(nodes):
            for w in nodes[i+1:]:
                # 計算
                weight = jaccard_coefficient_term(materials.get(u).get("text"), materials.get(w).get("text"), technical_terms)
                if weight > 0.0:
                    G.add_edge(u, w, weight=weight)
                pbar.update(1)
                
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
        
    # 平均重みの計算と表示
    if G.number_of_edges() > 0:
        total_weight = sum(data['weight'] for u, v, data in G.edges(data=True))
        average_weight = total_weight / G.number_of_edges()
        print(f"平均重み: {average_weight}")
    else:
        average_weight = 0.0
        
    # 重みの分布図
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
    
    for node in G.nodes(data=True): 
        print(f"Node: {node[0]}")
        print(f"Attr: {node[1]}")
    
    plt.figure(figsize=(10,6))
    nx.draw(G_ave, with_labels=True)
    plt.savefig("out/jaccard_ave_graph.png")
    nx.write_gml(G_ave, "out/jaccard_ave_graph.gml", stringizer=str)
    plt.close()
    
    return G_ave
    
def graph_remove_outliers_jaccard(materials, technical_terms):
    
    def filter_edges_by_percentile(G, lower_percentile=5, upper_percentile=95):
        weight = [data['weight'] for _, _, data in G.edges(data=True)]
        lower_bound = np.percentile(weights, lower_percentile)
        upper_bound = np.percentile(weights, upper_percentile)
        edges_to_remove = [(u, v) for u, v, data in G.edges(data=True)
                          if data['weight'] < lower_bound or data['weight'] > upper_bound]
        
        filtered_G = G.copy()
        filtered_G.remove_edges_from(edges_to_remove)
        return filtered_G
    
    nodes = list(materials.keys())
    G = nx.Graph()
    
    total = len(nodes)
    with tqdm(total=total * (total-1) // 2) as pbar:
        for i, u in enumerate(nodes):
            for w in nodes[i+1:]:
                # 計算
                weight = jaccard_coefficient_term(materials.get(u).get("text"), materials.get(w).get("text"), technical_terms)
                if weight > 0.0:
                    G.add_edge(u, w, weight=weight)
                pbar.update(1)

    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    weights = [data['weight'] for u, v, data in G.edges(data=True)]
    print(f"エッジの重みの範囲: {min(weights):.3f} - {max(weights):.3f}")
    
    G_filtered = filter_edges_by_percentile(G)
    filtered_weights = [data['weight'] for u, v, data in G_filtered.edges(data=True)]
    print(f"フィルタリング後の重みの範囲: {min(filtered_weights):.3f} - {max(filtered_weights):.3f}")
    
    # 孤立ノードの除去
    isolated_nodes = list(nx.isolates(G_filtered))
    G_filtered.remove_nodes_from(isolated_nodes)
    plt.figure(figsize=(10,6))
    nx.draw(G_filtered, with_labels=True)
    plt.savefig("out/jaccard_remove_outliers_graph.png")
    nx.write_gml(G_filtered, "out/jaccard_remove_outliers_graph.gml")
    plt.close()