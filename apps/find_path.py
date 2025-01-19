import networkx as nx

def find_node_by_label(G, label):
    for node, attr in G.nodes(data=True):
        # print(f"node: {node}, label: {attr.get('label')}")  # デバッグ用出力
        if attr.get('label') == label:
            return node
    return None
    
def find_max_weight_path(G, start_label, end_label):    
    start_node = find_node_by_label(G, start_label)
    print(f"始点ノード: {start_node}")
    print(f"始点ノードの属性: {G.nodes[start_node] if start_node in G else 'なし'}")
    
    end_node = find_node_by_label(G, end_label)
    print(f"終点ノード: {end_node}")
    print(f"終点ノードの属性: {G.nodes[end_node] if end_node in G else 'なし'}")
    
    if start_node is None:
        print(f"始点ラベル '{start_label}' に対応するノードが見つかりません。")
    if end_node is None:
        print(f"終点ラベル '{end_label}' に対応するノードが見つかりません。")
    if start_node is None or end_node is None:
        return None, 0
        
    G_temp = G.copy()
    
    for u, v, w in G_temp.edges(data='weight'):
        G_temp[u][v]['weight'] = -w
        
    try:
        paths = list(nx.shortest_simple_paths(G_temp, start_node, end_node, weight='weight'))
        if not paths:
            return None, 0
        path = paths[0]
        
        total_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
    
        # パスに含まれるノードの情報も出力
        print("\nパス上のノードの属性:")
        for node in path:
            print(f"ノード {node}: {G.nodes[node]}")
    
        return path, total_weight
    
    except nx.NetworkXNoPath:
        print("パスが見つかりませんでした")
        return None, 0
    
if __name__=='__main__':
    G = nx.DiGraph()
    G.add_node(1, label='A')
    G.add_node(2, label='B')
    G.add_node(3, label='C')
    G.add_edge(1, 2, weight=5)
    G.add_edge(2, 3, weight=3)

    path, weight = find_max_weight_path(G, 'A', 'C')
    
    print(f"見つかったパス: {path}")
    print(f"合計重み: {weight}")