import networkx as nx

def find_node_by_text_id(G, text_id):
    for node, attr in G.nodes(data=True):
        # print(f"node: {node} - label: {attr.get('label')}")  # デバッグ用出力
        if attr.get("label") == text_id:
            return node
    return None
    
def find_max_weight_path(G, start_text_id, end_text_id):    
    start_node = find_node_by_text_id(G, start_text_id)
    print(f"始点ノード: {start_node}")
    
    end_node = find_node_by_text_id(G, end_text_id)
    print(f"終点ノード: {end_node}")
    
    if start_node is None:
        print(f"始点テキストID: '{start_text_id}' に対応するノードが見つかりません。")
    if end_node is None:
        print(f"終点テキストID: '{end_text_id}' に対応するノードが見つかりません。")
    if start_node is None or end_node is None:
        return None, 0
        
    try:
        path = nx.dijkstra_path(G, start_node, end_node, weight=lambda u, v, d: 1 - d['weight'] if d['weight'] > 0 else float('inf'))

        total_weight = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))

        # パスに含まれるノードの情報も出力
        print("\nパス上のノードの属性:")
        for node in path:
            print(f"ノード {node}: {G.nodes[node]}")

        return path, total_weight
    
    except nx.NetworkXNoPath:
        print("パスが見つかりませんでした")
        return None, 0
    except ZeroDivisionError:
        print("重みが0のエッジが存在します")
        return None, 0
    
if __name__=='__main__':
    G = nx.DiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1, 2, weight=5)
    G.add_edge(2, 3, weight=3)

    path, weight = find_max_weight_path(G, 1, 2)
    
    print(f"見つかったパス: {path}")
    print(f"合計重み: {weight}")