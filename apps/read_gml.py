import networkx as nx

def read_gml(filepath):
    G = nx.Graph()
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        current_node = None
        for line in lines:
            line = line.strip()
            if line == "node [":
                current_node = {}
            elif current_node is not None:
                if line.startswith("id"):
                    current_node["id"] = int(line.split()[1])
                elif line.startswith("label"):
                    # ダブルクォートを除去して値を取得
                    current_node["label"] = line.split()[1].strip('"')
                elif line == "]":
                    # ノードを追加
                    node_id = current_node["id"]
                    G.add_node(node_id, label=current_node["label"])
                    current_node = None
                    
    return G

def test_read_gml():
    # テストデータの作成
    test_gml = '''graph [
  node [
    id 0
    label "1"
  ]
  node [
    id 1
    label "2"
  ]
]'''
    
    # テストファイルの作成
    with open('test.gml', 'w', encoding='utf-8') as f:
        f.write(test_gml)
    
    # GMLファイルの読み込み
    G = read_gml('test.gml')
    
    # テスト結果の表示
    print("ノード数:", len(G.nodes()))
    print("各ノードの属性:")
    for node in G.nodes(data=True):
        print(node)
    
    # テストファイルの削除
    import os
    os.remove('test.gml')

# テストの実行
if __name__=='__main__':
    test_read_gml()