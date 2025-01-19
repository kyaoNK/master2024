import preprocess
import technical_term
import graph_jaccard
from find_path import find_max_weight_path
import networkx as nx
import read_gml

raw_filepath = 'data/information1_raw.md'
materials = preprocess.preprocessing(raw_filepath)

technical_term_filepath = 'data/technical_term.xlsx'
technical_terms = technical_term.read_technical_term(technical_term_filepath)

G = graph_jaccard.graph_ave_jaccard(materials, technical_terms)

print("\nグラフの基本情報:")
print(f"ノード数: {G.number_of_nodes()}")
print(f"エッジ数: {G.number_of_edges()}")
    
# G = nx.read_gml("out/jaccard_ave_graph.gml", destringizer=str)

start_label = "251"
end_label = "293"
path, weight = find_max_weight_path(G, start_label, end_label)

if path:
    print(f"\n最大重みパス: {' -> '.join(path)}")
    print(f"総重み: {weight}")
else:
    print(f"\nパスが見つからなかった")