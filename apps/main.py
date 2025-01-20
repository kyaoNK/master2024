import preprocess
import technical_term
import graph_jaccard
from find_path import find_max_weight_path
import networkx as nx

raw_filepath = 'data/information1_raw.md'
materials = preprocess.preprocessing(raw_filepath)

technical_term_filepath = 'data/technical_term.xlsx'
technical_terms = technical_term.read_technical_term(technical_term_filepath)

G = graph_jaccard.graph_ave_jaccard_parallel(materials, technical_terms)

print("\nグラフの基本情報:")
print(f"ノード数: {G.number_of_nodes()}")
print(f"エッジ数: {G.number_of_edges()}")

r_G = nx.read_gml("out/jaccard_ave_graph.gml", label=None)

print("\nGraph nodes:")
for i, node in enumerate(r_G.nodes(data=True)):
    print(f"Node: {node[0]} - Attr: {node[1].get('label')}")
    if i == 100:
        break

# start_text_id = "251"
# end_text_id = "293"
# path, weight = find_max_weight_path(r_G, start_text_id, end_text_id)

# if path:
#     print(f"\n最大重みパス: {' -> '.join(path)}")
#     print(f"総重み: {weight}")
# else:
#     print(f"\nパスが見つからなかった")