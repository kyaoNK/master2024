import preprocess
import technical_term
import networkx as nx
import graph_jaccard
import graph_tfidf
# import graph_embedding
import graph_embedding_batch
from find_path import find_max_weight_path_in_jaccard, find_max_weight_path_in_tfidf, find_max_weight_path_in_embedding

raw_filepath = 'data/information1_raw.md'
materials = preprocess.preprocessing(raw_filepath)

technical_term_filepath = 'data/technical_term.xlsx'
technical_terms = technical_term.read_technical_term(technical_term_filepath)

# G = graph_jaccard.graph_ave_jaccard_parallel(materials, technical_terms)
# G = nx.read_gml("out/jaccard_ave_graph.gml", label=None)

# G = graph_jaccard.graph_remove_outliers_jaccard_parallel(materials, technical_terms)
# G = nx.read_gml("out/jaccard_remove_outliers_graph.gml", label=None)

# G = graph_tfidf.graph_ave_tfidf_parallel(materials)
# G = nx.read_gml("out/tfidf_ave_graph.gml", label=None)

# G = graph_tfidf.graph_remove_outliers_tfidf_parallel(materials)
G = nx.read_gml("out/tfidf_remove_outliers_graph.gml", label=None)

# G = graph_embedding_batch.graph_ave_openai_embedding(materials)
# G = nx.read_gml("out/embedding_ave_graph.gml", label=None)

# G = graph_embedding_batch.graph_remove_outliers_openai_embedding(materials)
# G = nx.read_gml("out/embedding_remove_outliers_graph.gml", label=None)


print("\nグラフの基本情報:")
print(f"ノード数: {G.number_of_nodes()}")
print(f"エッジ数: {G.number_of_edges()}")

for i, node in enumerate(G.nodes(data=True)):
    print(node)
    if i == 10:
        break

start_text_id = "75"
end_text_id = "135"

# path, weight = find_max_weight_path_in_jaccard(G, start_text_id, end_text_id)
path, weight = find_max_weight_path_in_tfidf(G, start_text_id, end_text_id)
# path, weight = find_max_weight_path_in_embedding(G, start_text_id, end_text_id)

path_labels = " -> ".join(str(G.nodes[node].get('label', node)) for node in path)

if path:
    print(f"\n最大重みパス: {path_labels}")
    print(f"総重み: {weight}")
else:
    print(f"\nパスが見つからなかった")