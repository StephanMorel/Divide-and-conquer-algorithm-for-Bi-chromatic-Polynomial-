import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import sympy






def cartesian_product_k2(graph):
    new_graph = {}
    label_map = {}
    #while cartesianing the graph by K_2 we have two labels one which was the vertcies before the graph was carteisiand and the 
    # other label which is the vertcies created not in the orignial graph
    for idx, u in enumerate(graph.keys()):
        new_u0, new_u1 = f"x{idx + 1}", f"y{idx + 1}"
        label_map[(u, 0)] = new_u0
        label_map[(u, 1)] = new_u1
        
        new_graph[(u, 0)] = []
        new_graph[(u, 1)] = []
        
        for v in graph[u]:
            new_graph[(u, 0)].append((v, 0))
            new_graph[(u, 1)].append((v, 1))
        
        new_graph[(u, 0)].append((u, 1))
        new_graph[(u, 1)].append((u, 0))
        
    return new_graph, label_map

def add_additional_edges(graph):
    # this function checks via the lables if the vertcies of the orignal graph are adjacent then creates adjacencies betwee the k_2 adjacent
    #vertex and the actual vertex
    to_add = []  # List to store the new edges to be added

    # Iterate over vertices and their neighbors in the graph
    for u, neighbors in graph.items():
        for v in neighbors:
            
            # Ignore edges where vertices u and v belong to different K_2 components
            if u[1] != v[1]:
                continue
            
            # Determine the alternate K_2 component of v (i.e., v') via labels in previous function
            v_tag = 1 if v[1] == 0 else 0
            
            # Schedule the edge (u, v') to be added
            to_add.append((u, (v[0], v_tag)))

    # Add the scheduled edges to the graph
    for u, v in to_add:
        if v not in graph[u]:
            graph[u].append(v)
        if u not in graph[v]:
            graph[v].append(u)
def draw_graph(graph, label_map):# I used this for drawing what graph I want to be tested
    G = nx.Graph()
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)
            
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=700, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels=label_map)
    plt.show()


def chromatic_polynomial(G): #deletion contraction algorthim 
    x = sympy.Symbol("x")
    stack = deque()
    stack.append(nx.MultiGraph(G, contraction_idx=0))
    polynomial = 0

    while stack:
        G = stack.pop()
        edges = list(G.edges)
        if not edges:
            polynomial += (-1) ** G.graph["contraction_idx"] * x ** len(G)
        else:
            e = edges[0]
            C = nx.contracted_edge(G, e, self_loops=True)
            C.graph["contraction_idx"] = G.graph["contraction_idx"] + 1
            C.remove_edge(e[0], e[0])
            G.remove_edge(*e)
            stack.append(G)
            stack.append(C)
        
        
    return polynomial






def multi_chromatic_polynomial(graph):# the K2 wreath product given a graph that converts a problem in the bichromatic case into the singular chroamtic case
    new_graph, label_map = cartesian_product_k2(graph)

    add_additional_edges(new_graph)
 
    #draw_graph(new_graph,label_map)

    return new_graph

def chrom(graph):
    new_graph, label_map = cartesian_product_k2(graph)

    add_additional_edges(new_graph)
 
    #draw_graph(new_graph,label_map)

    return chromatic_polynomial(new_graph)
