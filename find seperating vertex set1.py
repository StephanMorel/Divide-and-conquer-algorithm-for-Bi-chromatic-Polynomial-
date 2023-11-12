import networkx as nx
import matplotlib.pyplot as plt
import ifnotclique as i
import deletioncontraction as p
import sympy as sp
import copy
import time
    
def filter_graph_by_vertices(G, allowed_vertices):
    # Convert allowed_vertices list to set for faster lookups
    allowed_set = set(allowed_vertices)

    # Filter the graph based on the allowed vertices set
    filtered_graph = {v: [neighbor for neighbor in G[v] if neighbor in allowed_set] 
                      for v in allowed_set if v in G}

    return filtered_graph
def chromatic_polynomial_clique(n):
    x = sp.symbols('x')
    polynomial = 1
    for i in range(n):
        polynomial *= (x - i)
    return polynomial
# def chromatic_polynomial_clique(x):
   
#     n = symbols('x')
#     polynomial = 1
#     for i in range(x):
#         term = (n - 2 * i) * (n - 2 * i - 1)
#         polynomial *= term
#     polynomial = sp.simplify(polynomial / (2**x))
#     return polynomial

# Test the function with an example:



def is_clique(graph):
    n = len(graph)  # Number of vertices
    edge_count = sum(len(neighbors) for neighbors in graph.values()) // 2  # Number of edges

    # Check if the number of edges is equal to n choose 2
    vertices = list(graph.keys())
    n = len(vertices)

    # Loop over all pairs of vertices to ensure adjacency
    for i in range(n):
        for j in range(i + 1, n):
            v1 = vertices[i]
            v2 = vertices[j]

            # If the vertices aren't adjacent in either direction
            if v2 not in graph[v1] and v1 not in graph[v2]:
                return False
    return True

def remove_vertices(vertices_to_remove, graph_dict):
    """
    Remove the specified vertices from the graph represented as a dictionary.

    Parameters:
    - vertices_to_remove: A set or list of vertices to be removed.
    - graph_dict: A dictionary representing the graph from which the vertices will be removed.

    Returns:
    - Updated graph represented as a dictionary.
    """
    # Convert dictionary to networkx Graph
    G = nx.from_dict_of_lists(graph_dict)

    # Remove vertices
    for vertex in vertices_to_remove:
        if vertex in G:
            G.remove_node(vertex)
            
    # Convert back to dictionary
    return nx.to_dict_of_lists(G)

def undirected_to_digraph(undirected_graph):
    digraph = {vertex: [] for vertex in undirected_graph}
    
    for vertex, edges in undirected_graph.items():
        for edge in edges:
            if vertex < edge:
                digraph[vertex].append(edge)
            # In an undirected graph, both directions of an edge would be present
            # We don't need to add the reversed edge as it will be added when the larger vertex is the current vertex in the loop
                
    return digraph
  #    
graph1 ={
    1: [2, 4],
    2: [1, 3],
    3: [2, 4],
    4: [3, 1]
}

def undirected_to_digraph(graph):
    digraph = {v: [] for v in graph.keys()}  # Initialize empty digraph
    
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            if neighbor not in digraph[vertex]:  # Avoid adding duplicate edges
                digraph[vertex].append(neighbor)
    
    return digraph



#input a graph G
def ChromaticPoly(G):
    if is_clique(G)==True:#clique's chromatic polynomial is easy to calculate
        return chromatic_polynomial_clique(len(G))  
    #finds a vertex set that seperates the graph into disconnected subgraphs
    seperatingset =filter_graph_by_vertices(G, i.minimum_node_cut(nx.Graph((undirected_to_digraph(G)))))  
    remainder = remove_vertices(seperatingset,G)#the remainder is the subgraph without vetcies in seprating set in original graph
    return ChromaticPoly2(G,seperatingset,remainder)
def ChromaticPoly2(G, seperatingset, remainder=None):
    if is_clique(G) == True:
        return chromatic_polynomial_clique(len(G))
    if remainder == None:#this is to deal with the case where the remainder is not yet initlised
        remainder = remove_vertices(seperatingset, G)
        #if the seperating set is a cliue the program uses the clique glue method  
    if is_clique(seperatingset) == True:
        componets = i.identify_components((remainder))
        comp_1, comp_2 = componets[0], componets[1]
        return ChromaticPoly(i.filter_vertices_by_subgraphs(G, comp_1,
         seperatingset)) * ChromaticPoly(i.filter_vertices_by_subgraphs(G, comp_2, 
        seperatingset))/ chromatic_polynomial_clique(len(seperatingset))
    else:
        #find a pair of none-adjacent vertcies of distance more then 2 so the seperating set is more dense
        a, b = i.find_pair_or_min_degree(seperatingset)
        # Making deep copies before making any changes
        G_copy1 = copy.deepcopy(G)
        G_copy2 = copy.deepcopy(G)
        seperatingset_copy1 = copy.deepcopy(seperatingset)
        seperatingset_copy2 = copy.deepcopy(seperatingset)
        
        return ChromaticPoly2(i.add_edge(G_copy1, a, b), i.add_edge(seperatingset_copy1, 
        a, b)) + ChromaticPoly2(i.identify_vertices(G_copy2, a, b), i.identify_vertices(seperatingset_copy2, a, b))
       


from sympy import symbols, expand

# Define the variable
x = symbols('x')

# Define the polynomial


# Expand the polynomial
#expanded_polynomial = expand(ChromaticPoly(graph))



#c4x**8/16 - 5*x**7/4 + 87*x**6/8 - 211*x**5/4 + 2433*x**4/16 - 513*x**3/2 + 230*x**2 - 165*x/2
#c4 with an extra edge x**8/16 - 3*x**7/2 + 15*x**6 - 645*x**5/8 + 3999*x**4/16 - 3543*x**3/8 + 410*x**2 - 150*x
#P3 x**6/8 - 11*x**5/8 + 47*x**4/8 - 97*x**3/8 + 12*x**2 - 9*x/2
#P3 shared vertex edge  x**5/32 - x**4/4 + 23*x**3/32 - 7*x**2/8 + 3*x/8
#p2 x**4/4 - 3*x**3/2 + 11*x**2/4 - 3*x/2
#P3 WITH SQUIGGLE EDGE ADDED x**5/32 - 9*x**4/32 + 29*x**3/32 - 39*x**2/32 + 9*x/16
#P3 with all squiggle x**4/16 - 5*x**3/16 + x**2/2 - x/4
#P3 with double line edge x**4/16 - 3*x**3/8 + 11*x**2/16 - 3*x/8
import random
import networkx as nx
import matplotlib.pyplot as plt

import random

def generate_random_graph():
    n = random.randint(12,12 )
    graph = {i: [] for i in range(n)}

    # Create a spanning tree using a randomized version of Prim's algorithm
    in_tree = {0}  # start with an arbitrary vertex (e.g., vertex 0)
    while len(in_tree) < n:
        # Select a random edge that connects a vertex inside the tree to a vertex outside of the tree
        inner_vertex = random.choice(list(in_tree))
        outer_vertices = set(range(n)) - in_tree
        outer_vertex = random.choice(list(outer_vertices))

        # Add the edge to the graph and update in_tree
        graph[inner_vertex].append(outer_vertex)
        graph[outer_vertex].append(inner_vertex)
        in_tree.add(outer_vertex)

    # Add additional edges probabilistically
    for i in range(n):
        for j in range(i+1, n):
            if j not in graph[i] and random.random() < 0.5:
                graph[i].append(j)
                graph[j].append(i)

    return graph



def draw_graph(graph):
    G = nx.Graph()

    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    nx.draw(G, with_labels=True, node_size=500, node_color='skyblue', font_size=15, width=2.0, alpha=0.6)
    plt.show()

graph = generate_random_graph()
draw_graph(graph)
print(graph)
def print_graph_info(graph):
    # Number of vertices
    vertices = len(graph)
    
    # Number of edges
    edges = sum(len(v) for v in graph.values()) // 2
    
    print(f"Number of vertices: {vertices}")
    print(f"Number of edges: {edges}")
print_graph_info(graph)
start_cpu_time = time.process_time()
# Assuming the expand function and ChromaticPoly are defined earlier in your code.
# And 'graph' is a predefined graph object.
print(expand(ChromaticPoly(p.multi_chromatic_polynomial(graph)))/2**len(graph))
end_cpu_time = time.process_time()
print(f"sub_program1 took {end_cpu_time - start_cpu_time:.5f} seconds of CPU time")

# Measure CPU time for sub_program2
start_cpu_time = time.process_time()
# Assuming p.chrom is a function defined in your 'p' module that calculates chromatic polynomial.
print(expand(p.chrom(graph))/2**len(graph))
end_cpu_time = time.process_time()
print(f"sub_program2 took {end_cpu_time - start_cpu_time:.5f} seconds of CPU time")


import networkx as nx

def create_subgraphs_from_components(adjacency_list):
    # Create a graph from the adjacency list
    G = nx.Graph(adjacency_list)
    
    # Check if the graph is disconnected
    if not nx.is_connected(G):
        # Get the list of all connected components
        components = list(nx.connected_components(G))
        
        # Create a list to hold the adjacency list of each component
        components_adj_list = []
        
        # Iterate over each component
        for component in components:
            # Create a subgraph for the current component
            subgraph = G.subgraph(component).copy()
            
            # Convert the subgraph back to an adjacency list format
            subgraph_adj_list = {n: list(subgraph.neighbors(n)) for n in subgraph.nodes()}
            
            # Append the adjacency list of the current component to the list
            components_adj_list.append(subgraph_adj_list)
        
        return components_adj_list
    else:
        # If the graph is connected, return the graph itself as the only component
        return adjacency_list

# Example usage:

# if a disconnected graph use the above algorthim
#chromatic_polynomial=1
#list_of_components=create_subgraphs_from_components(G):
#for i in list_of_components:
#chromatic_polynomial*=expand(ChromaticPoly(p.multi_chromatic_polynomial(i)))/2**len(graph)
