import queue
import networkx as nx
def bfs_distance(graph, start, target):
  
    visited = set()
    distance = {v: float('inf') for v in graph}
    distance[start] = 0

    q = queue.Queue()
    q.put(start)

    while not q.empty():
        vertex = q.get()
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                q.put(neighbor)
                distance[neighbor] = distance[vertex] + 1

                if neighbor == target:
                    return distance[neighbor]
                    
    return float('inf')


def find_pair_or_min_degree(graph):
    # Check if two vertices are adjacent
    def are_adjacent(vertex_a, vertex_b):
        return vertex_b in graph[vertex_a] and vertex_a in graph[vertex_b]

    # Find a pair of non-adjacent vertices with distance > 2
    for vertex in graph:
        for target in graph:
            if vertex != target and not are_adjacent(vertex, target):
                distance = bfs_distance(graph, vertex, target)
                if distance > 2:
                    return vertex, target

    # If not found, then return two vertices with minimum degree
    degrees = {vertex: len(neighbors) for vertex, neighbors in graph.items()}
    min_degree_vertices = sorted(degrees, key=degrees.get)
    
    # Check pairs of min_degree_vertices for non-adjacency
    for i in range(len(min_degree_vertices)):
        for j in range(i+1, len(min_degree_vertices)):
            if not are_adjacent(min_degree_vertices[i], min_degree_vertices[j]):
                return min_degree_vertices[i], min_degree_vertices[j]

    return None


    
def identify_components(filtered_graph):
    """
    Identify the two distinct components in the filtered graph.
    Returns a tuple of components (graph1, graph2), where each component is represented as an adjacency list.
    Assumes the input graph can have more than 2 components due to isolated nodes.
    """
    G = nx.Graph(filtered_graph)
    components = list(nx.connected_components(G))
    
    # If there's only one component
    if len(components) == 1:
        return {node: list(G.neighbors(node)) for node in components[0]}, None

    graph1 = {node: list(G.neighbors(node)) for node in components[0]}
    graph2 = {node: list(G.neighbors(node)) for node in components[1]}
    
    # If there are isolated nodes, they will not be part of graph1 or graph2 yet.
    # So, append them to graph2.
    total_nodes_in_graphs = len(graph1) + len(graph2)
    if total_nodes_in_graphs < len(filtered_graph):
        isolated_nodes = set(filtered_graph) - set(graph1) - set(graph2)
        for node in isolated_nodes:
            graph2[node] = []

    return graph1, graph2

# Test
filtered_graph = {
    0: [1, 2],
    1: [0],
    2: [0],
    3: [4],
    4: [3]
   
}

def identify_vertices(graph, v1, v2):
    if v1 not in graph or v2 not in graph:
        raise ValueError("Vertices to be identified should be in the graph.")
    
    # Merge neighbors of v1 and v2, removing duplicates and references to v1 and v2
    new_edges = list(set(graph[v1] + [edge for edge in graph[v2] if edge != v1 and edge != v2]))
    graph[v1] = [edge for edge in new_edges if edge != v1 and edge != v2]  # remove any potential self-loop with v1 and v2
    
    # Remove v2 from the graph
    del graph[v2]

    # Update other vertices' neighbor lists
    for vertex, edges in graph.items():
        graph[vertex] = [v1 if edge == v2 else edge for edge in edges]  # Convert references from v2 to v1

        # Update undirected nature for those neighbors
        if vertex in graph[v1] and v1 not in graph[vertex]:
            graph[vertex].append(v1)
        
        # Remove any self-loop caused by v1 being its own neighbor
        graph[vertex] = [edge for edge in graph[vertex] if edge != vertex]

        # Remove duplicate edges for this vertex by converting to a set and then back to a list
        graph[vertex] = list(set(graph[vertex]))

    return graph


def add_edge(graph, src, dest):
    """Add an edge from src to dest to the graph and vice versa."""
    
    # Check if src is already in the graph
    if src not in graph:
        graph[src] = []
        
    # Check if dest is already in the graph
    if dest not in graph:
        graph[dest] = []

    # Check if dest is not already connected to src
    if dest not in graph[src]:
        graph[src].append(dest)
        
    # Check if src is not already connected to dest
    if src not in graph[dest]:
        graph[dest].append(src)

    return graph


def filter_vertices_by_subgraphs(G, vertices_set, subgraph):
    # Combine the set of vertices from the subgraph with the given vertices_set
    allowed_vertices = set(vertices_set).union(set(subgraph))

    # Create a new graph based on G but only with vertices that are in the allowed_vertices
    filtered_graph = {}
    
    for v in allowed_vertices:
        if v in G:
            for neighbor in G[v]:
                if neighbor in allowed_vertices:
                    # If v is not yet in the graph, initialize its adjacency list
                    if v not in filtered_graph:
                        filtered_graph[v] = []

                    # Ensure undirected nature and avoid self-loops
                    if neighbor != v and neighbor not in filtered_graph[v]:
                        filtered_graph[v].append(neighbor)

                    # Similarly, ensure undirected nature for the neighbor, avoiding self-loops and duplicates
                    if neighbor not in filtered_graph:
                        filtered_graph[neighbor] = []
                    if v != neighbor and v not in filtered_graph[neighbor]:
                        filtered_graph[neighbor].append(v)

    return filtered_graph



def subtract_graphs(main_graph, to_remove):
    """Subtract vertices and edges from 'to_remove' graph from the 'main_graph'."""
    
    # Remove edges
    for u, edges in to_remove.items():
        if u in main_graph:
            for edge in edges:
                if edge in main_graph[u]:
                    main_graph[u].remove(edge)
                    if u in main_graph[edge]:  # Remove the symmetric edge in the undirected graph
                        main_graph[edge].remove(u)
    
    # Remove vertices
    for vertex in to_remove.keys():
        if vertex in main_graph:
            del main_graph[vertex]
    
    return main_graph


graph = {
    0: [(1,1)],
    1: [(2,1)],
    2: [(3,1)],
    3: []
}

"""
Flow based cut algorithms
"""
import itertools

import networkx as nx
from networkx.algorithms.connectivity import build_auxiliary_edge_connectivity
from networkx.algorithms.connectivity.utils import build_auxiliary_node_connectivity

# Define the default maximum flow function to use in all flow based
# cut algorithms.
from networkx.algorithms.flow import build_residual_network, edmonds_karp

default_flow_func = edmonds_karp


__all__ = [
    "minimum_st_node_cut",
    "minimum_node_cut",
    "minimum_st_edge_cut",
    "minimum_edge_cut",
]


def minimum_st_edge_cut(G, s, t, flow_func=None, auxiliary=None, residual=None):
 
    if flow_func is None:
        flow_func = default_flow_func

    if auxiliary is None:
        H = build_auxiliary_edge_connectivity(G)
    else:
        H = auxiliary

    kwargs = {"capacity": "capacity", "flow_func": flow_func, "residual": residual}

    cut_value, partition = nx.minimum_cut(H, s, t, **kwargs)
    reachable, non_reachable = partition
    # Any edge in the original graph linking the two sets in the
    # partition is part of the edge cutset
    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    return cutset



def minimum_st_node_cut(G, s, t, flow_func=None, auxiliary=None, residual=None):
    
    if auxiliary is None:
        H = build_auxiliary_node_connectivity(G)
    else:
        H = auxiliary

    mapping = H.graph.get("mapping", None)
    if mapping is None:
        raise nx.NetworkXError("Invalid auxiliary digraph.")
    if G.has_edge(s, t) or G.has_edge(t, s):
        return {}
    kwargs = {"flow_func": flow_func, "residual": residual, "auxiliary": H}

    # The edge cut in the auxiliary digraph corresponds to the node cut in the
    # original graph.
    edge_cut = minimum_st_edge_cut(H, f"{mapping[s]}B", f"{mapping[t]}A", **kwargs)
    # Each node in the original graph maps to two nodes of the auxiliary graph
    node_cut = {H.nodes[node]["id"] for edge in edge_cut for node in edge}
    return node_cut - {s, t}



def minimum_node_cut(G, s=None, t=None, flow_func=None):
    
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # Local minimum node cut.
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return minimum_st_node_cut(G, s, t, flow_func=flow_func)

    # Global minimum node cut.
    # Analog to the algorithm 11 for global node connectivity in [1].
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            raise nx.NetworkXError("Input graph is not connected")
        iter_func = itertools.permutations

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v), G.successors(v)])

    else:
        if not nx.is_connected(G):
            raise nx.NetworkXError("Input graph is not connected")
        iter_func = itertools.combinations
        neighbors = G.neighbors

    # Reuse the auxiliary digraph and the residual network.
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "auxiliary": H, "residual": R}

    # Choose a node with minimum degree.
    v = min(G, key=G.degree)
    # Initial node cutset is all neighbors of the node with minimum degree.
    min_cut = set(G[v])
    # Compute st node cuts between v and all its non-neighbors nodes in G.
    for w in set(G) - set(neighbors(v)) - {v}:
        this_cut = minimum_st_node_cut(G, v, w, **kwargs)
        if len(min_cut) >= len(this_cut):
            min_cut = this_cut
    # Also for non adjacent pairs of neighbors of v.
    for x, y in iter_func(neighbors(v), 2):
        if y in G[x]:
            continue
        this_cut = minimum_st_node_cut(G, x, y, **kwargs)
        if len(min_cut) >= len(this_cut):
            min_cut = this_cut

    return min_cut



def minimum_edge_cut(G, s=None, t=None, flow_func=None):
    
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError("Both source and target must be specified.")

    # reuse auxiliary digraph and residual network
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, "capacity")
    kwargs = {"flow_func": flow_func, "residual": R, "auxiliary": H}

    # Local minimum edge cut if s and t are not None
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError(f"node {s} not in graph")
        if t not in G:
            raise nx.NetworkXError(f"node {t} not in graph")
        return minimum_st_edge_cut(H, s, t, **kwargs)

    # Global minimum edge cut
    # Analog to the algorithm for global edge connectivity
    if G.is_directed():
        # Based on algorithm 8 in [1]
        if not nx.is_weakly_connected(G):
            raise nx.NetworkXError("Input graph is not connected")

        # Initial cutset is all edges of a node with minimum degree
        node = min(G, key=G.degree)
        min_cut = set(G.edges(node))
        nodes = list(G)
        n = len(nodes)
        for i in range(n):
            try:
                this_cut = minimum_st_edge_cut(H, nodes[i], nodes[i + 1], **kwargs)
                if len(this_cut) <= len(min_cut):
                    min_cut = this_cut
            except IndexError:  # Last node!
                this_cut = minimum_st_edge_cut(H, nodes[i], nodes[0], **kwargs)
                if len(this_cut) <= len(min_cut):
                    min_cut = this_cut

        return min_cut

    else:  # undirected
        # Based on algorithm 6 in [1]
        if not nx.is_connected(G):
            raise nx.NetworkXError("Input graph is not connected")

        # Initial cutset is all edges of a node with minimum degree
        node = min(G, key=G.degree)
        min_cut = set(G.edges(node))
        # A dominating set is \lambda-covering
        # We need a dominating set with at least two nodes
        for node in G:
            D = nx.dominating_set(G, start_with=node)
            v = D.pop()
            if D:
                break
        else:
            # in complete graphs the dominating set will always be of one node
            # thus we return min_cut, which now contains the edges of a node
            # with minimum degree
            return min_cut
        for w in D:
            this_cut = minimum_st_edge_cut(H, v, w, **kwargs)
            if len(this_cut) <= len(min_cut):
                min_cut = this_cut

        return min_cut
G_undirected = nx.Graph({0: [1,3], 1: [2], 2: [3,1], 3: []})


