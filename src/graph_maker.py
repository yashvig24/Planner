import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from MapEnvironment import MapEnvironment


assert(nx.__version__ == '2.2' or nx.__version__ == '2.1')

def load_graph(filename):
    assert os.path.exists(filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        print('Loaded graph from {}'.format(f))
    return data['G']

def make_graph(env, sampler, connection_radius, num_vertices, lazy=False, saveto='graph.pkl'):
    """
    Returns a graph on the passed environment.
    All vertices in the graph must be collision-free.

    Graph should have node attribute "config" which keeps a configuration in tuple.
    E.g., for adding vertex "0" with configuration np.array([0, 1]),
    G.add_node(0, config=tuple(config))

    To add edges to the graph, call
    G.add_weighted_edges_from(edges)
    where edges is a list of tuples (node_i, node_j, weight),
    where weight is the distance between the two nodes.

    @param env: Map Environment for graph to be made on
    @param sampler: Sampler to sample configurations in the environment
    @param connection_radius: Maximum distance to connect vertices
    @param num_vertices: Minimum number of vertices in the graph.
    @param lazy: If true, edges are made without checking collision.
    @param saveto: File to save graph and the configurations
    """
    G = nx.Graph()
    vertices_to_add = num_vertices
    final_config = []

    # Implement here
    # 1. Sample vertices
    while vertices_to_add != 0:
        curr_config = sampler.sample(vertices_to_add)
        validity = env.state_validity_checker(curr_config)
        valid_config = curr_config[validity]
        final_config.extend(valid_config)
        vertices_added = np.sum(validity)
        vertices_to_add -= vertices_added
    
    for i in range(num_vertices):
        G.add_node(i, config=tuple(final_config[i]))

    edges = []
    # 2. Connect them with edges
    for i in range(num_vertices):
        curr_node = final_config[i]
        for j in range(num_vertices):
            if j == i:
                break
            neigh = final_config[j]
            valid, dist = env.edge_validity_checker(curr_node, neigh)
            if valid and dist <= connection_radius:
                tup = (i, j, dist)
                edges.append(tup)
    
    G.add_weighted_edges_from(edges)

    # Check for connectivity.
    num_connected_components = len(list(nx.connected_components(G)))
    if not num_connected_components == 1:
        print ("warning, Graph has {} components, not connected".format(num_connected_components))

    # Save the graph to reuse.
    if saveto is not None:
        data = dict(G=G)
        pickle.dump(data, open(saveto, 'wb'))
        print('Saved the graph to {}'.format(saveto))
    return G


def add_node(G, config, env, connection_radius):
    """
    This function should add a node to an existing graph G.
    @param G graph, constructed using make_graph
    @param config Configuration to add to the graph
    @param env Environment on which the graph is constructed
    @param connection_radius Maximum distance to connect vertices
    """
    # new index of the configuration
    index = G.number_of_nodes()
    G.add_node(index, config=tuple(config))
    G_configs = nx.get_node_attributes(G, 'config')
    G_configs = [G_configs[node] for node in G_configs]

    # Implement here
    # Add edges from the newly added node
    edges = []
    curr_node = np.array(list(G_configs[index]))
    for j in range(index):
        neigh = np.array(list(G_configs[j]))
        valid, dist = env.edge_validity_checker(curr_node, neigh)
        if valid and (dist <= connection_radius):
            tup = (index, j, dist)
            edges.append(tup)

    G.add_weighted_edges_from(edges)

    # Check for connectivity.
    num_connected_components = len(list(nx.connected_components(G)))
    if not num_connected_components == 1:
        print ("warning, Graph has {} components, not connected".format(num_connected_components))

    return G, index
