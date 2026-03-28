"""
Build similarity matrix for network partitioning.
Based on linkNode method in references/cd_main.m.
"""
import numpy as np
import networkx as nx


def build_similarity_matrix(wn, avg_pressure):
    """
    Build similarity matrix A, where A[i,j] = (avg_pressure[i] + avg_pressure[j]) / 2
    This follows the MATLAB routine cd_main.m using linkNode logic.
    
    Args:
        wn: WNTR water network model
        avg_pressure: Average pressure dictionary for each node
        
    Returns:
        G: NetworkX graph with edge weights based on pressure similarity
        node_list: List of node names
    """
    # Get all nodes (including tanks, reservoirs) to ensure connectivity in visualization
    junction_names = list(wn.junction_name_list)
    node_list = list(wn.node_name_list)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes
    for node in node_list:
        G.add_node(node)
    
    # Add edges based on pipes/links (similar to MATLAB linkNode)
    # For each link, add an edge with weight as the average of connected node pressures
    for link_name in wn.link_name_list:
        link = wn.get_link(link_name)
        start_node = link.start_node_name
        end_node = link.end_node_name
        
        # Consider edges between all nodes (including reservoirs/tanks)
        if start_node in node_list and end_node in node_list:
            # Calculate weight as average of node pressures (follows MATLAB pattern)
            if start_node in avg_pressure and end_node in avg_pressure:
                weight = (avg_pressure[start_node] + avg_pressure[end_node]) / 2
                # Handle negative pressures if necessary
                # Avoid zero values to prevent algorithmic errors
                G.add_edge(start_node, end_node, weight=weight)
    
    return G, node_list


def create_network_graph(wn, avg_pressure):
    """
    Create NetworkX graph with position information for partitioning.
    
    Args:
        wn: WNTR water network model
        avg_pressure: Average pressure dictionary
        
    Returns:
        G: NetworkX graph with weights and position information
        pos: Node positions dictionary
    """
    G, node_list = build_similarity_matrix(wn, avg_pressure)
    
    # Get node positions from network coordinates
    pos = {}
    for node_name in node_list:
        node = wn.get_node(node_name)
        pos[node_name] = (node.coordinates[0], node.coordinates[1])
    
    return G, pos

