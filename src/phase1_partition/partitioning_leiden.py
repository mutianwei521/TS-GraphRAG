"""
Network partitioning using Leiden algorithm.
Leiden is an improved version of Louvain, guaranteeing connected communities and stable convergence.
"""
import numpy as np
import networkx as nx
from cdlib import algorithms


def normalize_edge_weights(G):
    """
    Normalize edge weights to [0, 1] range.

    Args:
        G: NetworkX graph with edge weights

    Returns:
        G_normalized: New graph with normalized edge weights
    """
    G_normalized = G.copy()

    # Get all edge weights
    weights = [data.get('weight', 1.0) for u, v, data in G.edges(data=True)]

    if len(weights) == 0:
        return G_normalized

    min_weight = min(weights)
    max_weight = max(weights)

    # Normalize weights to [0, 1] range
    if max_weight > min_weight:
        for u, v, data in G_normalized.edges(data=True):
            original_weight = data.get('weight', 1.0)
            normalized_weight = (original_weight - min_weight) / (max_weight - min_weight)
            # Avoid zero weights (may cause issues with some algorithms)
            G_normalized[u][v]['weight'] = max(normalized_weight, 0.001)

    return G_normalized


def run_leiden_partitioning(G, resolution_range=None, num_iterations=50):
    """
    Run Leiden algorithm multiple times with different resolutions to get various partition sizes.

    Uses leidenalg library directly (via igraph) which supports resolution_parameter.

    Args:
        G: NetworkX graph
        resolution_range: Tuple of (min, max) for resolution parameter
        num_iterations: Number of different resolution values to try

    Returns:
        all_partitions: List of (num_communities, node_to_community, resolution) tuples
    """
    import igraph as ig
    import leidenalg

    if resolution_range is None:
        resolution_range = (0.1, 5.0)

    # Normalize edge weights before partitioning
    G_normalized = normalize_edge_weights(G)
    print(f"  Edge weights normalized to [0, 1] range")

    # Convert NetworkX graph to igraph graph
    # Preserve node name mapping
    node_list = list(G_normalized.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    ig_graph = ig.Graph(directed=False)
    ig_graph.add_vertices(len(node_list))
    ig_graph.vs['name'] = node_list

    edges = []
    weights = []
    for u, v, data in G_normalized.edges(data=True):
        edges.append((node_to_idx[u], node_to_idx[v]))
        weights.append(data.get('weight', 1.0))

    ig_graph.add_edges(edges)
    ig_graph.es['weight'] = weights

    print(f"  Converted to igraph: {ig_graph.vcount()} vertices, {ig_graph.ecount()} edges")

    all_partitions = []
    resolutions = np.linspace(resolution_range[0], resolution_range[1], num_iterations)

    for res in resolutions:
        try:
            # Run Leiden directly using leidenalg
            partition = leidenalg.find_partition(
                ig_graph,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=res
            )
            num_communities = len(partition)

            # Store partition result with node assignments (using original node names)
            node_to_community = {}
            for comm_id, community in enumerate(partition):
                for node_idx in community:
                    node_name = node_list[node_idx]
                    node_to_community[node_name] = comm_id

            all_partitions.append((num_communities, node_to_community, res))
        except Exception as e:
            print(f"Warning: Leiden failed at resolution {res}: {e}")
            continue

    return all_partitions


def extract_unique_partitions(all_partitions):
    """
    Extract unique partition counts, keeping only the last occurrence (optimal in iterations) for each partition count.

    Args:
        all_partitions: List of (num_communities, node_to_community, resolution) tuples

    Returns:
        unique_partitions: Dictionary {num_communities: node_to_community}
    """
    unique_partitions = {}

    # Iterate through all partitions, later ones will override earlier ones
    for num_comm, node_to_comm, res in all_partitions:
        unique_partitions[num_comm] = {
            'node_to_community': node_to_comm,
            'resolution': res
        }

    # Sort by number of communities
    sorted_partitions = dict(sorted(unique_partitions.items()))

    return sorted_partitions


def merge_communities_by_connectivity(G, node_to_community, target_num):
    """
    Merge the most tightly connected community pairs until reaching the target number.

    Args:
        G: NetworkX graph
        node_to_community: Dictionary mapping node names to community IDs
        target_num: Target number of communities

    Returns:
        new_node_to_community: Dictionary containing merged community assignments
    """
    current_communities = max(node_to_community.values()) + 1

    if current_communities <= target_num:
        return node_to_community

    # Create a copy to modify
    node_to_comm = node_to_community.copy()

    while current_communities > target_num:
        # Calculate connectivity between communities
        comm_connectivity = {}
        for u, v, data in G.edges(data=True):
            comm_u = node_to_comm.get(u)
            comm_v = node_to_comm.get(v)
            if comm_u is not None and comm_v is not None and comm_u != comm_v:
                pair = tuple(sorted([comm_u, comm_v]))
                weight = data.get('weight', 1.0)
                comm_connectivity[pair] = comm_connectivity.get(pair, 0) + weight

        if not comm_connectivity:
            break

        # Find the most tightly connected pair of communities
        best_pair = max(comm_connectivity, key=comm_connectivity.get)
        comm_to_merge, comm_to_keep = best_pair

        # Merge: reassign all nodes from comm_to_merge to comm_to_keep
        for node in node_to_comm:
            if node_to_comm[node] == comm_to_merge:
                node_to_comm[node] = comm_to_keep

        # Renumber communities to be contiguous
        unique_comms = sorted(set(node_to_comm.values()))
        comm_mapping = {old: new for new, old in enumerate(unique_comms)}
        node_to_comm = {node: comm_mapping[comm] for node, comm in node_to_comm.items()}

        current_communities = len(unique_comms)

    return node_to_comm


def generate_merged_partitions(G, base_partition, target_range=(2, 15)):
    """
    Generate partitions with fewer communities by merging from a base partition.

    Args:
        G: NetworkX graph
        base_partition: Dict {node: community_id} - base partition for merging
        target_range: Tuple (min, max) for target community counts

    Returns:
        merged_partitions: Dict {num_communities: {'node_to_community': dict, 'resolution': 'merged'}}
    """
    merged_partitions = {}

    for target_num in range(target_range[0], target_range[1] + 1):
        merged = merge_communities_by_connectivity(G, base_partition, target_num)
        actual_num = len(set(merged.values()))

        if actual_num not in merged_partitions:
            merged_partitions[actual_num] = {
                'node_to_community': merged,
                'resolution': 'merged'
            }

    return merged_partitions


def extract_partitions_with_merge(G, all_partitions, merge_range=(2, 15), target_k=None):
    """
    Extract unique partitions from Leiden results and generate merged partitions for smaller community counts.
    Ensures target_k exists if specified.

    Args:
        G: NetworkX graph
        all_partitions: List of (num_communities, node_to_community, resolution) tuples
        merge_range: Tuple (min, max) for merge target range
        target_k: Specific partition count required

    Returns:
        unique_partitions: Combined dictionary of Leiden and merged partitions
    """
    # Step 1: Extract unique Leiden partitions
    unique_partitions = extract_unique_partitions(all_partitions)
    partition_counts = sorted(unique_partitions.keys())

    print(f"  Leiden partition counts: {partition_counts}")
    print(f"  Leiden range: {min(partition_counts, default=0)} to {max(partition_counts, default=0)} communities")

    # Helper to find valid base partition needed to merge down to target count
    def find_best_base(target):
        # We need a partition with count greater than target
        candidates = [k for k in partition_counts if k > target]
        if not candidates:
            return None
        # Select the smallest count greater than target (i.e., closest upper neighbor)
        best_k = min(candidates)
        return unique_partitions[best_k]['node_to_community']

    # Step 2: Generate merged partitions for smaller community counts (default range)
    print(f"\n  Generating merged partitions for {merge_range[0]}-{merge_range[1]} communities...")

    targets_to_gen = set(range(merge_range[0], merge_range[1] + 1))
    if target_k is not None:
        targets_to_gen.add(target_k)

    merged_partitions = {}

    for t_k in sorted(list(targets_to_gen)):
        if t_k in unique_partitions:
            continue  # Already exists in Leiden results

        base_partition = find_best_base(t_k)
        if base_partition:
            merged = merge_communities_by_connectivity(G, base_partition, t_k)
            actual_num = len(set(merged.values()))
            if actual_num == t_k:
                merged_partitions[actual_num] = {
                    'node_to_community': merged,
                    'resolution': 'merged'
                }

    merged_counts = sorted(merged_partitions.keys())
    print(f"  Merged partition counts: {merged_counts}")

    # Step 3: Merge back into main dict
    for num_comm, data in merged_partitions.items():
        unique_partitions[num_comm] = data

    # Re-sort
    unique_partitions = dict(sorted(unique_partitions.items()))
    partition_counts = sorted(unique_partitions.keys())

    print(f"  Final partition counts: {partition_counts}")
    if partition_counts:
        print(f"  Final range: {min(partition_counts)} to {max(partition_counts)} communities")

    return unique_partitions
