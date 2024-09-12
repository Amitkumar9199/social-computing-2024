# Name : Amit Kumar
# Roll : 20CS30003

import snap
Rnd = snap.TRnd(42)
Rnd.Randomize()

import sys
import matplotlib.pyplot as plt
import os

def load_graph(file_path):
    """
    load the graph from the edge list file
    """

    # Load the undirected graph from an edge list
    graph = snap.LoadEdgeList(snap.PUNGraph, file_path, 0, 1)
    return graph

# 1. Size of the network
def print_network_size(graph):
    """
    print the number of nodes and edges in the network
    """

    # (a) Number of nodes
    num_nodes = graph.GetNodes()
    print(f"Number of nodes: {num_nodes}")

    # (b) Number of edges
    num_edges = graph.GetEdges()
    print(f"Number of edges: {num_edges}")


# 2. Degree of nodes in the network
def print_degree_info(graph):
    """
    print the number of nodes with degree = 7 and the node id(s) with the highest degree
    """

    # node_degrees: dictionary to store the degree of each node
    # used to plot the degree distribution
    node_degrees = {}
    for node in graph.Nodes():
        degree = node.GetDeg()
        node_degrees[node.GetId()] = degree

    # using SNAP's built-in functions
    # (a) Number of nodes which have degree = 7
    print(f"Number of nodes with degree=7: {snap.CntDegNodes(graph, 7)}")

    # (b) Node id(s) for the node with the highest degree. Note that there might be multiple nodes with highest degree
    max_degree_node_id = graph.GetMxDegNId() 
    max_degree = graph.GetNI(max_degree_node_id).GetDeg()
    highest_degree_nodes = []
    for NI in graph.Nodes(): # Iterate over all nodes
        if NI.GetDeg() == max_degree:  # Check if the node's degree equals the max degree
            highest_degree_nodes.append(NI.GetId())

    print(f"Node id(s) with highest degree: {', '.join(map(str, highest_degree_nodes))}")

    return node_degrees

# 2. Degree of nodes in the network
def plot_degree_distribution(graph, node_degrees, output_file, subgraph_name):
    """
    plot the distribution of degrees
    """

    # (c) Plot of the Degree distribution
    # using SNAP's built-in function
    snap.PlotOutDegDistr(graph, subgraph_name, "Degree Distribution for {}".format(subgraph_name))
    os.rename(f'outDeg.{subgraph_name}.png', f'./plots/deg_dist_{subgraph_name}.png')
    os.remove(f'outDeg.{subgraph_name}.plt')
    os.remove(f'outDeg.{subgraph_name}.tab')
    
    # using matplotlib to plot the degree distribution
    # Calculate the degree distribution
    degree_freq = {}
    for degree in node_degrees.values():
        if degree in degree_freq:
            degree_freq[degree] += 1
        else:
            degree_freq[degree] = 1

    # Extract the degrees and frequencies as lists
    degrees = list(degree_freq.keys())
    frequencies = list(degree_freq.values())

    # Plot the degree distribution
    plt.figure()
    plt.scatter(degrees, frequencies, color='blue', s=10)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution for {}'.format(subgraph_name))
    plt.savefig(output_file)
    plt.close()

# 3. Paths in the network
def compute_diameters(graph, num_samples=1000):
    """
    print the approximate full diameter and effective diameter of the network
    """

    # (a) Approximate full diameter (maximum shortest path length) starting from 1000 random test nodes.
    full_diam = graph.GetBfsFullDiam(num_samples, False)
    # assert(full_diam == snap.GetBfsFullDiam(graph, num_samples, False))
    print(f"Approximate full diameter: {full_diam:.4f}")

    # (b) Approximate effective diameter computed starting from 1000 random test nodes.
    eff_diam = graph.GetBfsEffDiam(num_samples, False)
    # assert(eff_diam == snap.GetBfsEffDiam(graph, num_samples, False))
    print(f"Approximate effective diameter: {eff_diam:.4f}")


# 3. Paths in the network
def plot_shortest_path_distribution(graph, output_file, subgraph_name):
    """
    plot the distribution of shortest path lengths
    """

    # (c) Plot of the distribution of the shortest path lengths in the network.
    # using SNAP's built-in function
    snap.PlotShortPathDistr(graph, subgraph_name, f"Shortest Path Length Distribution for {subgraph_name}")
    os.rename(f'diam.{subgraph_name}.png', f'./plots/shortest_path_{subgraph_name}.png')
    os.remove(f'diam.{subgraph_name}.plt')
    os.remove(f'diam.{subgraph_name}.tab')


    # using matplotlib to plot the shortest path length distribution
    dist_freq = {}
    
    # Calculate the shortest path lengths for all pairs of nodes
    for node in graph.Nodes():
        NIdToDistH = snap.TIntH()
        snap.GetShortPath(graph, node.GetId(), NIdToDistH, False)
        
        for dist in NIdToDistH:
            dist_val = NIdToDistH[dist]
            if dist_val in dist_freq:
                dist_freq[dist_val] += 1
            else:
                dist_freq[dist_val] = 1

    # Extract the distances and frequencies as lists
    distances = list(dist_freq.keys())
    frequencies = list(dist_freq.values())

    # Plot the shortest path length distribution
    plt.figure()
    plt.scatter(distances, frequencies, color='green', s=10)
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Frequency')
    plt.title('Shortest Path Length Distribution for {}'.format(subgraph_name))
    plt.savefig(output_file)
    plt.close()

# 4. Analyse connected components
def components_Analysis(graph):
    """
    print the fraction of nodes in the largest connected component, number of edge bridges, number of articulation points
    """

    # (a) Fraction of nodes in the largest connected component
    fraction = snap.GetMxWccSz(graph)
    print(f"Fraction of nodes in largest connected component: {fraction:.4f}")

    # (b) Number of edge bridges: An edge is a bridge if, when removed, increases the number of connected components.
    EdgeV = snap.TIntPrV()
    snap.GetEdgeBridges(graph, EdgeV)
    edge_bridges_count = EdgeV.Len()
    print(f"Number of edge bridges: {edge_bridges_count}")

    # (c) Number of articulation points: A node is a articulation point if, when removed, increases the number of connected components.
    ArtNIdV = snap.TIntV()
    snap.GetArtPoints(graph, ArtNIdV)
    articulation_points_count =  ArtNIdV.Len()
    print(f"Number of articulation points: {articulation_points_count}")

# 4. Analyse connected components
def plot_connected_component_distribution(graph, output_file, subgraph_name):
    """
    plot the distribution of connected components
    """

    # (d) Plot of the distribution of sizes of connected components
    # using SNAP's built-in function
    snap.PlotSccDistr(graph, subgraph_name, "Connected Component Size Distribution for {}".format(subgraph_name))
    os.rename(f'scc.{subgraph_name}.png', f'./plots/connected_comp_{subgraph_name}.png')
    os.remove(f'scc.{subgraph_name}.plt')
    os.remove(f'scc.{subgraph_name}.tab')

    # using matplotlib to plot the connected component size distribution
    # Get sizes of all connected components and their frequencies
    components = snap.TCnComV()
    snap.GetWccs(graph, components)  # Get all weakly connected components
    comp_sizes = [comp.Len() for comp in components]
    size_counts = {}
    for size in comp_sizes:
        if size in size_counts:
            size_counts[size] += 1
        else:
            size_counts[size] = 1
    
    # Extract the sizes and frequencies as lists
    sizes = list(size_counts.keys())
    frequencies = list(size_counts.values())
    
    # Plot the distribution of connected component sizes using a scatter plot
    plt.figure()
    plt.scatter(sizes, frequencies, color='purple', s=15)
    plt.xlabel('Component Size')
    plt.ylabel('Frequency')
    plt.title('Connected Component Size Distribution for {}'.format(subgraph_name))
    plt.savefig(output_file)
    plt.close()

# 5. Connectivity and clustering in the network 
def analyze_clustering(graph):
    """
    print the average clustering coefficient, number of triads, clustering coefficient of a randomly selected node,
    """

    # (a) Average clustering coefficient of the network
    CfVec = snap.TFltPrV()
    avg_clustering_coeff = snap.GetClustCf(graph, CfVec, -1)  # -1 means for the whole graph
    print(f"Average clustering coefficient: {avg_clustering_coeff:.4f}")

    # (b) Number of triads
    num_triads = graph.GetTriads(-1)  # -1 means for the whole graph
    print(f"Number of triads: {num_triads}")

    # (c) Clustering coefficient of a randomly selected node. Also report the selected node id.
    rnd_node_id = graph.GetRndNId()  # Get a random node ID
    node_clustering_coeff = graph.GetNodeClustCf(rnd_node_id)
    print(f"Clustering coefficient of random node {rnd_node_id}: {node_clustering_coeff:.4f}")

    # (d) Number of triads a randomly selected node participates in. Also report the selected node id.
    rnd_node_id = graph.GetRndNId()  # Get a random node ID
    num_triads_node = graph.GetNodeTriads(rnd_node_id)
    print(f"Number of triads random node {rnd_node_id} participates: {num_triads_node}")

# 5. Connectivity and clustering in the network 
def plot_clustering_coefficient_distribution(graph, output_file, subgraph_name):
    """
    plot the distribution of clustering coefficient
    """

    # (e) Plot of the distribution of clustering coefficient
    # using built-in function
    snap.PlotClustCf(graph, subgraph_name, "Clustering Coefficient Distribution for {}".format(subgraph_name))
    os.rename(f'ccf.{subgraph_name}.png', f'./plots/clustering_coeff_{subgraph_name}.png')
    os.remove(f'ccf.{subgraph_name}.plt')
    os.remove(f'ccf.{subgraph_name}.tab')

    # using matplotlib to plot the clustering coefficient distribution
    # Calculate clustering coefficients for all nodes
    cf_dist = snap.TFltPrV()
    coeff = snap.GetClustCf(graph, cf_dist, -1)
    degree_clust_coeffs = {}
    for pair in cf_dist:
        degree_clust_coeffs[pair.GetVal1()] = pair.GetVal2()
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.scatter(list(degree_clust_coeffs.keys()), list(degree_clust_coeffs.values()), color='orange', s=10)
    plt.xlabel('Degree')
    plt.ylabel('clustering coefficient')
    plt.title('Clustering Coefficient Distribution for {}'.format(subgraph_name))
    plt.savefig(output_file)
    plt.close()


# (a) Degree centrality
def get_top_degree_centrality(graph, top_n=5):
    """
    Get the top N nodes by degree centrality.
    """

    degrees = []
    for node in graph.Nodes(): # get degree centrality of each node
        degrees.append((node.GetId(), graph.GetDegreeCentr(node.GetId())))
    
    # Sort nodes by degree in descending order
    degrees = sorted(degrees, key=lambda x: x[1], reverse=True)
    
    # Get top N node ids
    top_nodes = [node_id for node_id, degree in degrees[:top_n]]
    return top_nodes

# (b) Closeness centrality
def get_top_closeness_centrality(graph, top_n=5):
    """
    Get the top N nodes by closeness centrality.
    """

    closeness = []
    for node in graph.Nodes():
        node_id = node.GetId()
        # Calculate closeness centrality using SNAP's GetClosenessCentr
        closeness_value = snap.GetClosenessCentr(graph, node_id, True, False)
        closeness.append((node_id, closeness_value))
    
    # Sort nodes by closeness in descending order
    closeness = sorted(closeness, key=lambda x: x[1], reverse=True)
    
    # Get top N node ids
    top_nodes = [node_id for node_id, _ in closeness[:top_n]]
    return top_nodes

# (c) Betweenness centrality
def get_top_betweenness_centrality(graph, top_n=5):
    """
    Get the top N nodes by betweenness centrality.
    """

    # Calculate betweenness centrality for all nodes
    Nodes, Edges = graph.GetBetweennessCentr(1.0, False)

    # Extract node ids and their centrality values
    betweenness = [(node_id, Nodes[node_id]) for node_id in Nodes]

    # Sort nodes by betweenness centrality in descending order
    betweenness = sorted(betweenness, key=lambda x: x[1], reverse=True)
    
    # Get top N node ids
    top_nodes = [node_id for node_id, _ in betweenness[:top_n]]
    return top_nodes

# 6. Centrality metrics of the network 
def centrality_measures(graph):
    """
    Print the top 5 nodes by degree, closeness, and betweenness centrality.
    """

    # (a) Degree centrality
    top_degree_centrality = get_top_degree_centrality(graph)
    print(f"Top 5 nodes by degree centrality: {' '.join(map(str, top_degree_centrality))}")
    
    # (b) Closeness centrality
    top_closeness_centrality = get_top_closeness_centrality(graph)
    print(f"Top 5 nodes by closeness centrality: {' '.join(map(str, top_closeness_centrality))}")
    
    # (c) Betweenness centrality
    top_betweenness_centrality = get_top_betweenness_centrality(graph)
    print(f"Top 5 nodes by betweenness centrality: {' '.join(map(str, top_betweenness_centrality))}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_structure.py <path_to_subgraph>")
        sys.exit(1)
    
    # mkdir -p plots
    os.makedirs("plots", exist_ok=True)
    os.makedirs("mat_plots", exist_ok=True)

    # Read the graph path from command line argument
    file_path = sys.argv[1]
    subgraph_name = os.path.basename(file_path).split('.')[0]

    # Load the graph
    graph = load_graph(file_path)
    
    # 1. Size of the network 
    print_network_size(graph)

    # 2. Degree of nodes in the network 
    node_degrees = print_degree_info(graph)
    plot_file = f"mat_plots/deg_dist_{subgraph_name}.png"
    plot_degree_distribution(graph, node_degrees, plot_file, subgraph_name)

    # 3. Paths in the network
    compute_diameters(graph)
    shortest_path_plot_file = f"mat_plots/shortest_path_{subgraph_name}.png"
    plot_shortest_path_distribution(graph, shortest_path_plot_file, subgraph_name)
    
    # 4. Analyse connected components
    components_Analysis(graph)
    output_file = f"mat_plots/connected_comp_{subgraph_name}.png"
    plot_connected_component_distribution(graph, output_file, subgraph_name)

    # 5. Connectivity and clustering in the network 
    analyze_clustering(graph)
    output_file = f"mat_plots/clustering_coeff_{subgraph_name}.png"
    plot_clustering_coefficient_distribution(graph, output_file, subgraph_name)

    # 6. Centrality metrics of the network 
    centrality_measures(graph)
    
if __name__ == "__main__":
    main()

