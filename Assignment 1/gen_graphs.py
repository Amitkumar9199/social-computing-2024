# Name : Amit Kumar
# Roll : 20CS30003

import snap
Rnd = snap.TRnd(42)
Rnd.Randomize()

import os

# # Function to save the graph to an edge list file
def save_signed_graph(nodes,edges, file_path):
    with open(file_path, 'w') as f:
        f.write("# graph: {}\n".format(file_path))
        f.write("# Nodes: {}\tEdges: {}\n".format(len(nodes), len(edges)))
        f.write("# FromNodeId\tToNodeId\tSign\n")
        for edge in edges:
            from_node = edge[0]
            to_node = edge[1]
            sign = edge[2]
            f.write(f"{from_node}\t{to_node}\t{sign}\n")
    return

# Function to process the epinions graph
def process_epinions_graph(file_path):
    edges = []
    nodes = set()
    # Add nodes and edges from the file, along with the sign
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip comment lines
            from_node, to_node, sign = map(int, line.strip().split())
            
            if ((from_node % 5) != 0) or ((to_node % 5) != 0):
                continue
            
            nodes.add(from_node)
            nodes.add(to_node)
            edges.append((from_node, to_node, sign))
    
    return nodes,edges



# # Function to save the graph to an edge list file
def save_facebook_graph(nodes,edges, file_path):
    with open(file_path, 'w') as f:
        f.write("# graph: {}\n".format(file_path))
        f.write("# Nodes: {}\tEdges: {}\n".format(len(nodes), len(edges)))
        f.write("# FromNodeId\tToNodeId\n")
        for edge in edges:
            from_node = edge[0]
            to_node = edge[1]
            f.write(f"{from_node}\t{to_node}\n")
    return

# Function to process the facebook graph
def process_facebook_graph(file_path):
    edges = []
    nodes = set()
    # Add nodes and edges from the file, along with the sign
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip comment lines
            from_node, to_node = map(int, line.strip().split())
            
            if ((from_node % 4) == 0) or ((to_node % 4) == 0):
                continue
            
            nodes.add(from_node)
            nodes.add(to_node)
            edges.append((from_node, to_node))
    
    return nodes,edges


# Function to write the graph to an edge list file
def save_graph(graph, file_path):
    snap.SaveEdgeList(graph, file_path)


# Random graph generation
def create_random_graph(num_nodes, num_edges):
    return snap.GenRndGnm(snap.PUNGraph, num_nodes, num_edges, False, Rnd)

# Small-World graph generation
def create_small_world_graph(num_nodes, node_degree, rewire_prob):
    return snap.GenSmallWorld(num_nodes, node_degree, rewire_prob, Rnd)


def main():
    # mkdir networks if it doesn't exist
    if not os.path.exists('networks'):
        os.makedirs('networks')
    
    # facebook
    nodes,edges = process_facebook_graph('./input_graphs/facebook_combined.txt')
    save_facebook_graph(nodes,edges, 'networks/facebook.elist')

    # epinions
    nodes,edges = process_epinions_graph('./input_graphs/soc-sign-epinions.txt')
    save_signed_graph(nodes, edges, 'networks/epinions.elist')

    # Random graph with 1000 nodes, 50000 edges
    random_graph = create_random_graph(1000, 50000)
    save_graph(random_graph, 'networks/random.elist')

    # Small-world graph with 1000 nodes, degree of 50, rewire probability 0.6
    small_world_graph = create_small_world_graph(1000, 50, 0.6)
    save_graph(small_world_graph, 'networks/smallworld.elist')


if __name__ == "__main__":
    main()