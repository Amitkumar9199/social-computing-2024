# Name : Amit Kumar
# Roll : 20CS30003

import snap
Rnd = snap.TRnd(42)
Rnd.Randomize()

import os
import sys
import numpy as np

# Function to load graph from an edge list with weights
def load_signed_graph(file_path):
    """
    loads a signed graph from an edge list with weights
    """

    # open the file and read the lines
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()

    nodes = set() # set of nodes
    for line in lines:
        if line.startswith('#'):
            continue
        from_node, to_node, sign = map(int, line.strip().split())
        nodes.add(from_node)
        nodes.add(to_node)

    nodes = list(nodes)
    nodes.sort()
    num_nodes = len(nodes)

    # adjacency list, why map , because nodes values can be greater than num_nodes
    adjacency_list = {node: [] for node in nodes}
    signed_adjacency_list = {node: [] for node in nodes}
    for line in lines:
        if line.startswith('#'):
            continue
        from_node, to_node, sign = map(int, line.strip().split())
        adjacency_list[from_node].append(to_node)
        adjacency_list[to_node].append(from_node)
        signed_adjacency_list[from_node].append((to_node, sign))
        signed_adjacency_list[to_node].append((from_node, sign))

    return num_nodes, nodes, adjacency_list, signed_adjacency_list

# Load the undirected graph from an edge list
def load_graph(file_path):
    """
    loads an undirected graph from an edge list
    """

    # open the file and read the lines
    file = open(file_path, 'r')
    lines = file.readlines()
    file.close()

    nodes = set() # set of nodes
    for line in lines:
        if line.startswith('#'):
            continue
        from_node, to_node = map(int, line.strip().split())
        nodes.add(from_node)
        nodes.add(to_node)

    nodes = list(nodes)
    nodes.sort()
    num_nodes = len(nodes)

    # adjacency list, why map , because nodes values can be greater than num_nodes
    adjacency_list = {node: [] for node in nodes}
    signed_adjacency_list = {node: [] for node in nodes}
    for line in lines:
        if line.startswith('#'):
            continue
        from_node, to_node = map(int, line.strip().split())
        adjacency_list[from_node].append(to_node)
        adjacency_list[to_node].append(from_node)
        signed_adjacency_list[from_node].append((to_node, 1))
        signed_adjacency_list[to_node].append((from_node, 1))

    return num_nodes, nodes, adjacency_list, signed_adjacency_list


# utility function to compute shortest paths using BFS
def bfs_shortest_paths(num_nodes, nodes, adjacency_list, start):
    """
    computes shortest paths from a source node to all other nodes using BFS
    """
    # initialize distances
    distances = {node: float('inf') for node in nodes}
    distances[start] = 0
    queue = [start]
    dist_sum=0
    cnt = 1  # count of nodes visited
    # BFS to compute shortest paths
    while len(queue) > 0:
        current = queue.pop(0)
        current_distance = distances[current]
        
        for neighbor in adjacency_list[current]:
            # if the neighbor has not been visited
            if distances[neighbor] == float('inf'):
                distances[neighbor] = current_distance + 1
                dist_sum += distances[neighbor]
                cnt += 1
                queue.append(neighbor)
    
    # return the sum of shortest paths and the count of nodes visited
    return dist_sum, cnt

# 1. Closeness centrality 
def closeness_centrality(num_nodes, nodes, adjacency_list):
    """
    calculates and stores closeness centrality values for all nodes in the graph
    
    heuristic: 
        1.  As per assignment, closeness centrality(node) = (n - 1) / total_distance
            But we need a normalisation facter, because we don't visit all nodes (graph may be disconnected)
            So, let centrality = (cnt-1)/total_distance, where cnt is the number of nodes visited
            then for normalisation, centrality = centrality*(cnt-1)/(n-1)
            this way if cnt is n, then centrality = (n-1)/total_distance
            otherwise centrality will get normalised as per the number of nodes visited

        2. if total_distance is 0, closeness centrality is 0
    """

    centrality = {}
    n = num_nodes
    
    for node in nodes:
        # compute shortest paths using BFS
        total_distance, cnt = bfs_shortest_paths(num_nodes, nodes, adjacency_list, node)
        ''' 
        # compute closeness centrality
        # centrality = (cnt-1)/total_distance
        # but we don't visit all nodes, so we need to normalize it
        # centrality = centrality*(cnt-1)/(n-1)
        # we can also see that, if cnt == n, then centrality = (n-1)/total_distance
        '''
        if total_distance > 0 and cnt > 0:
            centrality[node] = ((cnt - 1) / total_distance) * (cnt - 1) / (n - 1)
        else:
            centrality[node] = 0

    sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    # write to file, decimal precision 4
    with open(f"centralities/closeness.txt", "w") as file:
        for node, value in sorted_centrality:
            file.write(f"{node} {value:.4f}\n")

    return 

# 2. Betweenness centrality 
def betweenness_centrality(num_nodes, nodes, adjacency_list):
    """
    Calculates and stores betweenness centrality values for all nodes in the graph using Brandes' Algorithm.
    
    Heuristic:
        Uses Brandes' Algorithm to compute betweenness centrality values for all nodes in the graph.
        Currently, the fastest known algorithms require Θ(n^3) time and Θ(n^2) space, where n is the 
        number of actors in the network. 
        Brandes require O(n + m) space and run in O(n*m) and O(n*m + n^2 * log n) time on unweighted and 
        weighted networks, respectively, where m is the number of links. Experimental evidence is 
        provided that this substantially increases the range of networks for which centrality 
        analysis is feasible.
    Note: I have attached the reference paper(for algorithm) in the submission zip file.
    """

    # compute betweenness centralities using Brandes' Algorithm
    # Initialize betweenness centrality dictionary for all nodes with 0
    betweenness_centralities = {node: 0 for node in nodes}
    
    for node in nodes:
        stack = []                              # Stack to store the order of nodes processed
        paths = {node: [] for node in nodes}    # Stores shortest paths
        number_of_paths = {n:0 for n in nodes}  # Number of shortest paths to each node
        number_of_paths[node] = 1               # There is one shortest path from node to itself
        distances = {n:-1 for n in nodes}       # Distance from source node to other nodes
        distances[node] = 0 
        queue = []                              # Queue for BFS

        queue.append(node)
        # BFS to calculate shortest paths from the source node
        while len(queue) > 0:
            vertex = queue.pop(0)
            stack.append(vertex)                # Add node to stack after processing
            for node2 in adjacency_list[vertex]:
                # If neighbor has not been visited (distance < 0)
                if distances[node2] < 0:
                    queue.append(node2)
                    distances[node2] = distances[vertex] + 1
                # If the shortest path includes this vertex
                if distances[node2] == distances[vertex] + 1:
                    number_of_paths[node2] += number_of_paths[vertex] # Increment number of paths
                    paths[node2].append(vertex)  # Store path information

        # Dependency accumulation
        values = {n:0 for n in nodes}       # Store dependency values for back-propagation
        while len(stack) > 0:
            node2 = stack.pop()
            for vertex in paths[node2]:     # Backtrack through the paths
                values[vertex] += (number_of_paths[vertex] / number_of_paths[node2]) * (1 + values[node2])
            if node2 != node:               # Exclude the source node from its own betweenness centrality
                betweenness_centralities[node2] += values[node2]

    # normalize and sort betweenness centrality values
    betweenness_centralities_nodes = {}
    for node,centr in betweenness_centralities.items():
        betweenness_centralities[node] /= ((num_nodes - 1) * (num_nodes - 2))
        betweenness_centralities_nodes[node]=betweenness_centralities[node]

    betweenness_list = sorted(betweenness_centralities_nodes.items(), key=lambda x: x[1], reverse=True)

    with open(f"centralities/betweenness.txt", "w") as file:
        for node, value in betweenness_list:
            file.write(f"{node} {value:.4f}\n")

    return 

# 3. Biased PageRank
def pagerank(num_node,nodes,graph, alpha=0.8, tol=1e-6):
    """
    calculates and stores PageRank values for all nodes in the graph
    
    idea:
    uses standard power-iteration method to compute PageRank values until convergence
    d = bias vector
        d[node]=abs(sum(weights of edges to node))/mx
        where mx is the maximum of all d[i] for all nodes i
    """
    n = num_node
    ranks = {node: 1 / n for node in nodes}
    outgoing_counts = {node: len(graph[node]) for node in nodes}

    d = {node: 0 for node in nodes} # bias vector

    for node in nodes:
        for neighbor, sign in graph[node]:
            if sign == 1:
                d[neighbor] += 1 / outgoing_counts[node]
            else:
                d[neighbor] -= 1 / outgoing_counts[node]
    
    tot_d =0
    for node in nodes:
        d[node]=abs(d[node])
        tot_d = max(d[node],tot_d)
        
    for node in nodes:
        d[node] /= tot_d

    ranks = d.copy()
    
    # this fn updates the ranks using the formula nr[i] = alpha * t + (1 - alpha) * d[i]
    # where t = sum of all incoming ranks to node i,
    def update_rank(): 
        new_ranks = {}
        for node in nodes:
            incoming_sum = 0
            for neighbor, sign in graph[node]:
                if outgoing_counts[neighbor] >0:
                    incoming_sum = ranks[neighbor] / outgoing_counts[neighbor]
            new_ranks[node] = (1 - alpha) * d[node] + alpha * incoming_sum
        return new_ranks
    
    while True: # until convergence # for i in range(100):
        # compute new PageRanks
        new_ranks = update_rank()

        # L1 normalise PageRanks
        sum_l1 = sum(new_ranks.values())
        for key, value in new_ranks.items():
            new_ranks[key] = new_ranks[key] / sum_l1

        # check convergence with epsilon = 1e-6, break loop if converged
        if all(abs(new_ranks[node] - ranks[node]) < tol for node in ranks):
            break
            
        # update ranks
        ranks = new_ranks.copy()

    pagerank_list = sorted(ranks.items(), key=lambda x: x[1], reverse=True)

    with open(f"centralities/pagerank.txt", "w") as file:
        for node, value in pagerank_list:
            file.write(f"{node} {value:.4f}\n")

    return 

#  Compute Centrality Metrics
def gen_centrality(num_nodes, nodes, adjacency_list, signed_adjacency_list):  

    # 1. Closeness centrality 
    closeness_centrality(num_nodes, nodes, adjacency_list)

    # 2. Betweenness centrality 
    betweenness_centrality(num_nodes, nodes, adjacency_list)

    # 3. Biased PageRank
    pagerank(num_nodes, nodes, signed_adjacency_list)


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_centrality.py <path_to_subgraph>")
        sys.exit(1)
    
    # Read the graph path from command line argument
    file_path = sys.argv[1]
    subgraph_name = os.path.basename(file_path).split('.')[0]

    # mkdir -p centralities
    if not os.path.exists("centralities"):
        os.makedirs("centralities")

    # Load the graph based on the file path
    if "epinions" in file_path:
        num_nodes, nodes, adjacency_list, signed_adjacency_list = load_signed_graph(file_path)
    else:
        num_nodes, nodes, adjacency_list , signed_adjacency_list= load_graph(file_path)
    
    #  Compute Centrality Metrics
    gen_centrality(num_nodes, nodes, adjacency_list, signed_adjacency_list)
    
    path = "centralities/"
    files = ['closeness.txt', 'betweenness.txt', 'pagerank.txt']
    influencer = set()
    cnt = 0
    for file in files:
        tmp = set()
        with open(os.path.join(path, file), 'r') as f:
            lines = f.readlines()
            for line in lines[:200]:
                tmp.add(int(line.split()[0]))
        if cnt == 0:
            influencer = tmp
        else:
            influencer = influencer.intersection(tmp)
        cnt += 1

    print("Number of influencer nodes: ", len(influencer))
    
if __name__ == "__main__":
    main()