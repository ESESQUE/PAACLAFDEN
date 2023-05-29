import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
import random
np.random.seed(42)

label = 'activityID'

# A dynamic algorithm that computes node weights based on class (binary) distribution on nodes
def calculate_node_weights(train_nodes, epsilon=0.01):
    # find node_classes from train_nodes
    node_classes = []
    for node in train_nodes:
        node_classes.append(node[label].unique())

    num_nodes = len(node_classes)
    
    # Calculate class frequencies
    class_frequencies = {}
    for classes in node_classes:
        for cls in classes:
            class_frequencies[cls] = class_frequencies.get(cls, 0) + 1
    
    # Calculate diversity scores
    diversity_scores = []
    for classes in node_classes:
        diversity_score = sum(1 / (class_frequencies[cls] - 1 + epsilon) for cls in classes) * (1/len(classes))
        diversity_scores.append(diversity_score)
    
    # Normalize diversity scores to get weights
    total_diversity_score = sum(diversity_scores)
    weights = [score / total_diversity_score for score in diversity_scores]
    
    return weights

# A dynamic algorithm that computes node weights based on class distribution with frequencies on nodes
def calculate_node_weights_with_frequencies(train_nodes, epsilon=0.01):
    # find node_class_frequencies from train_nodes
    node_class_frequencies = {}
    for i, node in enumerate(train_nodes):
        node_class_frequencies[i+1] = node[label].value_counts().to_dict()
    
    num_nodes = len(node_class_frequencies)
    
    # Calculate class frequencies
    class_total_frequencies = {}
    for frequencies in node_class_frequencies.values():
        for cls, freq in frequencies.items():
            class_total_frequencies[cls] = class_total_frequencies.get(cls, 0) + freq
    
    # Calculate diversity scores
    diversity_scores = []
    for node, frequencies in node_class_frequencies.items():
        diversity_score = sum(freq / (class_total_frequencies[cls] - freq + epsilon) for cls, freq in frequencies.items()) * (1/len(frequencies))
        diversity_scores.append(diversity_score)
    
    # Normalize diversity scores to get weights
    total_diversity_score = sum(diversity_scores)
    weights = [score / total_diversity_score for score in diversity_scores]
    
    return weights

# Introduce an algorithm that calculates the class_weights
def calculate_class_weights(train_nodes, epsilon=0.01, threshold=0):
    # find node_class_frequencies from train_nodes
    node_class_frequencies = {}
    for i, node in enumerate(train_nodes):
        node_class_frequencies[i+1] = node[label].value_counts().to_dict()

    num_nodes = len(node_class_frequencies)
    
    # Calculate class total frequencies
    class_total_frequencies = {}
    for frequencies in node_class_frequencies.values():
        for cls, freq in frequencies.items():
            class_total_frequencies[cls] = class_total_frequencies.get(cls, 0) + freq
    
    # Calculate class_weights for each node
    class_weights = {}
    for node, frequencies in node_class_frequencies.items():
        class_weight = {}
        for cls, freq in frequencies.items():
            inv_freq = 1 / (class_total_frequencies[cls] - freq + epsilon)
            class_weight[cls] = inv_freq

        # Normalize class_weights
        total_class_weight = sum(class_weight.values())
        normalized_class_weight = {cls: max(weight / total_class_weight, threshold) for cls, weight in class_weight.items()}

        class_weights[node] = normalized_class_weight
    
    return class_weights

# An algorithm that utililzes a similarity graph to compute node weights
def fcngraph_from_nodes(train_nodes):
    # create a fully connected graph from train_nodes, where node name should be the index of the dataframe +1, and node attributes should be the values of the dataframe
    G = nx.Graph()
    for i in range(len(train_nodes)):
        G.add_node(i+1, data=train_nodes[i], classes = set(train_nodes[i][label].unique()))
    # add edges between all nodes
    for i in range(len(train_nodes)):
        for j in range(i+1, len(train_nodes)):
            G.add_edge(i+1, j+1)
    return G

# from train_nodes, create a connected graph with random edges
def random_graph_from_nodes(train_nodes):
    G = nx.Graph()
    for i in range(len(train_nodes)):
        G.add_node(i+1, train_data=train_nodes[i], test_data = test_nodes[i], classes = set(train_nodes[i][label].unique()))
    # add edges between the nodes, each node can have between 1 and len(train_nodes)-1 edges randomly, but not to itself
    for i in range(len(train_nodes)):
        num_edges = random.randint(1, len(train_nodes)-1)
        for j in range(num_edges):
            neighbor = random.randint(1, len(train_nodes))
            if neighbor != i+1:
                G.add_edge(i+1, neighbor)
    return G

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def bfs_traversal(graph, source, max_steps):
    visited = set()
    queue = [(source, 0)]

    while queue:
        node, step = queue.pop(0)
        if node not in visited:
            visited.add(node)
            if step < max_steps:
                for neighbor in graph.neighbors(node):
                    queue.append((neighbor, step + 1))

    return visited

def graph_node_weights(graph, max_steps=2, epsilon = 0.01):
    class_sets = nx.get_node_attributes(graph, 'classes')
    weights = defaultdict(float)
    uniqueness = defaultdict(float)

    for node in graph.nodes:
        neighbors = bfs_traversal(graph, node, max_steps)
        similarities = [jaccard_similarity(class_sets[node], class_sets[neighbor]) for neighbor in neighbors if neighbor != node]
        avg_similarity = sum(similarities) / len(similarities)
        uniqueness[node] = 1 - avg_similarity

    total_uniqueness = sum(uniqueness.values())
    for node in graph.nodes:
        weights[node] = uniqueness[node] / (total_uniqueness + epsilon)

    # create a list of weights from dictionary values
    weights_list = []
    for i in range(len(weights)):
        weights_list.append(weights[i+1])

    return weights_list