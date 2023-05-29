import os
import sys
sys.path.insert(0,"/Users/saeedalqubaisi/Desktop/thesis/Adaptive-Collaborative-Learning-for-Distrubted-Edge-Networks/src")

from models.models import *
from models.dynamic_algorithms import *
from evaluation.evals import *
from sklearn.metrics import accuracy_score
import pandas as pd

def evaluate_models(scenario_file):
    train_nodes = []
    test_nodes = []
    train_file_path = os.path.join(scenario_file, 'Train')
    test_file_path = os.path.join(scenario_file, 'Test')

    train_node_names = os.listdir(train_file_path)
    test_node_names = os.listdir(test_file_path)




    for node_name in train_node_names:
        file_path = os.path.join(train_file_path, node_name)
        # Check if the path is a file, not a directory or other type of file system object
        df = pd.read_csv(file_path)
        train_nodes.append(df)
    
    for node_name in test_node_names:
        file_path = os.path.join(test_file_path, node_name)
        # Check if the path is a file, not a directory or other type of file system object
        df = pd.read_csv(file_path)
        test_nodes.append(df)

    # Run Non-distibuted theoretical optimal
    # combine all traning data into one dataframe
    trainData = pd.concat(train_nodes)
    # train a random forest on trainData
    clf = sourceRFModel(trainData)
    # predict the labels of node3
    y_pred = predictTarNode(clf, test_nodes[0])
    # calculate the accuracy of the prediction
    print("Non-distributed Theoretical Optimal: ",accuracy_score(test_nodes[0]['data_sensitivity'], y_pred))

    # Run Distributed Theoretical Optimal
    # Run gridsearch to find the best combination of node weights and class weights

    # Dynamic Algorithm

    # Using binary class distribution to find node weights

    # without class weights
    node_weights = calculate_node_weights(train_nodes)
    # train source models for each node in train_nodes
    sourceModels = []
    for i in range(len(train_nodes)):
        sourceModels.append(sourceRFModel(train_nodes[i]))
    # find y_pred using weightedVoting
    y_pred = weightedVoting(sourceModels, test_nodes[0], node_weights)
    # calculate the accuracy of the prediction
    print("Binary class distribution without class weights: ",accuracy_score(test_nodes[0]['data_sensitivity'], y_pred))

    # with class weights
    node_weights = calculate_node_weights(train_nodes)
    class_weights = calculate_class_weights(train_nodes)
    # train source models for each node in train_nodes using sourceModelAdjWeight
    sourceModels = []
    for i in range(len(train_nodes)):
        sourceModels.append(sourceModelAdjWeight(train_nodes[i], class_weights[i+1]))
    # find y_pred using weightedVoting
    y_pred = weightedVoting(sourceModels, test_nodes[0], node_weights)
    # calculate the accuracy of the prediction
    print("Binary class distribution with class weights: ",accuracy_score(test_nodes[0]['data_sensitivity'], y_pred))

    # with a threshold of 0.1 for class weights
    node_weights = calculate_node_weights(train_nodes)
    class_weights = calculate_class_weights(train_nodes, threshold=0.1)
    # train source models for each node in train_nodes using sourceModelAdjWeight
    sourceModels = []
    for i in range(len(train_nodes)):
        sourceModels.append(sourceModelAdjWeight(train_nodes[i], class_weights[i+1]))
    # find y_pred using weightedVoting
    y_pred = weightedVoting(sourceModels, test_nodes[0], node_weights)
    # calculate the accuracy of the prediction
    print("Binary class distribution with class weights and 0.1 threshold: ",accuracy_score(test_nodes[0]['data_sensitivity'], y_pred))

    # using class distribution with frequencies to find node weights

    # without class weights
    node_weights = calculate_node_weights_with_frequencies(train_nodes)
    # train source models for each node in train_nodes
    sourceModels = []
    for i in range(len(train_nodes)):
        sourceModels.append(sourceRFModel(train_nodes[i]))
    # find y_pred using weightedVoting
    y_pred = weightedVoting(sourceModels, test_nodes[0], node_weights)
    # calculate the accuracy of the prediction
    print("Frequency class distribution without class weights: ",accuracy_score(test_nodes[0]['data_sensitivity'], y_pred))

    # with class weights
    node_weights = calculate_node_weights_with_frequencies(train_nodes)
    class_weights = calculate_class_weights(train_nodes)
    # train source models for each node in train_nodes using sourceModelAdjWeight
    sourceModels = []
    for i in range(len(train_nodes)):
        sourceModels.append(sourceModelAdjWeight(train_nodes[i], class_weights[i+1]))
    # find y_pred using weightedVoting
    y_pred = weightedVoting(sourceModels, test_nodes[0], node_weights)
    # calculate the accuracy of the prediction
    print("Frequency class distribution with class weights: ",accuracy_score(test_nodes[0]['data_sensitivity'], y_pred))

    # with a threshold of 0.1 for class weights
    node_weights = calculate_node_weights_with_frequencies(train_nodes)
    class_weights = calculate_class_weights(train_nodes, threshold=0.1)
    # train source models for each node in train_nodes using sourceModelAdjWeight
    sourceModels = []
    for i in range(len(train_nodes)):
        sourceModels.append(sourceModelAdjWeight(train_nodes[i], class_weights[i+1]))
    # find y_pred using weightedVoting
    y_pred = weightedVoting(sourceModels, test_nodes[0], node_weights)
    # calculate the accuracy of the prediction
    print("Frequency class distribution with class weights and 0.1 threshold: ",accuracy_score(test_nodes[0]['data_sensitivity'], y_pred))
    


