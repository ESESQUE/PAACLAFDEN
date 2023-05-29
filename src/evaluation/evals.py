from models.models import *

# Do a grid search for class_weights and weightedVoting weights to find the maximum accuracy
def gridSearch(train_nodes, targetData, class_weights_range, node_weights_range):
    # for each combination of class_weights and node_weights, train a model and test it on the target node
    # return the class_weights and node_weights that give the highest accuracy
    max_acc = 0
    max_acc_class_weights = []
    max_acc_node_weights = []
    for i in class_weights_range:
        for j in node_weights_range:
            sourceModels = []
            for k in range(len(train_nodes)):
                if k == 0:
                    sourceModels.append(sourceModelAdjWeight(train_nodes[k], {0: i, 1: 1-i}))
                elif k == 1:
                    sourceModels.append(sourceModelAdjWeight(train_nodes[k], {0: i, 2: 1-i}))
                else:
                    sourceModels.append(sourceRFModel(train_nodes[k]))
            acc = weightedVoting(sourceModels, targetData, j)
            if acc > max_acc:
                max_acc = acc
                max_acc_class_weights = i
                max_acc_node_weights = j
    return max_acc_class_weights, max_acc_node_weights, max_acc