import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
np.random.seed(42)

label = 'activityID'

# define a function that takes in a dataset and trains a random forest model and returns the model
def sourceRFModel(sourceData):
    # The labels are in the activityID column
    y = sourceData[label]
    # The features are all columns except activityID and timestamp
    featureNames = [x for x in sourceData.columns if x not in [label]]
    X = sourceData[featureNames]
    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    # Train the Classifier to take the training features and learn how they relate
    # to the training y
    clf.fit(X, y)
    return clf

def predictTarNode(sourceModels, targetNode):
    # for each sourceModel, test it on the target node and add the predicted labels as a column to the targetnode[100]
    targetNodePred = pd.DataFrame()
    for i,j in enumerate(sourceModels):
        targetNodePred['predicted' + str(i)] = sourceModels[i].predict(targetNode.drop([label], axis=1))
    # take the majority vote of the predicted labels
    targetNodePred['predicted'] = targetNodePred.mode(axis=1)[0]
    y_pred = targetNodePred['predicted']
    return y_pred

def weightedVoting(sourceModels, targetData, weights, return_proba=False):
    # The labels are in the activityID column
    y_test = targetData[label]
    # The features are all columns except activityID and timestamp
    featureNames = [x for x in targetData.columns if x not in [label]]
    X_test = targetData[featureNames]
    # Identify the classes predicted by each model
    classes = []
    for i in range(len(sourceModels)):
        classes.append(sourceModels[i].classes_)
    proba = []
    # create a dataframe for each model's predicted class probabilities, make sure to fill in the missing class with 0
    for i in range(len(sourceModels)):
        proba.append(pd.DataFrame(sourceModels[i].predict_proba(X_test), columns=classes[i]))
        for j in range(len(classes[i])):
            if classes[i][j] not in proba[i].columns:
                proba[i][classes[i][j]] = 0

    # Multiply the predicted class probabilities of each model by their respective weights
    weighted_proba = []
    for i in range(len(proba)):
        weighted_proba.append(proba[i].mul(weights[i], axis=1))
    # Sum up the weighted predicted class probabilities
    weighted_proba_sum = pd.DataFrame()
    for i in range(len(weighted_proba)):
        weighted_proba_sum = weighted_proba_sum.add(weighted_proba[i], fill_value=0)
    # Identify the class with the highest probability
    y_pred = weighted_proba_sum.idxmax(axis=1)
    if return_proba==False:
        return y_pred
    else:
        # convert weighted_proba_sum to a y_score for roc_auc_score
        weighted_proba_sum = weighted_proba_sum.to_numpy()
        return weighted_proba_sum

# define a function that takes in a dataset and train a model with class_weights
def sourceModelAdjWeight(sourceData, class_weight):
    # The labels are in the activityID column
    y = sourceData[label]
    # The features are all columns except activityID and timestamp
    featureNames = [x for x in sourceData.columns if x not in [label]]
    X = sourceData[featureNames]
    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42, class_weight=class_weight)
    # Train the Classifier to take the training features and learn how they relate
    # to the training y
    clf.fit(X, y)
    return clf

