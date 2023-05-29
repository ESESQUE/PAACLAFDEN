# Data Description

This document provides a detailed description of the dataset used in this project.

## Overview

The data used in this project is synthesized fire detection data from this [google colab document](https://colab.research.google.com/drive/1jFk1UtJltTvS1ZbByjTox-mzC0_dO5Oc?usp=sharing). It is then used to generate dataset for different scenarios as described below. The scenarios are for edge nodes that simulate different locations with a variety of fire detection context. Each location is represented by a node.


## Features

The data contains the following features:

1. **temp**: temperature of the environment
2. **humidity**: humidity of the environment
3. **aqi**: air quality index
4. **noise**: noise level of the environment
5. **occupancy**: the number of occupants in the environment
6. **CO**: carbon monoxide level of the environment
7. **CO2**: carbon dioxide level of the environment
8. **flame**: flame detection (binary: 1 if yes, 0 if no)

## Target Variable

The target variable is the data sensitivity, which represents the criticality of the scenario, i.e wether there is no fire, a small fire, or a very serious fire.

1. **data_sensitivity**: fire severity (0 if no fire, 1 if small fire, 2 if very serious fire)

## Scenarios
The predefined scenarios for the data nodes are as follows:

1. **Scenario 1**: This scenario resembles a situation where the sensitivity of 1 is initially only present in 1 source node, and the same goes for the sensitivity of 2. The target node is initially only exposed to sensitivity of 0, but then experiences both sensitivity of 1 and 2. Thus, the only only sensitivity 0 is initially shared between all nodes.
    - Source nodes:
        - node 1: has data with no fire and small fire
        - node 2: has data with no fire and severe fire
    - Target node:
        - node 3: initially only has data with no fire (can be used for training), but then experiences all sensitivities of no fire, small fire, and severe fire
2. **Scenario 2**: This scenario resembles a situation where the initially none of the sensitivities overlap between the nodes. It resembles a worst case scenario for collaborative learning due to the non-overlap of sensitivities.
    - Source nodes:
        - node 1: has data with small fire
        - node 2: has data with severe fire
    - Target node:
        - node 3: initially only has data with no fire (can be used for training), but then experiences all sensitivities of no fire, small fire, and severe fire
3. **Scenario 3**: This scenario resembles a situation where all nodes initially experience all sensitivities. It resembles the best case scenario for an ease of collaborative learning.
    - Source nodes:
        - node 1: has data with no fire, small fire, and severe fire
        - node 2: has data with no fire, small fire, and severe fire
    - Target node:
        - node 3: initially has data with no fire, small fire, and severe fire (can be used for training), and then experiences all sensitivities of no fire, small fire, and severe fire
4. **Scenario 4**: This scenario is similar to scenario 1, however, there is a slightly different overlap between the nodes. Each node has only one class that overlaps with another node.
    - Source nodes:
        - node 1: has data with no fire and small fire
        - node 2: has data with small fire and severe fire
    - Target node:
        - node 3: initially only has data with no fire and severe fire (can be used for training), but then experiences all sensitivities of no fire, small fire, and severe fire
5. **Scenario 5**: In this scenario, one node has experienced all the sensitivities, and the rest of the nodes have only experienced two sensitivities. In this scenario, we expect node 1 to heavily affect the model outcome.
    - Source nodes:
        - node 1: has data with no fire, small fire, and severe fire
        - node 2: has data with small fire and severe fire
    - Target node:
        - node 3: initially only has data with no fire and severe fire (can be used for training), but then experiences all sensitivities of no fire, small fire, and severe fire
6. **Scenario 6**: This scenario is different since there are 4 nodes instead of 3. Similar to scenario 5, node 1 has experienced all sensitivities, and the rest experience 2 types of sensitivities.
    - Source nodes:
        - node 1: has data with no fire, small fire, and severe fire
        - node 2: has data with no fire and small fire
        - node 3: has data with no fire and severe fire
    - Target node:
        - node 4: initially only has data with small fire and severe fire (can be used for training), but then experiences all sensitivities of no fire, small fire, and severe fire


## Data Split

In all scenarios, the machine learning models can only be trained on the source nodes, as well as the initial data points of the target node. To justify this approach to the data split, we can visualize a real fire detection situation. Lets assume each node represents a separate building in a city. Our goal is to develop a machine learning model for our target node to effectively detect fire by transferring the knowledge from other nodes. For example, in scenario 1, our target node has initially never experienced sensitivity 1 or 2, so the knowledge of small fire detection from node 1 and severe fire detection from node 2 should be effectively transferred to node 3. This has to be done without sharing the raw data between the nodes, thus keeping the data private. 
