from DecisionTree import *
import pandas as pd
import numpy as np


def test_gini():
    arr = np.array([12, 47, 3, 25, 33, 8, 50, 19, 41, 6])
    return round(gini_coefficient(arr), 5) == 0.38689


def test_is_leaf_node():
    node1 = Node()
    node2 = Node(left=node1)
    return node1.is_leaf_node() == True and node2.is_leaf_node() == False


def test_lowest_gini():
    pass


def test_best_split():
    data = {
        "Feature": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "Target": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
    }
    tree = DecisionTree()
    best_split, lowest_gini = tree._best_split(
        np.array(data["Feature"]), np.array(data["Target"])
    )
    return best_split == 30 and lowest_gini == 0.375


def test_lowest_gini_coefficient():
    X = pd.DataFrame(
        {
            "Feature1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Feature2": [0.1, 0.4, 0.7, 0.2, 0.5, 0.8, 0.3, 0.9, 0.6, 1.0],
        }
    )
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

    # Instantiate the class and call the method
    tree = DecisionTree()
    result_feature, result_threshold = tree._lowest_gini_coefficient(X, y)

    # Check if the result matches the expected
    return result_feature == "Feature2" and 0.5 == result_threshold


def test_split():
    X = pd.DataFrame(
        {
            "Feature1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Feature2": [0.1, 0.4, 0.7, 0.2, 0.5, 0.8, 0.3, 0.9, 0.6, 1.0],
        }
    )
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

    expected_left_X = pd.DataFrame(
        {"Feature1": [10, 20, 30], "Feature2": [0.1, 0.4, 0.7]}
    )
    expected_left_y = np.array([0, 0, 1])
    expected_right_X = pd.DataFrame(
        {
            "Feature1": [40, 50, 60, 70, 80, 90, 100],
            "Feature2": [0.2, 0.5, 0.8, 0.3, 0.9, 0.6, 1.0],
        }
    )
    expected_right_y = np.array([0, 1, 1, 0, 1, 0, 1])

    # Instantiate the class and call the method
    tree = DecisionTree()
    left_X, left_y, right_X, right_y = tree._split(X, y, "Feature1", 40)

    return (
        expected_left_X.equals(left_X)
        and np.array_equal(expected_left_y, left_y)
        and expected_right_X.equals(right_X)
        and np.array_equal(expected_right_y, right_y)
    )


def test_fit():
    X = pd.DataFrame(
        {
            "Feature1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "Feature2": [0.1, 0.4, 0.7, 0.2, 0.5, 0.8, 0.3, 0.9, 0.6, 1.0],
        }
    )
    y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])

    tree = DecisionTree()
    tree.fit(X, y)
    pass


# test
print("test_giniL: " + str(test_gini()))
print("test_is_leaf_node: " + str(test_is_leaf_node()))
print("test_lowest_gini: " + str(test_lowest_gini()))
print("test_best_split: " + str(test_best_split()))
print("test_lowest_gini_coefficient: " + str(test_lowest_gini_coefficient()))
print("test_split: " + str(test_split()))
print("test_fit: " + str(test_fit()))
