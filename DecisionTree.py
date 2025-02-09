import numpy as np
import pandas as pd
from collections import Counter  # for calculating node value


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
        sample_size=0,
    ):
        self.feature = feature  # feature name
        self.threshold = threshold  # The threshold value for the feature. x < threshold go left, x >= threshold go right
        self.left = left
        self.right = right
        self.value = value  # value to return if leaf node
        self.sample_size = sample_size  # Number of samples at node

    def is_leaf_node(self):
        """
        Returns true if node has no children

        Returns:
        boolean
        """
        return (
            self.right is None and self.left is None
        )  # A node is a leaf if it has not children


class DecisionTree:
    """Decision Tree Classifier"""

    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split  # minimum number of samples required to split an internal node
        self.max_depth = max_depth  # maximum depth of the tree.
        self.root = None  # root node

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Given dataframe X and target values y, fit decision tree to dataset.
        All data must be numeric and

        Parameters:
        X: Pandas dataframe containing features (columns) and entries (rows)
        y: 1D Numpy array of target values

        Returns:
        None
        """
        self.root = self._grow(X, np.array(y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Given a dataframe of entries, make a prediction for each entry

        Parameters:
        X: DataFrame of entries

        Returns:
        Array of predicted values
        """
        # If not fit to data yet, return none
        if self.root is None:
            return None

        # array to store results
        results = []

        # Get each row
        for (
            index,
            row,
        ) in (
            X.iterrows()
        ):  # rows is a series with labels, i.e. to access: row['Name'], row['Age']
            results.append(self._traverse(row, self.root))

        return np.array(results)

    def _traverse(self, x: pd.Series, node: Node):
        """
        Given a series representing one entry, and a current node, traverse the
        tree until a leaf node is hit and return prediction

        Parameters:
        x: Pandas Series of one entry to make prediction on
        node: Node object of current node

        Returns:
        Predicted value
        """
        # Check if at leaf
        if node.is_leaf_node():
            return node.value

        # Check if should go left
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left)
        # else go right
        return self._traverse(x, node.right)

    def _grow(self, X: pd.DataFrame, y: np.ndarray, depth: int = 0) -> Node:
        """
        Given the feature set and target recursively grow the decision tree
        until stopping condition hit (max_depth or min_samples_split)

        Parameters:
        X: Pandas dataframe containing features (columns) and entries (rows)
        y: 1D numpy array of target values
        depth: Current depth

        Returns:
        Node: Root node
        """

        # Calculate current value
        value = self._calculate_value(y)

        # find feature with lowest gini
        feature_to_split_on, threshold = self._lowest_gini_impurity(X, y)

        # Create Node
        current_node = Node(
            feature=feature_to_split_on,
            threshold=threshold,
            value=value,
            sample_size=len(y),
        )

        # Check if can split again (if enough samples remain, not at depth, and atleast 1 more feature)
        if (
            len(X) < self.min_samples_split
            or depth >= self.max_depth
            or len(X.columns) <= 1
        ):
            return current_node

        # Split data
        left_X, left_y, right_X, right_y = self._split(
            X, y, feature_to_split_on, threshold
        )

        # Check each child has atleast 1 sample else make current node leaf
        if len(left_X) == 0 or len(right_X) == 0:
            return current_node

        # Check if splitting again would worsen gini impurity
        current_gini = gini_impurity(y)
        left_gini = gini_impurity(left_y)
        right_gini = gini_impurity(right_y)
        weighted_gini_after_split = (
            len(left_y) / len(y) * left_gini + len(right_y) / len(y) * right_gini
        )
        # If splitting again worsens gini, stop splitting
        if current_gini <= weighted_gini_after_split:
            return current_node

        # drop feature from dataframe
        left_X.drop(feature_to_split_on, axis=1, inplace=True)
        right_X.drop(feature_to_split_on, axis=1, inplace=True)

        # calculate children
        current_node.left = self._grow(left_X, left_y, depth + 1)
        current_node.right = self._grow(right_X, right_y, depth + 1)

        return current_node

    def _lowest_gini_impurity(self, X: pd.DataFrame(), y: np.ndarray) -> (str, float):
        """
        Iterate through pandas dataframe columns and find the feature with
        the lowest gini coefficient

        Parameters:
        X: Pandas dataframe of features
        y: numpy array of target values

        Returns:
        String: column label with lowest gini coefficient
        Float: Threshold to split feature on
        """
        lowest_gini_feature = None
        lowest_gini_impurity = None
        best_threshold = None

        # iterate through all columns
        for feature in X:
            # calculte gini coefficient
            best_split, curr_gini = self._best_split(np.array(X[feature]), y)
            # check if new lowest gini
            if lowest_gini_feature is None or curr_gini < lowest_gini_impurity:
                lowest_gini_feature = feature
                lowest_gini_impurity = curr_gini
                best_threshold = best_split

        # return name of lowest gini feature
        return lowest_gini_feature, best_threshold

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> (float, float):
        """
        Given the array of integers and the array of target values, find the
        threshold which results in the lowest summed gini coefficients of the
        target when splitting the feature at that threshold.

        Paramaters:
        X: Numpy array of integers
        y: Numpy array of target values


        Returns:
        Float of threshold for best split
        Float of the gini coefficient
        """
        # Sort feature values along with their corresponding target values
        sorted_indices = np.argsort(X)
        sorted_X = X[sorted_indices]
        sorted_y = y[sorted_indices]

        # keep track of best threshold
        lowest_gini_impuritys = None
        best_split = None

        # iterate through each value
        for i in range(1, sorted_X.size):

            # skip duplicate values
            if i != 0 and sorted_X[i] == sorted_X[i - 1]:
                continue

            # split array
            left = sorted_y[:i]
            right = sorted_y[i:]

            # calculate weighted summed gini impurity
            left_gini = gini_impurity(left)
            right_gini = gini_impurity(right)
            summed_gini_impuritys = (
                len(left) / len(sorted_y) * left_gini
                + len(right) / len(sorted_y) * right_gini
            )

            # Check if lowest gini
            if best_split is None or summed_gini_impuritys < lowest_gini_impuritys:
                lowest_gini_impuritys = summed_gini_impuritys
                best_split = sorted_X[i]

        # return value of best split
        return best_split, lowest_gini_impuritys

    def _split(
        self, X: pd.DataFrame, y: np.ndarray, feature: str, threshold: float
    ) -> (pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray):
        """
        Splits the dataframe and corresponding target values by the given feature
        and threshold

        Parameters:
        X: Dataframe
        y: Array of target values
        feature: String of feature name to split on
        threshold: Float to split feature by

        Returns:
        Dataframe: Left dataframe
        Array: Left target array
        Dataframe: Right dataframe
        Array: Right target array
        """
        # return the boolean list
        left_indices = X[feature] < threshold
        right_indices = X[feature] >= threshold

        # get all the true values
        X_left = X.loc[left_indices].copy().reset_index(drop=True)
        X_right = X.loc[right_indices].copy().reset_index(drop=True)

        y_left = y[left_indices].copy()
        y_right = y[right_indices].copy()

        return X_left, y_left, X_right, y_right

    def _calculate_value(self, y: np.ndarray):
        """
        Calculate and return the most frequency value in the array

        Parameters:
        y: Numpy array

        Returns:
        Mode of y
        """
        # Count frequency of all values and get most common value & count
        most_common = Counter(y.tolist()).most_common(1)
        # Return most common value
        return most_common[0][0]  # Can not return mean if expecting non-numeric targets

    def print_tree(self, node=None, depth=0, prefix="Root:"):
        """Recursively prints the decision tree in a visually structured format."""
        if node is None:
            if depth == 0 and self.root != None:
                node = self.root
            else:
                return

        indent = "    " * depth  # Indentation based on tree depth

        # If it's a leaf node, print the value
        if node.is_leaf_node():
            print(
                f"{indent}{prefix} Leaf -> Value: {node.value}    (Sample size: {node.sample_size})"
            )
        else:
            # Print decision rule
            print(
                f"{indent}{prefix} Feature[{node.feature}] < {node.threshold:.2f} (Sample size: {node.sample_size})"
            )

        # Recursive calls for left and right child nodes
        self.print_tree(node.left, depth + 1, "L:")
        self.print_tree(node.right, depth + 1, "R:")


def gini_impurity(array: np.ndarray) -> float:
    """
    Calculate the Gini impurity of a NumPy array.

    Parameters:
    array: Input array class labels

    Returns:
    float: Gini impurity (0 = perfect classification, 1 = maximum impurity)
    """
    if array.size <= 1 or np.sum(array) == 0:
        return 0.0

    # Count occurrences of each class
    class_counts = Counter(array)
    total = len(array)

    # Calculate Gini impurity
    probabilities = [count / total for count in class_counts.values()]
    return 1 - sum(p**2 for p in probabilities)
