import numpy as np
import pandas as pd


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature  # feature name
        self.threshold = threshold  # The threshold value for the feature. x < threshold go left, x >= threshold go right
        self.left = left
        self.right = right
        self.value = value  # Probability distribution, i.e. [0.4, 0.6]

    def is_leaf_node(self):
        """
        Returns true if node has no children

        Returns:
        boolean
        """
        return (
            self.right == None and self.left == None
        )  # A node is a leaf if it has not children


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100):
        self.min_samples_split = min_samples_split  # minimum number of samples required to split an internal node
        self.max_depth = max_depth  # maximum depth of the tree.
        self.root = None  # root node

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.root = self._grow(X, y)

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
        value = 1 if np.sum(y) > len(y) / 2 else 0

        # find feature with lowest gini
        feature_to_split_on, threshold = self._lowest_gini_coefficient(X, y)

        # Create Node
        current_node = Node(
            feature=feature_to_split_on, threshold=threshold, value=value
        )

        # Check if can split again (if enough samples remain, not at depth, and atleast 1 feature)
        if (
            len(X) < self.min_samples_split
            or depth == self.max_depth
            or len(X.columns) == 0
        ):
            return current_node

        # Split data
        left_X, left_y, right_X, right_y = self._split(
            X, y, feature_to_split_on, threshold
        )

        # drop feature from dataframe
        left_X.drop(feature_to_split_on, axis=1, inplace=True)
        right_X.drop(feature_to_split_on, axis=1, inplace=True)

        # calculate children
        current_node.left = self._grow(left_X, left_y, depth + 1)
        current_node.right = self._grow(right_X, right_y, depth + 1)

        return current_node

    def _lowest_gini_coefficient(
        self, X: pd.DataFrame(), y: np.ndarray
    ) -> (str, float):
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
        lowest_gini_coefficient = None
        best_threshold = None

        # iterate through all columns
        for feature in X:
            # calculte gini coefficient
            best_split, curr_gini = self._best_split(np.array(X[feature]), y)
            # check if new lowest gini
            if lowest_gini_feature == None or curr_gini < lowest_gini_coefficient:
                lowest_gini_feature = feature
                lowest_gini_coefficient = curr_gini
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
        lowest_gini_coefficients = None
        best_split = None

        # iterate through each value
        for i in range(1, sorted_X.size - 1):

            # skip duplicate values
            if i != sorted_X.size - 1 and sorted_X[i] == sorted_X[i + 1]:
                continue

            # split array
            left = sorted_y[:i]
            right = sorted_y[i:]

            # calculate
            left_gini = gini_coefficient(left)
            right_gini = gini_coefficient(right)
            summed_gini_coefficients = left_gini + right_gini

            # Check if lowest gini
            if (
                best_split == None
                or summed_gini_coefficients < lowest_gini_coefficients
            ):
                lowest_gini_coefficients = summed_gini_coefficients
                best_split = sorted_X[i]

        # return value of best split
        return best_split, lowest_gini_coefficients

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
        X_left = X.loc[left_indices].reset_index(drop=True)
        X_right = X.loc[right_indices].reset_index(drop=True)

        y_left = y[left_indices]
        y_right = y[right_indices]

        return X_left, y_left, X_right, y_right


def gini_coefficient(array: np.ndarray) -> float:
    """
    Calculate the Gini coefficient of a NumPy array.

    Parameters:
    array: Input array containing numeric values

    Returns:
    float: Gini coefficient (0 = perfect equality, 1 = maximum inequality)
    """
    if array.size <= 1 or np.sum(array) == 0:
        return 0.0

    array = np.sort(array)  # Sort the array in ascending order
    n = array.size
    index = np.arange(1, n + 1)  # Create index array

    # Compute the Gini coefficient using the formula
    return (2 * np.sum(index * array) / (n * np.sum(array))) - (n + 1) / n
