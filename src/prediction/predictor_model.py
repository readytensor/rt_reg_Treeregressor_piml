import os
import warnings
from typing import Optional, Union, List

import joblib
import numpy as np
import pandas as pd
from piml.models import TreeRegressor
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Regressor:
    """A wrapper class for the TreeRegressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    model_name = "TreeRegressor"

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        feature_types: Optional[List[str]] = None,
        criterion: str = "squared_error",
        max_depth: int = 5,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 5,
        splitter: str = "best",
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        random_state: int = 0,
        max_leaf_nodes: int = None,
        **kwargs,
    ):
        """Construct a new TreeClassifier.

        Args:
            feature_names (Optional[List[str]]): The list of feature names.
            feature_types (Optional[List[str]]): The list of feature types. Available types include “numerical” and “categorical”.
            criterion (str): {“squared_error”, “friedman_mse”, “absolute_error”, “poisson”},
                The function to measure the quality of a split. Supported criteria are “squared_error” for the mean squared error,
                which is equal to variance reduction as feature selection criterion and minimizes the L2 loss using the mean of each terminal node,
                “friedman_mse”, which uses mean squared error with Friedman's improvement score for potential splits,
                “absolute_error” for the mean absolute error, which minimizes the L1 loss using the median of each terminal node, and
                “poisson” which uses reduction in Poisson deviance to find splits.

            splitter (str): {“best”, “random”},
                The strategy used to choose the split at each node.
                Supported strategies are “best” to choose the best split and “random” to choose the best random split.

            max_depth (int): The maximum depth of the tree. If None, then nodes are
            expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

            min_samples_split (Union[int, float]): The minimum number of samples required to split an internal node:
                If int, then consider min_samples_split as the minimum number.
                If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

            min_samples_leaf (Union[int, float]): The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches.
            This may have the effect of smoothing the model, especially in regression.
                If int, then consider min_samples_leaf as the minimum number.
                If float, then min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.

            min_weight_fraction_leaf (float): The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.
            Samples have equal weight when sample_weight is not provided.

            max_features (Union[int, float, str]): {“auto”, “sqrt”, “log2”},
            The number of features to consider when looking for the best split:
                If int, then consider max_features features at each split.
                If float, then max_features is a fraction and max(1, int(max_features * n_features_in_)) features are considered at each split.
                If “auto”, then max_features=sqrt(n_features).
                If “sqrt”, then max_features=sqrt(n_features).
                If “log2”, then max_features=log2(n_features).
                If None, then max_features=n_features.

            max_leaf_nodes (int): Grow a tree with max_leaf_nodes in best-first fashion.
            Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.

        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.splitter = splitter
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> TreeRegressor:
        """Build a new binary classifier."""
        model = TreeRegressor(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            splitter=self.splitter,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.random_state,
            **self.kwargs,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the regressor to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict regression targets for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.model.predict(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the regressor and return the r-squared score.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The targets of the test data.
        Returns:
            float: The r-squared score of the regressor.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the regressor to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Regressor":
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Regressor: A new instance of the loaded regressor.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"eta: {self.eta}, "
            f"gamma: {self.gamma}, "
            f"max_depth: {self.max_depth}, "
            f"n_estimators: {self.n_estimators})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Regressor:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data targets.
        hyperparameters (dict): Hyperparameters for the regressor.

    Returns:
        'Regressor': The regressor model
    """
    regressor = Regressor(**hyperparameters)
    regressor.fit(train_inputs=train_inputs, train_targets=train_targets)
    return regressor


def predict_with_model(regressor: Regressor, data: pd.DataFrame) -> np.ndarray:
    """
    Predict regression targets for the given data.

    Args:
        regressor (Regressor): The regressor model.
        data (pd.DataFrame): The input data.

    Returns:
        np.ndarray: The predicted regression targets.
    """
    return regressor.predict(data)


def save_predictor_model(model: Regressor, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Regressor:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Regressor, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the regressor model and return the r-squared value.

    Args:
        model (Regressor): The regressor model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The targets of the test data.

    Returns:
        float: The r-sq value of the regressor model.
    """
    return model.evaluate(x_test, y_test)
