"""
INLP debiasing projection. From shauli-ravfogel/nullspace_projection (master).
Uses local classifier module for SKlearnClassifier.
"""
from typing import Dict, List

import numpy as np
import scipy
import random
import warnings

from tqdm import tqdm

from src.core.ablation._vendor.inlp.classifier import SKlearnClassifier


def get_rowspace_projection(W: np.ndarray) -> np.ndarray:
    """
    :param W: the matrix over its nullspace to project
    :return: the projection matrix over the rowspace
    """
    if np.allclose(W, 0):
        w_basis = np.zeros_like(W.T)
    else:
        w_basis = scipy.linalg.orth(W.T)
    P_W = w_basis.dot(w_basis.T)
    return P_W


def get_projection_to_intersection_of_nullspaces(
    rowspace_projection_matrices: List[np.ndarray], input_dim: int
) -> np.ndarray:
    """
    Given a list of rowspace projection matrices P_R(w_1), ..., P_R(w_n),
    this function calculates the projection to the intersection of all nullspaces
    of the matrices w_1, ..., w_n.
    Uses the intersection-projection formula of Ben-Israel 2013.
    """
    I = np.eye(input_dim)
    Q = np.sum(rowspace_projection_matrices, axis=0)
    P = I - get_rowspace_projection(Q)
    return P


def debias_by_specific_directions(directions: List[np.ndarray], input_dim: int) -> np.ndarray:
    """
    Perform INLP on a set of user-provided directions (instead of learning those directions).
    """
    rowspace_projections = []
    for v in directions:
        P_v = get_rowspace_projection(v)
        rowspace_projections.append(P_v)
    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
    return P


def get_debiasing_projection(
    classifier_class,
    cls_params: Dict,
    num_classifiers: int,
    input_dim: int,
    is_autoregressive: bool,
    min_accuracy: float,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_dev: np.ndarray,
    Y_dev: np.ndarray,
    by_class: bool = False,
    Y_train_main=None,
    Y_dev_main=None,
    dropout_rate: float = 0,
):
    """
    :return: P (debiasing projection), rowspace_projections, Ws (classifiers).
    """
    if dropout_rate > 0 and is_autoregressive:
        warnings.warn(
            "Note: when using dropout with autoregressive training, "
            "the property w_i.dot(w_(i+1)) = 0 no longer holds."
        )

    if by_class:
        if (Y_train_main is None) or (Y_dev_main is None):
            raise Exception("Need main-task labels for by-class training.")
        main_task_labels = list(set(Y_train_main.tolist()))

    X_train_cp = X_train.copy()
    X_dev_cp = X_dev.copy()
    rowspace_projections = []
    Ws = []

    pbar = tqdm(range(num_classifiers))
    for i in pbar:
        clf = SKlearnClassifier(classifier_class(**cls_params))
        dropout_scale = 1.0 / (1 - dropout_rate + 1e-6)
        dropout_mask = (
            (np.random.rand(*X_train.shape) < (1 - dropout_rate)).astype(float) * dropout_scale
        )

        if by_class:
            cls = random.choice(main_task_labels)
            relevant_idx_train = Y_train_main == cls
            relevant_idx_dev = Y_dev_main == cls
        else:
            relevant_idx_train = np.ones(X_train_cp.shape[0], dtype=bool)
            relevant_idx_dev = np.ones(X_dev_cp.shape[0], dtype=bool)

        acc = clf.train_network(
            (X_train_cp * dropout_mask)[relevant_idx_train],
            Y_train[relevant_idx_train],
            X_dev_cp[relevant_idx_dev],
            Y_dev[relevant_idx_dev],
        )
        pbar.set_description("iteration: {}, accuracy: {}".format(i, acc))
        if acc < min_accuracy:
            continue

        W = clf.get_weights()
        Ws.append(W)
        P_rowspace_wi = get_rowspace_projection(W)
        rowspace_projections.append(P_rowspace_wi)

        if is_autoregressive:
            P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
            X_train_cp = (P.dot(X_train.T)).T
            X_dev_cp = (P.dot(X_dev.T)).T

    P = get_projection_to_intersection_of_nullspaces(rowspace_projections, input_dim)
    return P, rowspace_projections, Ws
