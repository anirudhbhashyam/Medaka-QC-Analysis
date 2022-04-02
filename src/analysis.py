import os
import argparse

from typing import Iterable, Union, Tuple, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from variables import *

qc_features = ["HROI Change Intensity", "Harmonic Intensity", "Heart size", "Movement detection max", "SNR", "Signal intensity", "Signal regional prominence", "Intensity/Harmonic Intensity (top 5 %)", "SNR Top 5%", "Signal Intensity Top 5%"]

def plot_qc_params(data: pd.DataFrame, 
                   limits: Union[dict, None] = None,
                   figsize: tuple = (10, 30), 
                   save_q: bool = False,
                   save_name: str = "abc") -> None:
    fig, ax = plt.subplots(nrows = len(qc_features), 
                        figsize = figsize,
                        gridspec_kw = dict(left = 0.01, right = 0.9,
                                            bottom = 0.0001, top = 0.9)) 

    for i, feature in enumerate(qc_features):
        ax[i].set(ylabel = feature) 
        sns.scatterplot(
            data = data,
            x = LABELS, 
            y = feature,
            s = 4,
            ax = ax[i]
        )
        if limits:
            if feature in limits:
                # Plot the threshold line for each feature.
                for threshold in limits[feature]:
                    ax[i].axhline(y = threshold, xmin = 0, xmax = 200, c = "red", lw = 0.7, label = threshold)    
                    ax[i].legend()
            
        
    if save_q:
        if not os.path.exists(PLOT_SAVE_DIR):
            os.makedirs(PLOT_SAVE_DIR)
        fig.savefig(os.path.join(PLOT_SAVE_DIR, ".".join([save_name, "png"])), 
                    dpi = 180,
                    bbox_inches = "tight")
    plt.show()
    
    return None

def convert_error_cat(actual: Iterable, desired: Iterable, threshold: float) -> list:
    return [1 if abs(a - d) >= threshold else 0 for a, d in zip(actual, desired)]

def process_data(raw_data: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, np.array]:
    columns_to_drop = ["DATASET", "Index", "WellID", "Well Name", "Loop", "Channel", "fps", "version", "Stop frame", "empty frames"]
    data = raw_data.drop(columns = columns_to_drop)
    
    actual = "Heartrate (BPM)"
    desired = "ground truth"  
    data[LABELS] = convert_error_cat(data[actual], data[desired], threshold)
    
    data = data.drop(columns = [actual, desired])
    
    scaler = MinMaxScaler()
    scaler.fit(data[qc_features])
    data[qc_features] = scaler.transform(data[qc_features])
    
    return data, scaler.scale_

def decision_tree(data: pd.DataFrame) -> sklearn.tree.DecisionTreeClassifier:
    Y = data.pop(LABELS)
    X_train, X_test, Y_train, Y_test = train_test_split(data, Y, test_size = 0.3, random_state = 104729)
    classifier = DecisionTreeClassifier(random_state = 224737, min_samples_split = 2)
    classifier.fit(X_train, Y_train)
    print(f"Accuracy achieved on test set: {(classifier.score(X_test, Y_test) * 100):.2f}%")
    return classifier
    
def plot_decision_tree(tree: sklearn.tree,
                       feature_names: Iterable[str],
                       class_names: Iterable[str] = ["no_error", "error"],
                       figsize: tuple = (16, 9),
                       save_q = True) -> None:
    
    
    plt.figure(figsize = figsize)
    sklearn.tree.plot_tree(tree,
                           feature_names = feature_names,
                           class_names = class_names,
                           fontsize = 14,
                           filled = True)
    
    if save_q:
        if not os.path.exists(PLOT_SAVE_DIR):
            os.makedirs(PLOT_SAVE_DIR)
        plt.figure().savefig(os.path.join(PLOT_SAVE_DIR, ".".join(["decision_tree", "jpg"])), format = "jpg", dpi = 180, bbox_inches = "tight")
    
def get_thresholds(unscaled_data: pd.DataFrame,
                   train_data_features: Iterable, 
                   classifier: sklearn.tree.DecisionTreeClassifier) -> dict:
    n_nodes = classifier.tree_.node_count
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold
    limits = dict()

    for i in range(n_nodes):
        if feature[i] != -2:
            curr_col = unscaled_data[train_data_features[feature[i]]]
            # Get the threshold for the current feature by reversing the scaling.
            actual_threshold = threshold[i] * (curr_col.max() - curr_col.min()) + curr_col.min() 
            if train_data_features[feature[i]] not in limits:
                limits[train_data_features[feature[i]]] = [actual_threshold]
            else:
                limits[train_data_features[feature[i]]].append(actual_threshold)
            
    # Clean limits.
    for _, v in limits.items():
        while (-2 in v):
            v.remove(-2)
            
    return limits

def process_limits(qc_thresolds: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(qc_thresolds, orient = "index")
    df["qc_mean"] = df.mean(axis = 1, skipna = True)
    df["qc_max"] = df.max(axis = 1, skipna = True)
    return df 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help = "Results from medaka", type = str, required = True)
    args = parser.parse_args()
    
    raw_data = pd.read_csv(args.file)
    # plot_qc_params(raw_data, save = True, save_name = "qc_params")
    data, scale = process_data(raw_data, 20)
    # print(data.head())
    classifier = decision_tree(data)
    plot_decision_tree(classifier, data.columns)
    limits = get_thresholds(raw_data, qc_features, classifier)
    plot_qc_params(raw_data, limits, save_q = True, save_name = "qc_params_thresholds", figsize = (10, 40))
    threshold_data = process_limits(limits)
    print(threshold_data)
    
if __name__ == "__main__":
    main()