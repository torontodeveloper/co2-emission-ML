"""
Creation:
    Date: 2023-12-05
Description:
    Reduce feature list to the most important features
"""

import os
import sys

sys.path.append(os.path.abspath("/../../src/"))
import src.data.get_dataset

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import pandas as pd

dataframe = src.data.get_dataset.get_merged_datasets()


def is_data_leak(s, to_exclude):
    for data_leak in to_exclude:
        if data_leak in s:
            return True
    return False


def get_features_no_data_leaks():
    numeric_cols = dataframe.select_dtypes(include=['number']).columns.difference(
        ['year']).tolist()  # we don't want year to be a feature in our prediction model
    # we want to limit our feature set to columns without any data leaks, since in real prediction we wouldn't know
    # those values.
    data_leak_list = ["co2", "ghg", "greenhouse_gas", "change"]
    possible_data_leak_list = ["nitrous_oxide", "methane", "ch4", "n2o"]
    to_exclude = data_leak_list + possible_data_leak_list

    feature_list = []
    for x in numeric_cols:
        is_leak = is_data_leak(x, to_exclude)
        if not is_leak:
            feature_list.append(x)

    return feature_list


def get_important_features(num=20):
    data_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ('std_scalar', StandardScaler())])
    feature_list = get_features_no_data_leaks()
    x = dataframe[feature_list]
    y = dataframe['co2']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    data_pipeline.fit(x_train)
    x_train = data_pipeline.transform(x_train)

    decision_tree = DecisionTreeRegressor()
    decision_tree.fit(x_train, y_train)

    feature_importance_data = pd.DataFrame(decision_tree.feature_importances_, index=feature_list,
                                           columns=["Importance"])
    feature_importance_data.sort_values(by='Importance', ascending=False, inplace=True)
    return feature_importance_data.head(num).index.tolist()
