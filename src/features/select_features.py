"""
Creation:
    Date: 2023-12-05
Description:
    Reduce feature list to the most important features
"""
import src.data.get_dataset

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Literal
import pandas as pd

dataframe = src.data.get_dataset.get_merged_datasets()


def is_data_leak(s, to_exclude):
    for data_leak in to_exclude:
        if data_leak in s:
            return True
    return False


def get_features_no_data_leaks(df=dataframe):
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


def combine_population_gendered_features(df=dataframe):
    features = df.columns
    fem_features = []
    male_features = []

    for feature in features:
        if 'male' in feature:
            if 'female' in feature:
                fem_features.append(feature)
            else:
                male_features.append(feature)

    fem_above_40 = []
    fem_below_40 = []
    male_above_40 = []
    male_below_40 = []

    cats_above_45 = ['40-44', '45-49', '50-54', '55-59', '60-64']
    for (feature_f, feature_m) in zip(fem_features, male_features):
        if any(x in feature_f for x in cats_above_45):
            fem_above_40.append(feature_f)
        else:
            fem_below_40.append(feature_f)

        if any(x in feature_m for x in cats_above_45):
            male_above_40.append(feature_m)
        else:
            male_below_40.append(feature_m)

    df['female_population_below_40_by_percent'] = df[fem_below_40].sum(axis=1)
    df['female_population_above_40_by_percent'] = df[fem_above_40].sum(axis=1)
    df['male_population_below_40_by_percent'] = df[male_below_40].sum(axis=1)
    df['male_population_above_40_by_percent'] = df[male_above_40].sum(axis=1)
    df.drop(columns=fem_features, inplace=True)
    df.drop(columns=male_features, inplace=True)
    return


def get_feature_importance(type : Literal["Linear", "RandomForest", "Tree"] = "Linear", df=dataframe):
    data_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ('std_scalar', StandardScaler())])
    combine_population_gendered_features(df)
    feature_list = get_features_no_data_leaks(df)
    x = dataframe[feature_list]
    y = dataframe['co2']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    data_pipeline.fit(x_train)
    x_train = data_pipeline.transform(x_train)
    if type == "Linear":
        model = LinearRegression()
        model.fit(x_train, y_train)
        feature_importance_data = pd.DataFrame(model.coef_, index=feature_list, columns=["Importance"])
        feature_importance_data.sort_values(by='Importance', ascending=False, inplace=True)
        return feature_importance_data
    elif type == "RandomForest":
        model = RandomForestRegressor(random_state=42)
    else:
        model = DecisionTreeRegressor(random_state=42)

    model.fit(x_train, y_train)
    feature_importance_data = pd.DataFrame(model.feature_importances_, index=feature_list, columns=["Importance"])
    feature_importance_data.sort_values(by='Importance', ascending=False, inplace=True)

    return feature_importance_data

