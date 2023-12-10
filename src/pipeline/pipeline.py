"""
Creation:
    Author: Martin Grunnill
    Date: 2023-12-09
Description: 
    
"""
import src.data.get_dataset

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

dataframe = src.data.get_dataset.get_merged_datasets()


def is_data_leak(s, to_exclude):
    for data_leak in to_exclude:
        if data_leak in s:
            return True
    return False


def get_features_no_data_leaks(df=dataframe):
    # we want to limit our feature set to columns without any data leaks, since in real prediction we wouldn't know
    # those values.
    data_leak_list = ["co2", "ghg", "greenhouse_gas", "change"]
    possible_data_leak_list = ["nitrous_oxide", "methane", "ch4", "n2o"]
    to_exclude = data_leak_list + possible_data_leak_list

    feature_list = []
    for x in df.columns:
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

    fem_child = []
    fem_work_age = []
    fem_retired = []
    male_child = []
    male_work_age = []
    male_retired = []

    children = ['0-4', '5-9', '10-14']
    retired = ['65']
    for (feature_f, feature_m) in zip(fem_features, male_features):
        if any(x in feature_f for x in children):
            fem_child.append(feature_f)
        elif any(x in feature_f for x in retired):
            fem_retired.append(feature_f)
        else:
            fem_work_age.append(feature_f)

        if any(x in feature_m for x in children):
            male_child.append(feature_m)
        elif any(x in feature_m for x in retired):
            male_retired.append(feature_m)
        else:
            male_work_age.append(feature_m)

    df['female_children_by_percent'] = df[fem_child].sum(axis=1)
    df['female_working_age_by_percent'] = df[fem_work_age].sum(axis=1)
    df['female_retired_by_percent'] = df[fem_retired].sum(axis=1)
    df['male_children_by_percent'] = df[male_child].sum(axis=1)
    df['male_working_age_by_percent'] = df[male_work_age].sum(axis=1)
    df['male_retired_by_percent'] = df[male_retired].sum(axis=1)
    df.drop(columns=fem_features, inplace=True)
    df.drop(columns=male_features, inplace=True)
    return


def drop_repeat_fuel_data(df=dataframe):
    drop_list = ['_per_', 'electricity', '_share_']
    for feature in df.columns:
        if any(item in feature for item in drop_list):
            df.drop(columns=feature, inplace=True)


def pipeline(df=dataframe, test_size=0.2, random_state=42):
    data_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean")), ('std_scalar', StandardScaler())])
    combine_population_gendered_features(df)
    drop_repeat_fuel_data(df)
    feature_list = get_features_no_data_leaks(df)
    x = df[feature_list]
    y = df['co2']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    data_pipeline.fit(x_train)
    x_train = data_pipeline.transform(x_train)
    x_test = data_pipeline.transform(x_test)
    return pd.DataFrame(x_train, columns=feature_list), pd.DataFrame(x_test, columns=feature_list), y_train, \
           y_test, feature_list
