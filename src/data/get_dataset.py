"""
Creation:
    Date: 2023-11-12
Description:
    Get Dataset from different sources for C02 and combine them into one CSV file
"""
from typing import Literal
import pandas as pd
from pandas import DataFrame

df: DataFrame


def get_merged_datasets(how: Literal['inner', 'left', 'right', 'outer', 'cross'] = 'inner'):
    """
    Get CO2 and energy datasets and merge them.

    Parameters ---------- :param how{‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’ Type of merge to be
    performed. left: use only keys from left frame, similar to a SQL left outer join; preserve key order. right: use
    only keys from right frame, similar to a SQL right outer join; preserve key order. outer: use union of keys from
    both frames, similar to a SQL full outer join; sort keys lexicographically. inner: use intersection of keys from
    both frames, similar to a SQL inner join; preserve the order of the left keys. cross: creates the cartesian
    product from both frames, preserves the order of the left keys.

    """
    C02_df = pd.read_csv('https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv')
    energy_df = pd.read_csv('https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv')
    to_merge_on = ['year', 'iso_code', 'country']
    C02_df.dropna(subset=to_merge_on, inplace=True)
    energy_df.dropna(subset=to_merge_on, inplace=True)
    cols_to_use = energy_df.columns.difference(C02_df.columns).tolist() + to_merge_on
    csv_data = C02_df.merge(energy_df[cols_to_use], how=how, on=to_merge_on)
    csv_data.dropna(subset=['gdp', 'co2'], inplace=True)
    csv_data.query('year >= 2000', inplace=True)

    return get_extra_features(csv_data, to_merge_on, how)


def get_extra_features(df: DataFrame, to_merge_on, how: Literal['inner', 'left', 'right', 'outer', 'cross'] = 'inner'):
    """
    Adds land-use, median age, and population data per country and per year to the main dataset.
    @param df: main data frame to add other features on
    @param to_merge_on: columns to merge data set on
    @param how: {'inner', 'left', 'right', 'outer', 'cross'}, default 'inner'
    """
    rename_dict = {'Entity': 'country', 'Year': 'year', 'Code': 'iso_code'}

    land_use_df = pd.read_csv(
        "https://dataset-ml-project.s3.us-east-2.amazonaws.com/land-use.csv")  # source: https://ourworldindata.org
    # /land-use
    land_use_df.rename(columns=rename_dict, inplace=True)
    # :DataFrame specifies what object type gets returned
    merged_df: DataFrame = pd.merge(df, land_use_df, on=to_merge_on, how=how, copy=False)

    median_age_df = pd.read_csv(
        "https://dataset-ml-project.s3.us-east-2.amazonaws.com/median-age.csv")  # source: https://ourworldindata.org
    # /grapher/median-age
    median_age_df.rename(columns=rename_dict, inplace=True)
    median_age_df.drop(columns='Median age - Sex: all - Age: all - Variant: medium', inplace=True)
    median_age_df.dropna(subset='Median age - Sex: all - Age: all - Variant: estimates', inplace=True)
    median_age_df.rename(columns={'Median age - Sex: all - Age: all - Variant: estimates': 'Median age'}, inplace=True)
    merged_df = pd.merge(merged_df, median_age_df, on=to_merge_on, how=how, copy=False)

    population = pd.read_csv(
        "https://dataset-ml-project.s3.us-east-2.amazonaws.com/population.csv")  # source: https://ourworldindata.org
    # /grapher/population
    population.rename(columns=rename_dict, inplace=True)
    merged_df = pd.merge(merged_df, population, on=to_merge_on, how=how, copy=False)

    return merged_df


if __name__ == '__main__':
    get_merged_datasets()
