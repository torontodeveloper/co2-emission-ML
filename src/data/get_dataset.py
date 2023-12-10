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

    return get_extra_features(csv_data, to_merge_on)


def get_extra_features(df: DataFrame, to_merge_on):
    """
    Adds land-use, median age, and population data per country and per year to the main dataset.
    @param df: main data frame to add other pipeline on
    @param to_merge_on: columns to merge data set on
    """
    how = 'left'
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
    median_age_df.rename(columns={'Median age - Sex: all - Age: all - Variant: estimates': 'median_age'}, inplace=True)
    median_age_df.dropna(subset='median_age', inplace=True)

    merged_df = pd.merge(merged_df, median_age_df, on=to_merge_on, how=how, copy=False)

    military_expenditure_df = pd.read_csv("https://dataset-ml-project.s3.us-east-2.amazonaws.com"
                                          "/military-expenditure-share-gdp.csv")
    # https://ourworldindata.org/grapher/military-expenditure-share-gdp
    military_expenditure_df.rename(columns=rename_dict, inplace=True)
    merged_df = pd.merge(merged_df, military_expenditure_df, on=to_merge_on, how=how, copy=False)

    rename_dict = {'Country Name': 'country', 'Time': 'year', 'Country Code': 'iso_code'}
    demographics_df = pd.read_csv("https://dataset-ml-project.s3.us-east-2.amazonaws.com/Demographics_WDI.csv")
    # source: https://databank.worldbank.org/reports.aspx?source=world-development-indicators
    demographics_df.rename(columns=rename_dict, inplace=True)
    demographics_df.drop(columns='Time Code', inplace=True)

    merged_df = pd.merge(merged_df, demographics_df, on=to_merge_on, how=how, copy=False)
    import_export_df = pd.read_csv("https://dataset-ml-project.s3.us-east-2.amazonaws.com"
                                   "/Imports_exports_from_WDI_Data.csv")
    # source: https://databank.worldbank.org/reports.aspx?source=world-development-indicators

    import_export_df.rename(columns=rename_dict, inplace=True)
    import_export_df.drop(columns=['Time Code'], inplace=True)

    merged_df = pd.merge(merged_df, import_export_df, on=to_merge_on, how=how, copy=False)

    columns = merged_df.columns
    new_columns = [x.lower().replace(" ", "_") for x in columns]
    for i in range(0, len(columns)):
        merged_df.rename(columns={columns[i]: new_columns[i]}, inplace=True)

    merged_df.dropna(subset=['co2'], inplace=True)
    remove_elipses(merged_df)
    return merged_df


def remove_elipses(df):
    cols = [i for i in df.columns if i not in ["country", "iso_code"]]
    for col in cols:
        df.drop(index=df[df[col] == '..'].index, inplace=True)
        df[col] = pd.to_numeric(df[col])


if __name__ == '__main__':
    print(get_merged_datasets().columns)
