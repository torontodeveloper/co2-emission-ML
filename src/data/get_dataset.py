"""
Creation:
    Author: Martin Grunnill
    Date: 2023-11-12
Description: 
    
"""

import pandas as pd

def get_merged_datasets(how='inner'):
    """
    Get CO2 and energy datasets and merge them.

    Parameters
    ----------
    how{‘left’, ‘right’, ‘outer’, ‘inner’, ‘cross’}, default ‘inner’
        Type of merge to be performed.
            left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
            right: use only keys from right frame, similar to a SQL right outer join; preserve key order.
            outer: use union of keys from both frames, similar to a SQL full outer join; sort keys lexicographically.
            inner: use intersection of keys from both frames, similar to a SQL inner join; preserve the order of the left keys.
            cross: creates the cartesian product from both frames, preserves the order of the left keys.

    """
    C02_df = pd.read_csv('https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv')
    energy_df =  pd.read_csv('https://nyc3.digitaloceanspaces.com/owid-public/data/energy/owid-energy-data.csv')
    to_merge_on = ['year', 'iso_code']
    cols_to_use = energy_df.columns.difference(C02_df.columns).tolist() + to_merge_on
    merged_df = C02_df.merge(energy_df[cols_to_use], how=how,on=to_merge_on)
    return merged_df


def get_merged_code_books():
    """
    Get CO2 and energy codebooks and merge them.
    """
    C02_cb = pd.read_csv('https://github.com/owid/co2-data/raw/master/owid-co2-codebook.csv')
    energy_cb = pd.read_csv('https://github.com/owid/energy-data/raw/master/owid-energy-codebook.csv')
    merged_cb = pd.concat([C02_cb,energy_cb[~energy_cb['column'].isin(C02_cb['column'])]])
    return merged_cb