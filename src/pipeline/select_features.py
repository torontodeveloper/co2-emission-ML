"""
Creation:
    Date: 2023-12-05
Description:
    Reduce feature list to the most important pipeline
"""
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import Literal
import pandas as pd



def get_feature_importance(x_train, y_train, feature_list,
                           type : Literal["Linear", "RandomForest", "Tree"] = "Linear",
                           random_state=42):

    if type == "Linear":
        model = LinearRegression()
        model.fit(x_train, y_train)
        feature_importance_data = pd.DataFrame(model.coef_, index=feature_list, columns=["Importance"])
        feature_importance_data.sort_values(by='Importance', ascending=False, inplace=True)
        return feature_importance_data
    elif type == "RandomForest":
        model = RandomForestRegressor(random_state=random_state)
    else:
        model = DecisionTreeRegressor(random_state=random_state)

    model.fit(x_train, y_train)
    feature_importance_data = pd.DataFrame(model.feature_importances_, index=feature_list, columns=["Importance"])
    feature_importance_data.sort_values(by='Importance', ascending=False, inplace=True)

    return feature_importance_data

