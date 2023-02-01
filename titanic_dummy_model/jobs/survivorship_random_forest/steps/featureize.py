from typing import Tuple

import pandas as pd
import pandas.api.types as ptypes
from edscommon.job_api import Context, DefaultResource
from sklearn.preprocessing import OneHotEncoder

from titanic_dummy_model.jobs.survivorship_random_forest.df_transformers import (
    add_column,
    add_cols,
    update_col,
)


def featureize_step(context: Context[DefaultResource], passengers: pd.DataFrame) -> pd.DataFrame:
    """ """
    context.log.debug("Starting featureize_step")
    passengers["Cabin"] = passengers["Cabin"].fillna("U0")
    passengers["Fare"] = passengers["Fare"].astype("int64")
    # passengers["Age"] = passengers["Age"].fillna(value=int(passengers["Age"].mean())).astype("int64")
    # passengers["Fare"] = passengers["Fare"].astype("int64")
    passengers = deck_column_(passengers)
    passengers = fill_missing_embarked_values_(passengers)
    passengers = add_is_male_column(passengers)

    return passengers


@add_cols(new_col_name="Deck", src_col="Cabin", ptype_fn=ptypes.is_categorical)
def deck_column_(cabin: pd.Series) -> pd.Series:
    cabin_to_deck_map = {cabin_letter: deck for deck, cabin_letter in enumerate("ABCDEFGU", start=1)}
    cabin_letters = cabin.str.extract("([a-zA-Z]+)")[0]
    cabin_letters = cabin_letters.replace(cabin_to_deck_map).fillna(0).astype("category")
    return cabin_letters


@add_cols(new_col_name="is_male", src_col="Sex")
def add_is_male_column(sex: pd.Series) -> pd.Series:
    male_ = sex == "male"
    return male_


@update_col(col="Embarked")
def fill_missing_embarked_values_(embarked: pd.Series) -> pd.Series:
    return embarked.fillna("S")


@update_col(col="Sex", ptype_fn=ptypes.is_categorical)
def make_sex_col_categorical(sex: pd.Series) -> pd.Series:
    return sex.astype("category")


def make_model_input_step(passenger_features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = passenger_features["Survived"]

    features = passenger_features[["is_male", "Fare"]]
    dummies = pd.get_dummies(features, "Deck")
    features = pd.concat([features, dummies], axis=1)

    return features, target
