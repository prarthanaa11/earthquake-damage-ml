from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "damage_grade"
ID_COLUMN = "building_id"


def load_training_data(data_dir="data"):
    values = pd.read_csv(
        Path(data_dir) / "train_values.csv"
    )

    labels = pd.read_csv(
        Path(data_dir) / "train_labels.csv"
    )

    df = values.merge(
        labels,
        on=ID_COLUMN,
        how="inner"
    )

    return df


def split_features_target(df):
    X = df.drop(
        columns=[TARGET, ID_COLUMN],
        errors="ignore"
    )

    y = df[TARGET]

    return X, y


def get_feature_types(X):
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    return numeric_features, categorical_features


def build_preprocessor(X):
    numeric_features, categorical_features = get_feature_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_features
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return (
        preprocessor,
        numeric_features,
        categorical_features
    )