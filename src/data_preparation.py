import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_data(data):
    # Check duplicates before removing IDs so distinct customers are retained.
    data = data.drop_duplicates().copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    if not data["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn must be Yes or No, with no missing values.")
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0}).astype(int)
    return data.drop(columns="customerID")


def prepare_features(data):
    X = data.drop(columns="Churn")
    y = data["Churn"]
    return X, y


def create_preprocessor(X):
    numerical_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([
        ("numeric", numerical_pipeline, numerical_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ])
