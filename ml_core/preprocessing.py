import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.final_columns_ = None

        self.edu_map = {
            "No_Education": 0,
            "Primary": 1,
            "Secondary": 2,
            "Higher": 3
        }

        self.wealth_map = {
            "Poorest": 0,
            "Poorer": 1,
            "Middle": 2,
            "Richer": 3,
            "Richest": 4
        }

        self.cat_cols = ["Gender", "Region", "Age_Group"]
        self.disease_cols = ["Anemia", "Malaria", "Diarrhea", "TB"]
        self.nutrition_flags = ["Stunting", "Underweight", "Overweight"]

    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["BMI"] = X["Weight_kg"] / (X["Height_cm"] / 100.0) ** 2
        X["Weight_per_Month"] = X["Weight_kg"] / (X["Age (months)"] + 1)
        X["Height_per_Month"] = X["Height_cm"] / (X["Age (months)"] + 1)

        age_bins = [0, 6, 12, 24, 36, 48, 60]
        age_labels = ["0-6", "6-12", "12-24", "24-36", "36-48", "48-60"]
        X["Age_Group"] = pd.cut(
            X["Age (months)"], bins=age_bins,
            labels=age_labels, include_lowest=True
        ).astype("object").fillna("Unknown")

        X["Disease_Count"] = X[self.disease_cols].sum(axis=1)
        X["Has_Multiple_Diseases"] = (X["Disease_Count"] >= 2).astype(int)
        X["Nutrition_Risk_Score"] = X[self.nutrition_flags].sum(axis=1)

        X["Mother_Education_Ord"] = X["Mother_Education"].map(self.edu_map).fillna(1)
        X["Wealth_Ord"] = X["Household_Wealth_Index"].map(self.wealth_map).fillna(2)

        X = X.drop(columns=["Mother_Education", "Household_Wealth_Index"])

        return X

    def fit(self, X, y=None):
        X = self._feature_engineering(X)
        X = pd.get_dummies(X, columns=self.cat_cols, drop_first=True)

        self.final_columns_ = X.columns.tolist()
        self.num_cols_ = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        self.num_imputer.fit(X[self.num_cols_])
        return self

    def transform(self, X):
        X = self._feature_engineering(X)
        X = pd.get_dummies(X, columns=self.cat_cols, drop_first=True)

        for col in self.final_columns_:
            if col not in X.columns:
                X[col] = 0

        X = X[self.final_columns_]
        X[self.num_cols_] = self.num_imputer.transform(X[self.num_cols_])
        return X
