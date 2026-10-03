"""Task 5 - Domain-driven feature engineering for Ames Housing.

Features are generated only from information available at/before sale and do not use SalePrice.
The transformer is intended to live inside a scikit-learn Pipeline so the same transformation
is applied independently inside every CV fold and to the final hold-out set.
"""

from sklearn.base import BaseEstimator, TransformerMixin


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add leakage-safe, domain-driven Ames Housing features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def col(name, default=0):
            if name in X.columns:
                return X[name]
            return default

        # 1. Total usable square footage: combines major finished/livable areas.
        X["TotalSF"] = col("TotalBsmtSF") + col("1stFlrSF") + col("2ndFlrSF")

        # 2. House age at sale: captures depreciation / maturity of the property.
        if "YrSold" in X.columns and "YearBuilt" in X.columns:
            X["HouseAge"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
        else:
            X["HouseAge"] = 0

        # 3. Age since most recent remodel at sale: captures renovation recency.
        if "YrSold" in X.columns and "YearRemodAdd" in X.columns:
            X["YearsSinceRemodel"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
        else:
            X["YearsSinceRemodel"] = 0

        # 4. Total bathroom-equivalent count: half baths receive half weight.
        X["TotalBathrooms"] = (
            col("FullBath")
            + 0.5 * col("HalfBath")
            + col("BsmtFullBath")
            + 0.5 * col("BsmtHalfBath")
        )

        # 5. Remodel flag: whether the house was remodeled after construction.
        if "YearBuilt" in X.columns and "YearRemodAdd" in X.columns:
            X["HasRemodeled"] = (X["YearRemodAdd"] > X["YearBuilt"]).astype(int)
        else:
            X["HasRemodeled"] = 0

        return X


FEATURE_DESCRIPTIONS = {
    "TotalSF": "Total basement + first-floor + second-floor square footage; summarizes overall usable size.",
    "HouseAge": "Year sold minus year built; captures property age at the time of sale.",
    "YearsSinceRemodel": "Year sold minus most recent remodel year; captures renovation recency.",
    "TotalBathrooms": "Full baths plus half baths weighted at 0.5, including basement bathrooms; captures bathroom capacity.",
    "HasRemodeled": "Binary indicator for whether YearRemodAdd is later than YearBuilt; captures whether the home has been remodeled.",
}


if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Preview engineered Ames Housing features.")
    parser.add_argument("input_csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    engineered = AmesFeatureEngineer().fit_transform(df)
    print("Added features:")
    for feature in FEATURE_DESCRIPTIONS:
        print(f"- {feature}")
    print("Output shape:", engineered.shape)
