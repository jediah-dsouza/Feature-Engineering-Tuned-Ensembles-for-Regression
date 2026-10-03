"""Task 5 — Feature Engineering & Tuned Ensembles for Regression."""
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from feature_engineering import AmesFeatureEngineer, FEATURE_DESCRIPTIONS

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


def prepare_features(df):
    y = df["SalePrice"].copy()
    X = df.drop(columns=["SalePrice"]).copy()
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])
    if "MSSubClass" in X.columns:
        X["MSSubClass"] = X["MSSubClass"].astype(str)
    return X, y


def build_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), lambda X: X.select_dtypes(exclude=["object", "category", "string"]).columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), lambda X: X.select_dtypes(include=["object", "category", "string"]).columns),
    ])


def make_pipeline(estimator):
    return Pipeline([
        ("features", AmesFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", estimator),
    ])


def make_log_target(estimator):
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def fold_metrics(estimator, X, y, cv):
    rmses, r2s = [], []
    for train_idx, valid_idx in cv.split(X, y):
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = estimator.predict(X.iloc[valid_idx])
        rmses.append(mean_squared_error(y.iloc[valid_idx], pred) ** 0.5)
        r2s.append(r2_score(y.iloc[valid_idx], pred))
    return np.array(rmses), np.array(r2s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Task 1 cleaned Ames CSV")
    parser.add_argument("report_file", nargs="?", default="MODEL_COMPARISON.md")
    parser.add_argument("--n-iter", type=int, default=10,
                        help="RandomizedSearchCV iterations per ensemble")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if "SalePrice" not in df.columns:
        raise ValueError("Input dataset must contain SalePrice.")

    X, y = prepare_features(df)
    # Exact Task 3 split: 80/20, random_state=42. Hold-out remains untouched during tuning.
    X_dev, X_holdout, y_dev, y_holdout = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Task 3-style Ridge baseline with the Task 5 engineered features and log target.
    baseline = make_log_target(make_pipeline(Ridge(alpha=0.1)))
    base_start = time.perf_counter()
    base_cv = cross_validate(
        baseline, X_dev, y_dev, cv=cv,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=1, return_train_score=False,
    )
    base_time = time.perf_counter() - base_start
    base_rmse = -base_cv["test_rmse"]
    base_r2 = base_cv["test_r2"]

    results = [{
        "Model": "Task 3-style Ridge (alpha=0.1)",
        "Mean CV RMSE": base_rmse.mean(), "CV RMSE Std": base_rmse.std(),
        "Mean CV R²": base_r2.mean(), "CV R² Std": base_r2.std(),
        "Training time (s)": base_time,
        "Note": "Task 3-style regularized linear baseline with Task 5 features; log target.",
    }]

    configs = [
        ("Random Forest", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1), {
            "regressor__model__n_estimators": [300, 500, 700],
            "regressor__model__max_depth": [None, 10, 20, 30],
            "regressor__model__min_samples_split": [2, 5, 10],
            "regressor__model__min_samples_leaf": [1, 2, 4],
            "regressor__model__max_features": [0.5, 0.8, 1.0],
        }),
        ("XGBoost", XGBRegressor(
            objective="reg:squarederror", eval_metric="rmse",
            random_state=RANDOM_STATE, n_jobs=1, tree_method="hist"
        ), {
            "regressor__model__n_estimators": [300, 500, 800, 1200],
            "regressor__model__max_depth": [2, 3, 4, 5, 6],
            "regressor__model__learning_rate": [0.02, 0.04, 0.06, 0.1],
            "regressor__model__subsample": [0.7, 0.85, 1.0],
            "regressor__model__colsample_bytree": [0.7, 0.85, 1.0],
            "regressor__model__min_child_weight": [1, 3, 5],
            "regressor__model__reg_alpha": [0.0, 0.01, 0.1],
            "regressor__model__reg_lambda": [1.0, 2.0, 5.0],
        }),
    ]

    searches = []
    for name, estimator, params in configs:
        pipe = make_pipeline(estimator)
        wrapped = make_log_target(pipe)
        start = time.perf_counter()
        search = RandomizedSearchCV(
            wrapped, params, n_iter=args.n_iter, scoring="neg_root_mean_squared_error",
            cv=cv, random_state=RANDOM_STATE, n_jobs=1, refit=True, verbose=0,
        )
        search.fit(X_dev, y_dev)
        elapsed = time.perf_counter() - start
        # Re-score the selected best estimator fold-by-fold to obtain RMSE and R² together.
        rmses, r2s = fold_metrics(search.best_estimator_, X_dev, y_dev, cv)
        results.append({
            "Model": name,
            "Mean CV RMSE": rmses.mean(), "CV RMSE Std": rmses.std(),
            "Mean CV R²": r2s.mean(), "CV R² Std": r2s.std(),
            "Training time (s)": elapsed,
            "Note": f"RandomizedSearchCV over {len(params)} hyperparameters; best params: {search.best_params_}.",
        })
        searches.append((name, search, elapsed))

    # Final hold-out evaluation, exactly once for the selected ensemble models.
    holdout = []
    base_final = make_log_target(make_pipeline(Ridge(alpha=0.1)))
    base_start = time.perf_counter(); base_final.fit(X_dev, y_dev); base_fit = time.perf_counter() - base_start
    base_pred = base_final.predict(X_holdout)
    holdout.append(("Task 3-style Ridge (alpha=0.1)", mean_squared_error(y_holdout, base_pred)**0.5,
                    r2_score(y_holdout, base_pred), base_fit))
    for name, search, elapsed in searches:
        pred = search.best_estimator_.predict(X_holdout)
        holdout.append((name, mean_squared_error(y_holdout, pred)**0.5,
                        r2_score(y_holdout, pred), elapsed))

    results_df = pd.DataFrame(results)
    holdout_df = pd.DataFrame(holdout, columns=["Model", "Hold-out RMSE", "Hold-out R²", "Training time (s)"])

    best = min(results[1:], key=lambda r: r["Mean CV RMSE"])
    improvement = (results[0]["Mean CV RMSE"] - best["Mean CV RMSE"]) / results[0]["Mean CV RMSE"] * 100

    report = [
        "# Task 5 — Feature Engineering & Tuned Ensembles for Regression\n",
        "## Dataset and validation strategy\n",
        f"- Input: `{Path(args.input_csv).name}` ({len(df):,} rows).",
        "- Target: `SalePrice`; all models use `log1p(SalePrice)` and return predictions to the original dollar scale for evaluation.",
        "- Task 3 hold-out preserved exactly: 80/20, `random_state=42`.",
        f"- Development data: **{len(X_dev):,}** rows; untouched hold-out: **{len(X_holdout):,}** rows.",
        "- Cross-validation: **5-fold shuffled KFold**, `random_state=42`.",
        "- Feature engineering, imputation, scaling, and one-hot encoding are inside the Pipeline and therefore fitted independently inside each CV fold.\n",
        "## Engineered features and justification\n",
    ]
    for name, desc in FEATURE_DESCRIPTIONS.items():
        report.append(f"- **{name}:** {desc}")
    report += [
        "\nAll engineered features are computed only from property characteristics and sale-year information available without `SalePrice`; no target-derived feature is used.\n",
        "## Required CV comparison\n",
        "Mean CV RMSE is shown as **mean ± standard deviation** across five folds.\n",
        "| Model | Mean CV RMSE ± SD | Mean CV R² | Training time (s) | Note |",
        "|---|---:|---:|---:|---|",
    ]
    for r in results:
        report.append(f"| {r['Model']} | ${r['Mean CV RMSE']:,.2f} ± ${r['CV RMSE Std']:,.2f} | {r['Mean CV R²']:.4f} | {r['Training time (s)']:.2f} | {r['Note']} |")
    report += [
        "\n## Final untouched hold-out check\n",
        "The hold-out was not used for hyperparameter tuning and is evaluated only after model selection.\n",
        "| Model | Hold-out RMSE | Hold-out R² | Training time (s) |",
        "|---|---:|---:|---:|",
    ]
    for _, r in holdout_df.iterrows():
        report.append(f"| {r['Model']} | ${r['Hold-out RMSE']:,.2f} | {r['Hold-out R²']:.4f} | {r['Training time (s)']:.2f} |")
    report += [
        "\n## Verdict\n",
        f"The tuned **{best['Model']}** produced the lowest mean 5-fold CV RMSE (${best['Mean CV RMSE']:,.2f} ± ${best['CV RMSE Std']:,.2f}). Relative to the Task 3-style Ridge baseline with the same engineered features, the CV RMSE reduction is **{improvement:.2f}%**.",
        "\nThe final shipping decision should consider both the size and stability of the error reduction and the additional training/implementation complexity. A small reduction relative to the fold-to-fold variation should not be presented as a meaningful improvement without qualification. The untouched hold-out results above provide the final generalization check.\n",
        "## Leakage and quality checks\n",
        "- `SalePrice` is never used to create engineered features.",
        "- `Id` is removed as an identifier.",
        "- Preprocessing is inside the Pipeline and fitted only on each CV training fold.",
        "- RandomizedSearchCV tunes on the development partition only.",
        "- The 20% Task 3 hold-out remains untouched until final evaluation.\n",
        "## Reproduce\n",
        "```bash",
        "pip install pandas numpy scikit-learn xgboost",
        "python tuned_models.py ames_cleaned.csv",
        "```",
    ]
    Path(args.report_file).write_text("\n".join(report), encoding="utf-8")

    print(f"Input shape:       {df.shape}")
    print(f"Development shape: {X_dev.shape}")
    print(f"Hold-out shape:    {X_holdout.shape}")
    print("\nCV comparison:")
    print(results_df.to_string(index=False))
    print("\nFinal hold-out results:")
    print(holdout_df.to_string(index=False))
    print(f"\nSaved report to: {args.report_file}")


if __name__ == "__main__":
    main()
