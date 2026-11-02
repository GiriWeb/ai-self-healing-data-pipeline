
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_validate
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


# ==================================================
# CONFIGURATION
# ==================================================

MAX_TRAIN_ROWS = 20000
CV_FOLDS = 5
RANDOM_STATE = 42


# ==================================================
# RAW INPUT MODEL
# ==================================================

class RawInputModel:

    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)

        if not hasattr(self.model, "predict_proba"):
            raise ValueError(
                "This model does not support prediction probabilities."
            )

        return self.model.predict_proba(X_processed)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

    @property
    def coef_(self):
        return self.model.coef_

    @property
    def classes_(self):
        return self.model.classes_


# ==================================================
# PROBLEM TYPE DETECTION
# ==================================================

def detect_problem_type(df, target_column):

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    target = df[target_column].dropna()

    if target.empty:
        raise ValueError("Target column has no valid values.")

    if target.nunique() < 2:
        raise ValueError("Target must contain at least 2 unique values.")

    # Text and categorical targets
    if (
        pd.api.types.is_object_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(target)
    ):
        return "Classification"

    # Numeric targets
    if pd.api.types.is_numeric_dtype(target):

        unique_count = target.nunique()

        # Binary numeric target
        if unique_count == 2:
            return "Classification"

        # Small integer-valued targets are treated as classes
        values = pd.to_numeric(target, errors="coerce").dropna()

        is_integer_target = np.all(
            np.isclose(values % 1, 0)
        )

        if unique_count <= 15 and is_integer_target:
            return "Classification"

        return "Regression"

    return "Classification"


# ==================================================
# DATA PREPARATION
# ==================================================

def prepare_data(df, target_column, problem_type):

    if target_column not in df.columns:
        raise ValueError("Target column not found.")

    data = df.copy()

    # Remove missing target rows
    data = data.dropna(subset=[target_column])

    if problem_type == "Regression":

        y = pd.to_numeric(
            data[target_column],
            errors="coerce"
        )

        valid = y.notna() & np.isfinite(y)

        data = data.loc[valid].copy()
        y = y.loc[valid]

    else:

        y = data[target_column].astype(str)

    X = data.drop(columns=[target_column])

    # Remove completely empty columns
    X = X.dropna(axis=1, how="all")

    # Remove constant columns
    X = X.loc[:, X.nunique(dropna=False) > 1]

    if X.shape[1] == 0:
        raise ValueError("No useful feature columns found.")

    if len(X) < 10:
        raise ValueError(
            "At least 10 valid rows are required."
        )

    if problem_type == "Classification":

        if y.nunique() < 2:
            raise ValueError(
                "Classification needs at least 2 classes."
            )

        if y.value_counts().min() < 2:
            raise ValueError(
                "Each classification class needs at least 2 rows."
            )

    return X, y


# ==================================================
# PREPROCESSING
# ==================================================

def create_preprocessor(X):

    numeric_cols = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    transformers = []

    if numeric_cols:

        numeric_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ])

        transformers.append(
            ("num", numeric_pipeline, numeric_cols)
        )

    if categorical_cols:

        categorical_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ])

        transformers.append(
            ("cat", categorical_pipeline, categorical_cols)
        )

    if not transformers:
        raise ValueError("No usable features found.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ==================================================
# TRAINING ROW LIMIT
# ==================================================

def limit_training_rows(X_train, y_train, problem_type):

    if len(X_train) <= MAX_TRAIN_ROWS:
        return X_train, y_train

    if problem_type == "Classification":

        X_train, _, y_train, _ = train_test_split(
            X_train,
            y_train,
            train_size=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE,
            stratify=y_train
        )

    else:

        X_train = X_train.sample(
            n=MAX_TRAIN_ROWS,
            random_state=RANDOM_STATE
        )

        y_train = y_train.loc[X_train.index]

    return X_train, y_train


# ==================================================
# CROSS VALIDATION
# ==================================================

def perform_cross_validation(X, y, models, problem_type):

    results = []

    if problem_type == "Regression":

        folds = min(CV_FOLDS, len(X))

        if folds < 2:
            raise ValueError(
                "Not enough rows for cross-validation."
            )

        cv = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        scoring = {
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        }

    else:

        min_class_count = int(y.value_counts().min())

        folds = min(CV_FOLDS, min_class_count)

        # CV unavailable for very small class counts
        if folds < 2:

            for name in models:

                results.append({
                    "Model": name,
                    "CV Accuracy": np.nan,
                    "CV F1 Score": np.nan,
                    "CV F1 Std": np.nan
                })

            return pd.DataFrame(results)

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        scoring = {
            "accuracy": "accuracy",
            "f1": "f1_weighted"
        }

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", create_preprocessor(X)),
            ("model", model)
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise"
        )

        if problem_type == "Regression":

            results.append({
                "Model": name,
                "CV MAE": -scores["test_mae"].mean(),
                "CV RMSE": -scores["test_rmse"].mean(),
                "CV R2": scores["test_r2"].mean(),
                "CV RMSE Std": scores["test_rmse"].std()
            })

        else:

            results.append({
                "Model": name,
                "CV Accuracy": scores["test_accuracy"].mean(),
                "CV F1 Score": scores["test_f1"].mean(),
                "CV F1 Std": scores["test_f1"].std()
            })

    return pd.DataFrame(results)


# ==================================================
# OVERFITTING DETECTION
# ==================================================

def detect_overfitting(
    model,
    X_train_processed,
    y_train,
    X_test_processed,
    y_test,
    problem_type
):

    train_predictions = model.predict(X_train_processed)

    test_predictions = model.predict(X_test_processed)

    if problem_type == "Regression":

        train_score = r2_score(
            y_train,
            train_predictions
        )

        test_score = r2_score(
            y_test,
            test_predictions
        )

        gap = train_score - test_score

        if gap > 0.20:
            status = "Possible Overfitting"

        elif gap < -0.20:
            status = "Check Data Distribution"

        else:
            status = "Good Generalization"

    else:

        train_score = accuracy_score(
            y_train,
            train_predictions
        )

        test_score = accuracy_score(
            y_test,
            test_predictions
        )

        gap = train_score - test_score

        if gap > 0.15:
            status = "Possible Overfitting"

        elif gap < -0.15:
            status = "Check Data Distribution"

        else:
            status = "Good Generalization"

    return {
        "Training Score": train_score,
        "Testing Score": test_score,
        "Performance Gap": gap,
        "Overfitting Status": status
    }


# ==================================================
# REGRESSION TRAINING
# ==================================================

def train_regression_models(df, target_column):

    X, y = prepare_data(
        df,
        target_column,
        "Regression"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    X_train, y_train = limit_training_rows(
        X_train,
        y_train,
        "Regression"
    )

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=10,
            random_state=RANDOM_STATE
        )
    }

    cv_results = perform_cross_validation(
        X_train,
        y_train,
        models,
        "Regression"
    )

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    results = []
    trained_models = {}
    overfitting_results = {}

    for name, model in models.items():

        model.fit(
            X_train_processed,
            y_train
        )

        predictions = model.predict(
            X_test_processed
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        overfit_info = detect_overfitting(
            model,
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
            "Regression"
        )

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2 Score": r2,
            **overfit_info
        })

        trained_models[name] = RawInputModel(
            preprocessor,
            model
        )

        overfitting_results[name] = overfit_info

    results_df = pd.DataFrame(results)

    results_df = results_df.merge(
        cv_results,
        on="Model"
    )

    # Primary selection: lowest cross-validation RMSE.
    # Fallback: lowest test RMSE if CV is unavailable.
    if results_df["CV RMSE"].notna().any():

        results_df = results_df.sort_values(
            ["CV RMSE", "CV RMSE Std"],
            ascending=[True, True]
        )

    else:

        results_df = results_df.sort_values("RMSE")

    best_model_name = results_df.iloc[0]["Model"]

    return {
        "problem_type": "Regression",
        "results": results_df,
        "cv_results": cv_results,
        "overfitting_results": overfitting_results,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
        "trained_models": trained_models,
        "preprocessor": preprocessor,
        "feature_columns": X.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test,
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }


# ==================================================
# CLASSIFICATION TRAINING
# ==================================================

def train_classification_models(df, target_column):

    X, y = prepare_data(
        df,
        target_column,
        "Classification"
    )

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y
        )

    except ValueError as error:

        raise ValueError(
            f"Unable to split classification data: {error}"
        )

    X_train, y_train = limit_training_rows(
        X_train,
        y_train,
        "Classification"
    )

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE
        )
    }

    cv_results = perform_cross_validation(
        X_train,
        y_train,
        models,
        "Classification"
    )

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    results = []
    trained_models = {}
    overfitting_results = {}

    for name, model in models.items():

        model.fit(
            X_train_processed,
            y_train
        )

        predictions = model.predict(
            X_test_processed
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )

        overfit_info = detect_overfitting(
            model,
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
            "Classification"
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            **overfit_info
        })

        trained_models[name] = RawInputModel(
            preprocessor,
            model
        )

        overfitting_results[name] = overfit_info

    results_df = pd.DataFrame(results)

    results_df = results_df.merge(
        cv_results,
        on="Model"
    )

    # Primary selection: highest cross-validation F1.
    # Fallback: highest test F1 if CV is unavailable.
    if results_df["CV F1 Score"].notna().any():

        results_df = results_df.sort_values(
            ["CV F1 Score", "CV F1 Std"],
            ascending=[False, True]
        )

    else:

        results_df = results_df.sort_values(
            "F1 Score",
            ascending=False
        )

    best_model_name = results_df.iloc[0]["Model"]

    return {
        "problem_type": "Classification",
        "results": results_df,
        "cv_results": cv_results,
        "overfitting_results": overfitting_results,
        "best_model_name": best_model_name,
        "best_model": trained_models[best_model_name],
        "trained_models": trained_models,
        "preprocessor": preprocessor,
        "feature_columns": X.columns.tolist(),
        "X_test": X_test,
        "y_test": y_test,
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }


# ==================================================
# INTELLIGENT MODEL SELECTION
# ==================================================

def train_suitable_models(
    df,
    target_column,
    problem_type="Automatic"
):

    if problem_type == "Automatic":

        detected_type = detect_problem_type(
            df,
            target_column
        )

    elif problem_type in ["Regression", "Classification"]:

        detected_type = problem_type

    else:

        raise ValueError(
            "Problem type must be Automatic, Regression or Classification."
        )

    print("=" * 50)
    print("INTELLIGENT MODEL SELECTION")
    print("=" * 50)

    print(f"Target Column: {target_column}")
    print(f"Detected Problem Type: {detected_type}")

    if detected_type == "Regression":

        output = train_regression_models(
            df,
            target_column
        )

    else:

        output = train_classification_models(
            df,
            target_column
        )

    print("\nMODEL COMPARISON")
    print(output["results"].to_string(index=False))

    print("\nBEST MODEL:", output["best_model_name"])

    return output
