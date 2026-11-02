
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px

from models.model_selection import train_suitable_models
from models.model_saver import save_model, load_model


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Self-Healing ML Pipeline",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Self-Healing ML Pipeline")

st.write(
    "Upload datasets, clean and analyze data, select suitable "
    "ML models and generate predictions."
)


# ==================================================
# SESSION STATE
# ==================================================

DEFAULT_STATE = {
    "ml_output": None,
    "model_config": None,
    "loaded_model_package": None,
    "dataset_signature": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# HELPER: CREATE PREDICTION FORM
# ==================================================

def create_prediction_form(metadata, form_key):

    prediction_input = {}

    col1, col2 = st.columns(2)

    with st.form(form_key):

        for index, (feature, details) in enumerate(metadata.items()):

            container = col1 if index % 2 == 0 else col2

            with container:

                feature_type = details.get("type")

                if feature_type == "numeric":

                    prediction_input[feature] = st.number_input(
                        str(feature),
                        value=float(details.get("default", 0.0)),
                        step=(
                            1.0 if details.get("integer", False)
                            else 0.1
                        ),
                        key=f"{form_key}_{feature}"
                    )

                elif feature_type == "boolean":

                    prediction_input[feature] = st.selectbox(
                        str(feature),
                        [False, True],
                        key=f"{form_key}_{feature}"
                    )

                else:

                    options = details.get("options", [])

                    if options:

                        prediction_input[feature] = st.selectbox(
                            str(feature),
                            options,
                            key=f"{form_key}_{feature}"
                        )

                    else:

                        prediction_input[feature] = st.text_input(
                            str(feature),
                            value=str(details.get("default", "")),
                            key=f"{form_key}_{feature}"
                        )

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:

        return pd.DataFrame(
            [prediction_input],
            columns=list(metadata.keys())
        )

    return None


# ==================================================
# HELPER: CREATE FEATURE METADATA
# ==================================================

def create_feature_metadata(df, feature_columns):

    feature_metadata = {}

    for feature in feature_columns:

        series = df[feature]

        if pd.api.types.is_bool_dtype(series):

            feature_metadata[feature] = {
                "type": "boolean",
                "default": False
            }

        elif pd.api.types.is_numeric_dtype(series):

            values = pd.to_numeric(
                series,
                errors="coerce"
            ).replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            feature_metadata[feature] = {
                "type": "numeric",
                "default": (
                    float(values.median())
                    if len(values) else 0.0
                ),
                "integer": pd.api.types.is_integer_dtype(series)
            }

        else:

            unique_values = series.dropna().unique().tolist()

            feature_metadata[feature] = {
                "type": "categorical",
                "options": unique_values[:50],
                "default": (
                    unique_values[0]
                    if len(unique_values) else ""
                )
            }

    return feature_metadata


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🤖 Model Management")

st.sidebar.info(
    "Upload a dataset, train suitable models, "
    "save the best model and generate predictions."
)


# ==================================================
# DATASET UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload CSV or Excel Dataset",
    type=["csv", "xlsx", "xls"]
)


# ==================================================
# DATASET WORKFLOW
# ==================================================

if uploaded_file is not None:

    try:

        # --------------------------------------------------
        # LOAD DATASET
        # --------------------------------------------------

        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if df.empty:
            st.error("Uploaded dataset is empty.")
            st.stop()

        dataset_signature = (
            uploaded_file.name,
            uploaded_file.size
        )

        if st.session_state["dataset_signature"] != dataset_signature:

            st.session_state["dataset_signature"] = dataset_signature
            st.session_state["ml_output"] = None
            st.session_state["model_config"] = None
            st.session_state["loaded_model_package"] = None

        st.success("Dataset uploaded successfully!")

        # --------------------------------------------------
        # DATASET PREVIEW
        # --------------------------------------------------

        st.subheader("📊 Dataset Preview")

        st.dataframe(
            df,
            use_container_width=True
        )

        # --------------------------------------------------
        # DATASET INFORMATION
        # --------------------------------------------------

        st.subheader("📋 Dataset Information")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

        st.write("**Duplicate Rows:**", int(df.duplicated().sum()))

        # --------------------------------------------------
        # TARGET COLUMN SELECTION
        # --------------------------------------------------

        st.subheader("🎯 Target Column Selection")

        selection_mode = st.radio(
            "Choose Target Selection Mode",
            ["Automatic Detection", "Manual Selection"],
            horizontal=True
        )

        columns = df.columns.tolist()

        if not columns:
            st.error("No columns found in the dataset.")
            st.stop()

        if selection_mode == "Automatic Detection":

            target_keywords = [
                "target", "label", "class", "outcome",
                "result", "prediction", "price",
                "sales", "attrition", "status"
            ]

            excluded_keywords = [
                "id", "index", "serial", "s.no"
            ]

            scores = {}

            for column in columns:

                column_name = (
                    str(column)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                )

                score = 0

                if any(word in column_name for word in target_keywords):
                    score += 10

                if any(word in column_name for word in excluded_keywords):
                    score -= 10

                scores[column] = score

            best_score = max(scores.values())

            if best_score > 0:
                suggested_target = max(scores, key=scores.get)
            else:
                suggested_target = columns[-1]

            st.info(f"🤖 Suggested Target: {suggested_target}")

            target_column = st.selectbox(
                "Confirm Target Column",
                options=columns,
                index=columns.index(suggested_target)
            )

        else:

            target_column = st.selectbox(
                "Choose the column you want to predict",
                options=columns
            )

        st.success(f"Selected Target: {target_column}")

        # --------------------------------------------------
        # PROBLEM TYPE SELECTION
        # --------------------------------------------------

        st.subheader("🧠 Problem Type Selection")

        problem_mode = st.radio(
            "Choose problem detection mode",
            ["Automatic Detection", "Manual Selection"],
            horizontal=True,
            key="problem_mode"
        )

        if problem_mode == "Manual Selection":

            problem_type = st.selectbox(
                "Select ML Problem Type",
                ["Classification", "Regression"]
            )

        else:
            problem_type = None

        # --------------------------------------------------
        # MODEL CONFIGURATION
        # --------------------------------------------------

        current_config = (
            dataset_signature,
            target_column,
            problem_mode,
            problem_type
        )

        if st.session_state["model_config"] != current_config:

            st.session_state["ml_output"] = None
            st.session_state["model_config"] = current_config

        # --------------------------------------------------
        # TRAIN MODEL
        # --------------------------------------------------

        if st.button("🚀 Train & Compare ML Models"):

            with st.spinner("Training suitable ML models..."):

                try:

                    if problem_type is None:

                        output = train_suitable_models(
                            df,
                            target_column
                        )

                    elif problem_type == "Regression":

                        from models.model_selection import train_regression_models

                        output = train_regression_models(
                            df,
                            target_column
                        )

                    else:

                        from models.model_selection import train_classification_models

                        output = train_classification_models(
                            df,
                            target_column
                        )

                    st.session_state["ml_output"] = output

                    st.success(
                        "Model training completed successfully!"
                    )

                except Exception as e:

                    st.error(f"Model training failed: {e}")

        # ==================================================
        # DISPLAY TRAINING RESULTS
        # ==================================================

        output = st.session_state.get("ml_output")

        if output is not None:

            st.divider()

            st.subheader("🧠 Problem Type")
            st.info(output["problem_type"])

            # --------------------------------------------------
            # MODEL COMPARISON
            # --------------------------------------------------

            st.subheader("🏆 Model Comparison")

            results = output["results"].copy()

            st.dataframe(
                results,
                use_container_width=True
            )

            # ==================================================
            # ADVANCED PERFORMANCE DASHBOARD
            # ==================================================

            st.divider()

            st.header("📊 Advanced Model Performance Dashboard")

            problem_type = output["problem_type"]

            # --------------------------------------------------
            # CLASSIFICATION METRICS
            # --------------------------------------------------

            if problem_type == "Classification":

                st.subheader("🎯 Classification Metrics")

                metric_columns = [
                    col for col in [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score"
                    ]
                    if col in results.columns
                ]

                if metric_columns:

                    metric_cols = st.columns(len(metric_columns))

                    for i, metric in enumerate(metric_columns):

                        best_value = results[metric].max()

                        metric_cols[i].metric(
                            f"Best {metric}",
                            f"{best_value:.3f}"
                        )

                    chart_data = results.melt(
                        id_vars=["Model"],
                        value_vars=metric_columns,
                        var_name="Metric",
                        value_name="Score"
                    )

                    fig = px.bar(
                        chart_data,
                        x="Model",
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Classification Model Comparison",
                        text_auto=".3f"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:
                    st.warning("Classification metrics unavailable.")

            # --------------------------------------------------
            # REGRESSION METRICS
            # --------------------------------------------------

            elif problem_type == "Regression":

                st.subheader("📈 Regression Metrics")

                metric_specs = [
                    ("MAE", "min"),
                    ("RMSE", "min"),
                    ("R2", "max"),
                    ("R2 Score", "max")
                ]

                available_metrics = []

                for metric, direction in metric_specs:

                    if metric in results.columns:

                        if metric not in [
                            item[0] for item in available_metrics
                        ]:

                            available_metrics.append(
                                (metric, direction)
                            )

                if available_metrics:

                    metric_cols = st.columns(len(available_metrics))

                    for i, (metric, direction) in enumerate(available_metrics):

                        value = (
                            results[metric].min()
                            if direction == "min"
                            else results[metric].max()
                        )

                        metric_cols[i].metric(
                            f"Best {metric}",
                            f"{value:,.4f}"
                        )

                    for metric, direction in available_metrics:

                        fig = px.bar(
                            results,
                            x="Model",
                            y=metric,
                            title=f"{metric} Model Comparison",
                            text_auto=".3f"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                else:
                    st.warning("Regression metrics unavailable.")

            # ==================================================
            # ACTUAL VS PREDICTED
            # ==================================================

            st.subheader("🔍 Actual vs Predicted Analysis")

            try:

                X_test = output["X_test"]
                y_test = output["y_test"]
                best_model = output["best_model"]

                predictions = best_model.predict(X_test)

                comparison_df = pd.DataFrame({
                    "Actual": np.asarray(y_test),
                    "Predicted": np.asarray(predictions)
                })

                if problem_type == "Regression":

                    fig = px.scatter(
                        comparison_df,
                        x="Actual",
                        y="Predicted",
                        title="Actual vs Predicted Values",
                        opacity=0.7
                    )

                    low = min(
                        comparison_df["Actual"].min(),
                        comparison_df["Predicted"].min()
                    )

                    high = max(
                        comparison_df["Actual"].max(),
                        comparison_df["Predicted"].max()
                    )

                    fig.add_shape(
                        type="line",
                        x0=low,
                        y0=low,
                        x1=high,
                        y1=high,
                        line=dict(dash="dash")
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.dataframe(
                        comparison_df.head(20),
                        use_container_width=True
                    )

                    class_counts = (
                        comparison_df
                        .groupby(["Actual", "Predicted"])
                        .size()
                        .reset_index(name="Count")
                    )

                    fig = px.bar(
                        class_counts,
                        x="Actual",
                        y="Count",
                        color="Predicted",
                        barmode="group",
                        title="Actual vs Predicted Classes"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            except Exception as e:

                st.warning(
                    f"Actual vs Predicted unavailable: {e}"
                )

            # ==================================================
            # FEATURE IMPORTANCE
            # ==================================================

            st.subheader("⭐ Feature Importance")

            try:

                estimator = output["best_model"]

                for attribute in [
                    "model",
                    "estimator",
                    "trained_model",
                    "regressor",
                    "classifier"
                ]:

                    candidate = getattr(
                        estimator,
                        attribute,
                        None
                    )

                    if candidate is not None:
                        estimator = candidate
                        break

                if hasattr(estimator, "feature_importances_"):

                    importance = estimator.feature_importances_
                    feature_names = output["feature_columns"]

                    if len(importance) == len(feature_names):

                        importance_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importance
                        })

                        importance_df = importance_df.sort_values(
                            "Importance",
                            ascending=True
                        )

                        fig = px.bar(
                            importance_df,
                            x="Importance",
                            y="Feature",
                            orientation="h",
                            title="Feature Importance",
                            text_auto=".3f"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    else:

                        st.info(
                            "Encoded feature names differ from original "
                            "feature names. Importance chart unavailable."
                        )

                else:

                    st.info(
                        "Selected model does not expose built-in "
                        "feature importance."
                    )

            except Exception as e:

                st.warning(f"Feature importance unavailable: {e}")

            # ==================================================
            # BEST MODEL
            # ==================================================

            st.subheader("🥇 Best Model")

            st.success(output["best_model_name"])

            # ==================================================
            # TRAINING INFORMATION
            # ==================================================

            st.subheader("📊 Training Information")

            col1, col2, col3 = st.columns(3)

            col1.metric("Training Rows", output["training_rows"])
            col2.metric("Testing Rows", output["testing_rows"])
            col3.metric(
                "Features Used",
                len(output["feature_columns"])
            )

            # ==================================================
            # SAMPLE PREDICTION
            # ==================================================

            st.subheader("🔍 Sample Prediction")

            try:

                sample_data = output["X_test"].iloc[[0]]

                sample_prediction = output["best_model"].predict(
                    sample_data
                )

                st.write("Predicted Value:", sample_prediction[0])
                st.write("Actual Value:", output["y_test"].iloc[0])

            except Exception as e:

                st.warning(f"Sample prediction unavailable: {e}")

            # ==================================================
            # SAVE TRAINED MODEL
            # ==================================================

            st.divider()

            st.header("💾 Save Trained Model")

            if st.button("💾 Save Best Model"):

                try:

                    feature_metadata = create_feature_metadata(
                        df,
                        output["feature_columns"]
                    )

                    saved_path = save_model(
                        model=output["best_model"],
                        model_name=output["best_model_name"],
                        target_column=target_column,
                        problem_type=output["problem_type"],
                        feature_metadata=feature_metadata
                    )

                    st.success("Model saved successfully!")
                    st.code(saved_path)

                except Exception as e:

                    st.error(f"Model saving failed: {e}")

            # ==================================================
            # WEB-BASED PREDICTION
            # ==================================================

            st.divider()

            st.header("🔮 Web-Based Prediction")

            st.write(
                "Enter new feature values to get a prediction."
            )

            feature_metadata = create_feature_metadata(
                df,
                output["feature_columns"]
            )

            new_data = create_prediction_form(
                feature_metadata,
                "current_model_prediction_form"
            )

            if new_data is not None:

                try:

                    best_model = output["best_model"]

                    prediction = best_model.predict(new_data)

                    st.subheader("🎯 Prediction Result")

                    # ==================================================
                    # REGRESSION RELIABILITY
                    # ==================================================

                    if output["problem_type"] == "Regression":

                        predicted_value = float(prediction[0])

                        st.success(
                            f"Predicted Value: {predicted_value:,.2f}"
                        )

                        X_test = output["X_test"]
                        y_test = output["y_test"]

                        test_predictions = best_model.predict(X_test)

                        actual_values = np.asarray(
                            y_test,
                            dtype=float
                        )

                        predicted_values = np.asarray(
                            test_predictions,
                            dtype=float
                        )

                        absolute_errors = np.abs(
                            actual_values - predicted_values
                        )

                        mae = float(np.mean(absolute_errors))

                        rmse = float(
                            np.sqrt(
                                np.mean(
                                    (actual_values - predicted_values) ** 2
                                )
                            )
                        )

                        ss_total = np.sum(
                            (actual_values - np.mean(actual_values)) ** 2
                        )

                        ss_residual = np.sum(
                            (actual_values - predicted_values) ** 2
                        )

                        r2 = (
                            1 - (ss_residual / ss_total)
                            if ss_total > 0 else 0.0
                        )

                        # ------------------------------------------
                        # RELIABILITY DASHBOARD
                        # ------------------------------------------

                        st.divider()

                        st.subheader(
                            "🛡️ Prediction Reliability Analysis"
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric("Test MAE", f"{mae:,.2f}")
                        col2.metric("Test RMSE", f"{rmse:,.2f}")
                        col3.metric("Test R² Score", f"{r2:.4f}")

                        # ------------------------------------------
                        # ESTIMATED PREDICTION RANGE
                        # ------------------------------------------

                        st.markdown("### 📊 Estimated Prediction Range")

                        lower_limit = predicted_value - mae
                        upper_limit = predicted_value + mae

                        st.info(
                            f"Predicted Value: {predicted_value:,.2f}\n\n"
                            f"Indicative Range: {lower_limit:,.2f} "
                            f"to {upper_limit:,.2f}"
                        )

                        st.caption(
                            "This range uses test-set MAE as an "
                            "indicative error margin. It is not a "
                            "guaranteed confidence interval."
                        )

                        # ------------------------------------------
                        # ERROR DISTRIBUTION
                        # ------------------------------------------

                        st.markdown("### 📉 Prediction Error Distribution")

                        error_df = pd.DataFrame({
                            "Absolute Error": absolute_errors
                        })

                        fig = px.histogram(
                            error_df,
                            x="Absolute Error",
                            nbins=30,
                            title="Distribution of Test Prediction Errors"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        # ------------------------------------------
                        # RELIABILITY ASSESSMENT
                        # ------------------------------------------

                        st.markdown("### 🤖 Reliability Assessment")

                        if r2 >= 0.90:

                            st.success(
                                "Strong test-set fit. Predictions performed "
                                "well on the held-out test data."
                            )

                        elif r2 >= 0.70:

                            st.info(
                                "Moderate test-set fit. Predictions may "
                                "have noticeable errors."
                            )

                        else:

                            st.warning(
                                "Low test-set fit. Consider improving "
                                "the dataset or selecting another model."
                            )

                    # ==================================================
                    # CLASSIFICATION RELIABILITY
                    # ==================================================

                    else:

                        st.success(
                            f"Predicted Class: {prediction[0]}"
                        )

                        st.divider()

                        st.subheader("🛡️ Classification Reliability")

                        if hasattr(best_model, "predict_proba"):

                            probabilities = (
                                best_model.predict_proba(new_data)[0]
                            )

                            classes = best_model.classes_

                            probability_df = pd.DataFrame({
                                "Class": classes,
                                "Probability": probabilities
                            })

                            probability_df["Probability (%)"] = (
                                probability_df["Probability"] * 100
                            )

                            st.dataframe(
                                probability_df,
                                use_container_width=True
                            )

                            confidence = float(
                                np.max(probabilities) * 100
                            )

                            st.metric(
                                "Prediction Confidence",
                                f"{confidence:.2f}%"
                            )

                            st.caption(
                                "Model probability is not a guarantee "
                                "that the prediction is correct."
                            )

                        else:

                            st.info(
                                "The selected classification model does "
                                "not provide prediction probabilities."
                            )

                except Exception as e:

                    st.error(
                        f"Prediction reliability analysis failed: {e}"
                    )

    # ==================================================
    # OUTER DATASET WORKFLOW EXCEPTION HANDLER
    # ==================================================

    except Exception as e:

        st.error(f"Dataset processing failed: {e}")


# ==================================================
# SAVED MODEL MANAGEMENT
# ==================================================

st.divider()

st.header("📂 Saved Model Management")

saved_files = []

if os.path.exists("saved_models"):

    saved_files = sorted(
        [
            file for file in os.listdir("saved_models")
            if file.endswith(".joblib")
        ],
        reverse=True
    )

if saved_files:

    selected_file = st.selectbox(
        "Select a saved model",
        saved_files,
        key="saved_model_selector"
    )

    # --------------------------------------------------
    # LOAD SAVED MODEL
    # --------------------------------------------------

    if st.button("📂 Load Saved Model"):

        try:

            file_path = os.path.join(
                "saved_models",
                selected_file
            )

            loaded_package = load_model(file_path)

            st.session_state["loaded_model_package"] = loaded_package

            st.success("Saved model loaded successfully!")

        except Exception as e:

            st.error(f"Model loading failed: {e}")

    # --------------------------------------------------
    # PREDICTION WITHOUT DATASET UPLOAD
    # --------------------------------------------------

    loaded_package = st.session_state.get(
        "loaded_model_package"
    )

    if loaded_package is not None:

        st.divider()

        st.subheader("🔮 Predict Without Dataset Upload")

        st.write("Loaded model:", loaded_package["model_name"])
        st.write("Target:", loaded_package["target_column"])
        st.write("Problem Type:", loaded_package["problem_type"])

        metadata = loaded_package.get("feature_metadata", {})

        if metadata:

            new_data = create_prediction_form(
                metadata,
                "saved_model_prediction_form"
            )

            if new_data is not None:

                try:

                    saved_model = loaded_package["model"]

                    prediction = saved_model.predict(new_data)

                    st.subheader("🎯 Prediction Result")

                    if loaded_package["problem_type"] == "Regression":

                        st.success(
                            f"Predicted Value: {prediction[0]:,.2f}"
                        )

                    else:

                        st.success(
                            f"Predicted Class: {prediction[0]}"
                        )

                        if hasattr(saved_model, "predict_proba"):

                            probabilities = (
                                saved_model.predict_proba(new_data)[0]
                            )

                            confidence = float(
                                np.max(probabilities) * 100
                            )

                            st.metric(
                                "Prediction Confidence",
                                f"{confidence:.2f}%"
                            )

                except Exception as e:

                    st.error(
                        f"Saved model prediction failed: {e}"
                    )

        else:

            st.warning(
                "This is an older saved model without feature metadata. "
                "Train and save the model again to enable prediction "
                "without dataset upload."
            )

else:

    st.info(
        "No saved models found. Upload a dataset, train and save a model first."
    )
