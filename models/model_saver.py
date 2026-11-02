
import os
import joblib
from datetime import datetime


MODEL_DIR = "saved_models"


def save_model(
    model,
    model_name,
    target_column,
    problem_type,
    feature_metadata=None
):

    os.makedirs(MODEL_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"best_model_{timestamp}.joblib"

    file_path = os.path.join(MODEL_DIR, filename)

    model_package = {
        "model": model,
        "model_name": model_name,
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_metadata": feature_metadata or {},
        "saved_at": timestamp
    }

    joblib.dump(model_package, file_path)

    return file_path


def load_model(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError("Saved model file not found.")

    return joblib.load(file_path)
