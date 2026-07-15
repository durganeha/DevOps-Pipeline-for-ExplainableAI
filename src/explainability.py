"""
explainability.py

Thin SHAP wrapper used by app.py (for the interactive UI) and optionally by
CI to log explanation artifacts per model version. Kept separate from
train_model.py so explanations can be regenerated without retraining.
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

MODEL_PATH = Path("models/baseline_model.pkl")


def load_model_bundle(path: Path = MODEL_PATH) -> dict:
    """Loads the {model, scaler, feature_names, ...} bundle saved by train_model.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python src/train_model.py` first to train and save a model."
        )
    return joblib.load(path)


def get_shap_explainer(bundle: dict, background_data: pd.DataFrame):
    """
    Builds a SHAP explainer for the model in `bundle`. Uses LinearExplainer
    since the baseline model is Logistic Regression; swap for
    shap.TreeExplainer or shap.Explainer(model.predict, ...) if you later
    move to a tree-based or black-box model.
    """
    scaler = bundle["scaler"]
    model = bundle["model"]
    background_scaled = scaler.transform(background_data[bundle["feature_names"]])
    return shap.LinearExplainer(model, background_scaled)


def explain_instance(bundle: dict, instance: pd.DataFrame, explainer=None, background_data: pd.DataFrame = None):
    """
    Returns SHAP values for a single row (or small batch) of applicant data.
    `instance` must be a DataFrame with the same feature columns used in
    training (see FEATURE_COLS in src/data_loader.py).

    Returns:
        shap_values: np.ndarray of shape (n_instances, n_features)
        base_value: float, the model's average prediction over the background
    """
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]

    if explainer is None:
        if background_data is None:
            raise ValueError("Provide either a prebuilt `explainer` or `background_data` to build one.")
        explainer = get_shap_explainer(bundle, background_data)

    instance_scaled = scaler.transform(instance[feature_names])
    shap_values = explainer.shap_values(instance_scaled)
    base_value = explainer.expected_value

    return shap_values, base_value


def plot_shap_summary(bundle: dict, data: pd.DataFrame, out_path: str = "shap_summary.png"):
    """
    Generates and saves a SHAP summary bar plot (mean |SHAP value| per
    feature) across a dataset — the "which features matter most overall"
    view used in the app's Explainability tab.
    """
    feature_names = bundle["feature_names"]
    scaler = bundle["scaler"]
    data_scaled = scaler.transform(data[feature_names])

    explainer = shap.LinearExplainer(bundle["model"], data_scaled)
    shap_values = explainer.shap_values(data_scaled)

    plt.figure()
    shap.summary_plot(shap_values, data_scaled, feature_names=feature_names, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def explain_single_prediction_text(bundle: dict, instance: pd.DataFrame, background_data: pd.DataFrame, top_n: int = 3) -> str:
    """
    Produces a plain-English explanation of the top contributing features
    for a single applicant's prediction — used for the "why was this loan
    approved/rejected" card in the UI.
    """
    explainer = get_shap_explainer(bundle, background_data)
    shap_values, base_value = explain_instance(bundle, instance, explainer=explainer)

    row_shap = shap_values[0] if shap_values.ndim > 1 else shap_values
    feature_names = bundle["feature_names"]

    contributions = sorted(
        zip(feature_names, row_shap, instance.iloc[0][feature_names].values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    lines = []
    for name, value, raw in contributions:
        direction = "increased" if value > 0 else "decreased"
        lines.append(f"- {name} = {raw} {direction} the approval likelihood (impact: {value:+.3f})")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.data_loader import load_processed_data

    bundle = load_model_bundle()
    df = load_processed_data()
    sample = df[bundle["feature_names"]].sample(200, random_state=1)

    path = plot_shap_summary(bundle, sample)
    print(f"Saved SHAP summary plot to {path}")

    one_applicant = df[bundle["feature_names"]].iloc[[0]]
    explanation = explain_single_prediction_text(bundle, one_applicant, background_data=sample)
    print("\nExample single-prediction explanation:")
    print(explanation)