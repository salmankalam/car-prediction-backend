from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PIPELINE_V3 = ROOT / "pipeline_v3"
WEB_APP = ROOT / "web_app"


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    """Read a small CSV artifact into JSON-safe row dictionaries."""

    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON artifact if it exists."""

    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas objects into values json.dumps can write."""

    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def load_ale_results(path: Path) -> dict[str, Any]:
    """Convert cached ALE curves into JSON-friendly data.

    ALE is preferred over PDP in v3 because it is better behaved for correlated
    features. The frontend can still render it as the same line-chart concept.
    """

    if not path.exists():
        return {}
    with path.open("rb") as file:
        raw = pickle.load(file)

    curves: dict[str, Any] = {}
    for feature, value in raw.items():
        if isinstance(value, pd.DataFrame):
            curves[feature] = value.to_dict(orient="records")
        elif isinstance(value, dict):
            curves[feature] = {
                key: item.tolist() if hasattr(item, "tolist") else item
                for key, item in value.items()
            }
        else:
            curves[feature] = value
    return json_safe(curves)


def write_sample(source: Path, destination: Path, rows: int) -> None:
    """Write a small deployable CSV sample and skip missing heavy sources."""

    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.read_parquet(source).head(rows).to_csv(destination, index=False)


def build_cache(rows: int) -> None:
    """Create all compact backend artifacts needed for Render deployment."""

    cache_dir = PIPELINE_V3 / "04_cache"
    metrics_dir = PIPELINE_V3 / "02_metrics"
    models_dir = PIPELINE_V3 / "01_models"
    out_models = WEB_APP / "models"
    out_data = WEB_APP / "data"
    out_models.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    cache = {
        "version": "v3",
        "feature_metadata": read_json(cache_dir / "feature_metadata.json"),
        "encoding_maps": read_json(cache_dir / "encoding_maps.json"),
        "model_comparison": read_csv_records(metrics_dir / "model_comparison.csv"),
        "xai_quality": read_csv_records(metrics_dir / "rq3_xai_quality.csv"),
        "feature_actionability": read_csv_records(metrics_dir / "rq3_feature_actionability.csv"),
        "shap_importance": read_csv_records(cache_dir / "shap_importance.csv"),
        "permutation_importance": read_csv_records(cache_dir / "permutation_importance.csv"),
        "feature_importance": read_csv_records(cache_dir / "feature_importance.csv"),
        "inference_timing": read_csv_records(metrics_dir / "inference_timing_v3.csv"),
        "carbon_lgbm": read_csv_records(metrics_dir / "carbon_lgbm.csv"),
        "carbon_xgb": read_csv_records(metrics_dir / "carbon_xgb.csv"),
        "ale_results": load_ale_results(cache_dir / "ale_results.pkl"),
    }
    (out_models / "v3_backend_cache.json").write_text(json.dumps(cache, indent=2), encoding="utf-8")

    # The LightGBM v3 model is small enough for deployment and is the selected model.
    shutil.copy2(models_dir / "lgb_model_v3.pkl", out_models / "lgb_model_v3.pkl")

    # Samples are optional helpers for recommendations/debugging, not full training data.
    write_sample(cache_dir / "X_v3.parquet", out_models / "X_v3_sample.csv", rows)
    write_sample(cache_dir / "rq1_raw_features.parquet", out_data / "raw_features_v3_sample.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deployable v3 backend artifacts.")
    parser.add_argument("--rows", type=int, default=50_000)
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args().rows)
