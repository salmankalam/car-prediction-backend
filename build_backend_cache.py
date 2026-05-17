from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_CSV = ROOT / "data" / "used_cars_10M_2025.csv"
DEFAULT_FEATURE_CSV = ROOT / "data_cleaning" / "v2_data_cleaning" / "feature_engineered_data.csv"
DEFAULT_CACHE_JSON = ROOT / "web_app" / "models" / "backend_cache.json"
DEFAULT_RAW_SAMPLE = ROOT / "web_app" / "data" / "used_cars_sample_50000.csv"
DEFAULT_FEATURE_SAMPLE = ROOT / "web_app" / "models" / "feature_engineered_sample_50000.csv"


def normalize_text(value: object) -> str:
    """Normalize category text exactly like the FastAPI app does."""

    return str(value).strip().lower()


def normalize_model(value: object) -> str:
    """Model names use hyphens because training preprocessing did that."""

    return normalize_text(value).replace(" ", "-")


def calculate_target_maps(raw_csv: Path, chunksize: int) -> tuple[Dict[str, Dict[str, float]], float]:
    """Compute target-encoding maps once from the full original CSV.

    The API uses these means to convert brand/model/city/color into numeric
    values. Building them at request time requires scanning the 3GB CSV, so we
    store them in JSON for deployment.
    """

    cols = ["brand", "model", "city", "color"]
    maps: Dict[str, Dict[str, List[float]]] = {col: {} for col in cols}
    total_sum = 0.0
    total_count = 0

    for chunk in pd.read_csv(raw_csv, usecols=cols + ["price_usd"], chunksize=chunksize):
        chunk = chunk.fillna("missing")
        chunk["brand"] = chunk["brand"].astype(str).map(normalize_text)
        chunk["model"] = chunk["model"].astype(str).map(normalize_model)
        chunk["city"] = chunk["city"].astype(str).map(lambda value: normalize_text(value).replace(" ", "-"))
        chunk["color"] = chunk["color"].astype(str).map(normalize_text)

        total_sum += float(chunk["price_usd"].sum())
        total_count += len(chunk)

        for col in cols:
            stats = chunk.groupby(col)["price_usd"].agg(["sum", "count"])
            for key, row in stats.iterrows():
                maps[col].setdefault(str(key), [0.0, 0.0])
                maps[col][str(key)][0] += float(row["sum"])
                maps[col][str(key)][1] += float(row["count"])

    default_mean = total_sum / total_count if total_count else 0.0
    target_maps = {
        col: {key: values[0] / values[1] for key, values in maps[col].items() if values[1]}
        for col in cols
    }
    return target_maps, default_mean


def calculate_scaler_params(raw_csv: Path, chunksize: int, current_year: int) -> Dict[str, Dict[str, float]]:
    """Compute StandardScaler-style means/stds used by API feature engineering."""

    base_cols = ["year", "mileage_km", "horsepower", "doors", "condition_score"]
    stats = {col: {"sum": 0.0, "sum_sq": 0.0, "count": 0} for col in base_cols}
    stats["age"] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    stats["mileage_per_year"] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

    for chunk in pd.read_csv(raw_csv, usecols=base_cols, chunksize=chunksize):
        chunk = chunk.astype({col: float for col in base_cols})
        age = (current_year - chunk["year"]).where(lambda value: value > 0, 1)
        mileage_per_year = chunk["mileage_km"] / age.replace(0, 1)
        values = {
            "year": chunk["year"],
            "mileage_km": chunk["mileage_km"],
            "horsepower": chunk["horsepower"],
            "doors": chunk["doors"],
            "condition_score": chunk["condition_score"],
            "age": age,
            "mileage_per_year": mileage_per_year,
        }

        for key, series in values.items():
            stats[key]["sum"] += float(series.sum())
            stats[key]["sum_sq"] += float((series ** 2).sum())
            stats[key]["count"] += len(series)

    scaler = {}
    for key, stat in stats.items():
        mean = stat["sum"] / stat["count"] if stat["count"] else 0.0
        variance = (stat["sum_sq"] / stat["count"] - mean * mean) if stat["count"] else 0.0
        scaler[key] = {"mean": mean, "std": float(np.sqrt(variance) if variance > 0 else 1.0)}
    return scaler


def write_sample(source: Path, destination: Path, sample_size: int) -> None:
    """Write a deployable sample CSV while keeping column names unchanged."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.read_csv(source, nrows=sample_size).to_csv(destination, index=False)


def build_cache(args: argparse.Namespace) -> None:
    """Build every small artifact needed by the deployed backend."""

    raw_csv = Path(args.raw_csv)
    feature_csv = Path(args.feature_csv)
    cache_json = Path(args.cache_json)
    current_year = datetime.now().year

    cache_json.parent.mkdir(parents=True, exist_ok=True)
    target_maps, default_target_mean = calculate_target_maps(raw_csv, args.chunksize)
    scaler_params = calculate_scaler_params(raw_csv, args.chunksize, current_year)

    cache = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_raw_csv": str(raw_csv),
        "current_year": current_year,
        "default_target_mean": default_target_mean,
        "target_maps": target_maps,
        "scaler_params": scaler_params,
    }
    cache_json.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    write_sample(raw_csv, Path(args.raw_sample_csv), args.sample_size)
    write_sample(feature_csv, Path(args.feature_sample_csv), args.sample_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build small deployable backend cache files.")
    parser.add_argument("--raw-csv", default=str(DEFAULT_RAW_CSV))
    parser.add_argument("--feature-csv", default=str(DEFAULT_FEATURE_CSV))
    parser.add_argument("--cache-json", default=str(DEFAULT_CACHE_JSON))
    parser.add_argument("--raw-sample-csv", default=str(DEFAULT_RAW_SAMPLE))
    parser.add_argument("--feature-sample-csv", default=str(DEFAULT_FEATURE_SAMPLE))
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--chunksize", type=int, default=200_000)
    return parser.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
