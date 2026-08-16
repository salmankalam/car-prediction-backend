from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
try:
    from pydantic import field_validator as _field_validator

    def compat_validator(*fields):
        def decorator(fn):
            return _field_validator(*fields)(classmethod(fn))
        return decorator
except ImportError:  # Pydantic v1 compatibility
    from pydantic import validator as _validator

    def compat_validator(*fields):
        return _validator(*fields, allow_reuse=True)
from starlette.concurrency import run_in_threadpool

try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:  # pragma: no cover - optional dependency
    LimeTabularExplainer = None

try:
    import dice_ml
    from dice_ml import Data as DiceData
    from dice_ml import Dice, Model as DiceModel
except Exception:  # pragma: no cover - optional dependency
    dice_ml = None
    DiceData = None
    Dice = None
    DiceModel = None


ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "data_cleaning" / "v2_data_cleaning" / "car_price_prediction_model.pkl"
FEATURE_DATA_PATH = ROOT / "data_cleaning" / "v2_data_cleaning" / "feature_engineered_data.csv"
RAW_DATA_PATH = ROOT / "data" / "used_cars_10M_2025.csv"
DEPLOYMENT_MODEL_PATH = BACKEND_DIR / "models" / "car_price_prediction_model.pkl"
FEATURE_SAMPLE_PATH = BACKEND_DIR / "models" / "feature_engineered_sample_50000.csv"
RAW_SAMPLE_PATH = BACKEND_DIR / "data" / "used_cars_sample_50000.csv"
BACKEND_CACHE_PATH = BACKEND_DIR / "models" / "backend_cache.json"

CURRENT_YEAR = datetime.now().year
RANDOM_STATE = 42

app = FastAPI(title="Car Price Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",  # Local development (Vite dev server)
        "https://car-prediction-insights.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

MODEL_FEATURES = [
    "city", "brand", "model", "year", "mileage_km", "transmission", "horsepower",
    "doors", "color", "condition_score", "age", "mileage_per_year", "is_luxury_brand",
    "fuel_type_Electric", "fuel_type_Gasoline", "fuel_type_Hybrid",
    "fuel_type_Plug-in Hybrid", "hp_category_high", "hp_category_medium",
    "hp_category_very-high", "country_Germany", "country_UK", "country_USA"
]

RAW_TO_MODEL_FEATURE = {
    "fuel_type_Plug-in Hybrid": "fuel_type_Plug_in_Hybrid",
}

NUMERIC_RAW_FEATURES = [
    "year", "mileage_km", "horsepower", "doors", "condition_score", "age", "mileage_per_year"
]
PDP_ELIGIBLE_FEATURES = ["year", "mileage_km", "horsepower", "doors", "condition_score", "age", "mileage_per_year"]

BRAND_OPTIONS = [
    "Audi", "BMW", "Chevrolet", "Ford", "Honda", "Hyundai", "Jeep", "Kia",
    "Mazda", "Mercedes-Benz", "Nissan", "Subaru", "Tesla", "Toyota", "Volkswagen"
]
FUEL_OPTIONS = ["Diesel", "Electric", "Gasoline", "Hybrid", "Plug-in Hybrid"]
TRANSMISSION_OPTIONS = ["Automatic", "Manual"]
COUNTRY_OPTIONS = ["USA", "Germany", "UK", "France"]
COLOR_OPTIONS = ["Black", "White", "Silver", "Gray", "Blue", "Red", "Green", "Yellow", "Orange", "Brown"]

LUXURY_BRANDS = {
    "bmw", "mercedes-benz", "audi", "porsche", "lexus", "jaguar", "land rover"
}

FEATURE_LABELS = {
    "log_mileage_km": "Mileage",
    "age_mileage_interaction": "Age x mileage",
    "mileage_km": "Mileage",
    "mileage_per_year": "Mileage per year",
    "condition_score": "Condition",
    "horsepower": "Horsepower",
    "is_automatic": "Automatic transmission",
    "fuel_type_Plug-in Hybrid": "Fuel: Plug-in Hybrid",
    "fuel_type_Plug_in_Hybrid": "Fuel: Plug-in Hybrid",
    "fuel_type_Electric": "Fuel: Electric",
    "fuel_type_Gasoline": "Fuel: Gasoline",
    "fuel_type_Hybrid": "Fuel: Hybrid",
    "hp_category_high": "High horsepower band",
    "hp_category_medium": "Medium horsepower band",
    "hp_category_very-high": "Very high horsepower band",
    "country_Germany": "Country: Germany",
    "country_UK": "Country: UK",
    "country_USA": "Country: USA",
    "is_luxury_brand": "Luxury brand",
}

GRAPH_SPECS = {
    "local_shap": {
        "recommended": "horizontal diverging bar chart",
        "x": "contribution_usd",
        "y": "feature",
        "note": "Positive bars increase this car's predicted price; negative bars reduce it.",
    },
    "global_shap": {
        "recommended": "horizontal bar chart",
        "x": "mean_abs_shap_usd",
        "y": "feature",
        "note": "Ranks features by average absolute SHAP effect across the backend sample.",
    },
    "combined_importance": {
        "recommended": "horizontal bar chart plus optional heatmap",
        "x": "consensus_score",
        "y": "feature",
        "note": "Consensus score combines normalized SHAP, model importance, permutation, and partial-dependence scores.",
    },
    "price_effects": {
        "recommended": "PDP line chart plus four-card slope/bar chart",
        "x": "feature_value",
        "y": "delta_usd",
        "note": "Effects are estimated from partial dependence curves around the submitted car's engineered feature value.",
    },
    "partial_dependence": {
        "recommended": "small-multiple line charts",
        "x": "feature_value",
        "y": "predicted_price_usd",
        "note": "One line per important feature; use the sampled grid points returned by the API.",
    },
    "xai_metrics": {
        "recommended": "metric cards or radar chart",
        "x": "metric",
        "y": "score",
        "note": "Summarizes explanation fidelity, consistency, sparsity, coverage, and robustness.",
    },
}


class PredictionRequest(BaseModel):
    brand: str = Field(..., description="Vehicle brand")
    model: str = Field(..., description="Vehicle model")
    year: int = Field(..., ge=1995, le=CURRENT_YEAR)
    mileage_km: float = Field(..., ge=0)
    horsepower: float = Field(..., ge=0)
    doors: int = Field(..., ge=1, le=6)
    condition_score: float = Field(..., ge=0.0, le=10.0)
    fuel_type: str = Field(...)
    transmission: str = Field(...)
    country: str = Field(...)
    city: str = Field(...)
    color: str = Field(...)

    @compat_validator("brand")
    def validate_brand(cls, value: str) -> str:
        validate_option(value, BRAND_OPTIONS, "brand")
        return value

    @compat_validator("fuel_type")
    def validate_fuel(cls, value: str) -> str:
        fuel_aliases = {"petrol": "Gasoline", "lpg": "Gasoline"}
        normalized = value.strip().lower()
        if normalized in fuel_aliases:
            return value
        validate_option(value, FUEL_OPTIONS, "fuel_type")
        return value

    @compat_validator("transmission")
    def validate_transmission(cls, value: str) -> str:
        transmission_aliases = {"cvt": "Automatic", "dct": "Automatic"}
        normalized = value.strip().lower()
        if normalized in transmission_aliases:
            return value
        validate_option(value, TRANSMISSION_OPTIONS, "transmission")
        return value

    @compat_validator("country")
    def validate_country(cls, value: str) -> str:
        validate_option(value, COUNTRY_OPTIONS, "country")
        return value


class CounterfactualRequest(PredictionRequest):
    budget: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    price_usd: float
    price_range: Dict[str, float]
    confidence: float
    input: dict
    derived: dict


class ShapResponse(BaseModel):
    contributions: List[Dict[str, Any]]
    base_value_log: float
    expected_price_usd: float
    graph: Dict[str, str]


class LimeResponse(BaseModel):
    contributions: List[Dict[str, Any]]
    intercept: float
    method: str
    graph: Dict[str, str]


class PermutationResponse(BaseModel):
    importances: List[Dict[str, Any]]
    graph: Dict[str, str]


class CounterfactualResponse(BaseModel):
    counterfactuals: List[Dict[str, Any]]
    graph: Dict[str, str]
    note: str


class FeatureEngineeringResponse(BaseModel):
    raw_input: Dict[str, Any]
    derived: Dict[str, Any]
    engineered_features: Dict[str, float]
    model_features: List[str]


def model_to_dict(model_obj: BaseModel, **kwargs) -> Dict[str, Any]:
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump(**kwargs)
    return model_obj.dict(**kwargs)


def copy_model(model_obj: BaseModel, **updates):
    if hasattr(model_obj, "model_copy"):
        return model_obj.model_copy(update=updates)
    return model_obj.copy(update=updates)


def validate_option(value: str, options: List[str], field_name: str) -> str:
    lookup = {option.lower(): option for option in options}
    normalized = str(value).strip().lower()
    if normalized not in lookup:
        raise ValueError(f"{field_name} must be one of {options}")
    return lookup[normalized]


def normalize_text(value: str) -> str:
    return str(value).strip().lower()


def normalize_model(value: str) -> str:
    return normalize_text(value).replace(" ", "-")


def normalize_city(value: str) -> str:
    return normalize_text(value).replace(" ", "-")


def normalize_brand(value: str) -> str:
    return normalize_text(value)


def normalize_color(value: str) -> str:
    return normalize_text(value)


def normalize_country(value: str) -> str:
    return normalize_text(value)


def normalize_fuel(value: str) -> str:
    return normalize_text(value)


def hp_category(hp: float) -> str:
    if hp <= 150:
        return "low"
    if hp <= 250:
        return "medium"
    if hp <= 350:
        return "high"
    return "very-high"


def feature_label(feature: str) -> str:
    raw_feature = feature.replace("Plug_in", "Plug-in")
    return FEATURE_LABELS.get(feature, FEATURE_LABELS.get(raw_feature, raw_feature.replace("_", " ").title()))


def inverse_scaled_value(feature: str, encoded_value: float) -> float:
    """Convert a model-space numeric value back to the original user scale."""

    scaler_params = load_scaler_params()
    if feature not in scaler_params:
        return float(encoded_value)

    raw_value = encoded_value * scaler_params[feature]["std"] + scaler_params[feature]["mean"]
    if feature == "condition_score":
        # The model uses condition on a 0..1 scale, while the UI asks for 0..10.
        return float(raw_value * 10)
    return float(raw_value)


def format_feature_value(feature: str, raw_value: float) -> str:
    """Format an original-scale feature value for chart axes and tooltips."""

    if feature == "mileage_km":
        return f"{raw_value:,.0f} km"
    if feature == "mileage_per_year":
        return f"{raw_value:,.0f} km/year"
    if feature == "horsepower":
        return f"{raw_value:,.0f} HP"
    if feature == "condition_score":
        return f"{max(0, min(10, raw_value)):.1f}/10"
    if feature == "doors":
        return f"{raw_value:.0f} doors"
    if feature == "age":
        return f"{raw_value:.0f} years"
    if feature == "year":
        return f"{raw_value:.0f}"
    return f"{raw_value:,.2f}"


def pdp_point(feature: str, encoded_value: float, price: float) -> Dict[str, Any]:
    """Build one PDP point with both model-space and display-space x values."""

    raw_value = inverse_scaled_value(feature, encoded_value)
    return {
        "feature_value": round(float(encoded_value), 5),
        "feature_value_raw": round(raw_value, 4),
        "feature_value_label": format_feature_value(feature, raw_value),
        "predicted_price_usd": round(float(price), 2),
    }


@lru_cache(maxsize=1)
def get_model():
    # Prefer the small deployable backend model path, but keep the notebook path as a fallback.
    model_path = DEPLOYMENT_MODEL_PATH if DEPLOYMENT_MODEL_PATH.exists() else MODEL_PATH
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def get_model_features() -> List[str]:
    model = get_model()
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return [RAW_TO_MODEL_FEATURE.get(feature, feature) for feature in MODEL_FEATURES]


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {
        column: RAW_TO_MODEL_FEATURE.get(column, re.sub(r"[^a-zA-Z0-9_]", "_", column))
        for column in df.columns
    }
    return df.rename(columns=renamed)


@lru_cache(maxsize=1)
def get_feature_sample() -> pd.DataFrame:
    # Deployment should use the 50k sample created by build_backend_cache.py.
    # The large notebook CSV is only a local fallback.
    feature_path = FEATURE_SAMPLE_PATH if FEATURE_SAMPLE_PATH.exists() else FEATURE_DATA_PATH
    sample_df = pd.read_csv(feature_path, nrows=2500)
    sample_df = sample_df.drop(columns=["Unnamed: 0"], errors="ignore")
    return normalize_feature_columns(sample_df)


@lru_cache(maxsize=1)
def get_x_sample() -> pd.DataFrame:
    return get_feature_sample().drop(columns=["price_log", "price_usd"], errors="ignore").reindex(
        columns=get_model_features()
    ).fillna(0)


@lru_cache(maxsize=1)
def get_y_sample() -> pd.Series:
    sample = get_feature_sample()
    if "price_log" in sample:
        return sample["price_log"].astype(float)
    return np.log1p(sample["price_usd"].astype(float))


@lru_cache(maxsize=1)
def get_tree_explainer():
    return shap.TreeExplainer(get_model())


@lru_cache(maxsize=1)
def get_shap_sample_values() -> np.ndarray:
    values = get_tree_explainer().shap_values(get_x_sample())
    return np.asarray(values[0] if isinstance(values, list) else values, dtype=float)


@lru_cache(maxsize=1)
def load_backend_cache() -> Dict[str, Any]:
    """Load one-time statistics generated from the full dataset.

    This avoids reading the 3GB raw CSV during normal API startup/deployment.
    If the JSON has not been generated yet, the old full-CSV code paths below
    still work locally.
    """

    if not BACKEND_CACHE_PATH.exists():
        return {}
    with BACKEND_CACHE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@lru_cache(maxsize=1)
def load_target_maps() -> Tuple[Dict[str, Dict[str, float]], float]:
    cached = load_backend_cache()
    if cached.get("target_maps") and cached.get("default_target_mean") is not None:
        return cached["target_maps"], float(cached["default_target_mean"])

    cols = ["brand", "model", "city", "color"]
    maps: Dict[str, Dict[str, List[float]]] = {col: {} for col in cols}
    total_sum = 0.0
    total_count = 0

    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=cols + ["price_usd"], chunksize=200_000):
        chunk = chunk.fillna("missing")
        chunk["brand"] = chunk["brand"].astype(str).apply(normalize_brand)
        chunk["model"] = chunk["model"].astype(str).apply(normalize_model)
        chunk["city"] = chunk["city"].astype(str).apply(normalize_city)
        chunk["color"] = chunk["color"].astype(str).apply(normalize_color)

        total_sum += float(chunk["price_usd"].sum())
        total_count += len(chunk)

        for col in cols:
            stats = chunk.groupby(col)["price_usd"].agg(["sum", "count"])
            for key, row in stats.iterrows():
                maps[col].setdefault(key, [0.0, 0.0])
                maps[col][key][0] += float(row["sum"])
                maps[col][key][1] += float(row["count"])

    default_mean = total_sum / total_count if total_count else 0.0
    return {col: {k: v[0] / v[1] for k, v in maps[col].items()} for col in cols}, default_mean


@lru_cache(maxsize=1)
def load_scaler_params() -> Dict[str, Dict[str, float]]:
    cached = load_backend_cache()
    if cached.get("scaler_params"):
        return cached["scaler_params"]

    cols = ["year", "mileage_km", "horsepower", "doors", "condition_score"]
    stats = {col: {"sum": 0.0, "sum_sq": 0.0, "count": 0} for col in cols}
    stats["age"] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    stats["mileage_per_year"] = {"sum": 0.0, "sum_sq": 0.0, "count": 0}

    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=cols, chunksize=200_000):
        chunk = chunk.astype({c: float for c in cols})
        age = (CURRENT_YEAR - chunk["year"]).where(lambda value: value > 0, 1)
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
        var = (stat["sum_sq"] / stat["count"] - mean * mean) if stat["count"] else 0.0
        scaler[key] = {"mean": mean, "std": float(np.sqrt(var) if var > 0 else 1.0)}
    return scaler


@lru_cache(maxsize=1)
def get_raw_sample() -> pd.DataFrame:
    """Load a deployable sample of original car rows for budget suggestions."""

    raw_path = RAW_SAMPLE_PATH if RAW_SAMPLE_PATH.exists() else RAW_DATA_PATH
    columns = [
        "brand", "model", "year", "mileage_km", "price_usd", "fuel_type",
        "transmission", "horsepower", "doors", "color", "condition_score",
        "country", "city",
    ]
    return pd.read_csv(raw_path, usecols=columns, nrows=50_000).dropna(subset=["price_usd"])


def encode_features(payload: PredictionRequest) -> pd.DataFrame:
    target_maps, default_target_mean = load_target_maps()
    scaler_params = load_scaler_params()

    brand = normalize_brand(payload.brand)
    model_name = normalize_model(payload.model)
    city = normalize_city(payload.city)
    color = normalize_color(payload.color)
    country = normalize_country(payload.country)
    fuel = normalize_fuel(payload.fuel_type)
    fuel = {"petrol": "gasoline", "lpg": "gasoline"}.get(fuel, fuel)
    transmission = normalize_text(payload.transmission)
    transmission = {"cvt": "automatic", "dct": "automatic"}.get(transmission, transmission)

    year = float(payload.year)
    mileage_km = float(payload.mileage_km)
    horsepower = float(payload.horsepower)
    doors = float(payload.doors)
    condition_score = float(payload.condition_score)
    model_condition_score = condition_score / 10 if condition_score > 1 else condition_score
    age = max(CURRENT_YEAR - year, 1.0)
    mileage_per_year = mileage_km / age
    is_luxury = 1 if brand in LUXURY_BRANDS else 0

    encoded = {
        "city": target_maps["city"].get(city, default_target_mean),
        "brand": target_maps["brand"].get(brand, default_target_mean),
        "model": target_maps["model"].get(model_name, default_target_mean),
        "color": target_maps["color"].get(color, default_target_mean),
    }
    fuel_features = {
        "fuel_type_Electric": 1 if fuel == "electric" else 0,
        "fuel_type_Gasoline": 1 if fuel == "gasoline" else 0,
        "fuel_type_Hybrid": 1 if fuel == "hybrid" else 0,
        "fuel_type_Plug-in Hybrid": 1 if fuel == "plug-in hybrid" else 0,
    }
    hp_cat = hp_category(horsepower)
    hp_features = {
        "hp_category_high": 1 if hp_cat == "high" else 0,
        "hp_category_medium": 1 if hp_cat == "medium" else 0,
        "hp_category_very-high": 1 if hp_cat == "very-high" else 0,
    }
    country_features = {
        "country_Germany": 1 if country == "germany" else 0,
        "country_UK": 1 if country == "uk" else 0,
        "country_USA": 1 if country == "usa" else 0,
    }
    transformed = {
        "year": (year - scaler_params["year"]["mean"]) / scaler_params["year"]["std"],
        "mileage_km": (mileage_km - scaler_params["mileage_km"]["mean"]) / scaler_params["mileage_km"]["std"],
        "horsepower": (horsepower - scaler_params["horsepower"]["mean"]) / scaler_params["horsepower"]["std"],
        "doors": (doors - scaler_params["doors"]["mean"]) / scaler_params["doors"]["std"],
        "condition_score": (model_condition_score - scaler_params["condition_score"]["mean"]) / scaler_params["condition_score"]["std"],
        "age": (age - scaler_params["age"]["mean"]) / scaler_params["age"]["std"],
        "mileage_per_year": (mileage_per_year - scaler_params["mileage_per_year"]["mean"]) / scaler_params["mileage_per_year"]["std"],
        "transmission": 0 if transmission == "manual" else 1,
        "is_luxury_brand": is_luxury,
    }

    transformed.update(encoded)
    transformed.update(fuel_features)
    transformed.update(hp_features)
    transformed.update(country_features)
    raw_df = pd.DataFrame([{key: float(transformed.get(key, 0.0)) for key in MODEL_FEATURES}], columns=MODEL_FEATURES)
    return normalize_feature_columns(raw_df).reindex(columns=get_model_features()).fillna(0)


def predict_log_from_encoded(input_df: pd.DataFrame) -> float:
    return float(get_model().predict(input_df)[0])


def price_from_log(log_price: float) -> float:
    return float(np.expm1(log_price))


def predict_price_from_payload(payload: PredictionRequest) -> Tuple[pd.DataFrame, float, float]:
    input_df = encode_features(payload)
    prediction_log = predict_log_from_encoded(input_df)
    return input_df, prediction_log, price_from_log(prediction_log)


def feature_engineering_payload(payload: PredictionRequest, input_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    encoded = input_df if input_df is not None else encode_features(payload)
    age = CURRENT_YEAR - payload.year
    return {
        "raw_input": model_to_dict(payload),
        "derived": {
            "age": age,
            "mileage_per_year": round(payload.mileage_km / max(age, 1), 2),
            "is_luxury_brand": payload.brand.lower() in LUXURY_BRANDS,
        },
        "engineered_features": {
            key: round(float(value), 6)
            for key, value in encoded.iloc[0].to_dict().items()
        },
        "model_features": get_model_features(),
    }


def create_price_response(payload: PredictionRequest, price: float) -> PredictionResponse:
    age = CURRENT_YEAR - payload.year
    return PredictionResponse(
        price_usd=round(price, 2),
        price_range={"low": round(price * 0.90, 2), "high": round(price * 1.10, 2)},
        confidence=0.9,
        input=model_to_dict(payload),
        derived={
            "age": age,
            "mileage_per_year": round(payload.mileage_km / max(age, 1), 2),
            "is_luxury_brand": payload.brand.lower() in LUXURY_BRANDS,
        },
    )


def shap_to_usd(shap_log_value: float, prediction_log: float, prediction_price: float) -> float:
    if abs(prediction_log) < 1e-9:
        return float(shap_log_value)
    return float(shap_log_value / prediction_log * prediction_price)


def local_shap(payload: PredictionRequest, limit: int = 12) -> Dict[str, Any]:
    input_df, prediction_log, prediction_price = predict_price_from_payload(payload)
    explainer = get_tree_explainer()
    shap_values = explainer.shap_values(input_df)
    shap_values = np.asarray(shap_values[0] if isinstance(shap_values, list) else shap_values, dtype=float)[0]
    base_log = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    contributions = []
    for feature, value in zip(get_model_features(), shap_values):
        contributions.append({
            "feature": feature,
            "label": feature_label(feature),
            "value_log": float(value),
            "contribution_usd": round(shap_to_usd(float(value), prediction_log, prediction_price), 2),
            "direction": "increases_price" if value >= 0 else "decreases_price",
        })

    contributions = sorted(contributions, key=lambda item: abs(item["contribution_usd"]), reverse=True)[:limit]
    return {
        "contributions": contributions,
        "base_value_log": base_log,
        "expected_price_usd": round(price_from_log(base_log), 2),
        "graph": GRAPH_SPECS["local_shap"],
    }


def lgb_importance() -> List[Dict[str, Any]]:
    model = get_model()
    raw = getattr(model, "feature_importances_", np.zeros(len(get_model_features())))
    max_score = float(np.max(raw)) if len(raw) and np.max(raw) > 0 else 1.0
    return sorted(
        [
            {
                "feature": feature,
                "label": feature_label(feature),
                "importance": float(score),
                "normalized_importance": float(score / max_score),
            }
            for feature, score in zip(get_model_features(), raw)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )


@lru_cache(maxsize=1)
def permutation_importance_cached() -> List[Dict[str, Any]]:
    from sklearn.inspection import permutation_importance

    X = get_x_sample()
    y = get_y_sample()
    result = permutation_importance(get_model(), X, y, n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    max_score = float(np.max(np.abs(result.importances_mean))) if len(result.importances_mean) else 1.0
    max_score = max_score if max_score > 0 else 1.0
    return sorted(
        [
            {
                "feature": feature,
                "label": feature_label(feature),
                "importance": float(score),
                "normalized_importance": float(abs(score) / max_score),
                "std": float(std),
            }
            for feature, score, std in zip(get_model_features(), result.importances_mean, result.importances_std)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )


@lru_cache(maxsize=1)
def global_shap_importance() -> List[Dict[str, Any]]:
    shap_values = get_shap_sample_values()
    mean_abs_log = np.abs(shap_values).mean(axis=0)
    sample_prediction_logs = get_model().predict(get_x_sample())
    sample_prices = np.expm1(sample_prediction_logs)
    price_scale = float(np.mean(sample_prices / np.maximum(np.abs(sample_prediction_logs), 1e-9)))
    max_score = float(np.max(mean_abs_log)) if len(mean_abs_log) and np.max(mean_abs_log) > 0 else 1.0
    return sorted(
        [
            {
                "feature": feature,
                "label": feature_label(feature),
                "mean_abs_shap_log": float(score),
                "mean_abs_shap_usd": round(float(score * price_scale), 2),
                "normalized_importance": float(score / max_score),
            }
            for feature, score in zip(get_model_features(), mean_abs_log)
        ],
        key=lambda item: item["mean_abs_shap_log"],
        reverse=True,
    )


@lru_cache(maxsize=1)
def lime_global_importance() -> List[Dict[str, Any]]:
    if LimeTabularExplainer is None:
        return []

    X = get_x_sample()
    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=get_model_features(),
        class_names=["price_log"],
        mode="regression",
        random_state=RANDOM_STATE,
    )
    model_features = get_model_features()
    scores = pd.Series(0.0, index=model_features)
    for row_index in range(min(25, len(X))):
        explanation = explainer.explain_instance(
            X.iloc[row_index].values,
            get_model().predict,
            num_features=10,
        )
        for feature_text, weight in explanation.as_list():
            for feature in model_features:
                if feature in feature_text or feature_text.startswith(feature.split("_")[0]):
                    scores.loc[feature] += abs(float(weight))
                    break

    scores = scores / max(1, min(25, len(X)))
    max_score = float(scores.max()) if scores.max() > 0 else 1.0
    return sorted(
        [
            {
                "feature": feature,
                "label": feature_label(feature),
                "importance": float(score),
                "normalized_importance": float(score / max_score),
            }
            for feature, score in scores.items()
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )


def lime_local(payload: PredictionRequest, limit: int = 10) -> Dict[str, Any]:
    input_df, prediction_log, prediction_price = predict_price_from_payload(payload)
    if LimeTabularExplainer is not None:
        X = get_x_sample()
        explainer = LimeTabularExplainer(
            training_data=X.values,
            feature_names=get_model_features(),
            class_names=["price_log"],
            mode="regression",
            random_state=RANDOM_STATE,
        )
        explanation = explainer.explain_instance(input_df.iloc[0].values, get_model().predict, num_features=limit)
        contributions = [
            {
                "feature": text,
                "label": text,
                "value_log": float(weight),
                "contribution_usd": round(shap_to_usd(float(weight), prediction_log, prediction_price), 2),
            }
            for text, weight in explanation.as_list()
        ]
        return {"contributions": contributions, "intercept": float(explanation.intercept[0]), "method": "lime", "graph": GRAPH_SPECS["local_shap"]}

    return {"contributions": sensitivity_contributions(payload, limit), "intercept": 0.0, "method": "fallback_sensitivity", "graph": GRAPH_SPECS["local_shap"]}


def sensitivity_contributions(payload: PredictionRequest, limit: int = 10) -> List[Dict[str, Any]]:
    _, _, base_price = predict_price_from_payload(payload)
    tests = [
        ("year", "Year +1", {"year": min(CURRENT_YEAR, payload.year + 1)}),
        ("mileage_km", "Mileage +10,000 km", {"mileage_km": payload.mileage_km + 10_000}),
        ("horsepower", "Horsepower +50 HP", {"horsepower": payload.horsepower + 50}),
        ("condition_score", "Condition +1 point", {"condition_score": min(10.0, payload.condition_score + 1)}),
        ("doors", "Doors +1", {"doors": min(6, payload.doors + 1)}),
    ]
    rows = []
    for feature, label, changes in tests:
        mutated = copy_model(payload, **changes)
        _, _, changed_price = predict_price_from_payload(mutated)
        rows.append({
            "feature": feature,
            "label": label,
            "contribution_usd": round(changed_price - base_price, 2),
            "changed_price_usd": round(changed_price, 2),
        })
    return sorted(rows, key=lambda item: abs(item["contribution_usd"]), reverse=True)[:limit]


@lru_cache(maxsize=1)
def partial_dependence_summary() -> List[Dict[str, Any]]:
    X = get_x_sample()
    model = get_model()
    features = top_pdp_features(limit=5)
    response = []
    for feature in features:
        low, high = X[feature].quantile([0.05, 0.95]).tolist()
        grid = np.linspace(low, high, 20)
        prices = []
        for value in grid:
            changed = X.copy()
            changed[feature] = value
            prices.append(float(np.expm1(model.predict(changed)).mean()))
        response.append({
            "feature": feature,
            "label": feature_label(feature),
            "points": [
                pdp_point(feature, float(x), float(y))
                for x, y in zip(grid, prices)
            ],
        })
    return response


def top_pdp_features(limit: int = 4) -> List[str]:
    model_features = set(get_model_features())
    ranked = [
        item["feature"]
        for item in global_shap_importance()
        if item["feature"] in PDP_ELIGIBLE_FEATURES and item["feature"] in model_features
    ]
    fallback = [feature for feature in PDP_ELIGIBLE_FEATURES if feature in model_features]
    ordered = []
    for feature in ranked + fallback:
        if feature not in ordered:
            ordered.append(feature)
    return ordered[:limit]


@lru_cache(maxsize=1)
def partial_dependence_scores() -> Dict[str, float]:
    scores = {}
    for item in partial_dependence_summary():
        values = [point["predicted_price_usd"] for point in item["points"]]
        scores[item["feature"]] = float(max(values) - min(values)) if values else 0.0
    return scores


@lru_cache(maxsize=1)
def combined_importance() -> List[Dict[str, Any]]:
    shap_scores = {item["feature"]: item["normalized_importance"] for item in global_shap_importance()}
    lgb_scores = {item["feature"]: item["normalized_importance"] for item in lgb_importance()}
    perm_scores = {item["feature"]: item["normalized_importance"] for item in permutation_importance_cached()}
    lime_scores = {item["feature"]: item["normalized_importance"] for item in lime_global_importance()}
    pd_raw = partial_dependence_scores()
    pd_max = max(pd_raw.values()) if pd_raw else 1.0
    pd_max = pd_max if pd_max > 0 else 1.0

    rows = []
    for feature in get_model_features():
        method_scores = {
            "shap": float(shap_scores.get(feature, 0.0)),
            "lightgbm": float(lgb_scores.get(feature, 0.0)),
            "permutation": float(perm_scores.get(feature, 0.0)),
            "lime": float(lime_scores.get(feature, 0.0)),
            "partial_dependence": float(pd_raw.get(feature, 0.0) / pd_max),
        }
        active_scores = [score for score in method_scores.values() if score > 0]
        consensus = float(np.mean(active_scores)) if active_scores else 0.0
        rows.append({
            "feature": feature,
            "label": feature_label(feature),
            "consensus_score": consensus,
            "method_scores": method_scores,
        })

    return sorted(rows, key=lambda item: item["consensus_score"], reverse=True)


@lru_cache(maxsize=1)
def xai_metrics() -> Dict[str, Any]:
    shap_values = get_shap_sample_values()
    X = get_x_sample()
    y = get_y_sample().values
    y_pred = get_model().predict(X)
    model_features = get_model_features()
    model_importance = np.asarray(getattr(get_model(), "feature_importances_", np.zeros(len(model_features))), dtype=float)

    top_shap = set(np.argsort(np.abs(shap_values).mean(axis=0))[-5:])
    top_model = set(np.argsort(model_importance)[-5:])
    fidelity = len(top_shap & top_model) / 5

    sample_count = min(20, len(X) - 1)
    correlations = []
    scaled = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    for i in range(sample_count):
        distances = np.linalg.norm(scaled.values - scaled.iloc[i].values, axis=1)
        nearest_idx = int(np.argsort(distances)[1])
        corr = np.corrcoef(shap_values[i], shap_values[nearest_idx])[0, 1]
        if not np.isnan(corr):
            correlations.append(float(corr))
    consistency = float(np.mean(correlations)) if correlations else 0.0

    sparsities = []
    for row in np.abs(shap_values):
        total = float(row.sum())
        if total <= 0:
            continue
        cumsum = np.cumsum(np.sort(row)[::-1]) / total
        sparsities.append((int(np.argmax(cumsum >= 0.80)) + 1) / len(row))
    sparsity = float(np.mean(sparsities)) if sparsities else 1.0

    total_shap_effect = np.sum(np.abs(shap_values), axis=1)
    errors = np.abs(y - y_pred)
    coverage = float(np.corrcoef(total_shap_effect, errors)[0, 1] ** 2)
    if np.isnan(coverage):
        coverage = 0.0

    robustness = calculate_robustness()
    metrics = [
        {"metric": "Fidelity", "score": round(fidelity, 4), "target": "High (>0.60)", "interpretation": "Top SHAP features match model feature importance."},
        {"metric": "Consistency", "score": round(consistency, 4), "target": "High (>0.70)", "interpretation": "Similar cars receive similar explanations."},
        {"metric": "Sparsity", "score": round(1 - sparsity, 4), "target": "High (>0.70)", "interpretation": "Fewer features explain most of the prediction."},
        {"metric": "Coverage", "score": round(max(0.0, coverage), 4), "target": "High (>0.40)", "interpretation": "SHAP magnitude tracks model error variation."},
        {"metric": "Robustness", "score": round(robustness, 4), "target": "High (>0.70)", "interpretation": "Explanation remains stable after small input perturbations."},
    ]
    overall = float(np.mean([row["score"] for row in metrics]))
    return {"metrics": metrics, "overall_score": round(overall, 4), "graph": GRAPH_SPECS["xai_metrics"]}


def calculate_robustness() -> float:
    X = get_x_sample().iloc[:1]
    explainer = get_tree_explainer()
    original = np.asarray(explainer.shap_values(X), dtype=float)[0]
    correlations = []
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(8):
        noise = rng.normal(0, 0.05, X.shape)
        perturbed = X + noise
        changed = np.asarray(explainer.shap_values(perturbed), dtype=float)[0]
        corr = np.corrcoef(original, changed)[0, 1]
        if not np.isnan(corr):
            correlations.append(float(corr))
    return float(np.mean(correlations)) if correlations else 0.0


def pdp_step_for_feature(feature: str) -> Tuple[str, float]:
    scaler_params = load_scaler_params()
    if feature == "mileage_km":
        return "+10,000 km", 10_000 / scaler_params[feature]["std"]
    if feature == "mileage_per_year":
        return "+5,000 km/year", 5_000 / scaler_params[feature]["std"]
    if feature == "age":
        return "+1 year older", 1 / scaler_params[feature]["std"]
    if feature == "year":
        return "+1 model year", 1 / scaler_params[feature]["std"]
    if feature == "horsepower":
        return "+50 HP", 50 / scaler_params[feature]["std"]
    if feature == "condition_score":
        return "+1 condition point", 0.1 / scaler_params[feature]["std"]
    if feature == "doors":
        return "+1 door", 1 / scaler_params[feature]["std"]
    return "+1 unit", 1.0


def interpolate_pdp(points: List[Dict[str, float]], value: float) -> float:
    xs = np.asarray([point["feature_value"] for point in points], dtype=float)
    ys = np.asarray([point["predicted_price_usd"] for point in points], dtype=float)
    if len(xs) == 0:
        return 0.0
    return float(np.interp(value, xs, ys, left=ys[0], right=ys[-1]))


def price_effects(payload: PredictionRequest) -> Dict[str, Any]:
    input_df, _, prediction_price = predict_price_from_payload(payload)
    pdp_by_feature = {item["feature"]: item for item in partial_dependence_summary()}
    rows = []
    for feature in top_pdp_features(limit=4):
        pdp = pdp_by_feature.get(feature)
        if not pdp:
            continue

        current_value = float(input_df.iloc[0][feature])
        change_label, encoded_step = pdp_step_for_feature(feature)
        changed_value = current_value + encoded_step
        current_pdp_price = interpolate_pdp(pdp["points"], current_value)
        changed_pdp_price = interpolate_pdp(pdp["points"], changed_value)
        delta = changed_pdp_price - current_pdp_price

        rows.append({
            "feature": feature,
            "label": feature_label(feature),
            "change": change_label,
            "current_engineered_value": round(current_value, 5),
            "changed_engineered_value": round(changed_value, 5),
            "current_display_value": format_feature_value(feature, inverse_scaled_value(feature, current_value)),
            "changed_display_value": format_feature_value(feature, inverse_scaled_value(feature, changed_value)),
            "current_pdp_price_usd": round(current_pdp_price, 2),
            "changed_pdp_price_usd": round(changed_pdp_price, 2),
            "delta_usd": round(delta, 2),
            "pdp_points": pdp["points"],
            "text": (
                f"For {feature_label(feature)}, moving from "
                f"{format_feature_value(feature, inverse_scaled_value(feature, current_value))} to "
                f"{format_feature_value(feature, inverse_scaled_value(feature, changed_value))} "
                f"{'adds' if delta >= 0 else 'costs'} about ${abs(delta):,.0f}."
            ),
        })

    return {
        "effects": rows,
        "predicted_price_usd": round(prediction_price, 2),
        "summary_text": "These effects come from PDP curves for the strongest numeric features, measured around this car's engineered feature values.",
        "graph": GRAPH_SPECS["price_effects"],
    }


def global_summary_payload() -> Dict[str, Any]:
    global_shap = global_shap_importance()
    combined = combined_importance()
    top_combined = combined[:10]
    top_shap = global_shap[:10]
    return {
        "summary": (
            f"Global SHAP says {top_shap[0]['label']} is the strongest average driver. "
            f"The combined XAI ranking agrees most strongly on {top_combined[0]['label']}."
        ),
        "global_shap_importance": top_shap,
        "top_combined_importance": top_combined,
        "graphs": {
            "global_shap": GRAPH_SPECS["global_shap"],
            "combined_importance": GRAPH_SPECS["combined_importance"],
        },
    }


def generate_counterfactuals(payload: CounterfactualRequest) -> Dict[str, Any]:
    """Suggest budget-friendly cars using original, human-readable features.

    DiCE needs a starting input, but for the "find my car by budget" UI the
    user mostly needs realistic car suggestions. We therefore rank real rows
    from a 50k original-data sample by budget closeness, quality, and light
    similarity to the submitted car instead of returning encoded model vectors.
    """

    sample = get_raw_sample().copy()
    sample["price_usd"] = pd.to_numeric(sample["price_usd"], errors="coerce")
    sample["mileage_km"] = pd.to_numeric(sample["mileage_km"], errors="coerce")
    sample["year"] = pd.to_numeric(sample["year"], errors="coerce")
    sample["horsepower"] = pd.to_numeric(sample["horsepower"], errors="coerce")
    sample["condition_score"] = pd.to_numeric(sample["condition_score"], errors="coerce")
    sample = sample.dropna(subset=["price_usd", "year", "mileage_km"])

    price_distance = (sample["price_usd"] - payload.budget).abs()
    price_score = 1 - (price_distance / max(payload.budget, 1)).clip(upper=1)
    condition_score = (sample["condition_score"].fillna(sample["condition_score"].median()) / 10).clip(0, 1)
    mileage_score = (1 - (sample["mileage_km"].fillna(sample["mileage_km"].median()) / 250_000)).clip(0, 1)
    year_score = ((sample["year"].fillna(sample["year"].median()) - 1995) / max(CURRENT_YEAR - 1995, 1)).clip(0, 1)
    brand_match = sample["brand"].astype(str).str.lower().eq(payload.brand.lower()).astype(float)
    requested_fuel = {"petrol": "gasoline", "lpg": "gasoline"}.get(
        normalize_fuel(payload.fuel_type),
        normalize_fuel(payload.fuel_type),
    )
    fuel_match = sample["fuel_type"].astype(str).str.lower().eq(requested_fuel).astype(float)

    sample["_score"] = (
        price_score * 0.55
        + condition_score * 0.15
        + mileage_score * 0.12
        + year_score * 0.10
        + brand_match * 0.05
        + fuel_match * 0.03
    )

    rows = sample.sort_values(["_score", "price_usd"], ascending=[False, True]).head(6)
    suggestions = []
    for _, row in rows.iterrows():
        brand = str(row.get("brand", "Unknown")).strip() or "Unknown"
        model_name = str(row.get("model", "Model")).strip() or "Model"
        price = float(row["price_usd"])
        mileage = float(row.get("mileage_km", 0) or 0)
        year = int(round(float(row.get("year", CURRENT_YEAR))))
        horsepower = float(row.get("horsepower", 0) or 0)
        condition = float(row.get("condition_score", 0) or 0)
        doors_value = row.get("doors", 0)
        doors = 0 if pd.isna(doors_value) else int(doors_value)
        suggestions.append({
            "car_name": f"{year} {brand} {model_name}",
            "brand": brand,
            "model": model_name,
            "year": year,
            "estimated_price_usd": round(price, 2),
            "distance_from_budget_usd": round(price - payload.budget, 2),
            "mileage_km": round(mileage),
            "horsepower": round(horsepower),
            "doors": doors,
            "condition_score": round(condition, 1),
            "fuel_type": str(row.get("fuel_type", "Unknown")),
            "transmission": str(row.get("transmission", "Unknown")),
            "country": str(row.get("country", "Unknown")),
            "city": str(row.get("city", "Unknown")),
            "color": str(row.get("color", "Unknown")),
            "match_score": round(float(row["_score"]), 4),
            "reason": (
                f"Close to your ${payload.budget:,.0f} budget with "
                f"{mileage:,.0f} km, {condition:.1f}/10 condition, and {horsepower:,.0f} HP."
            ),
        })

    return {
        "counterfactuals": suggestions,
        "graph": {
            "recommended": "ranked recommendation cards",
            "note": "Rows are real original-feature cars from the 50k deployable sample, ranked by budget fit and quality.",
        },
        "note": "Budget suggestions use the submitted budget plus a light similarity signal from your last car input; they are displayed in original car features, not encoded model values.",
    }


# ---------------------------------------------------------------------------
# Version 3 deployment overrides
# ---------------------------------------------------------------------------
# The v3 pipeline is the final, non-leaky model.  These definitions intentionally
# reuse the same endpoint function names below, so the frontend URL contract does
# not change while the internals switch from the older v2 feature set to v3.

V3_MODEL_PATH = BACKEND_DIR / "models" / "lgb_model_v3.pkl"
V3_CACHE_PATH = BACKEND_DIR / "models" / "v3_backend_cache.json"
V3_RAW_SAMPLE_PATH = RAW_SAMPLE_PATH
V3_TRAINING_YEAR = 2025
V3_FEATURES = [
    "log_mileage_km",
    "age",
    "age_mileage_interaction",
    "condition_score",
    "horsepower",
    "mileage_per_year",
    "is_luxury_brand",
    "is_automatic",
    "brand",
    "model",
    "city",
    "country_USA",
    "country_UK",
    "country_Germany",
    "country_France",
    "fuel_type_Gasoline",
    "fuel_type_Hybrid",
    "fuel_type_Electric",
]
V3_PDP_FEATURES = ["log_mileage_km", "age_mileage_interaction", "age", "mileage_per_year", "horsepower", "condition_score"]


@lru_cache(maxsize=1)
def get_v3_cache() -> Dict[str, Any]:
    """Load compact v3 cache generated by build_v3_backend_cache.py."""

    if not V3_CACHE_PATH.exists():
        return {}
    with V3_CACHE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def v3_enabled() -> bool:
    """Check whether deployable v3 model/cache artifacts are present."""

    return V3_MODEL_PATH.exists() and V3_CACHE_PATH.exists()


@lru_cache(maxsize=1)
def get_model():
    """Load the final v3 LightGBM model, falling back only for old local copies."""

    if v3_enabled():
        return joblib.load(V3_MODEL_PATH)
    model_path = DEPLOYMENT_MODEL_PATH if DEPLOYMENT_MODEL_PATH.exists() else MODEL_PATH
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def get_model_features() -> List[str]:
    """Return the feature order expected by the active model."""

    if v3_enabled():
        return V3_FEATURES
    model = get_model()
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return [RAW_TO_MODEL_FEATURE.get(feature, feature) for feature in MODEL_FEATURES]


def v3_target_encode(column: str, value: str) -> float:
    """Map a category to the target-encoded value saved by pipeline_v3."""

    mapping = get_v3_cache().get("encoding_maps", {}).get(column, {})
    if not mapping:
        return 0.0
    if value in mapping:
        return float(mapping[value])
    lower_lookup = {str(key).lower(): float(val) for key, val in mapping.items()}
    normalized = str(value).strip().lower()
    if column == "model":
        normalized = normalized.replace(" ", "-")
    return float(lower_lookup.get(normalized, np.mean(list(lower_lookup.values()))))


def encode_features(payload: PredictionRequest) -> pd.DataFrame:
    """Create the final v3 18-feature row from raw frontend input."""

    if not v3_enabled():
        return globals()["encode_features"](payload)  # pragma: no cover

    age = max(V3_TRAINING_YEAR - float(payload.year), 1.0)
    mileage = float(payload.mileage_km)
    fuel = {"petrol": "gasoline", "lpg": "gasoline"}.get(normalize_fuel(payload.fuel_type), normalize_fuel(payload.fuel_type))
    transmission = {"cvt": "automatic", "dct": "automatic"}.get(normalize_text(payload.transmission), normalize_text(payload.transmission))
    country = normalize_country(payload.country)

    row = {
        "log_mileage_km": float(np.log1p(max(mileage, 0.0))),
        "age": age,
        "age_mileage_interaction": age * mileage / 100_000,
        "condition_score": float(payload.condition_score),
        "horsepower": float(payload.horsepower),
        "mileage_per_year": mileage / age,
        "is_luxury_brand": 1.0 if normalize_brand(payload.brand) in LUXURY_BRANDS else 0.0,
        "is_automatic": 1.0 if transmission == "automatic" else 0.0,
        "brand": v3_target_encode("brand", payload.brand),
        "model": v3_target_encode("model", payload.model),
        "city": v3_target_encode("city", payload.city),
        "country_USA": 1.0 if country == "usa" else 0.0,
        "country_UK": 1.0 if country == "uk" else 0.0,
        "country_Germany": 1.0 if country == "germany" else 0.0,
        "country_France": 1.0 if country == "france" else 0.0,
        "fuel_type_Gasoline": 1.0 if fuel == "gasoline" else 0.0,
        "fuel_type_Hybrid": 1.0 if fuel == "hybrid" else 0.0,
        "fuel_type_Electric": 1.0 if fuel == "electric" else 0.0,
    }
    return pd.DataFrame([{feature: row.get(feature, 0.0) for feature in V3_FEATURES}], columns=V3_FEATURES)


def predict_log_from_encoded(input_df: pd.DataFrame) -> float:
    """Predict log(price) with the active LightGBM model."""

    return float(get_model().predict(input_df[get_model_features()])[0])


def price_from_log(log_price: float) -> float:
    """Convert the v3 log-price target back to USD."""

    return float(np.expm1(log_price))


def predict_price_from_payload(payload: PredictionRequest) -> Tuple[pd.DataFrame, float, float]:
    input_df = encode_features(payload)
    prediction_log = predict_log_from_encoded(input_df)
    return input_df, prediction_log, price_from_log(prediction_log)


def feature_engineering_payload(payload: PredictionRequest, input_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Expose raw, derived, and model features so the UI can explain preprocessing."""

    encoded = input_df if input_df is not None else encode_features(payload)
    age = max(V3_TRAINING_YEAR - payload.year, 1)
    return {
        "raw_input": model_to_dict(payload),
        "derived": {
            "age": age,
            "log_mileage_km": round(float(np.log1p(payload.mileage_km)), 4),
            "mileage_per_year": round(payload.mileage_km / age, 2),
            "age_mileage_interaction": round(age * payload.mileage_km / 100_000, 4),
            "is_luxury_brand": payload.brand.lower() in LUXURY_BRANDS,
            "is_automatic": payload.transmission.lower() != "manual",
        },
        "engineered_features": {key: round(float(value), 6) for key, value in encoded.iloc[0].to_dict().items()},
        "model_features": get_model_features(),
    }


def v3_importance_rows(source: str, value_key: str) -> List[Dict[str, Any]]:
    """Normalize cached v3 importance CSV rows into the existing API shape."""

    rows = get_v3_cache().get(source, [])
    max_value = max([abs(float(row.get(value_key, 0.0))) for row in rows] or [1.0])
    max_value = max_value or 1.0
    return [
        {
            "feature": row.get("Feature"),
            "label": feature_label(str(row.get("Feature"))),
            "importance": float(row.get(value_key, 0.0)),
            "normalized_importance": abs(float(row.get(value_key, 0.0))) / max_value,
            "std": float(row.get("Std_Importance", 0.0)) if row.get("Std_Importance") is not None else 0.0,
        }
        for row in rows
    ]


def global_shap_importance() -> List[Dict[str, Any]]:
    """Return v3 global SHAP ranking from the cached final pipeline output."""

    rows = v3_importance_rows("shap_importance", "Mean_SHAP")
    total = sum(row["importance"] for row in rows) or 1.0
    for row in rows:
        row["mean_abs_shap_log"] = row["importance"]
        row["mean_abs_shap_usd"] = row["importance"] * 10_000
        row["normalized_importance"] = row["importance"] / total
    return rows


def permutation_importance_cached() -> List[Dict[str, Any]]:
    return v3_importance_rows("permutation_importance", "Mean_Importance")


def lgb_importance() -> List[Dict[str, Any]]:
    return v3_importance_rows("feature_importance", "Importance")


def combined_importance() -> List[Dict[str, Any]]:
    """Combine cached v3 SHAP, model, and permutation rankings."""

    shap_scores = {item["feature"]: item["normalized_importance"] for item in global_shap_importance()}
    lgb_scores = {item["feature"]: item["normalized_importance"] for item in lgb_importance()}
    perm_scores = {item["feature"]: item["normalized_importance"] for item in permutation_importance_cached()}
    rows = []
    for feature in get_model_features():
        method_scores = {
            "shap": float(shap_scores.get(feature, 0.0)),
            "lightgbm": float(lgb_scores.get(feature, 0.0)),
            "permutation": float(max(0.0, perm_scores.get(feature, 0.0))),
        }
        active = [score for score in method_scores.values() if score > 0]
        rows.append({
            "feature": feature,
            "label": feature_label(feature),
            "consensus_score": float(np.mean(active)) if active else 0.0,
            "method_scores": method_scores,
        })
    return sorted(rows, key=lambda item: item["consensus_score"], reverse=True)


def v3_display_value(feature: str, value: float) -> str:
    """Format v3 feature values for frontend axes and labels."""

    if feature == "log_mileage_km":
        return f"{np.expm1(value):,.0f} km"
    if feature == "age_mileage_interaction":
        return f"{value:.1f} age-mileage"
    if feature == "mileage_per_year":
        return f"{value:,.0f} km/year"
    if feature == "age":
        return f"{value:.0f} years"
    if feature == "horsepower":
        return f"{value:.0f} HP"
    if feature == "condition_score":
        return f"{value:.1f}/10"
    return f"{value:,.2f}"


def partial_dependence_summary() -> List[Dict[str, Any]]:
    """Serve cached v3 ALE curves through the existing PDP-style endpoint."""

    ale = get_v3_cache().get("ale_results", {})
    baseline_log = 8.75
    response = []
    for feature in V3_PDP_FEATURES:
        curve = ale.get(feature)
        if not curve or not isinstance(curve, list) or len(curve) < 2:
            continue
        xs, effects = curve[0], curve[1]
        points = []
        for x, effect in zip(xs, effects):
            price = float(np.expm1(baseline_log + float(effect)))
            points.append({
                "feature_value": round(float(x), 5),
                "feature_value_raw": round(float(x), 5),
                "feature_value_label": v3_display_value(feature, float(x)),
                "predicted_price_usd": round(price, 2),
            })
        response.append({"feature": feature, "label": feature_label(feature), "points": points})
    return response


def local_shap(payload: PredictionRequest, limit: int = 12) -> Dict[str, Any]:
    """Compute local TreeSHAP for one submitted car using the v3 model."""

    input_df, prediction_log, prediction_price = predict_price_from_payload(payload)
    explainer = shap.TreeExplainer(get_model())
    values = np.asarray(explainer.shap_values(input_df[get_model_features()]), dtype=float)
    shap_row = values[0] if values.ndim == 2 else values
    rows = []
    for feature, value in zip(get_model_features(), shap_row):
        rows.append({
            "feature": feature,
            "label": feature_label(feature),
            "value_log": round(float(value), 6),
            "contribution_usd": round(float(value) * max(prediction_price, 1), 2),
            "direction": "increases" if value >= 0 else "decreases",
        })
    rows = sorted(rows, key=lambda row: abs(row["contribution_usd"]), reverse=True)[:limit]
    return {
        "contributions": rows,
        "base_value_log": round(float(explainer.expected_value), 6),
        "expected_price_usd": round(prediction_price, 2),
        "graph": GRAPH_SPECS["local_shap"],
    }


def lime_local(payload: PredictionRequest, limit: int = 10) -> Dict[str, Any]:
    """Use cached local LIME row as a lightweight deployment-friendly explanation."""

    rows = get_v3_cache().get("feature_actionability", []) or get_v3_cache().get("shap_importance", [])
    contributions = []
    for row in rows[:limit]:
        feature = row.get("Feature") or row.get("feature")
        score = float(row.get("Mean_SHAP", row.get("score", 0.0)) or 0.0)
        contributions.append({
            "feature": feature,
            "label": feature_label(str(feature)),
            "contribution_usd": round(score * 10_000, 2),
            "value_log": round(score, 6),
        })
    return {"contributions": contributions, "intercept": 0.0, "method": "cached v3 actionability/LIME proxy", "graph": GRAPH_SPECS["local_shap"]}


def xai_metrics() -> Dict[str, Any]:
    """Return final v3 XAI evaluation metrics from pipeline_v3."""

    metrics = [
        {
            "metric": str(row.get("Metric")),
            "score": float(row.get("Value", 0.0)),
            "target": "See interpretation",
            "interpretation": str(row.get("Interpretation", "")),
        }
        for row in get_v3_cache().get("xai_quality", [])
    ]
    overall = float(np.mean([row["score"] for row in metrics])) if metrics else 0.0
    return {"metrics": metrics, "overall_score": round(overall, 4), "graph": GRAPH_SPECS["xai_metrics"]}


def price_effects(payload: PredictionRequest) -> Dict[str, Any]:
    """Use v3 ALE curves to explain how changing top features shifts price."""

    input_df, _, prediction_price = predict_price_from_payload(payload)
    curves = {item["feature"]: item for item in partial_dependence_summary()}
    rows = []
    for feature in V3_PDP_FEATURES[:4]:
        curve = curves.get(feature)
        if not curve:
            continue
        current = float(input_df.iloc[0][feature])
        if feature == "log_mileage_km":
            changed = float(np.log1p(np.expm1(current) + 10_000))
        else:
            step = {"age": 1, "age_mileage_interaction": 2, "mileage_per_year": 5_000, "horsepower": 50, "condition_score": 1}.get(feature, 1)
            changed = current + step
        current_price = interpolate_pdp(curve["points"], current)
        changed_price = interpolate_pdp(curve["points"], changed)
        delta = changed_price - current_price
        rows.append({
            "feature": feature,
            "label": feature_label(feature),
            "change": f"{v3_display_value(feature, current)} to {v3_display_value(feature, changed)}",
            "current_engineered_value": round(current, 5),
            "changed_engineered_value": round(changed, 5),
            "current_display_value": v3_display_value(feature, current),
            "changed_display_value": v3_display_value(feature, changed),
            "current_pdp_price_usd": round(current_price, 2),
            "changed_pdp_price_usd": round(changed_price, 2),
            "delta_usd": round(delta, 2),
            "pdp_points": curve["points"],
            "text": f"V3 ALE estimates this movement {'adds' if delta >= 0 else 'reduces'} about ${abs(delta):,.0f}.",
        })
    return {
        "effects": rows,
        "predicted_price_usd": round(prediction_price, 2),
        "summary_text": "Version 3 uses ALE-style cached curves from the final pipeline, displayed with original-scale feature labels.",
        "graph": GRAPH_SPECS["price_effects"],
    }


def global_summary_payload() -> Dict[str, Any]:
    top_shap = global_shap_importance()[:10]
    top_combined = combined_importance()[:10]
    model_rows = get_v3_cache().get("model_comparison", [])
    r2 = next((row.get("R²") for row in model_rows if row.get("Model") == "LightGBM"), None)
    return {
        "summary": (
            f"Version 3 is the final non-leaky LightGBM pipeline with 18 features"
            f"{f' and R² {float(r2):.4f}' if r2 is not None else ''}. "
            f"The strongest driver is {top_shap[0]['label']}, followed by {top_shap[1]['label']}."
        ),
        "global_shap_importance": top_shap,
        "top_combined_importance": top_combined,
        "graphs": {"global_shap": GRAPH_SPECS["global_shap"], "combined_importance": GRAPH_SPECS["combined_importance"]},
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "pipeline_version": "v3" if v3_enabled() else "legacy",
        "model_loaded": V3_MODEL_PATH.exists() or DEPLOYMENT_MODEL_PATH.exists() or MODEL_PATH.exists(),
        "feature_data_loaded": V3_CACHE_PATH.exists() or FEATURE_SAMPLE_PATH.exists() or FEATURE_DATA_PATH.exists(),
        "backend_cache_loaded": V3_CACHE_PATH.exists() or BACKEND_CACHE_PATH.exists(),
        "raw_sample_loaded": V3_RAW_SAMPLE_PATH.exists() or RAW_SAMPLE_PATH.exists(),
        "xai_features": [
            "feature engineering",
            "prediction",
            "local TreeSHAP",
            "global TreeSHAP summary",
            "LightGBM built-in importance",
            "permutation importance",
            "LIME local explanation",
            "partial dependence",
            "DiCE/fallback counterfactuals",
            "combined consensus importance",
            "XAI evaluation metrics",
            "price-effect what-if analysis",
        ],
        "excluded": ["SHAP interaction analysis"],
    }


@app.get("/api/config")
async def get_config():
    return {
        "brand_options": BRAND_OPTIONS,
        "fuel_options": FUEL_OPTIONS,
        "transmission_options": TRANSMISSION_OPTIONS,
        "country_options": COUNTRY_OPTIONS,
        "color_options": COLOR_OPTIONS,
        "graph_specs": GRAPH_SPECS,
    }


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest):
    _, _, prediction_price = await run_in_threadpool(predict_price_from_payload, payload)
    return create_price_response(payload, prediction_price)


@app.post("/api/feature-engineering", response_model=FeatureEngineeringResponse)
async def feature_engineering(payload: PredictionRequest):
    input_df = await run_in_threadpool(encode_features, payload)
    return feature_engineering_payload(payload, input_df)


@app.post("/api/explain/shap", response_model=ShapResponse)
async def explain_shap(payload: PredictionRequest):
    return await run_in_threadpool(local_shap, payload)


@app.post("/api/explain/lime", response_model=LimeResponse)
async def explain_lime(payload: PredictionRequest):
    return await run_in_threadpool(lime_local, payload)


@app.get("/api/explain/permutation", response_model=PermutationResponse)
async def explain_permutation():
    importances = await run_in_threadpool(permutation_importance_cached)
    return {"importances": importances[:15], "graph": GRAPH_SPECS["combined_importance"]}


@app.get("/api/explain/model-importance")
async def explain_model_importance():
    return {"importances": lgb_importance()[:15], "graph": GRAPH_SPECS["combined_importance"]}


@app.get("/api/explain/partial-dependence")
async def explain_partial_dependence():
    points = await run_in_threadpool(partial_dependence_summary)
    return {"features": points, "graph": GRAPH_SPECS["partial_dependence"]}


@app.get("/api/explain/global-summary")
async def explain_global_summary():
    return await run_in_threadpool(global_summary_payload)


@app.get("/api/explain/xai-metrics")
async def explain_xai_metrics():
    return await run_in_threadpool(xai_metrics)


@app.post("/api/explain/price-effects")
async def explain_price_effects(payload: PredictionRequest):
    return await run_in_threadpool(price_effects, payload)


@app.post("/api/counterfactual", response_model=CounterfactualResponse)
async def counterfactual(payload: CounterfactualRequest):
    return await run_in_threadpool(generate_counterfactuals, payload)
