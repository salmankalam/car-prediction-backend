from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import re
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
MODEL_PATH = ROOT / "data_cleaning" / "v2_data_cleaning" / "car_price_prediction_model.pkl"
FEATURE_DATA_PATH = ROOT / "data_cleaning" / "v2_data_cleaning" / "feature_engineered_data.csv"
RAW_DATA_PATH = ROOT / "data" / "used_cars_10M_2025.csv"

CURRENT_YEAR = datetime.now().year
RANDOM_STATE = 42

app = FastAPI(title="Car Price Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    "mileage_km": "Mileage",
    "mileage_per_year": "Mileage per year",
    "condition_score": "Condition",
    "horsepower": "Horsepower",
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


@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH)


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
    sample_df = pd.read_csv(FEATURE_DATA_PATH, nrows=2500)
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
def load_target_maps() -> Tuple[Dict[str, Dict[str, float]], float]:
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
        confidence=0.85,
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
                {"feature_value": round(float(x), 5), "predicted_price_usd": round(float(y), 2)}
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
            "current_pdp_price_usd": round(current_pdp_price, 2),
            "changed_pdp_price_usd": round(changed_pdp_price, 2),
            "delta_usd": round(delta, 2),
            "pdp_points": pdp["points"],
            "text": (
                f"For {feature_label(feature)}, the PDP curve estimates that {change_label} "
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
    request_payload = PredictionRequest(**model_to_dict(payload, exclude={"budget"}))
    input_df, _, base_price = predict_price_from_payload(request_payload)

    if dice_ml is not None:
        try:
            X = get_x_sample().astype(np.float32)
            dice_df = pd.concat(
                [X.reset_index(drop=True), pd.Series(np.expm1(get_y_sample()), name="price_usd").reset_index(drop=True)],
                axis=1,
            )
            dice_data = DiceData(
                dataframe=dice_df,
                continuous_features=list(X.columns),
                outcome_name="price_usd",
            )
            dice_model = DiceModel(model=get_model(), backend="sklearn", model_type="regressor")
            dice_explainer = Dice(dice_data, dice_model, method="random")
            counterfactuals = dice_explainer.generate_counterfactuals(
                input_df.astype(np.float32),
                total_CFs=3,
                desired_range=[payload.budget * 0.97, payload.budget * 1.03],
                features_to_vary=list(input_df.columns),
            )
            cf_df = counterfactuals.cf_examples_list[0].final_cfs_df
            return {
                "counterfactuals": [
                    {key: round(float(value), 4) for key, value in row.items()}
                    for row in cf_df.to_dict(orient="records")
                ],
                "graph": {"recommended": "comparison table", "note": "Show current encoded values next to each counterfactual row."},
                "note": "Generated with DiCE around the requested budget.",
            }
        except Exception:
            pass

    suggestions = []
    candidate_updates = [
        {"mileage_km": max(0.0, request_payload.mileage_km - 20_000)},
        {"year": min(CURRENT_YEAR, request_payload.year + 1)},
        {"condition_score": min(10.0, request_payload.condition_score + 1)},
        {"horsepower": request_payload.horsepower + 50},
        {"mileage_km": request_payload.mileage_km + 20_000, "year": max(1995, request_payload.year - 1)},
    ]
    for update in candidate_updates:
        candidate = copy_model(request_payload, **update)
        _, _, candidate_price = predict_price_from_payload(candidate)
        suggestions.append({
            "changed_fields": update,
            "estimated_price_usd": round(candidate_price, 2),
            "distance_from_budget_usd": round(candidate_price - payload.budget, 2),
            "current_price_delta_usd": round(candidate_price - base_price, 2),
        })

    suggestions = sorted(suggestions, key=lambda row: abs(row["distance_from_budget_usd"]))[:3]
    return {
        "counterfactuals": suggestions,
        "graph": {"recommended": "ranked comparison table", "note": "Sort by absolute distance_from_budget_usd."},
        "note": "DiCE was unavailable or could not find valid rows, so these are deterministic what-if suggestions.",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "feature_data_loaded": FEATURE_DATA_PATH.exists(),
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
