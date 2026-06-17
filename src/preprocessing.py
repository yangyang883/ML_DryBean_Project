from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import MODEL_DIR, TARGET_COLUMN
from .data_loader import get_feature_columns
from .utils import load_joblib, save_joblib


PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.joblib"
RAW_FEATURE_COLUMNS_PATH = MODEL_DIR / "raw_feature_columns.joblib"


def normalize_label(value: object) -> str:
    """Normalize dirty labels: spaces, case, and common 0/O and 3/E pollution."""
    label = str(value).strip().upper()
    label = label.replace("0", "O").replace("3", "E")
    valid_labels = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}
    return label if label in valid_labels else label


def clean_dataframe(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """Clean dirty input labels and numeric feature types before feature engineering."""
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]
    for column in get_feature_columns(cleaned):
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map(normalize_label)
    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    return cleaned


def add_morphology_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stable morphology ratio features derived from Dry Bean shape measurements."""
    engineered = df.copy()
    eps = 1e-9

    def ratio(name: str, numerator: str, denominator: str) -> None:
        if numerator in engineered.columns and denominator in engineered.columns:
            engineered[name] = engineered[numerator] / (engineered[denominator] + eps)

    ratio("Area_Perimeter_Ratio", "Area", "Perimeter")
    ratio("Major_Minor_Axis_Ratio", "MajorAxisLength", "MinorAxisLength")
    ratio("ConvexArea_Area_Ratio", "ConvexArea", "Area")
    ratio("EquivDiameter_Perimeter_Ratio", "EquivDiameter", "Perimeter")
    ratio("Area_ConvexArea_Ratio", "Area", "ConvexArea")
    return engineered


@dataclass
class DryBeanPreprocessor:
    """Fit-only-on-train preprocessor: feature engineering, median impute, IQR clip, scale."""

    raw_feature_columns: list[str] | None = None
    feature_columns: list[str] | None = None
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    clip_lower_: pd.Series | None = None
    clip_upper_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "DryBeanPreprocessor":
        cleaned = clean_dataframe(df, drop_duplicates=True)
        self.raw_feature_columns = get_feature_columns(cleaned)
        X = add_morphology_features(cleaned[self.raw_feature_columns])
        self.feature_columns = list(X.columns)

        self.imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=self.feature_columns,
            index=X.index,
        )

        q1 = X_imputed.quantile(0.25)
        q3 = X_imputed.quantile(0.75)
        iqr = q3 - q1
        self.clip_lower_ = q1 - 1.5 * iqr
        self.clip_upper_ = q3 + 1.5 * iqr
        X_clipped = X_imputed.clip(lower=self.clip_lower_, upper=self.clip_upper_, axis=1)

        self.scaler = StandardScaler()
        self.scaler.fit(X_clipped)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not all([self.raw_feature_columns, self.feature_columns, self.imputer, self.scaler]):
            raise RuntimeError("Preprocessor has not been fitted.")

        cleaned = df.copy()
        cleaned.columns = [column.strip() for column in cleaned.columns]
        missing = sorted(set(self.raw_feature_columns or []) - set(cleaned.columns))
        if missing:
            raise ValueError(f"Input data is missing required feature columns: {missing}")

        for column in self.raw_feature_columns or []:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        X = add_morphology_features(cleaned[self.raw_feature_columns])
        X = X.reindex(columns=self.feature_columns)
        X_imputed = pd.DataFrame(
            self.imputer.transform(X),
            columns=self.feature_columns,
            index=X.index,
        )
        X_clipped = X_imputed.clip(lower=self.clip_lower_, upper=self.clip_upper_, axis=1)
        return self.scaler.transform(X_clipped)


def fit_transform_train(train_df: pd.DataFrame):
    """Fit all preprocessing artifacts only on training data, then save them."""
    train_df = clean_dataframe(train_df, drop_duplicates=True)
    preprocessor = DryBeanPreprocessor().fit(train_df)
    label_encoder = LabelEncoder()

    X_train = preprocessor.transform(train_df)
    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN])

    save_joblib(preprocessor, PREPROCESSOR_PATH)
    save_joblib(label_encoder, LABEL_ENCODER_PATH)
    save_joblib(preprocessor.feature_columns, FEATURE_COLUMNS_PATH)
    save_joblib(preprocessor.raw_feature_columns, RAW_FEATURE_COLUMNS_PATH)
    return X_train, y_train, preprocessor.feature_columns, label_encoder


def transform_dataset(df: pd.DataFrame):
    """Transform validation/test data with saved training-fitted artifacts."""
    df = clean_dataframe(df, drop_duplicates=False)
    preprocessor: DryBeanPreprocessor = load_joblib(PREPROCESSOR_PATH)
    label_encoder: LabelEncoder = load_joblib(LABEL_ENCODER_PATH)
    X = preprocessor.transform(df)
    y = label_encoder.transform(df[TARGET_COLUMN]) if TARGET_COLUMN in df.columns else None
    return X, y, preprocessor.feature_columns, label_encoder


def transform_features_for_prediction(df: pd.DataFrame):
    """Transform uploaded feature rows for UI prediction with the saved training preprocessor."""
    preprocessor: DryBeanPreprocessor = load_joblib(PREPROCESSOR_PATH)
    X = preprocessor.transform(df)
    return X, preprocessor.feature_columns
