from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET_COLUMN, TEST_PATH, TRAIN_PATH, VAL_PATH, ensure_dirs
from .utils import friendly_file_check


_OVERLAP_WARNING_PRINTED = False


def load_train_test(include_val_in_train: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSV files and optionally merge validation data into training data."""
    friendly_file_check(TRAIN_PATH, "training data")
    friendly_file_check(TEST_PATH, "test data")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if include_val_in_train and VAL_PATH.exists():
        val_df = pd.read_csv(VAL_PATH)
        train_df = pd.concat([train_df, val_df], ignore_index=True)

    _validate_schema(train_df, "train")
    _validate_schema(test_df, "test")
    train_df = remove_test_overlaps(train_df, test_df)
    return train_df, test_df


def load_all_available() -> pd.DataFrame:
    """Load all available project CSV files for EDA."""
    friendly_file_check(TRAIN_PATH, "training data")
    frames = [pd.read_csv(TRAIN_PATH)]
    if VAL_PATH.exists():
        frames.append(pd.read_csv(VAL_PATH))
    if TEST_PATH.exists():
        frames.append(pd.read_csv(TEST_PATH))
    df = pd.concat(frames, ignore_index=True)
    _validate_schema(df, "combined")
    return df


def split_raw_dataset(raw_path, test_size: float = 0.2) -> None:
    """Split a single raw Dry Bean CSV into data/train.csv and data/test.csv."""
    ensure_dirs()
    raw_df = pd.read_csv(raw_path)
    _validate_schema(raw_df, "raw")
    train_df, test_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=raw_df[TARGET_COLUMN],
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column != TARGET_COLUMN]


def _normalized_overlap_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize rows only for train/test overlap detection, without fitting anything."""
    normalized = df.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    for column in get_feature_columns(normalized):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if TARGET_COLUMN in normalized.columns:
        normalized[TARGET_COLUMN] = (
            normalized[TARGET_COLUMN].astype(str).str.strip().str.upper().str.replace("0", "O").str.replace("3", "E")
        )
    return normalized


def _row_keys(df: pd.DataFrame) -> pd.Series:
    normalized = _normalized_overlap_frame(df)
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    return normalized.astype(str).agg("|".join, axis=1)


def remove_test_overlaps(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows from training data that duplicate clean-normalized test rows."""
    global _OVERLAP_WARNING_PRINTED
    test_keys = set(_row_keys(test_df))
    train_keys = _row_keys(train_df)
    overlap_mask = train_keys.isin(test_keys)
    overlap_count = int(overlap_mask.sum())
    if overlap_count and not _OVERLAP_WARNING_PRINTED:
        print(f"[WARN] Removed {overlap_count} train rows duplicated in test set to avoid data leakage.")
        _OVERLAP_WARNING_PRINTED = True
    return train_df.loc[~overlap_mask].reset_index(drop=True)


def _validate_schema(df: pd.DataFrame, name: str) -> None:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{name} data must contain target column '{TARGET_COLUMN}'.")
    if df.shape[1] < 2:
        raise ValueError(f"{name} data must contain features and target.")
