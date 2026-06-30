from __future__ import annotations

import pandas as pd


def observation_only(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def physical_semantic(df: pd.DataFrame) -> pd.Series:
    return df["physical_semantic_match"].astype(bool) & df["is_temporally_valid"].astype(bool)


def full_fcsl(df: pd.DataFrame, epsilon: float = 0.68) -> pd.Series:
    return df["is_temporally_valid"].astype(bool) & df["satisfies_requirement"].notna() & (df["rel_score"] >= epsilon)
