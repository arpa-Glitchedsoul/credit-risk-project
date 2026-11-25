import numpy as np
import pandas as pd

LATE_COLS = ["NumberOfTime30-59DaysPastDueNotWorse",
             "NumberOfTimes90DaysLate",
             "NumberOfTime60-89DaysPastDueNotWorse"]
SENTINELS = [96, 98]

def clean_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based cleaning. Learns nothing from the data, so it is safe before the split."""
    df = df.copy()
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Flag FIRST, then neutralise the codes (otherwise we lose track of which rows had them)
    df["late_code_flag"] = df[LATE_COLS].isin(SENTINELS).any(axis=1).astype(int)
    df[LATE_COLS] = df[LATE_COLS].replace(SENTINELS, np.nan)

    # Flag missing income before any filling happens
    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)

    # Drop impossible ages
    return df[df["age"] >= 18]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple same-row features. Learns nothing from data, never uses the target."""
    df = df.copy()
    # min_count=1 keeps NaN when all three are missing (the 96/98 rows)
    df["total_late"] = df[LATE_COLS].sum(axis=1, min_count=1)
    df["income_per_person"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["over_limit"] = (df["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)
    return df