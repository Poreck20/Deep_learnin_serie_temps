import pandas as pd
import numpy as np

# =============================================================================
# JOURS FÉRIÉS – FRANCE MÉTROPOLITAINE
# =============================================================================

DATA_FERIES = pd.read_csv(
    "./data/jours_feries_metropole.csv",
    sep=",",
    encoding="utf-8"
)
DATA_FERIES["date"] = pd.to_datetime(DATA_FERIES["date"]).dt.normalize()


def add_public_holiday_feature(
    df,
    holidays_df=DATA_FERIES,
    date_col="DATE",
    add_name=True
):
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    holidays = holidays_df["date"]

    df["is_holiday"] = df[date_col].isin(holidays)

    if add_name:
        holiday_map = (
            holidays_df
            .set_index("date")["nom_jour_ferie"]
            .to_dict()
        )
        df["holiday_name"] = df[date_col].map(holiday_map)

    return df


# =============================================================================
# VACANCES SCOLAIRES – ÎLE-DE-FRANCE (ZONE C)
# =============================================================================

DATA_SCOLAIRE = pd.read_csv(
    "./data/fr-en-calendrier-scolaire.csv",
    sep=";",
    encoding="utf-8"
)

DATA_SCOLAIRE = DATA_SCOLAIRE[
    (DATA_SCOLAIRE["Zones"] == "Zone C") &
    (DATA_SCOLAIRE["Académies"].isin(["Paris", "Créteil", "Versailles"]))
]

DATA_SCOLAIRE["Date de début"] = pd.to_datetime(
    DATA_SCOLAIRE["Date de début"], utc=True
).dt.tz_localize(None)

DATA_SCOLAIRE["Date de fin"] = pd.to_datetime(
    DATA_SCOLAIRE["Date de fin"], utc=True
).dt.tz_localize(None)


def add_school_holiday_feature(df, vacances_df=DATA_SCOLAIRE, date_col="DATE"):
    df = df.copy()
    df["is_school_holiday"] = False

    for start, end in zip(
        vacances_df["Date de début"],
        vacances_df["Date de fin"]
    ):
        df.loc[
            (df[date_col] >= start) & (df[date_col] <= end),
            "is_school_holiday"
        ] = True

    return df


# =============================================================================
# FEATURES CYCLIQUES (SEMAINE / MOIS)
# =============================================================================

def add_cyclical_features(df, date_col="DATE"):
    df = df.copy()

    dayofweek = df[date_col].dt.dayofweek
    month = df[date_col].dt.month

    df["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7)

    df["dayofmonth_sin"] = np.sin(2 * np.pi * (df[date_col].dt.day - 1) / 31)
    df["dayofmonth_cos"] = np.cos(2 * np.pi * (df[date_col].dt.day - 1) / 31)

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


# =============================================================================
# WEEK-END
# =============================================================================

def add_weekend_feature(df, date_col="DATE"):
    df = df.copy()
    df["is_weekend"] = df[date_col].dt.dayofweek >= 5
    return df


# =============================================================================
# SAISONS
# =============================================================================

def add_season_features(df, date_col="DATE"):
    df = df.copy()
    month = df[date_col].dt.month

    df["is_winter"] = month.isin([12, 1, 2])
    df["is_spring"] = month.isin([3, 4, 5])
    df["is_summer"] = month.isin([6, 7, 8])
    df["is_autumn"] = month.isin([9, 10, 11])

    return df


# =============================================================================
# COVID – CONFINEMENTS
# =============================================================================

LOCKDOWNS = [
    (pd.Timestamp("2020-03-17"), pd.Timestamp("2020-05-11")),
    (pd.Timestamp("2020-10-30"), pd.Timestamp("2020-12-15")),
    (pd.Timestamp("2021-04-03"), pd.Timestamp("2021-05-03")),
]


def add_covid_feature(df, date_col="DATE"):
    df = df.copy()
    df["is_lockdown"] = False

    for start, end in LOCKDOWNS:
        df.loc[
            (df[date_col] >= start) & (df[date_col] <= end),
            "is_lockdown"
        ] = True

    return df


# =============================================================================
# INTERACTIONS
# =============================================================================

def add_interaction_features(df):
    df = df.copy()

    if {"is_weekend", "is_school_holiday"}.issubset(df.columns):
        df["weekend_school_holiday"] = (
            df["is_weekend"] & df["is_school_holiday"]
        )

    if {"is_weekend", "is_holiday"}.issubset(df.columns):
        df["weekend_holiday"] = (
            df["is_weekend"] & df["is_holiday"]
        )

    return df


# =============================================================================
# PIPELINE FINAL
# =============================================================================

def pipeline_feature_engineering(df, date_col="DATE"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()

    df = add_public_holiday_feature(df, date_col=date_col)
    df = add_school_holiday_feature(df, date_col=date_col)
    df = add_weekend_feature(df, date_col=date_col)
    df = add_season_features(df, date_col=date_col)
    df = add_covid_feature(df, date_col=date_col)
    df = add_interaction_features(df)
    df = add_cyclical_features(df, date_col=date_col)

    return df
