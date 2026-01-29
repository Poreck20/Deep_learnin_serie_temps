import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ===================================  Time Series Plots ==========================================

def plot_validations_by_station(
    df,
    stations=None,
    start_year=2015,
    end_year=2025,
    freq='D'
):
    """
    Trace les validations par station avec agrégation temporelle.

    Args:
        df (pd.DataFrame): DataFrame source
        stations (list): liste des stations à afficher (None = toutes)
        start_year (int): année de début
        end_year (int): année de fin
        freq (str): fréquence temporelle ('D', 'W', 'M')
    """

    df = df.copy()

    # Filtrage temporel
    df = df[(df['DATE'].dt.year >= start_year) &
            (df['DATE'].dt.year <= end_year)]

    # Agrégation temporelle par station
    df_agg = (
        df
        .set_index('DATE')
        .groupby('LIBELLE_ARRET')
        .resample(freq)['VALD_TOTAL']
        .sum()
        .reset_index()
    )

    if stations is None:
        stations = df_agg['LIBELLE_ARRET'].unique().tolist()

    figs = []

    freq_label = {
        'D': 'quotidiennes',
        'W': 'hebdomadaires',
        'M': 'mensuelles'
    }.get(freq, freq)

    for station in stations:
        df_station = df_agg[df_agg['LIBELLE_ARRET'] == station]

        fig = px.line(
            df_station,
            x='DATE',
            y='VALD_TOTAL',
            title=f'Validations {freq_label} – {station} ({start_year}-{end_year})',
            labels={
                'VALD_TOTAL': 'Nombre de validations',
                'DATE': 'Date'
            },
            template='plotly'
        )

        fig.update_xaxes(rangeslider_visible=True)
        fig.update_yaxes(title='Nombre de validations')

        figs.append(fig)

    for fig in figs:
        fig.show()

    #return figs


#===================================  Distribution + Gaussienne ==========================================

def plot_vald_total_gaussian(df, date_col='DATE', value_col='VALD_TOTAL', bins=30):
    """
    Trace la distribution journalière de VALD_TOTAL et la compare
    à une loi gaussienne + Q-Q plot.

    Args:
        df (pd.DataFrame): DataFrame source
        date_col (str): colonne de date
        value_col (str): colonne des valeurs
        bins (int): nombre de classes de l'histogramme
    """

    # Agrégation journalière
    df_jour = (
        df.groupby(date_col, as_index=False)[value_col]
          .sum()
    )

    values = df_jour[value_col]

    # Paramètres de la loi normale
    mu = values.mean()
    sigma = values.std()

    # ---------- Histogramme + gaussienne ----------
    plt.figure()
    plt.hist(values, bins=bins, density=True)

    x = np.linspace(values.min(), values.max(), 1000)
    gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

    plt.plot(x, gauss)
    plt.xlabel("VALD_TOTAL par jour")
    plt.ylabel("Densité")
    plt.title("Distribution journalière de VALD_TOTAL vs loi gaussienne")
    plt.show()

    # ---------- Q-Q plot ----------
    plt.figure()
    stats.probplot(values, dist="norm", plot=plt)
    plt.title("Q-Q plot de VALD_TOTAL journalier")
    plt.show()

    return {
        "mean": mu,
        "std": sigma,
        "nb_days": len(values)
    }



#=================================== ACF PCA ==========================================


def plot_acf_pacf_groups(
    df,
    stations="GARE DE LYON",
    date_col="DATE",
    target_col="VALD_TOTAL",
    station_col="LIBELLE_ARRET",
    max_lag=60
):
    """
    Trace ACF et PACF (avec bandes de confiance) pour :
    - Toutes les stations (global)
    - Top 5 stations
    - Bottom 5 stations
    - Une ou plusieurs stations choisies
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # ================== gestion stations input ==================
    if isinstance(stations, str):
        stations = [stations]

    # ================== classement des gares ==================
    mean_by_station = (
        df.groupby(station_col)[target_col]
          .mean()
          .sort_values(ascending=False)
    )

    top5 = mean_by_station.head(5).index.tolist()
    bottom5 = mean_by_station.tail(5).index.tolist()

    # ================== séries temporelles ==================
    series_dict = {
        "Toutes les stations": (
            df.groupby(date_col)[target_col]
              .sum()
              .sort_index()
        ),
        "Top 5 stations": (
            df[df[station_col].isin(top5)]
              .groupby(date_col)[target_col]
              .sum()
              .sort_index()
        ),
        "Bottom 5 stations": (
            df[df[station_col].isin(bottom5)]
              .groupby(date_col)[target_col]
              .sum()
              .sort_index()
        )
    }

    # ================== stations personnalisées ==================
    for station in stations:
        if station not in df[station_col].unique():
            print(f"⚠️ Station inconnue : {station}")
            continue

        series_dict[f"Station – {station}"] = (
            df[df[station_col] == station]
              .groupby(date_col)[target_col]
              .sum()
              .sort_index()
        )

    # ================== ACF / PACF ==================
    for name, series in series_dict.items():
        series = series.dropna()

        if len(series) < max_lag + 5:
            print(f"⚠️ Série trop courte pour {name}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        plot_acf(
            series,
            lags=max_lag,
            ax=axes[0],
            alpha=0.05
        )

        plot_pacf(
            series,
            lags=max_lag,
            ax=axes[1],
            method="ywm",
            alpha=0.05
        )

        axes[0].set_title(f"ACF – {name}")
        axes[0].set_xlabel("Lag (jours)")
        axes[0].set_ylabel("Autocorrélation")

        axes[1].set_title(f"PACF – {name}")
        axes[1].set_xlabel("Lag (jours)")
        axes[1].set_ylabel("Autocorrélation partielle")

        plt.tight_layout()
        plt.show()
