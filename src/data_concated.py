import pandas as pd
import numpy as np 

from src.importation_data import load_data_csv
from src.importation_data import load_data_txt


#===================Donne 2015====================  

df_S1_2015 = load_data_csv("2015S1_NB_FER")
df_S2_2015 = load_data_csv("2015S2_NB_FER")


#===================Donne 2016====================  

df_S1_2016 = load_data_txt("2016S1_NB_FER")
df_S2_2016 = load_data_txt("2016S2_NB_FER")


#===================Donne 2017====================  

df_S1_2017 = load_data_txt("2017S1_NB_FER")
df_S2_2017 = load_data_txt("2017_S2_NB_FER")


#===================Donne 2018====================  

df_S1_2018 = load_data_txt("2018_S1_NB_FER")
df_S2_2018 = load_data_txt("2018_S2_NB_FER")

#===================Donne 2019====================  

df_S1_2019 = load_data_txt("2019_S1_NB_FER")
df_S2_2019 = load_data_txt("2019_S2_NB_FER")

#===================Donne 2020====================  

df_S1_2020 = load_data_txt("2020_S1_NB_FER")
df_S2_2020 = load_data_txt("2020_S2_NB_FER")

#===================Donne 2021====================  

df_S1_2021 = load_data_txt("2021_S1_NB_FER")
df_S2_2021 = load_data_txt("2021_S2_NB_FER")


# ================Donne 2022=====================
df_S1_2022 = load_data_txt("2022_S1_NB_FER")
df_S2_2022 = load_data_txt("2022_S2_NB_FER", sep=';')

#===================Donne 2023====================

df_S1_2023 = load_data_txt("2023_S1_NB_FER", encoding="ANSI")
df_S2_2023 = load_data_txt("2023_S2_NB_FER", encoding="utf-16le")

#=====================Donnee 2024=================

df_S1_2024 = load_data_txt("2024_S1_NB_FER", encoding="ANSI")
df_T3_2024 = load_data_txt("2024_T3_NB_FER")
df_T4_2024 = load_data_csv("2024_T4_NB_FER")

#=====================Donnee 2025=================
df_T1_2025 = load_data_csv("2025_T1_NB_FER")
df_T2_2025 = load_data_csv("2025_T2_NB_FER")
df_T3_2025 = load_data_csv("2025_T3_NB_FER")


data_gare  = pd.read_csv("./data/schema_gares-gf.csv", sep=";", encoding="utf-8")
data_gare = data_gare[["NOM_GARE", "X", "Y", "Geo Point"]]
data_gare['NOM_GARE'] = data_gare['NOM_GARE'].str.strip().str.upper()
data_gare = data_gare.drop_duplicates()

#====================limitations du nombre de gare ==================



def df_concated( min_observations = 3888):
    """
    Concatène une liste de DataFrames en un seul DataFrame.
    
    Args:
        dataframes (list of pd.DataFrame): Liste des DataFrames à concaténer.
        
    Returns:
        pd.DataFrame: DataFrame résultant de la concaténation.
    """
    df = pd.concat([df_S1_2015, df_S2_2015, df_S1_2016, df_S2_2016, df_S1_2017, df_S2_2017, df_S1_2018, df_S2_2018, df_S1_2019, df_S2_2019, df_S1_2020, df_S2_2020, df_S1_2021, df_S2_2021, df_S1_2022, df_S2_2022, df_S1_2023, df_S2_2023, df_S1_2024, df_T3_2024, df_T4_2024, df_T1_2025, df_T2_2025, df_T3_2025], ignore_index=True)
    df['JOUR'] = pd.to_datetime(df['JOUR'], format='mixed', errors='coerce')
    df.rename(columns={'JOUR': 'DATE'}, inplace=True)
    df['LIBELLE_ARRET'] = df['LIBELLE_ARRET'].str.strip().str.upper()
    df = df.drop_duplicates()
    #df = df.drop(columns=["CATEGORIE_TITRE", "NB_VALD"])
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = (
    df
    .groupby(["DATE", "LIBELLE_ARRET"], as_index=False)["NB_VALD"]
    .sum()
    .rename(columns={"NB_VALD": "VALD_TOTAL"}))

    #df = df.merge(data_gare, left_on='LIBELLE_ARRET', right_on='NOM_GARE', how='inner')
    df = df.assign(
    ANNEE=df['DATE'].dt.year,
    MOIS=df['DATE'].dt.month,
    JOUR=df['DATE'].dt.day,
    dayofweek = df['DATE'].dt.dayofweek) 

    gare_counts = df["LIBELLE_ARRET"].value_counts()
    gare_valide = gare_counts[gare_counts >= min_observations].index.tolist()
    print(f" {len (gare_valide)} boutiques avec ≥{min_observations} observations")
    df = df[df["LIBELLE_ARRET"].isin (gare_valide)]
    

    return df