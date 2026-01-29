import pandas as pd
import os
from src.traitement_nbvald import clean_nb_vald

def load_data_txt(filename, encoding="utf-8", sep="\t"):
    if not filename.endswith(".txt"):
        filename += ".txt"
    
    file_path = os.path.join("./data", filename)

    df = pd.read_csv(file_path, sep=sep, encoding=encoding)
    df.columns = df.columns.str.upper()

    # Nettoyage NB_VALD
    df["NB_VALD"] = clean_nb_vald(df["NB_VALD"])

    # Types
    df["CODE_STIF_ARRET"] = df["CODE_STIF_ARRET"].astype(str)
    df["LIBELLE_ARRET"] = df["LIBELLE_ARRET"].astype(str)
    df["CATEGORIE_TITRE"] = pd.Categorical(df["CATEGORIE_TITRE"])
    df = df.drop_duplicates()
    # Agrégation
    #df_daily = (
    #df
    #.groupby(["JOUR", "LIBELLE_ARRET"], as_index=False)["NB_VALD"]
    #.sum()
    #.rename(columns={"NB_VALD": "VALD_TOTAL"})
#)

    return df[[
        "JOUR",
        #"CODE_STIF_ARRET",
        #"CODE_STIF_TRNS",
        "LIBELLE_ARRET",
        #"CATEGORIE_TITRE",
        "NB_VALD"
    ]]


   
def load_data_csv(filename, encoding="utf-8", sep=";"):
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    file_path = os.path.join("./data", filename)

    df = pd.read_csv(file_path, sep=sep, encoding=encoding)
    df.columns = df.columns.str.upper()

    # Nettoyage NB_VALD
    df["NB_VALD"] = clean_nb_vald(df["NB_VALD"])

    # Types
    df["CODE_STIF_ARRET"] = df["CODE_STIF_ARRET"].astype(str)
    df["LIBELLE_ARRET"] = df["LIBELLE_ARRET"].astype(str)
    df["CATEGORIE_TITRE"] = pd.Categorical(df["CATEGORIE_TITRE"])
    df = df.drop_duplicates()
    # Agrégation

    return df[[
        "JOUR",
        #"CODE_STIF_ARRET",
        #"CODE_STIF_TRNS",
        "LIBELLE_ARRET",
        #"CATEGORIE_TITRE",
        "NB_VALD"
    ]]


