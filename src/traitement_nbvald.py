import pandas as pd 
import numpy as np  


def clean_nb_vald(series, remplacement_moins_5=2):
    """
    Nettoyage adaptatif de NB_VALD :
    - si déjà numérique → retourne tel quel
    - si texte :
        - supprime espaces visibles et invisibles
        - remplace 'Moins de 5' si présent par la mediane (2 par défaut)
        - convertit en numérique
    """
    # Cas 1 — déjà numérique
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    # Cas 2 — texte
    s = series.astype(str)

    # Nettoyage minimal
    s = (
        s.str.strip()
         .str.replace("\xa0", " ", regex=False)
    )

    # Remplacement seulement si nécessaire
    if s.str.contains("Moins de 5", regex=False, na=False).any():
        s = s.replace("Moins de 5", str(remplacement_moins_5))

    # Suppression des séparateurs de milliers si présents
    if s.str.contains(" ", regex=False, na=False).any():
        s = s.str.replace(" ", "", regex=False)

    # Conversion finale
    s = pd.to_numeric(s, errors="coerce")

    # Sécurité finale (rare)
    if s.isna().any():
        s = s.fillna(remplacement_moins_5)

    return s.astype(int)

