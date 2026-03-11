# data_loader.py - Funções para download e carregamento de dados SDSS e Galaxy Zoo 2

import pandas as pd
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# ─────────────────────────────────────────────────────────────────────────────
# Download de Dados Fotométricos e Espectroscópicos do SDSS
# ─────────────────────────────────────────────────────────────────────────────

def download_sdss_galaxy_data(ra_center, dec_center, radius_deg,
                               max_records=5000):
    """
    Downloads photometric and spectroscopic data for galaxies from SDSS DR17.

    Args:
        ra_center  (float): Right Ascension of the center of the search area (degrees).
        dec_center (float): Declination of the center of the search area (degrees).
        radius_deg (float): Radius of the search area (degrees).
        max_records  (int): Maximum number of records to retrieve.

    Returns:
        pd.DataFrame: DataFrame containing the downloaded galaxy data.
    """
    # SQL query to select relevant galaxy properties
    query = f"""
    SELECT TOP {max_records}
        p.objid, p.ra, p.dec,
        p.u, p.g, p.r, p.i, p.z,
        p.petroRad_r, p.petroR50_r, p.petroR90_r,
        p.deVAB_r, p.expAB_r,
        p.lnLDeV_r, p.lnLExp_r, p.lnLStar_r,
        p.fracDeV_r,
        s.z AS redshift, s.zErr,
        s.velDisp, s.velDispErr
    FROM PhotoObj AS p
    JOIN SpecObj AS s ON s.bestobjid = p.objid
    WHERE
        p.clean = 1 AND
        p.r BETWEEN 14.0 AND 19.5 AND
        s.z BETWEEN 0.02 AND 0.25 AND
        s.zWarning = 0 AND
        dbo.fDistanceArcMinEq(p.ra, p.dec, {ra_center}, {dec_center}) < {radius_deg * 60}
    """

    # Execute the query
    result = SDSS.query_sql(query)

    if result is not None:
        df = result.to_pandas()
        print(f"Downloaded {len(df)} records from SDSS.")
        return df
    else:
        print("No results found for the given query.")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Função auxiliar: salvar resultado em data/raw/
# ─────────────────────────────────────────────────────────────────────────────

def save_sdss_data(df, filename="sdss_galaxies.csv", raw_dir=None):
    """
    Salva o DataFrame do SDSS em data/raw/.

    Args:
        df        (pd.DataFrame): DataFrame retornado por download_sdss_galaxy_data.
        filename         (str): Nome do arquivo CSV de saída.
        raw_dir          (str): Caminho para o diretório raw/.
                                Se None, usa '../data/raw' relativo a este arquivo.
    """
    if raw_dir is None:
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, filename)
    df.to_csv(out_path, index=False)
    print(f"Dados salvos em: {out_path}")
    return out_path

# ─────────────────────────────────────────────────────────────────────────────
# Carregamento e Filtragem de Dados do Galaxy Zoo 2
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter_gz2(file_path="gz2_hart16.csv"):
    """
    Loads the Galaxy Zoo 2 dataset and filters relevant columns for classification.

    Args:
        file_path (str): Path to the gz2_hart16.csv file.

    Returns:
        pd.DataFrame: Filtered DataFrame with GZ2 classifications.
    """
    # Columns to keep from the GZ2 dataset
    gz2_cols = [
        'dr8objid',                                          # Object ID for cross-matching with SDSS
        't01_smooth_or_features_a01_smooth_fraction',
        't01_smooth_or_features_a02_features_or_disk_fraction',
        't01_smooth_or_features_a03_star_or_artifact_fraction',
        't03_bar_a06_bar_fraction',
        't03_bar_a07_no_bar_fraction',
        't04_spiral_a08_spiral_fraction',
        't04_spiral_a09_no_spiral_fraction',
        't01_smooth_or_features_total_weight'                # Total votes for confidence
    ]

    gz2_df = pd.read_csv(file_path, usecols=gz2_cols)

    # Rename 'dr8objid' to 'objid' for easier merging with SDSS data
    gz2_df.rename(columns={'dr8objid': 'objid'}, inplace=True)

    print(f"Loaded {len(gz2_df)} records from GZ2.")
    return gz2_df


def assign_morphological_class(row, min_vote_fraction=0.8, min_total_votes=20):
    """
    Assigns a morphological class based on GZ2 vote fractions.

    Args:
        row               (pd.Series): A row from the GZ2 DataFrame.
        min_vote_fraction   (float): Minimum fraction of votes for a class to be assigned.
        min_total_votes       (int): Minimum total votes for a reliable classification.

    Returns:
        str: Assigned class ('Elliptical', 'Spiral', 'Lenticular', 'Irregular', 'Uncertain').
    """
    if row['t01_smooth_or_features_total_weight'] < min_total_votes:
        return 'Uncertain'

    if row['t01_smooth_or_features_a01_smooth_fraction'] >= min_vote_fraction:
        # Smooth galaxies: check for disk features to distinguish Elliptical from Lenticular
        if row['t01_smooth_or_features_a02_features_or_disk_fraction'] >= min_vote_fraction:
            return 'Lenticular'   # Smooth with disk features
        else:
            return 'Elliptical'   # Purely smooth
    elif row['t01_smooth_or_features_a02_features_or_disk_fraction'] >= min_vote_fraction:
        # Galaxies with features/disk: check for spiral arms
        if row['t04_spiral_a08_spiral_fraction'] >= min_vote_fraction:
            return 'Spiral'
        else:
            return 'Irregular'    # Disk but no clear spiral arms

    return 'Uncertain'            # Default if no clear class is assigned


# ─────────────────────────────────────────────────────────────────────────────
# Função auxiliar: merge SDSS + GZ2
# ─────────────────────────────────────────────────────────────────────────────

def merge_sdss_gz2(sdss_df, gz2_df):
    """
    Faz o merge entre os dados do SDSS e do Galaxy Zoo 2 pelo objid,
    atribui as classes morfológicas e remove as entradas 'Uncertain'.

    Args:
        sdss_df (pd.DataFrame): DataFrame do SDSS.
        gz2_df  (pd.DataFrame): DataFrame do GZ2 já carregado por load_and_filter_gz2().

    Returns:
        pd.DataFrame: Catálogo mesclado com coluna 'morph_class'.
    """
    merged = pd.merge(sdss_df, gz2_df, on='objid', how='inner')
    print(f"Registros após merge: {len(merged)}")

    # Atribuir classes morfológicas
    merged['morph_class'] = merged.apply(assign_morphological_class, axis=1)

    # Distribuição antes de filtrar incertos
    print("\nDistribuição de classes (incluindo Uncertain):")
    print(merged['morph_class'].value_counts())

    # Remover classificações incertas do conjunto de treino
    merged_clean = merged[merged['morph_class'] != 'Uncertain'].copy()
    print(f"\nRegistros após remover 'Uncertain': {len(merged_clean)}")
    print("\nDistribuição final de classes:")
    print(merged_clean['morph_class'].value_counts())

    return merged_clean

