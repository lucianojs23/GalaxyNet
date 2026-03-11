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
    # Nota: o arquivo gz2_hart16.csv usa 'dr7objid' e 'total_votes'
    gz2_cols = [
        'dr7objid',                                          # Object ID for cross-matching with SDSS
        't01_smooth_or_features_a01_smooth_fraction',
        't01_smooth_or_features_a02_features_or_disk_fraction',
        't01_smooth_or_features_a03_star_or_artifact_fraction',
        't03_bar_a06_bar_fraction',
        't03_bar_a07_no_bar_fraction',
        't04_spiral_a08_spiral_fraction',
        't04_spiral_a09_no_spiral_fraction',
        'total_votes'                                        # Total votes for confidence
    ]

    gz2_df = pd.read_csv(file_path, usecols=gz2_cols)

    # Renomear para nomes padronizados usados no resto do pipeline
    gz2_df.rename(columns={
        'dr7objid': 'objid',
        'total_votes': 't01_smooth_or_features_total_weight'
    }, inplace=True)

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

# ─────────────────────────────────────────────────────────────────────────────
# Download de Imagens FITS do SDSS
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from astropy.io import fits
from tqdm import tqdm

def download_galaxy_image_cutout(ra, dec, objid, size_pixels=64,
                                  band_list=['g', 'r', 'i']):
    """
    Downloads a cutout image for a galaxy from SDSS in specified bands.

    Args:
        ra          (float): Right Ascension of the galaxy (degrees).
        dec         (float): Declination of the galaxy (degrees).
        objid         (int): SDSS object ID.
        size_pixels   (int): Size of the square cutout image in pixels (e.g., 64 for 64x64).
        band_list    (list): List of photometric bands to download (e.g., ['g', 'r', 'i']).

    Returns:
        np.ndarray: A 3D NumPy array (size_pixels, size_pixels, len(band_list))
                    containing the image data, or None if download fails.
    """
    pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    try:
        # Get images for the specified bands
        images = SDSS.get_images(
            coordinates=pos,
            radius=size_pixels * 0.396 * u.arcsec,  # 0.396 arcsec/pixel for SDSS
            band=band_list,
            data_release=17,
            cutout_size=size_pixels * u.pixel
        )

        if not images:
            print(f"Warning: No images found for objid {objid} at RA={ra}, Dec={dec}")
            return None

        # Stack images from different bands into a single array
        channels = []
        for img_hdu in images:
            # img_hdu is a list of HDU objects, usually the first one contains the image data
            data = img_hdu[0].data
            if data.shape[0] != size_pixels or data.shape[1] != size_pixels:
                print(f"Warning: Image cutout for objid {objid} has unexpected size {data.shape}. "
                      f"Expected {size_pixels}x{size_pixels}.")
                return None
            channels.append(data)

        # Ensure all channels were successfully retrieved
        if len(channels) == len(band_list):
            return np.stack(channels, axis=-1)  # Stack along the last axis (channels)
        else:
            print(f"Warning: Missing bands for objid {objid}. "
                  f"Expected {len(band_list)}, got {len(channels)}.")
            return None

    except Exception as e:
        print(f"Error downloading image for objid {objid}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Download em lote com tqdm
# ─────────────────────────────────────────────────────────────────────────────

def download_images_batch(catalog_df, images_dir, size_pixels=64,
                           band_list=['g', 'r', 'i'], max_images=None):
    """
    Baixa imagens FITS para todas as galáxias do catálogo em lote,
    salvando cada imagem como arquivo .npy em images_dir.
    Pula imagens já existentes para evitar downloads repetidos.

    Args:
        catalog_df  (pd.DataFrame): Catálogo com colunas 'objid', 'ra', 'dec'.
        images_dir        (str): Diretório onde salvar as imagens (.npy).
        size_pixels       (int): Tamanho do cutout em pixels.
        band_list        (list): Bandas fotométricas a baixar.
        max_images        (int): Limite de imagens (None = todas). Use 500 para teste.

    Returns:
        list: Lista de objids cujo download foi bem-sucedido.
    """
    os.makedirs(images_dir, exist_ok=True)

    df = catalog_df.copy()
    if max_images is not None:
        df = df.head(max_images)

    successful = []
    failed     = []

    print(f"Iniciando download de até {len(df)} imagens em '{images_dir}'...")
    print("(Imagens já existentes serão puladas)\n")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Baixando imagens"):
        objid = int(row['objid'])
        out_path = os.path.join(images_dir, f"{objid}.npy")

        # Pular se já existir
        if os.path.exists(out_path):
            successful.append(objid)
            continue

        img = download_galaxy_image_cutout(
            ra=float(row['ra']),
            dec=float(row['dec']),
            objid=objid,
            size_pixels=size_pixels,
            band_list=band_list
        )

        if img is not None:
            np.save(out_path, img)
            successful.append(objid)
        else:
            failed.append(objid)

    print(f"\n✓ Download concluído:")
    print(f"  Bem-sucedidos : {len(successful)}")
    print(f"  Falhos        : {len(failed)}")

    return successful
