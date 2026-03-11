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

