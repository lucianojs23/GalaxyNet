# data_loader.py - Funções para download e carregamento de dados SDSS e Galaxy Zoo 2

import pandas as pd
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from astropy.io import fits
from tqdm import tqdm
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
    query = f"""
    SELECT
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

    O arquivo gz2_hart16.csv contém as colunas 'ra', 'dec' e 'dr7objid'.
    O merge com o SDSS é feito via cruzamento espacial por coordenadas (ra, dec),
    pois o GZ2 usa objids do DR7 enquanto o SDSS foi baixado no DR17.

    Args:
        file_path (str): Caminho para o arquivo gz2_hart16.csv.

    Returns:
        pd.DataFrame: DataFrame filtrado com classificações do GZ2.
    """
    gz2_cols = [
        'dr7objid',
        'ra',
        'dec',
        't01_smooth_or_features_a01_smooth_fraction',
        't01_smooth_or_features_a02_features_or_disk_fraction',
        't01_smooth_or_features_a03_star_or_artifact_fraction',
        't03_bar_a06_bar_fraction',
        't03_bar_a07_no_bar_fraction',
        't04_spiral_a08_spiral_fraction',
        't04_spiral_a09_no_spiral_fraction',
        'total_votes',
    ]

    gz2_df = pd.read_csv(file_path, usecols=gz2_cols)

    # Renomear para nomes padronizados usados no resto do pipeline
    gz2_df.rename(columns={
        'dr7objid'   : 'gz2_objid',
        'ra'         : 'gz2_ra',
        'dec'        : 'gz2_dec',
        'total_votes': 'total_votes_gz2',
    }, inplace=True)

    print(f"Loaded {len(gz2_df)} records from GZ2.")
    return gz2_df


# ─────────────────────────────────────────────────────────────────────────────
# Atribuição de classes morfológicas
# ─────────────────────────────────────────────────────────────────────────────

def assign_morphological_class(row, min_vote_fraction=0.8, min_total_votes=20):
    """
    Assigns a morphological class based on GZ2 vote fractions.

    Args:
        row               (pd.Series): Uma linha do DataFrame mesclado.
        min_vote_fraction   (float): Fração mínima de votos para confirmar classe.
        min_total_votes       (int): Mínimo de votos para classificação confiável.

    Returns:
        str: Classe atribuída ('Elliptical', 'Spiral', 'Lenticular', 'Irregular', 'Uncertain').
    """
    if row['total_votes_gz2'] < min_total_votes:
        return 'Uncertain'

    if row['t01_smooth_or_features_a01_smooth_fraction'] >= min_vote_fraction:
        if row['t01_smooth_or_features_a02_features_or_disk_fraction'] >= min_vote_fraction:
            return 'Lenticular'
        else:
            return 'Elliptical'
    elif row['t01_smooth_or_features_a02_features_or_disk_fraction'] >= min_vote_fraction:
        if row['t04_spiral_a08_spiral_fraction'] >= min_vote_fraction:
            return 'Spiral'
        else:
            return 'Irregular'

    return 'Uncertain'


# ─────────────────────────────────────────────────────────────────────────────
# Merge SDSS + GZ2 por cruzamento espacial de coordenadas (ra, dec)
# ─────────────────────────────────────────────────────────────────────────────

def merge_sdss_gz2(sdss_df, gz2_df, max_sep_arcsec=1.0):
    """
    Faz o cruzamento espacial entre SDSS e GZ2 usando as coordenadas ra/dec,
    atribui classes morfológicas e remove entradas 'Uncertain'.

    O GZ2 usa objids do DR7 e o SDSS foi baixado no DR17 — por isso o merge
    é feito por proximidade angular (cross-match) e não por objid diretamente.

    Args:
        sdss_df         (pd.DataFrame): DataFrame do SDSS com colunas 'ra', 'dec'.
        gz2_df          (pd.DataFrame): DataFrame do GZ2 carregado por load_and_filter_gz2().
        max_sep_arcsec        (float): Separação máxima em arcsec para considerar
                                       dois objetos como o mesmo (padrão: 1.0 arcsec).

    Returns:
        pd.DataFrame: Catálogo mesclado com coluna 'morph_class'.
    """
    print("Construindo catálogos de coordenadas para cross-match...")

    sdss_coords = SkyCoord(
        ra=sdss_df['ra'].values * u.deg,
        dec=sdss_df['dec'].values * u.deg
    )
    gz2_coords = SkyCoord(
        ra=gz2_df['gz2_ra'].values * u.deg,
        dec=gz2_df['gz2_dec'].values * u.deg
    )

    print(f"Executando cross-match (separação máxima: {max_sep_arcsec} arcsec)...")
    idx, sep2d, _ = sdss_coords.match_to_catalog_sky(gz2_coords)

    # Filtrar apenas pares dentro da separação máxima
    mask = sep2d.arcsec <= max_sep_arcsec
    n_matched = mask.sum()
    print(f"Pares encontrados dentro de {max_sep_arcsec} arcsec: {n_matched} / {len(sdss_df)}")

    # Montar DataFrame mesclado
    sdss_matched = sdss_df[mask].copy().reset_index(drop=True)
    gz2_matched  = gz2_df.iloc[idx[mask]].copy().reset_index(drop=True)

    merged = pd.concat([sdss_matched, gz2_matched], axis=1)
    print(f"Registros após merge: {len(merged)}")

    # Atribuir classes morfológicas
    merged['morph_class'] = merged.apply(assign_morphological_class, axis=1)

    print("\nDistribuição de classes (incluindo Uncertain):")
    print(merged['morph_class'].value_counts())

    # Remover classificações incertas
    merged_clean = merged[merged['morph_class'] != 'Uncertain'].copy()
    print(f"\nRegistros após remover 'Uncertain': {len(merged_clean)}")
    print("\nDistribuição final de classes:")
    print(merged_clean['morph_class'].value_counts())

    return merged_clean


# ─────────────────────────────────────────────────────────────────────────────
# Download de Imagens FITS do SDSS
# ─────────────────────────────────────────────────────────────────────────────

def download_galaxy_image_cutout(ra, dec, objid, size_pixels=64,
                                  band_list=['g', 'r', 'i'],
                                  timeout=60, max_retries=3):
    """
    Downloads a cutout image for a galaxy from SDSS in specified bands.

    O SDSS.get_images() retorna a placa inteira (~2048x1489 px). O recorte
    centrado na posição da galáxia é feito com astropy.nddata.Cutout2D.

    Args:
        ra          (float): Right Ascension of the galaxy (degrees).
        dec         (float): Declination of the galaxy (degrees).
        objid         (int): SDSS object ID.
        size_pixels   (int): Tamanho do cutout quadrado em pixels (padrão: 64).
        band_list    (list): Bandas fotométricas a baixar (padrão: ['g','r','i']).
        timeout       (int): Timeout em segundos por tentativa (padrão: 60).
        max_retries   (int): Número máximo de tentativas em caso de falha (padrão: 3).

    Returns:
        np.ndarray: Array (size_pixels, size_pixels, len(band_list)) float32, ou None.
    """
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    import time

    pos = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    for attempt in range(1, max_retries + 1):
        try:
            hdu_lists = SDSS.get_images(
                coordinates=pos,
                radius=size_pixels * 0.396 * u.arcsec,
                band=band_list,
                data_release=17,
                timeout=timeout,
                show_progress=False
            )

            if not hdu_lists:
                return None

            channels = []
            for hdu_list in hdu_lists:
                hdu  = hdu_list[0]
                wcs  = WCS(hdu.header)
                data = hdu.data

                if data is None:
                    return None

                cutout = Cutout2D(
                    data,
                    position=pos,
                    size=(size_pixels, size_pixels),
                    wcs=wcs,
                    mode='partial',
                    fill_value=0.0
                )
                channels.append(cutout.data.astype(np.float32))

            if len(channels) != len(band_list):
                return None

            return np.stack(channels, axis=-1)

        except Exception as e:
            err = str(e)
            if attempt < max_retries:
                wait = attempt * 5  # 5s, 10s entre tentativas
                print(f"  [objid {objid}] Tentativa {attempt}/{max_retries} falhou: {err[:60]} — aguardando {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [objid {objid}] Falhou após {max_retries} tentativas: {err[:80]}")
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
        max_images        (int): Limite de imagens (None = todas).

    Returns:
        list: Lista de objids com download bem-sucedido.
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
        objid    = int(row['objid'])
        out_path = os.path.join(images_dir, f"{objid}.npy")

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
