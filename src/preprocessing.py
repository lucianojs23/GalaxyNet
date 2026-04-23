# preprocessing.py - Funções de pré-processamento tabular e de imagens

import os
import pickle

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Features tabulares usadas pelos modelos MLP e pelo ramo tabular do Híbrido.
# Inclui magnitudes brutas, índices de cor derivados, parâmetros estruturais
# e dados espectroscópicos.
TABULAR_FEATURES = [
    # Magnitudes brutas (5 bandas SDSS)
    'u', 'g', 'r', 'i', 'z',
    # Índices de cor derivados (Seção 5.2 do enunciado)
    'u_g', 'g_r', 'r_i', 'i_z',
    # Espectroscópicos
    'redshift', 'velDisp',
    # Morfológicos / estruturais
    'concentration', 'deVAB_r', 'expAB_r', 'fracDeV_r',
]


# ─────────────────────────────────────────────────────────────────────────────
# Limpeza do Catálogo
# ─────────────────────────────────────────────────────────────────────────────

# Faixas físicas esperadas para colunas SDSS.
# Usadas para detectar e corrigir valores sem ponto decimal (artefato de locale).
_DECIMAL_FIX_RANGES = {
    'u': (10.0, 25.0), 'g': (10.0, 25.0), 'r': (10.0, 25.0),
    'i': (10.0, 25.0), 'z': (10.0, 25.0),
    'petroRad_r':  (0.1, 200.0),
    'petroR50_r':  (0.1, 100.0),
    'petroR90_r':  (0.1, 200.0),
    'velDisp':    (10.0, 600.0),
    'velDispErr':  (0.5, 200.0),
}


def fix_missing_decimal(df):
    """
    Corrige valores SDSS onde o ponto decimal foi perdido (artefato de locale).

    Heurística: se um valor não possui parte fracionária E o valor dividido por
    1000 cai dentro da faixa física esperada da coluna, o ponto decimal foi
    removido durante a exportação/importação do CSV.

    Exemplos:
        u = 16106   → 16.106  (magnitude SDSS, faixa 10–25)
        petroR50_r = 1988  → 1.988  (raio em arcsec, faixa 0.1–100)
        velDisp = 230421   → 230.421  (dispersão de velocidades, faixa 10–600 km/s)

    Valores sem ponto decimal que não se encaixam em nenhuma coluna (ou cuja
    divisão por 1000 ainda cai fora da faixa) não são alterados — serão
    capturados e removidos pelo filtro de precisão subsequente.

    Args:
        df (pd.DataFrame): DataFrame lido do CSV (colunas já em float).

    Returns:
        pd.DataFrame: Cópia com os valores corrigidos e contagem impressa.
    """
    df = df.copy()
    n_corrected = 0
    for col, (lo, hi) in _DECIMAL_FIX_RANGES.items():
        if col not in df.columns:
            continue
        # Valores sem parte fracionária E fora da faixa esperada
        mask = (df[col] % 1 == 0) & (df[col] > hi)
        corrected = df.loc[mask, col] / 1000.0
        valid = (corrected >= lo) & (corrected <= hi)
        idx = corrected[valid].index
        df.loc[idx, col] = corrected[valid]
        n_corrected += len(idx)
    print(f'fix_missing_decimal: {n_corrected} valor(es) corrigido(s)')
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_tabular_features(df):
    """
    Adiciona features derivadas ao DataFrame do catálogo mesclado.

    Não altera o DataFrame original — opera sobre uma cópia.

    Features criadas:
        u_g, g_r, r_i, i_z  — índices de cor (diferença de magnitudes).
        concentration        — índice de concentração C = petroR90_r / petroR50_r.
                               Galáxias elípticas têm C alto (~4–5);
                               espirais têm C mais baixo (~2–3).

    Args:
        df (pd.DataFrame): DataFrame com colunas u, g, r, i, z,
                           petroR90_r, petroR50_r.

    Returns:
        pd.DataFrame: Cópia do DataFrame com as novas colunas adicionadas.
    """
    df = df.copy()

    # Índices de cor
    df['u_g'] = df['u'] - df['g']
    df['g_r'] = df['g'] - df['r']
    df['r_i'] = df['r'] - df['i']
    df['i_z'] = df['i'] - df['z']

    # Índice de concentração — substituir 0 por NaN para evitar divisão por zero
    df['concentration'] = df['petroR90_r'] / df['petroR50_r'].replace(0, np.nan)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Pré-processamento Tabular Principal
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_tabular_data(df, features=None, scaler=None, scale=True):
    """
    Realiza engenharia de features, limpeza de NaN e opcionalmente escalonamento.

    Fluxo:
        1. Chama engineer_tabular_features() para criar colunas derivadas.
        2. Remove linhas com NaN em qualquer feature ou em morph_class.
        3. Se scale=True, escala com StandardScaler.

    Args:
        df       (pd.DataFrame): DataFrame de merged_catalog.csv.
        features       (list): Colunas a usar. Padrão: TABULAR_FEATURES.
        scaler (StandardScaler): Scaler já ajustado. Se None e scale=True,
                                 ajusta um novo no df recebido.
        scale          (bool): Se False, retorna X bruto sem escalonamento e
                               scaler=None. Use False no notebook de pré-
                               processamento para escalonar depois do split.

    Returns:
        X        (np.ndarray):    Matriz de features (bruta ou escalonada).
        y        (pd.Series):     Rótulos de classe ('morph_class') alinhados.
        scaler   (StandardScaler | None): Scaler ajustado, ou None se scale=False.
        objids   (pd.Series):     objids alinhados com X.
    """
    if features is None:
        features = TABULAR_FEATURES

    df = engineer_tabular_features(df)

    # Remove linhas com NaN em qualquer feature relevante ou sem rótulo
    df_clean = df.dropna(subset=features + ['morph_class']).copy()

    n_dropped = len(df) - len(df_clean)
    print(f"Registros originais : {len(df)}")
    print(f"Removidos (NaN)     : {n_dropped}")
    print(f"Registros finais    : {len(df_clean)}")

    X = df_clean[features].values.astype(np.float32)
    y = df_clean['morph_class'].reset_index(drop=True)
    objids = df_clean['objid'].reset_index(drop=True)

    if not scale:
        return X, y, None, objids

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)

    return X, y, scaler, objids


# ─────────────────────────────────────────────────────────────────────────────
# Persistência
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessed_tabular(X_scaled, y, objids, out_dir, scaler=None):
    """
    Salva os artefatos do pré-processamento tabular em out_dir.

    Arquivos gerados:
        X_tabular.npy  — matriz de features (float32; bruta ou escalonada)
        y_labels.npy   — rótulos morph_class como array de strings
        objids.npy     — objids alinhados com X_tabular
        scaler.pkl     — StandardScaler ajustado (apenas se scaler não for None)

    Args:
        X_scaled (np.ndarray):         Saída de preprocess_tabular_data().
        y        (pd.Series|ndarray):  Rótulos de classe.
        objids   (pd.Series|ndarray):  objids correspondentes.
        out_dir        (str):          Diretório de destino (criado se não existir).
        scaler   (StandardScaler):     Scaler ajustado, ou None (padrão) para não salvar.
                                       Após a correção do pipeline, o scaler é ajustado
                                       no notebook 03 (pós-split) e salvo em models/.
    """
    os.makedirs(out_dir, exist_ok=True)

    y_arr = y.values if hasattr(y, 'values') else y
    ids_arr = objids.values if hasattr(objids, 'values') else objids

    np.save(os.path.join(out_dir, 'X_tabular.npy'), X_scaled.astype(np.float32))
    np.save(os.path.join(out_dir, 'y_labels.npy'),  y_arr)
    np.save(os.path.join(out_dir, 'objids.npy'),    ids_arr)

    if scaler is not None:
        with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    print(f"\nArtefatos salvos em '{out_dir}':")
    print(f"  X_tabular.npy  → shape {X_scaled.shape}, dtype float32")
    print(f"  y_labels.npy   → {len(y_arr)} rótulos")
    print(f"  objids.npy     → {len(ids_arr)} IDs")
    if scaler is not None:
        print(f"  scaler.pkl     → StandardScaler ({X_scaled.shape[1]} features)")
    else:
        print(f"  scaler.pkl     → não salvo (scaler será ajustado no notebook 03)")


def load_preprocessed_tabular(out_dir):
    """
    Carrega os artefatos salvos por save_preprocessed_tabular().

    Args:
        out_dir (str): Diretório onde os artefatos foram salvos.

    Returns:
        X_scaled (np.ndarray):    Matriz de features.
        y        (np.ndarray):    Array de rótulos (strings).
        objids   (np.ndarray):    Array de objids.
        scaler   (StandardScaler): Scaler ajustado.
    """
    X_scaled = np.load(os.path.join(out_dir, 'X_tabular.npy'))
    y        = np.load(os.path.join(out_dir, 'y_labels.npy'),  allow_pickle=True)
    objids   = np.load(os.path.join(out_dir, 'objids.npy'),    allow_pickle=True)

    scaler_path = os.path.join(out_dir, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    print(f"Carregado de '{out_dir}':")
    print(f"  X_tabular : {X_scaled.shape}")
    print(f"  y_labels  : {len(y)} rótulos")
    print(f"  objids    : {len(objids)} IDs")

    return X_scaled, y, objids, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Pré-processamento de Imagens (Seção 5.3)
# ─────────────────────────────────────────────────────────────────────────────

# Parâmetros padrão para o pipeline de imagens
IMAGE_TARGET_SIZE    = (64, 64)   # (height, width) — mesmo tamanho do download
IMAGE_STRETCH_FACTOR = 1000       # divisor do arcsinh; menor = mais contraste


def preprocess_galaxy_image(image_data, target_size=(64, 64), stretch_factor=1000):
    """
    Pré-processa um array de imagem de galáxia (H, W, C) para entrada de CNN.

    Pipeline (por canal independentemente):
        1. Arcsinh stretch: np.arcsinh(canal / stretch_factor)
           Realça feições tênues sem saturar o núcleo brilhante.
           Equivalente à normalização padrão de pipelines astronômicos.
        2. Escalonamento min-max para [0, 1].
        3. Redimensionamento para target_size com cv2.INTER_AREA
           (preserva energia, ideal para downscaling).

    Args:
        image_data    (np.ndarray): Array (H, W, C) float32 — saída de
                                    download_galaxy_image_cutout().
        target_size         (tuple): (height, width) desejado. Padrão: (64, 64).
        stretch_factor      (float): Divisor do arcsinh. Padrão: 1000.

    Returns:
        np.ndarray: Array (H', W', C) float32, valores em [0, 1].
                    None se image_data for None.
    """
    if image_data is None:
        return None

    normalized_channels = []
    for i in range(image_data.shape[-1]):
        channel = image_data[:, :, i]

        # 1. Arcsinh stretch
        stretched = np.arcsinh(channel / stretch_factor)

        # 2. Min-max scaling para [0, 1]
        ch_min = stretched.min()
        ch_max = stretched.max()
        if ch_max - ch_min > 0:
            stretched = (stretched - ch_min) / (ch_max - ch_min)
        else:
            # Canal completamente uniforme (ex.: borda de imagem parcial)
            stretched = np.zeros_like(stretched)

        normalized_channels.append(stretched)

    normalized_image = np.stack(normalized_channels, axis=-1)

    # 3. Redimensionamento — cv2.resize usa (width, height), inverso do numpy
    resized = cv2.resize(
        normalized_image,
        (target_size[1], target_size[0]),
        interpolation=cv2.INTER_AREA,
    )

    return resized.astype(np.float32)


def preprocess_images_batch(catalog_df, images_dir,
                            target_size=(64, 64), stretch_factor=1000):
    """
    Pré-processa em lote as imagens de todas as galáxias do catálogo.

    Lê <objid>.npy de images_dir, aplica preprocess_galaxy_image() e empilha
    em um array 4D. Galáxias sem arquivo de imagem são ignoradas e reportadas.

    Args:
        catalog_df  (pd.DataFrame): Catálogo com coluna 'objid'.
        images_dir        (str): Diretório com arquivos <objid>.npy.
        target_size     (tuple): (height, width) para redimensionamento.
        stretch_factor  (float): Parâmetro de stretch arcsinh.

    Returns:
        X_images   (np.ndarray): Shape (N, H, W, C) float32.
        img_objids (np.ndarray): objids correspondentes às N imagens processadas.
    """
    X_images   = []
    img_objids = []
    missing    = []

    for objid in catalog_df['objid'].values:
        path = os.path.join(images_dir, f"{int(objid)}.npy")

        if not os.path.exists(path):
            missing.append(objid)
            continue

        raw = np.load(path)
        processed = preprocess_galaxy_image(raw, target_size=target_size,
                                            stretch_factor=stretch_factor)
        if processed is not None:
            X_images.append(processed)
            img_objids.append(objid)

    print(f"Imagens processadas      : {len(X_images)}")
    print(f"Imagens não encontradas  : {len(missing)}")
    if missing:
        print(f"  (primeiros 5 ausentes: {missing[:5]})")

    if not X_images:
        return np.array([]), np.array([])

    return np.stack(X_images, axis=0), np.array(img_objids)


def save_preprocessed_images(X_images, img_objids, out_dir):
    """
    Salva o array de imagens pré-processadas em out_dir.

    Arquivos gerados:
        X_images.npy    — array (N, H, W, C) float32
        img_objids.npy  — objids alinhados com X_images

    Args:
        X_images   (np.ndarray): Shape (N, H, W, C).
        img_objids (np.ndarray): objids alinhados.
        out_dir          (str): Diretório de destino (criado se não existir).
    """
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'X_images.npy'),   X_images.astype(np.float32))
    np.save(os.path.join(out_dir, 'img_objids.npy'), img_objids)

    size_mb = X_images.nbytes / 1e6
    print(f"\nImagens salvas em '{out_dir}':")
    print(f"  X_images.npy   → shape {X_images.shape}, dtype float32")
    print(f"  img_objids.npy → {len(img_objids)} IDs")
    print(f"  Tamanho em memória: {size_mb:.1f} MB")


def load_preprocessed_images(out_dir):
    """
    Carrega os artefatos salvos por save_preprocessed_images().

    Args:
        out_dir (str): Diretório onde os artefatos foram salvos.

    Returns:
        X_images   (np.ndarray): Shape (N, H, W, C) float32.
        img_objids (np.ndarray): objids alinhados.
    """
    X_images   = np.load(os.path.join(out_dir, 'X_images.npy'))
    img_objids = np.load(os.path.join(out_dir, 'img_objids.npy'), allow_pickle=True)

    print(f"Carregado de '{out_dir}':")
    print(f"  X_images   : {X_images.shape}")
    print(f"  img_objids : {len(img_objids)} IDs")

    return X_images, img_objids
