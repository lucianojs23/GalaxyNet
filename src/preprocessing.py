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

def preprocess_tabular_data(df, features=None, scaler=None):
    """
    Realiza engenharia de features, limpeza de NaN e escalonamento tabular.

    Fluxo:
        1. Chama engineer_tabular_features() para criar colunas derivadas.
        2. Remove linhas com NaN em qualquer feature ou em morph_class.
        3. Escala com StandardScaler (fit no treino, transform no val/teste).

    Args:
        df       (pd.DataFrame): DataFrame de merged_catalog.csv.
        features       (list): Colunas a usar. Padrão: TABULAR_FEATURES.
        scaler (StandardScaler): Scaler já ajustado. Se None, ajusta um novo
                                 no df recebido (use apenas para treino).

    Returns:
        X_scaled (np.ndarray):    Matriz de features escalonada (float64).
        y        (pd.Series):     Rótulos de classe ('morph_class') alinhados.
        scaler   (StandardScaler): Scaler ajustado (reutilizar em val/teste).
        objids   (pd.Series):     objids alinhados com X_scaled (para join
                                  com imagens no modelo Híbrido).
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

    X = df_clean[features].values
    y = df_clean['morph_class'].reset_index(drop=True)
    objids = df_clean['objid'].reset_index(drop=True)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler, objids


# ─────────────────────────────────────────────────────────────────────────────
# Persistência
# ─────────────────────────────────────────────────────────────────────────────

def save_preprocessed_tabular(X_scaled, y, scaler, objids, out_dir):
    """
    Salva os artefatos do pré-processamento tabular em out_dir.

    Arquivos gerados:
        X_tabular.npy  — matriz de features escalonada (float32)
        y_labels.npy   — rótulos morph_class como array de strings
        objids.npy     — objids alinhados com X_tabular
        scaler.pkl     — StandardScaler ajustado

    Args:
        X_scaled (np.ndarray):    Saída de preprocess_tabular_data().
        y        (pd.Series):     Rótulos de classe.
        scaler   (StandardScaler): Scaler ajustado.
        objids   (pd.Series):     objids correspondentes.
        out_dir        (str):     Diretório de destino (criado se não existir).
    """
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'X_tabular.npy'), X_scaled.astype(np.float32))
    np.save(os.path.join(out_dir, 'y_labels.npy'),  y.values)
    np.save(os.path.join(out_dir, 'objids.npy'),    objids.values)

    with open(os.path.join(out_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nArtefatos salvos em '{out_dir}':")
    print(f"  X_tabular.npy  → shape {X_scaled.shape}, dtype float32")
    print(f"  y_labels.npy   → {len(y)} rótulos: {sorted(y.unique())}")
    print(f"  objids.npy     → {len(objids)} IDs")
    print(f"  scaler.pkl     → StandardScaler (mean/scale para {X_scaled.shape[1]} features)")


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

    with open(os.path.join(out_dir, 'scaler.pkl'), 'rb') as f:
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
