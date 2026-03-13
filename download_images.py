"""
download_images.py
  - Lê o catálogo mesclado (SDSS + GZ2)
  - Baixa imagens FITS nas bandas g, r, i para cada galáxia
  - Salva cada imagem como arquivo .npy em data/images/
  - Pula imagens já existentes (evita downloads repetidos)

Pré-requisito:
  - data/processed/merged_catalog.csv  (gerado por load_gz2_and_merge.py)

Uso:
    python download_images.py --max 500

    # Download completo
    python download_images.py
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from data_loader import download_images_batch

BASE_DIR    = os.path.dirname(__file__)
CATALOG_CSV = os.path.join(BASE_DIR, "data", "processed", "merged_catalog.csv")
IMAGES_DIR  = os.path.join(BASE_DIR, "data", "images")

# ─── Argumento opcional --max ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--max", type=int, default=500,
                    help="Número máximo de imagens a baixar (padrão: 500 para teste)")
args = parser.parse_args()

print("=" * 60)
print("GalaxyNet — Download de Imagens FITS")
print("=" * 60)

# ─── Verificar catálogo ───────────────────────────────────────────────────────
if not os.path.exists(CATALOG_CSV):
    print(f"\n[ERRO] Catálogo não encontrado: {CATALOG_CSV}")
    print("Execute primeiro: python load_gz2_and_merge.py")
    sys.exit(1)

catalog_df = pd.read_csv(CATALOG_CSV)
print(f"Catálogo carregado: {len(catalog_df)} galáxias")
print(f"Baixando até {args.max} imagens (bandas g, r, i — 64x64 pixels)\n")

# ─── Download em lote ─────────────────────────────────────────────────────────
successful = download_images_batch(
    catalog_df  = catalog_df,
    images_dir  = IMAGES_DIR,
    size_pixels = 64,
    band_list   = ['g', 'r', 'i'],
    max_images  = args.max
)

print(f"\nImagens salvas em: {IMAGES_DIR}")
print(f"Formato de cada arquivo: (64, 64, 3) — float32, bandas [g, r, i]")
