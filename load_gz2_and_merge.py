"""
load_gz2_and_merge.py
Passo 2 do projeto GalaxyNet:
  - Carrega o Galaxy Zoo 2 (gz2_hart16.csv)
  - Atribui classes morfológicas (Listing 2 do PDF)
  - Faz o merge com os dados do SDSS já baixados
  - Salva o catálogo mesclado em data/processed/merged_catalog.csv

Pré-requisito:
  - data/raw/sdss_galaxies.csv  (gerado por download_sdss.py)
  - data/raw/gz2_hart16.csv     (baixado de https://zenodo.org/record/3565489)

Uso:
    python load_gz2_and_merge.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pandas as pd
from data_loader import load_and_filter_gz2, merge_sdss_gz2

BASE_DIR      = os.path.dirname(__file__)
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

SDSS_CSV = os.path.join(RAW_DIR, "sdss_galaxies.csv")
GZ2_CSV  = os.path.join(RAW_DIR, "gz2_hart16.csv")
OUT_CSV  = os.path.join(PROCESSED_DIR, "merged_catalog.csv")

print("=" * 60)
print("GalaxyNet — Carregamento GZ2 e Merge com SDSS")
print("=" * 60)

# ─── Verificar arquivos de entrada ────────────────────────────────────────────
if not os.path.exists(SDSS_CSV):
    print(f"\n[ERRO] Arquivo não encontrado: {SDSS_CSV}")
    print("Execute primeiro: python download_sdss.py")
    sys.exit(1)

if not os.path.exists(GZ2_CSV):
    print(f"\n[ERRO] Arquivo não encontrado: {GZ2_CSV}")
    print("Baixe o gz2_hart16.csv em: https://zenodo.org/record/3565489")
    print(f"E salve em: {RAW_DIR}/")
    sys.exit(1)

# ─── Carregar SDSS ────────────────────────────────────────────────────────────
print(f"\n[1/3] Carregando dados do SDSS...")
sdss_df = pd.read_csv(SDSS_CSV)
print(f"      {len(sdss_df)} galáxias carregadas de {SDSS_CSV}")

# ─── Carregar e filtrar GZ2 (Listing 2 do PDF) ────────────────────────────────
print(f"\n[2/3] Carregando Galaxy Zoo 2...")
gz2_df = load_and_filter_gz2(GZ2_CSV)

# ─── Merge + atribuição de classes ────────────────────────────────────────────
print(f"\n[3/3] Fazendo merge SDSS × GZ2 e atribuindo classes morfológicas...")
merged_df = merge_sdss_gz2(sdss_df, gz2_df)

# ─── Salvar catálogo mesclado ─────────────────────────────────────────────────
os.makedirs(PROCESSED_DIR, exist_ok=True)
merged_df.to_csv(OUT_CSV, index=False)

print(f"\n✓ Catálogo mesclado salvo em:\n  {OUT_CSV}")
print(f"  Shape: {merged_df.shape}")
print(f"\nColunas disponíveis:")
print(f"  {list(merged_df.columns)}")

print("\nPróximo passo:")
print("  git add data/processed/merged_catalog.csv")
print("  git commit -m 'data: merge SDSS + GZ2 concluído'")
