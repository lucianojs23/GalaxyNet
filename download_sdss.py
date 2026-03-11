"""
download_sdss.py
Executa o download dos dados fotométricos e espectroscópicos do SDSS DR17


Uso:
    python download_sdss.py

O arquivo CSV resultante é salvo em data/raw/sdss_galaxies.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import download_sdss_galaxy_data, save_sdss_data

# ─── Parâmetros da busca ───────────────────────────────────────────────────────


RA_CENTER   = 180.0   # graus — centro do campo de busca
DEC_CENTER  =   0.0   # graus
RADIUS_DEG  =   2.0   # graus
MAX_RECORDS = 1000    # TOP 1000 para teste inicial (trocar para 5000 na versão final)

print("=" * 60)
print("GalaxyNet — Download de dados do SDSS DR17")
print("=" * 60)
print(f"  RA center  : {RA_CENTER}°")
print(f"  Dec center : {DEC_CENTER}°")
print(f"  Radius     : {RADIUS_DEG}°")
print(f"  Max records: {MAX_RECORDS}")
print("=" * 60)
print("Iniciando query SQL no servidor SDSS...")
print("(pode levar alguns minutos \n")

# ─── Download ─────────────────────────────────────────────────────────────────
df = download_sdss_galaxy_data(
    ra_center   = RA_CENTER,
    dec_center  = DEC_CENTER,
    radius_deg  = RADIUS_DEG,
    max_records = MAX_RECORDS
)

if df.empty:
    print("\nNenhum dado retornado. Verifique a conexão ou os parâmetros.")
    sys.exit(1)

# ─── Inspeção rápida ──────────────────────────────────────────────────────────
print("\n--- Primeiras linhas ---")
print(df.head())

print("\n--- Colunas e tipos ---")
print(df.dtypes)

print("\n--- Valores nulos por coluna ---")
print(df.isnull().sum())

print("\n--- Estatísticas básicas ---")
print(df.describe())

# ─── Salvar em data/raw/ ──────────────────────────────────────────────────────
raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
out_path = save_sdss_data(df, filename="sdss_galaxies.csv", raw_dir=raw_dir)

print(f"\n✓ Download concluído! {len(df)} galáxias salvas em:\n  {out_path}")


