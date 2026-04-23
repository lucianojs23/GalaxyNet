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
# Modo all-sky: RA_CENTER, DEC_CENTER e RADIUS_DEG = None → busca em todo o footprint SDSS
# Para busca por cone: definir RA_CENTER, DEC_CENTER (graus) e RADIUS_DEG (graus)

RA_CENTER   = None    # None = busca em todo o céu SDSS
DEC_CENTER  = None    # None = busca em todo o céu SDSS
RADIUS_DEG  = None    # None = sem restrição espacial
MAX_RECORDS = 50000   # Número máximo de galáxias a baixar

print("=" * 60)
print("GalaxyNet — Download de dados do SDSS DR17")
print("=" * 60)
if RA_CENTER is not None:
    print(f"  Modo       : Cone")
    print(f"  RA center  : {RA_CENTER}°")
    print(f"  Dec center : {DEC_CENTER}°")
    print(f"  Radius     : {RADIUS_DEG}°")
else:
    print(f"  Modo       : All-sky (todo o footprint SDSS)")
print(f"  Max records: {MAX_RECORDS}")
print("=" * 60)
print("Iniciando query SQL no servidor SDSS...")
print("(pode levar alguns minutos)\n")

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


