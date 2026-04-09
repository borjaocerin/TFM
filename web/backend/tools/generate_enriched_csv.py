#!/usr/bin/env python
"""
Script simple para generar el dataset procesado desde el histórico crudo.
"""
import pandas as pd
from pathlib import Path

# Rutas - buscar desde arriba del backend
repo_root = Path(__file__).resolve()
while repo_root.name != "TFM" and repo_root.parent != repo_root:
    repo_root = repo_root.parent

HIST_CSV = repo_root / "data" / "historical" / "laliga_merged_matches.csv"
OUT_CSV = repo_root / "data" / "out" / "laliga_enriched_model.csv"

def main():
    print(f"Leyendo histórico desde: {HIST_CSV}")
    df = pd.read_csv(HIST_CSV)
    print(f"Registros: {len(df)}, Columnas: {len(df.columns)}")
    print(f"Formato detectado: columnas = {df.columns.tolist()[:10]}")
    
    # Normalizar nombres de columna
    df.columns = df.columns.str.lower().str.strip()
    
    # El CSV está en formato team-vs-opponent (a nivel de equipo por partido)
    # Transformar a formato partido único si es necesario
    if 'team' in df.columns and 'opponent' in df.columns:
        print("Convertiendo de formato team-opponent a formato partido único (home-away)...")
        df = df.rename(columns={
            'team': 'home_team',
            'opponent': 'away_team',
            'gf': 'home_goals',
            'ga': 'away_goals',
            'xg': 'xg_home',
            'xga': 'xga_home',
            'poss': 'poss_home',
            'sh': 'sh_home',
            'sot': 'sot_home',
            'fk': 'fk_home',
            'pk': 'pk_home',
            'pkatt': 'pkatt_home',
        })
    
    # Asegurar que están las columnas clave
    required = ['date', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']
    for col in required:
        if col not in df.columns:
            print(f"AVISO: Columna '{col}' no encontrada. Creando vacía.")
            df[col] = None
    
    # Limpiar y ordenar
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date')
    df = df.dropna(subset=['date', 'home_team', 'away_team'])
    
    # Seleccionar columnas relevantes
    output_cols = [c for c in ['date', 'season', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result'] 
                   if c in df.columns]
    df_out = df[output_cols].copy()
    
    # Guardar
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nDataset procesado guardado en: {OUT_CSV}")
    print(f"Registros finales: {len(df_out)}")
    print(f"Columnas finales: {list(df_out.columns)}")
    print(f"\nMuestra de primeros registros:")
    print(df_out.head(3))

if __name__ == "__main__":
    main()
