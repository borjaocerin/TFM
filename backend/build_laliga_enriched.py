#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Construye datasets enriquecidos para LaLiga a partir de:
 - Histórico unificado (nivel partido) con xG/posesión/formaciones: laliga_merged_matches.csv
 - Football-Data.co.uk por temporada (odds apertura/cierre, árbitro, asistencia, stats clásicas)
 - (Opcional) ELO snapshots (ClubElo consolidado) ELO_RATINGS.csv
 - (Opcional) Fixtures próximos para generar features pre-partido

Salida:
 - out/laliga_enriched_all.csv
 - out/laliga_enriched_model.csv (subset útil para entrenamiento 1X2)
 - out/fixtures_enriched.csv (si se proporciona --fixtures)

Uso:
  python build_laliga_enriched.py \
    --hist data/historical/laliga_merged_matches.csv \
    --fdata_dir data/football-data/ \
    --elo data/elo/ELO_RATINGS.csv \
    --fixtures data/fixtures/fixtures.csv \
    --outdir out

Notas columnas Football-Data: ver https://www.football-data.co.uk/notes.txt
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# ---------------------- Utilidades ----------------------

def _load_csvs_from_dir(dir_path: Path) -> List[pd.DataFrame]:
    frames = []
    for p in sorted(dir_path.glob('*.csv')):
        try:
            df = pd.read_csv(p)
            df['__srcfile'] = p.name
            frames.append(df)
        except Exception as e:
            print(f"[WARN] No se pudo leer {p}: {e}")
    return frames


def _std_team(s: str, name_map: dict) -> str:
    if pd.isna(s):
        return s
    s2 = name_map.get(s, s)
    return s2


def _parse_date_series(series: pd.Series) -> pd.Series:
    # Football-Data usa dd/mm/yy o dd/mm/yyyy; nuestro histórico va en ISO
    return pd.to_datetime(series, errors='coerce', dayfirst=True).dt.date.astype('string')


# ---------------------- Carga histórico base ----------------------

def load_historical(hist_path: Path, name_map: dict) -> pd.DataFrame:
    df = pd.read_csv(hist_path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Estandarizar equipos
    if 'home_team' in df.columns:
        df['home_team'] = df['home_team'].map(lambda x: _std_team(x, name_map))
    if 'away_team' in df.columns:
        df['away_team'] = df['away_team'].map(lambda x: _std_team(x, name_map))
    # Fecha ISO
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date.astype('string')
    # Resultado consistente
    # Mantener columnas esperadas aunque falten
    expected = ['date','season','home_team','away_team','home_goals','away_goals','result']
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    return df


# ---------------------- Football-Data (odds + stats clásicas) ----------------------

FD_ODDS_BOOKS = [
    # principales (pueden no estar todas)
    'B365','PS','Avg','Max','Pinnacle','BW','IW','LB','WH','VC','SB','GB','BS','BMD','SO','SJ','SY','Bb'
]

FD_RESULT_COLS = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR']
FD_STATS_COLS = ['Referee','Attendance','HS','AS','HST','AST','HC','AC','HY','AY','HR','AR']


def load_football_data(fdata_dir: Path, name_map: dict) -> pd.DataFrame:
    frames = _load_csvs_from_dir(fdata_dir)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    # Normalizar columnas a mayúscula por compatibilidad Football-Data
    cols = {c: c for c in df.columns}
    # Extraer odds H/D/A y sus variantes de cierre con sufijo C (según notes.txt)
    # Ejem: B365H,B365D,B365A  y  B365CH,B365CD,B365CA
    def pick_odds_prefix(prefix: str, close=False):
        suf = 'C' if close else ''
        return [f'{prefix}{suf}H', f'{prefix}{suf}D', f'{prefix}{suf}A']

    keep_cols = set(FD_RESULT_COLS + FD_STATS_COLS)
    for pref in FD_ODDS_BOOKS:
        for close in (False, True):
            h, d, a = pick_odds_prefix(pref, close)
            for c in (h, d, a):
                if c in df.columns:
                    keep_cols.add(c)

    # Filtrar solo columnas conocidas
    df = df[[c for c in df.columns if c in keep_cols]].copy()

    # Renombrar y estandarizar
    df.rename(columns={
        'Date':'date_fd','HomeTeam':'home_team','AwayTeam':'away_team',
        'FTHG':'home_goals_fd','FTAG':'away_goals_fd','FTR':'result_fd',
        'HTHG':'ht_home_goals_fd','HTAG':'ht_away_goals_fd','HTR':'ht_result_fd'
    }, inplace=True)

    # Mapear equipos
    df['home_team'] = df['home_team'].map(lambda x: _std_team(x, name_map))
    df['away_team'] = df['away_team'].map(lambda x: _std_team(x, name_map))

    # Fecha ISO
    df['date'] = _parse_date_series(df['date_fd'])

    # Construir columnas agregadas de mercado promedio/máximo si existen
    def first_available(row, cols: List[str]):
        for c in cols:
            if c in row and pd.notna(row[c]):
                return row[c]
        return np.nan

    # Promedios apertura/cierre
    for typ, close in [('avg', False), ('avg_close', True)]:
        suf = 'C' if close else ''
        cols_h = [f'Avg{suf}H'] if f'Avg{suf}H' in df.columns else []
        cols_d = [f'Avg{suf}D'] if f'Avg{suf}D' in df.columns else []
        cols_a = [f'Avg{suf}A'] if f'Avg{suf}A' in df.columns else []
        if cols_h or cols_d or cols_a:
            df[f'odds_{typ}_h'] = df.apply(lambda r: first_available(r, cols_h), axis=1)
            df[f'odds_{typ}_d'] = df.apply(lambda r: first_available(r, cols_d), axis=1)
            df[f'odds_{typ}_a'] = df.apply(lambda r: first_available(r, cols_a), axis=1)

    # Selección final
    base_keep = ['date','home_team','away_team','Referee','Attendance','HS','AS','HST','AST','HC','AC','HY','AY','HR','AR',
                 'home_goals_fd','away_goals_fd','result_fd','ht_home_goals_fd','ht_away_goals_fd','ht_result_fd']
    odds_keep = [c for c in df.columns if c.startswith('odds_')]
    keep = [c for c in base_keep if c in df.columns] + odds_keep
    return df[keep].drop_duplicates(subset=['date','home_team','away_team'])


# ---------------------- Enriquecimiento ELO ----------------------

def enrich_with_elo(df_matches: pd.DataFrame, elo_path: Path, name_map: dict) -> pd.DataFrame:
    elo = pd.read_csv(elo_path)
    # Esperado: columnas ['Date','Club','Elo', ...] (repo consolidado)
    cols = [c.lower() for c in elo.columns]
    elo.columns = cols
    # Normalizar nombres
    if 'club' in elo.columns:
        elo['club'] = elo['club'].map(lambda x: _std_team(x, name_map))
    # Fecha a ISO
    if 'date' in elo.columns:
        elo['date'] = pd.to_datetime(elo['date'], errors='coerce').dt.date
    else:
        raise ValueError('ELO file must contain a date column')

    # Para cada match, emparejar el snapshot inmediatamente anterior para home/away
    df = df_matches.copy()
    df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')

    def lookup_elo(team: str, d: pd.Timestamp) -> float:
        if pd.isna(team) or pd.isna(d):
            return np.nan
        sub = elo[(elo['club'] == team) & (elo['date'] <= d.date())]
        if sub.empty:
            return np.nan
        return float(sub.sort_values('date').iloc[-1]['elo']) if 'elo' in sub.columns else np.nan

    df['elo_home'] = df.apply(lambda r: lookup_elo(r.get('home_team'), r.get('date_dt')), axis=1)
    df['elo_away'] = df.apply(lambda r: lookup_elo(r.get('away_team'), r.get('date_dt')), axis=1)
    df['elo_diff'] = df['elo_home'] - df['elo_away']
    df = df.drop(columns=['date_dt'])
    return df


# ---------------------- Rolling features para fixtures ----------------------

def _compute_points(hg, ag):
    if pd.isna(hg) or pd.isna(ag):
        return np.nan
    if hg > ag:
        return 3
    if hg < ag:
        return 0
    return 1


def rolling_team_features(hist: pd.DataFrame, windows=(5,10)) -> pd.DataFrame:
    """Devuelve tabla a nivel (team, date) con rolling features hasta t-1."""
    req = ['date','home_team','away_team','home_goals','away_goals','xg_home','xg_away','xga_home','xga_away',
           'poss_home','poss_away','sh_home','sh_away','sot_home','sot_away']
    for c in req:
        if c not in hist.columns:
            hist[c] = np.nan
    # Explode a formato equipo-partido con perspectiva local/visitante
    parts = []
    base = hist.copy()
    base['date_dt'] = pd.to_datetime(base['date'], errors='coerce')

    # Home perspective
    home = pd.DataFrame({
        'team': base['home_team'],
        'date': base['date_dt'],
        'gf': base['home_goals'],
        'ga': base['away_goals'],
        'xg': base['xg_home'],
        'xga': base['xga_home'],
        'poss': base['poss_home'],
        'sh': base['sh_home'],
        'sot': base['sot_home'],
    })
    # Away perspective
    away = pd.DataFrame({
        'team': base['away_team'],
        'date': base['date_dt'],
        'gf': base['away_goals'],
        'ga': base['home_goals'],
        'xg': base['xg_away'],
        'xga': base['xga_away'],
        'poss': base['poss_away'],
        'sh': base['sh_away'],
        'sot': base['sot_away'],
    })
    parts.append(home)
    parts.append(away)
    tall = pd.concat(parts, ignore_index=True)

    tall['points'] = [_compute_points(r.gf, r.ga) for r in tall.itertuples()]
    tall = tall.dropna(subset=['team','date'])
    tall = tall.sort_values(['team','date'])

    out_frames = []
    for w in windows:
        g = tall.groupby('team', group_keys=False).apply(
            lambda df: df.assign(**{
                f'xg_last{w}': df['xg'].rolling(w, min_periods=1).mean().shift(1),
                f'xga_last{w}': df['xga'].rolling(w, min_periods=1).mean().shift(1),
                f'poss_last{w}': df['poss'].rolling(w, min_periods=1).mean().shift(1),
                f'sh_last{w}': df['sh'].rolling(w, min_periods=1).mean().shift(1),
                f'sot_last{w}': df['sot'].rolling(w, min_periods=1).mean().shift(1),
                f'gf_last{w}': df['gf'].rolling(w, min_periods=1).mean().shift(1),
                f'ga_last{w}': df['ga'].rolling(w, min_periods=1).mean().shift(1),
                f'points_last{w}': df['points'].rolling(w, min_periods=1).sum().shift(1),
            })
        )
        out_frames.append(g)
    feats = pd.concat(out_frames, axis=0)
    feats = feats.drop(columns=['gf','ga','xg','xga','poss','sh','sot','points'])
    # Mantener la última fila por (team,date) por si hay duplicados de ventanas
    feats = feats.drop_duplicates(subset=['team','date'], keep='last')
    return feats


def enrich_fixtures(fixtures: pd.DataFrame, hist: pd.DataFrame, elo_path: Path, name_map: dict, windows=(5,10)) -> pd.DataFrame:
    fx = fixtures.copy()
    fx.columns = [c.strip().lower() for c in fx.columns]
    # Esperados: date, home_team, away_team (u otros que mapearemos)
    # Soportar Football-Data fixtures: HomeTeam/AwayTeam, Date, etc.
    if 'date' not in fx.columns and 'Date' in fixtures.columns:
        fx['date'] = _parse_date_series(fixtures['Date'])
    else:
        fx['date'] = pd.to_datetime(fx.get('date', pd.NaT), errors='coerce').dt.date.astype('string')

    # Mapear nombres
    # soportar HomeTeam/AwayTeam capitalizados
    if 'home_team' not in fx.columns and 'HomeTeam' in fixtures.columns:
        fx['home_team'] = fixtures['HomeTeam']
    if 'away_team' not in fx.columns and 'AwayTeam' in fixtures.columns:
        fx['away_team'] = fixtures['AwayTeam']
    fx['home_team'] = fx['home_team'].map(lambda x: _std_team(x, name_map))
    fx['away_team'] = fx['away_team'].map(lambda x: _std_team(x, name_map))

    # Rolling features
    feats = rolling_team_features(hist, windows=windows)

    fx['date_dt'] = pd.to_datetime(fx['date'], errors='coerce')

    # Join por (team, fecha) usando la última observación anterior (merge_asof)
    feats = feats.sort_values('date')
    fx_home = fx[['date_dt','home_team']].rename(columns={'home_team':'team'})
    fx_away = fx[['date_dt','away_team']].rename(columns={'away_team':'team'})

    home_join = pd.merge_asof(
        fx_home.sort_values('date_dt'),
        feats.rename(columns={'date':'feat_date'}).sort_values('feat_date'),
        left_on='date_dt', right_on='feat_date', by='team', direction='backward'
    )
    away_join = pd.merge_asof(
        fx_away.sort_values('date_dt'),
        feats.rename(columns={'date':'feat_date'}).sort_values('feat_date'),
        left_on='date_dt', right_on='feat_date', by='team', direction='backward'
    )

    # Sufijos
    def add_suffix(df: pd.DataFrame, suff: str) -> pd.DataFrame:
        keep = [c for c in df.columns if c not in ('team','date_dt','feat_date')]
        ren = {c: f"{c}_{suff}" for c in keep}
        return df.rename(columns=ren)

    home_feat = add_suffix(home_join, 'home')
    away_feat = add_suffix(away_join, 'away')

    # Combinar en fixtures
    out = fx.copy()
    out = pd.concat([out.reset_index(drop=True),
                     home_feat.filter(regex='_home$').reset_index(drop=True),
                     away_feat.filter(regex='_away$').reset_index(drop=True)], axis=1)

    # ELO
    if elo_path and Path(elo_path).exists():
        out = enrich_with_elo(out, Path(elo_path), name_map)

    # Diferenciales
    for base in ['xg_last5','xga_last5','poss_last5','sh_last5','sot_last5','gf_last5','ga_last5','points_last5',
                 'xg_last10','xga_last10','poss_last10','sh_last10','sot_last10','gf_last10','ga_last10','points_last10']:
        h = f'{base}_home'
        a = f'{base}_away'
        if h in out.columns and a in out.columns:
            out[f'{base}_diff'] = out[h] - out[a]

    # Ordenar columnas
    first = ['date','home_team','away_team']
    model_cols = [c for c in out.columns if any(c.endswith(s) for s in ['_home','_away','_diff'])] + \
                 [c for c in ['elo_home','elo_away','elo_diff'] if c in out.columns]
    out = out[first + model_cols]
    return out


# ---------------------- Orquestador principal ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hist', type=str, required=True, help='Ruta a laliga_merged_matches.csv (histórico unificado)')
    ap.add_argument('--fdata_dir', type=str, required=True, help='Directorio con CSV por temporada de Football-Data (LaLiga)')
    ap.add_argument('--elo', type=str, default='', help='(Opcional) Ruta a ELO_RATINGS.csv')
    ap.add_argument('--fixtures', type=str, default='', help='(Opcional) CSV de fixtures para enriquecer pre-partido')
    ap.add_argument('--team_map', type=str, default='team_name_map_es.json', help='JSON de mapeo de nombres de equipos')
    ap.add_argument('--outdir', type=str, default='out', help='Directorio de salida')
    ap.add_argument('--windows', type=str, default='5,10', help='Ventanas rolling, p.ej. 5,10')

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Cargar mapeo de equipos
    team_map_path = Path(args.team_map)
    if team_map_path.exists():
        with open(team_map_path, 'r', encoding='utf-8') as f:
            name_map = json.load(f)
    else:
        name_map = {}

    # 1) Histórico base
    hist = load_historical(Path(args.hist), name_map)

    # 2) Football-Data odds & stats
    fdata = load_football_data(Path(args.fdata_dir), name_map)

    # 3) Merge
    df = hist.merge(fdata, on=['date','home_team','away_team'], how='left', suffixes=('',''))

    # 4) ELO (opcional)
    if args.elo:
        df = enrich_with_elo(df, Path(args.elo), name_map)

    # 5) Salidas histórico
    #   - completo
    full_out = outdir / 'laliga_enriched_all.csv'
    df.to_csv(full_out, index=False)

    #   - subset para modelado 1X2
    model_cols = [
        'date','season','home_team','away_team','home_goals','away_goals','result',
        'xg_home','xg_away','xga_home','xga_away','poss_home','poss_away','sh_home','sh_away','sot_home','sot_away',
        'odds_avg_h','odds_avg_d','odds_avg_a','odds_avg_close_h','odds_avg_close_d','odds_avg_close_a',
        'elo_home','elo_away','elo_diff'
    ]
    model_keep = [c for c in model_cols if c in df.columns]
    model_out_df = df[model_keep].copy()
    model_out = outdir / 'laliga_enriched_model.csv'
    model_out_df.to_csv(model_out, index=False)

    # 6) Fixtures enriquecidos (opcional)
    if args.fixtures:
        fixtures_df = pd.read_csv(args.fixtures)
        windows = tuple(int(x) for x in args.windows.split(','))
        fx_en = enrich_fixtures(fixtures_df, df, Path(args.elo) if args.elo else '', name_map, windows=windows)
        fx_out = outdir / 'fixtures_enriched.csv'
        fx_en.to_csv(fx_out, index=False)

    print('[OK] He generado los archivos en', outdir.resolve())


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('[ERROR]', e)
        sys.exit(1)
