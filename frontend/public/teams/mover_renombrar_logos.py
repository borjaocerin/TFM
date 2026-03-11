import os
import shutil
import re

# Mapeo manual de nombres de archivo a nombre normalizado esperado por el frontend
TEAM_MAP = {
    "spain_athletic-club.football-logos.cc.svg": "athletic",
    "spain_atletico-madrid.football-logos.cc.svg": "atletico_madrid",
    "spain_barcelona.football-logos.cc.svg": "barcelona",
    "spain_cadiz.football-logos.cc.svg": "cadiz",
    "spain_celta.football-logos.cc.svg": "celta",
    "spain_deportivo.football-logos.cc.svg": "alaves",
    "spain_getafe.football-logos.cc.svg": "getafe",
    "spain_girona.football-logos.cc.svg": "girona",
    "spain_granada.football-logos.cc.svg": "granada",
    "spain_las-palmas.football-logos.cc.svg": "las_palmas",
    "spain_mallorca.football-logos.cc.svg": "mallorca",
    "spain_osasuna.football-logos.cc.svg": "osasuna",
    "spain_rayo-vallecano.football-logos.cc.svg": "rayo_vallecano",
    "spain_real-betis.football-logos.cc.svg": "betis",
    "spain_real-madrid.football-logos.cc.svg": "real_madrid",
    "spain_real-sociedad.football-logos.cc.svg": "real_sociedad",
    "spain_sevilla.football-logos.cc.svg": "sevilla",
    "spain_valencia.football-logos.cc.svg": "valencia",
    "spain_villarreal.football-logos.cc.svg": "villarreal",
    # Si hay otros equipos, añadir aquí
}

SRC = os.path.join(os.path.dirname(__file__), "logos_laliga")
DST = os.path.dirname(__file__)

for src_file, team_name in TEAM_MAP.items():
    src_path = os.path.join(SRC, src_file)
    # El frontend busca .svg y .png, pero estos son .svg
    dst_path = os.path.join(DST, f"{team_name}.svg")
    if os.path.exists(src_path):
        shutil.copyfile(src_path, dst_path)
        print(f"Copiado {src_file} -> {team_name}.svg")
    else:
        print(f"No encontrado: {src_file}")
