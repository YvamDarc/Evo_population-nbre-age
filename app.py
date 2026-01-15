import io
import csv
import time
import math
import re
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Zone + Démographie (INSEE upload)", layout="wide")

UA = {"User-Agent": "zone-demographie-upload/1.0"}
GEO_COMMUNES_URL = "https://geo.api.gouv.fr/communes"
GEOCODE_URL = "https://data.geopf.fr/geocodage/search/"


# =========================
# Utils
# =========================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def extract_insee_5digits(value: str) -> Optional[str]:
    if value is None:
        return None
    m = re.search(r"(\d{5})", str(value))
    return m.group(1) if m else None


def sort_years(vals: List[str]) -> List[str]:
    def k(v):
        s = str(v)
        return int(s) if s.isdigit() else s
    return sorted([str(v) for v in vals], key=k)


def safe_int_series(s: pd.Series) -> pd.Series:
    # OBS_VALUE parfois float (34.52) => on arrondit
    out = pd.to_numeric(s, errors="coerce").fillna(0.0)
    return out.round().astype(int)


# =========================
# CSV reader
# =========================
@st.cache_data(show_spinner=False)
def read_uploaded_csv_smart(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict]:
    size_mb = round(len(file_bytes) / (1024 * 1024), 2)

    sample = file_bytes[:300_000].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except Exception:
        sep = ";"

    def _try_read(enc: str):
        t0 = time.time()
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=sep,
            engine="python",
            quotechar='"',
            encoding=enc,
            on_bad_lines="skip",
        )
        diag = {
            "format": "csv",
            "size_mb": size_mb,
            "sep_used": sep,
            "encoding_used": enc,
            "seconds": round(time.time() - t0, 2),
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": df.columns.tolist(),
        }
        return df, diag

    try:
        return _try_read("utf-8")
    except Exception:
        return _try_read("latin-1")


# =========================
# Geo helpers
# =========================
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode_search(q: str, limit: int = 8):
    params = {"q": q, "limit": limit}
    r = requests.get(GEOCODE_URL, params=params, timeout=25, headers=UA)
    r.raise_for_status()
    data = r.json()
    feats = data.get("features", [])
    out = []
    for f in feats:
        props = f.get("properties", {}) or {}
        coords = (f.get("geometry") or {}).get("coordinates", None)
        if not coords:
            continue
        lon, lat = coords[0], coords[1]
        out.append({"label": props.get("label", ""), "lat": float(lat), "lon": float(lon)})
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geo_commune_by_latlon(lat: float, lon: float) -> Optional[dict]:
    params = {"lat": lat, "lon": lon, "fields": "nom,code,codeDepartement", "format": "json"}
    r = requests.get(GEO_COMMUNES_URL, params=params, timeout=25, headers=UA)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and data:
        return data[0]
    return None


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=True)
def geo_communes_dept(code_dept: str) -> pd.DataFrame:
    params = {
        "codeDepartement": code_dept,
        "fields": "nom,code,codeDepartement,codesPostaux,population,centre",
        "format": "geojson",
        "geometry": "centre",
    }
    r = requests.get(GEO_COMMUNES_URL, params=params, timeout=60, headers=UA)
    r.raise_for_status()
    geojson = r.json()

    feats = geojson.get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {}) or {}
        coords = (f.get("geometry") or {}).get("coordinates", None)
        if not coords:
            continue
        lon, lat = coords[0], coords[1]
        rows.append(
            {
                "code_insee": str(props.get("code")).zfill(5),
                "nom": props.get("nom"),
                "code_dept": props.get("codeDepartement"),
                "codes_postaux": ", ".join(props.get("codesPostaux") or []),
                "population_geoapi": props.get("population"),
                "lat": float(lat),
                "lon": float(lon),
            }
        )
    return pd.DataFrame(rows)


# =========================
# Age decoding (SDMX-ish)
# =========================
def parse_age_code(age_code: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Gère les cas fréquents INSEE:
    - Y0T4, Y5T9, Y10T14 ...
    - Y_LT15
    - Y_GE80
    - Y15T29 (si présent)
    """
    if age_code is None:
        return None, None
    s = str(age_code).strip()

    # Y0T4, Y15T29 etc
    m = re.fullmatch(r"Y(\d{1,3})T(\d{1,3})", s)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Y_LT15
    m = re.fullmatch(r"Y_LT(\d{1,3})", s)
    if m:
        return 0.0, float(m.group(1)) - 1.0

    # Y_GE80
    m = re.fullmatch(r"Y_GE(\d{1,3})", s)
    if m:
        return float(m.group(1)), None

    return None, None


def midpoint(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is not None and b is not None:
        return (a + b) / 2.0
    if a is not None and b is None:
        return a + 2.5  # approx
    return None


def bucket_from_mid(mid: Optional[float]) -> str:
    if mid is None or (isinstance(mid, float) and np.isnan(mid)):
        return "Inconnu"
    if mid < 15:
        return "0-14"
    if mid < 30:
        return "15-29"
    if mid < 45:
        return "30-44"
    if mid < 60:
        return "45-59"
    if mid < 75:
        return "60-74"
    return "75+"


def detect_total_age_value(age_values: List[str]) -> Optional[str]:
    """
    Cherche une modalité AGE qui représente le TOTAL population.
    Ex: Y_TOT / Y_TOTAL / TOTAL / ALL / _T etc (selon fichiers).
    """
    if not age_values:
        return None
    priorities = [
        "Y_TOT", "Y_TOTAL", "TOTAL", "ALL", "Y_ALL", "Y_T", "_T", "T"
    ]
    sset = set(age_values)
    for p in priorities:
        if p in sset:
            return p

    # Heuristique: contient TOT
    for v in age_values:
        if "TOT" in v.upper() or "TOTAL" in v.upper():
            return v

    return None


def pick_non_overlapping_age_base(age_values: List[str]) -> List[str]:
    """
    Pour calculer des tranches sans recouvrement, on préfère les classes 'YxTy' (5 ans).
    Si pas dispo, on renvoie vide -> l'app forcera l'utilisateur à choisir une stratégie.
    """
    vals = [str(v) for v in age_values]
    yxtys = [v for v in vals if re.fullmatch(r"Y\d{1,3}T\d{1,3}", v)]
    # S'il y en a assez, on considère que c'est la bonne base
    if len(yxtys) >= 10:
        return yxtys
    return []


# =========================
# Calculs fiables
# =========================
def filter_base_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Colonnes essentielles
    needed = [c for c in ["AGE", "GEO", "GEO_OBJECT", "RP_MEASURE", "SEX", "TIME_PERIOD", "OBS_VALUE"] if c in d.columns]
    d = d[needed].copy()

    # Normalisations
    d["GEO"] = d["GEO"].astype(str).str.zfill(5)
    d["TIME_PERIOD"] = d["TIME_PERIOD"].astype(str)

    # Filtrages sûrs (si colonnes présentes)
    if "GEO_OBJECT" in d.columns:
        d = d[d["GEO_OBJECT"].astype(str).str.upper().eq("COM")].copy()
    if "RP_MEASURE" in d.columns:
        d = d[d["RP_MEASURE"].astype(str).str.upper().eq("POP")].copy()

    if "SEX" in d.columns:
        # Sur ton fichier : "_T"
        d = d[d["SEX"].astype(str).isin(["_T", "T", "TOTAL"])].copy()

    d["OBS_VALUE"] = safe_int_series(d["OBS_VALUE"])
    return d


def zone_total_population_by_year(df_zone: pd.DataFrame, total_age_value: Optional[str], age_base_for_sum: List[str]) -> pd.Series:
    """
    Population totale zone par année.
    Priorité:
      1) AGE total si dispo (évite recouvrement)
      2) sinon somme des classes YxTy (non recouvrantes) si dispo
    """
    d = df_zone.copy()

    if total_age_value is not None:
        dd = d[d["AGE"].astype(str) == total_age_value].copy()
        if dd.empty:
            # fallback
            total_age_value = None
        else:
            s = dd.groupby("TIME_PERIOD")["OBS_VALUE"].sum()
            s = s.loc[sorted(s.index, key=lambda x: int(x) if str(x).isdigit() else str(x))]
            return s

    if age_base_for_sum:
        dd = d[d["AGE"].astype(str).isin(age_base_for_sum)].copy()
        s = dd.groupby("TIME_PERIOD")["OBS_VALUE"].sum()
        s = s.loc[sorted(s.index, key=lambda x: int(x) if str(x).isdigit() else str(x))]
        return s

    # dernier recours (dangereux): somme tout (recouvrement possible)
    s = d.groupby("TIME_PERIOD")["OBS_VALUE"].sum()
    s = s.loc[sorted(s.index, key=lambda x: int(x) if str(x).isdigit() else str(x))]
    return s


def zone_age_buckets_by_year(df_zone: pd.DataFrame, age_base: List[str]) -> pd.DataFrame:
    """
    Évolution zone par grandes tranches d'âge.
    On construit à partir des classes non recouvrantes (YxTy) si dispo.
    """
    if not age_base:
        return pd.DataFrame()

    d = df_zone[df_zone["AGE"].astype(str).isin(age_base)].copy()

    # map AGE -> bucket
    ages = d["AGE"].astype(str).unique().tolist()
    map_bucket = {}
    for a in ages:
        amin, amax = parse_age_code(a)
        mid = midpoint(amin, amax)
        map_bucket[a] = bucket_from_mid(mid)

    d["bucket"] = d["AGE"].astype(str).map(map_bucket).fillna("Inconnu")

    g = d.groupby(["TIME_PERIOD", "bucket"])["OBS_VALUE"].sum().reset_index()
    pivot = g.pivot(index="TIME_PERIOD", columns="bucket", values="OBS_VALUE").fillna(0).astype(int)

    # ordre des années
    pivot = pivot.loc[sort_years(pivot.index.tolist())]

    # ordre des buckets
    bucket_order = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+", "Inconnu"]
    cols = [c for c in bucket_order if c in pivot.columns] + [c for c in pivot.columns if c not in bucket_order]
    pivot = pivot[cols]
    return pivot


# =========================
# UI
# =========================
st.title("Population totale de zone (fiable) + Tranches d’âge (fiable) — INSEE upload")

diag_mode = st.checkbox("🧪 Mode diagnostic", value=True)

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) Zone (point + rayon)")
    q = st.text_input("Adresse / code postal / ville", value="")
    radius_km = st.slider("Rayon (km)", 1, 80, 15, 1)

    if st.button("🔎 Rechercher le point"):
        if not q.strip():
            st.warning("Entre une adresse/ville.")
        else:
            st.session_state["geo_results"] = geocode_search(q.strip(), limit=8)

    results = st.session_state.get("geo_results", [])
    if not results:
        st.info("➡️ Tape une ville/adresse puis clique **Rechercher le point**.")
        st.stop()

    idx = st.selectbox("Choisir le point", list(range(len(results))), format_func=lambda i: results[i]["label"])
    center = results[idx]

    st.caption(f"Centre: {center['label']} (lat={center['lat']:.5f}, lon={center['lon']:.5f})")

    if st.button("📍 Charger communes du rayon"):
        com = geo_commune_by_latlon(center["lat"], center["lon"])
        if not com:
            st.error("Commune introuvable à partir du point.")
            st.stop()
        dept = com["codeDepartement"]
        df_communes = geo_communes_dept(dept)

        df_communes["dist_km"] = df_communes.apply(
            lambda r: haversine_km(center["lat"], center["lon"], r["lat"], r["lon"]), axis=1
        )
        in_radius = df_communes[df_communes["dist_km"] <= radius_km].copy()
        in_radius = in_radius.sort_values(["dist_km", "population_geoapi"], ascending=[True, False])
        st.session_state["in_radius"] = in_radius

    in_radius = st.session_state.get("in_radius", None)
    if in_radius is None:
        st.info("➡️ Clique **Charger communes du rayon**.")
        st.stop()

    st.write(f"Communes dans {radius_km} km : **{len(in_radius):,}**")

    if "selected_codes" not in st.session_state:
        st.session_state["selected_codes"] = set()

    view = in_radius[["code_insee", "nom", "codes_postaux", "code_dept", "population_geoapi", "dist_km"]].copy()
    view["ajouter"] = view["code_insee"].isin(st.session_state["selected_codes"])

    edited = st.data_editor(
        view,
        hide_index=True,
        use_container_width=True,
        column_config={
            "ajouter": st.column_config.CheckboxColumn("Ajouter"),
            "dist_km": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
        },
        disabled=["code_insee", "nom", "codes_postaux", "code_dept", "population_geoapi", "dist_km"],
        key="editor_communes",
    )

    st.session_state["selected_codes"] = set(
        edited.loc[edited["ajouter"] == True, "code_insee"].astype(str).tolist()
    )
    sel = sorted(list(st.session_state["selected_codes"]))

    st.info(f"Communes sélectionnées : **{len(sel)}**")

    st.subheader("2) Upload INSEE (CSV)")
    age_file = st.file_uploader("CSV INSEE (AGE;GEO;...;TIME_PERIOD;OBS_VALUE)", type=["csv"])

with right:
    st.subheader("Carte")
    m = folium.Map(location=[center["lat"], center["lon"]], zoom_start=10, control_scale=True)
    folium.Marker([center["lat"], center["lon"]], tooltip="Centre", popup=center["label"]).add_to(m)
    folium.Circle([center["lat"], center["lon"]], radius=radius_km * 1000, fill=False).add_to(m)

    sample = in_radius.head(250)
    for _, r in sample.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=4, tooltip=f"{r['nom']} ({r['code_insee']})").add_to(m)

    st_folium(m, width=900, height=420)

    if not sel:
        st.warning("Sélectionne au moins une commune.")
        st.stop()

    if age_file is None:
        st.info("➡️ Upload ton CSV INSEE pour calculer les graphiques.")
        st.stop()

    # --- Read & filter base dims
    with st.spinner("Lecture + nettoyage CSV (peut être long)..."):
        df_raw, diag = read_uploaded_csv_smart(age_file.getvalue())
        df_all = filter_base_dimensions(df_raw)

    # filter zone communes
    df_zone = df_all[df_all["GEO"].isin(sel)].copy()
    if df_zone.empty:
        st.error("Aucune ligne ne correspond aux communes sélectionnées (vérifie que GEO contient bien le code commune).")
        st.stop()

    if diag_mode:
        st.markdown("### Diagnostic fichier / filtres")
        st.json(
            {
                "filename": age_file.name,
                "sep_used": diag.get("sep_used"),
                "encoding_used": diag.get("encoding_used"),
                "raw_rows": diag.get("rows"),
                "raw_cols": diag.get("cols"),
                "rows_after_base_filters": int(df_all.shape[0]),
                "rows_zone": int(df_zone.shape[0]),
                "unique_years_zone": sort_years(df_zone["TIME_PERIOD"].unique().tolist()),
                "unique_age_values_sample": sorted(df_zone["AGE"].astype(str).unique().tolist())[:25],
                "unique_sex_values": sorted(df_raw["SEX"].astype(str).unique().tolist()) if "SEX" in df_raw.columns else None,
            }
        )

    # --- Choose TOTAL age code (VERY IMPORTANT)
    age_values = sorted(df_zone["AGE"].astype(str).unique().tolist())
    guessed_total_age = detect_total_age_value(age_values)

    st.subheader("Paramètres de calcul (anti-doublons)")
    st.caption(
        "Important : sur certains fichiers, des modalités AGE se recouvrent (ex: Y_LT15, Y_GE80). "
        "Pour la population totale, on doit utiliser la modalité 'TOTAL' si elle existe."
    )

    total_age_choice = st.selectbox(
        "Modalité AGE à utiliser pour la population totale (idéalement TOTAL)",
        options=(["(Auto)"] + age_values),
        index=0,
        help="Si Auto se trompe (ou si tu veux contrôler), choisis une modalité AGE explicitement.",
    )

    # base non-recouvrante pour tranches: classes YxTy
    age_base = pick_non_overlapping_age_base(age_values)

    # résolution total_age_value
    total_age_value = guessed_total_age
    if total_age_choice != "(Auto)":
        total_age_value = total_age_choice

    if diag_mode:
        st.markdown("### Diagnostic choix total / tranches")
        st.write(f"AGE total (auto) détecté : **{guessed_total_age}**")
        st.write(f"AGE total (utilisé) : **{total_age_value}**")
        st.write(f"Nb classes non-recouvrantes détectées (YxTy) : **{len(age_base)}**")

    # =========================
    # 1) Population totale zone par année (fiable)
    # =========================
    st.markdown("## 1) Population totale de la zone par année")

    pop_zone = zone_total_population_by_year(df_zone, total_age_value, age_base)
    pop_zone = pop_zone.astype(int)

    # Affichage
    pop_df = pop_zone.rename("population_zone").reset_index().rename(columns={"TIME_PERIOD": "annee"})
    st.dataframe(pop_df, use_container_width=True, height=260)
    st.line_chart(pop_zone, use_container_width=True)

    st.download_button(
        "⬇️ Télécharger CSV - population zone par année",
        data=pop_df.to_csv(index=False).encode("utf-8"),
        file_name="population_zone_par_annee.csv",
        mime="text/csv",
    )

    # =========================
    # 2) Tranches d’âge (fiable) zone par année
    # =========================
    st.markdown("## 2) Évolution de la population par grandes tranches d’âge (zone)")

    if not age_base:
        st.warning(
            "Je ne trouve pas de classes d'âge non-recouvrantes de type Y0T4/Y5T9/... dans ce fichier.\n"
            "Dans ce cas, les tranches d’âge risquent d’être fausses (recouvrement). "
            "👉 Solution: utilise un fichier INSEE qui contient les classes YxTy, ou prépare un CSV déjà agrégé dans Colab."
        )
        st.stop()

    bucket_pivot = zone_age_buckets_by_year(df_zone, age_base)

    st.dataframe(bucket_pivot.reset_index().rename(columns={"TIME_PERIOD": "annee"}), use_container_width=True, height=260)

    # Visu: aire (souvent plus lisible pour répartition)
    st.area_chart(bucket_pivot, use_container_width=True)

    st.download_button(
        "⬇️ Télécharger CSV - tranches d’âge zone par année",
        data=bucket_pivot.reset_index().rename(columns={"TIME_PERIOD": "annee"}).to_csv(index=False).encode("utf-8"),
        file_name="tranches_age_zone_par_annee.csv",
        mime="text/csv",
    )
