import io
import math
import time
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Zone + Démographie (upload INSEE)", layout="wide")

UA = {"User-Agent": "zone-demographie-upload/1.0"}
GEO_COMMUNES_URL = "https://geo.api.gouv.fr/communes"
GEOCODE_URL = "https://data.geopf.fr/geocodage/search/"

# -----------------------------
# Utils
# -----------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def safe_int(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace("\u202f", "").replace(" ", "").replace(",", ".")
        return int(float(x))
    except Exception:
        return np.nan


# -----------------------------
# Geo helpers
# -----------------------------
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
        out.append(
            {
                "label": props.get("label", ""),
                "lat": float(lat),
                "lon": float(lon),
            }
        )
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


# -----------------------------
# Upload parsing helpers
# -----------------------------
def detect_code_insee_column(cols: List[str]) -> Optional[str]:
    # plus robustes + tolérants
    lowered = {c: str(c).lower() for c in cols}

    preferred = [
        "codgeo", "code insee", "code_insee", "insee", "code géographique", "code geographique",
        "insee code géographique", "insee code geographique"
    ]
    for p in preferred:
        for c, cl in lowered.items():
            if cl.strip() == p:
                return c

    # heuristique
    for c, cl in lowered.items():
        if ("geo" in cl or "gé" in cl or "insee" in cl) and "code" in cl:
            return c

    return None


def detect_age_columns(cols: List[str]) -> List[str]:
    # colonnes type "0-4", "5-9", ..., "95+" ou "100+"
    out = []
    for c in cols:
        s = str(c).strip()
        if any(ch.isdigit() for ch in s) and (("-" in s) or ("+" in s)):
            out.append(c)
    return out


@st.cache_data(show_spinner=False)
def read_uploaded_excel_bytes(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict]:
    """
    Lecture Excel "standard".
    (On la garde en cache Streamlit pour éviter de relire si l'utilisateur ne change pas le fichier.)
    """
    t0 = time.time()
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet = xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet)
    diag = {
        "sheet_used": sheet,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "seconds": round(time.time() - t0, 2),
        "columns": df.columns.tolist()[:50],  # aperçu
    }
    return df, diag


def midpoint_from_age_group(age_groupe: str) -> Optional[float]:
    s = str(age_groupe).strip()
    if "+" in s:
        try:
            base = float(s.replace("+", ""))
            return base + 2.5
        except Exception:
            return None
    if "-" in s:
        try:
            a, b = s.split("-")
            return (float(a) + float(b)) / 2.0
        except Exception:
            return None
    return None


def age_buckets_from_row(row: pd.Series, age_cols: List[str]) -> pd.DataFrame:
    """
    Transforme une ligne wide (0-4,5-9,...) en table (age_groupe,total),
    puis en grands groupes 0-14,15-29,...
    """
    tmp = pd.DataFrame({"age_groupe": [str(c) for c in age_cols], "total": [row[c] for c in age_cols]})
    tmp["total"] = tmp["total"].apply(safe_int)
    tmp = tmp.dropna(subset=["total"])

    tmp["mid"] = tmp["age_groupe"].apply(midpoint_from_age_group)

    def bucket(mid):
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

    tmp["bucket"] = tmp["mid"].apply(bucket)
    g = tmp.groupby("bucket", as_index=False)["total"].sum()
    return g


def approx_mean_age_from_row(row: pd.Series, age_cols: List[str]) -> Optional[float]:
    tmp = pd.DataFrame({"age_groupe": [str(c) for c in age_cols], "total": [row[c] for c in age_cols]})
    tmp["total"] = tmp["total"].apply(safe_int)
    tmp["mid"] = tmp["age_groupe"].apply(midpoint_from_age_group)
    tmp = tmp.dropna(subset=["total", "mid"])
    denom = tmp["total"].sum()
    if denom <= 0:
        return None
    return float((tmp["mid"] * tmp["total"]).sum() / denom)


# -----------------------------
# UI
# -----------------------------
st.title("Zone de chalandise (carte + rayon) + INSEE depuis fichier uploadé")

diag_mode = st.checkbox("🧪 Mode diagnostic", value=True)

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) Point + rayon")
    q = st.text_input("Adresse / code postal / ville", value="")
    radius_km = st.slider("Rayon (km)", 1, 80, 15, 1)

    if st.button("🔎 Rechercher"):
        if not q.strip():
            st.warning("Entre une adresse/ville.")
        else:
            try:
                results = geocode_search(q.strip(), limit=8)
                st.session_state["geo_results"] = results
            except Exception as e:
                st.error(f"Géocodage KO : {e}")

    results = st.session_state.get("geo_results", [])
    if not results:
        st.info("➡️ Tape une ville/adresse puis clique Rechercher.")
        st.stop()

    idx = st.selectbox("Choisir le point", list(range(len(results))), format_func=lambda i: results[i]["label"])
    center = results[idx]

    st.caption(f"Centre: {center['label']} (lat={center['lat']:.5f}, lon={center['lon']:.5f})")

    if st.button("📍 Charger communes du rayon"):
        try:
            com = geo_commune_by_latlon(center["lat"], center["lon"])
            if not com:
                raise RuntimeError("Commune introuvable à partir du point.")
            dept = com["codeDepartement"]
            df_communes = geo_communes_dept(dept)

            df_communes["dist_km"] = df_communes.apply(
                lambda r: haversine_km(center["lat"], center["lon"], r["lat"], r["lon"]), axis=1
            )
            in_radius = df_communes[df_communes["dist_km"] <= radius_km].copy()
            in_radius = in_radius.sort_values(["dist_km", "population_geoapi"], ascending=[True, False])
            st.session_state["in_radius"] = in_radius
        except Exception as e:
            st.error(f"Erreur communes : {e}")

    in_radius = st.session_state.get("in_radius", None)
    if in_radius is None:
        st.info("➡️ Clique 'Charger communes du rayon'.")
        st.stop()

    st.write(f"Communes dans {radius_km} km : **{len(in_radius):,}**")

    if "selected_codes" not in st.session_state:
        st.session_state["selected_codes"] = set()

    view = in_radius[["code_insee", "nom", "codes_postaux", "code_dept", "population_geoapi", "dist_km"]].copy()
    view["ajouter"] = view["code_insee"].isin(st.session_state["selected_codes"])

    edited = st.data_editor(
        view,
        hide_index=True,
        width="stretch",
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

    st.info(f"Communes sélectionnées : **{len(st.session_state['selected_codes'])}**")

    st.subheader("2) Upload fichier INSEE âges (XLSX/CSV)")
    age_file = st.file_uploader("Fichier âges (communes)", type=["xlsx", "xls", "csv"])

with right:
    st.subheader("Carte")
    m = folium.Map(location=[center["lat"], center["lon"]], zoom_start=10, control_scale=True)
    folium.Marker([center["lat"], center["lon"]], tooltip="Centre", popup=center["label"]).add_to(m)
    folium.Circle([center["lat"], center["lon"]], radius=radius_km * 1000, fill=False).add_to(m)

    sample = in_radius.head(250)
    for _, r in sample.iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=4, tooltip=f"{r['nom']} ({r['code_insee']})").add_to(m)

    st_folium(m, width="stretch", height=420)

    sel = sorted(list(st.session_state["selected_codes"]))
    if not sel:
        st.warning("Sélectionne au moins une commune.")
        st.stop()

    st.subheader("Analyses âges (depuis upload)")
    if age_file is None:
        st.info("➡️ Upload ton fichier INSEE âges pour continuer.")
        st.stop()

    # Lecture fichier
    try:
        t0 = time.time()
        if age_file.name.lower().endswith(".csv"):
            df_age = pd.read_csv(age_file)
            read_diag = {"format": "csv", "seconds": round(time.time() - t0, 2), "rows": df_age.shape[0], "cols": df_age.shape[1]}
        else:
            file_bytes = age_file.getvalue()
            df_age, read_diag = read_uploaded_excel_bytes(file_bytes)
            read_diag["format"] = "excel"
            read_diag["size_mb"] = round(len(file_bytes) / (1024 * 1024), 2)
    except Exception as e:
        st.error("Impossible de lire le fichier uploadé.")
        st.exception(e)
        st.stop()

    # Détection colonnes
    code_col = detect_code_insee_column(df_age.columns.tolist())
    age_cols = detect_age_columns(df_age.columns.tolist())

    if diag_mode:
        st.markdown("### Diagnostic fichier")
        st.json({
            "filename": age_file.name,
            **read_diag,
            "detected_code_col": code_col,
            "nb_age_columns_detected": len(age_cols),
            "age_columns_sample": [str(c) for c in age_cols[:20]],
        })

    if code_col is None:
        st.error("Je ne trouve pas la colonne code INSEE dans ton fichier. (ex: CODGEO / Code géographique / INSEE...)")
        st.write("Colonnes détectées :")
        st.write(df_age.columns.tolist())
        st.stop()

    if len(age_cols) < 5:
        st.error("Je ne détecte pas les colonnes d'âges (ex: '0-4', '5-9', '95+').")
        st.write("Colonnes détectées :")
        st.write(df_age.columns.tolist())
        st.stop()

    # Filtrage sur communes sélectionnées
    df_age[code_col] = df_age[code_col].astype(str).str.zfill(5)
    df_zone = df_age[df_age[code_col].isin(sel)].copy()

    st.markdown("### Données brutes filtrées (zone)")
    st.caption(f"Lignes trouvées : {len(df_zone)} / communes sélectionnées : {len(sel)}")
    st.dataframe(df_zone.head(200), width="stretch", height=260)

    missing = sorted(list(set(sel) - set(df_zone[code_col].unique().tolist())))
    if missing and diag_mode:
        st.warning(f"Codes INSEE manquants dans le fichier (extrait) : {missing[:25]}")

    # Agrégation zone : structure d'âge + âge moyen approx
    st.markdown("### Indicateurs zone")
    # somme des colonnes âge (toutes communes)
    sums = df_zone[age_cols].applymap(safe_int).sum(axis=0, numeric_only=False)
    row_sums = sums.to_dict()
    # construire une "ligne" virtuelle pour réutiliser fonctions
    pseudo_row = pd.Series(row_sums)

    buckets = age_buckets_from_row(pseudo_row, age_cols)
    mean_age = approx_mean_age_from_row(pseudo_row, age_cols)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Structure par grands groupes**")
        st.dataframe(buckets, width="stretch", hide_index=True)
    with c2:
        st.markdown("**Âge moyen approximatif (zone)**")
        st.metric("Âge moyen approx", f"{mean_age:.1f} ans" if mean_age is not None else "N/A")

    st.download_button(
        "⬇️ Télécharger CSV brut (zone filtrée)",
        data=df_zone.to_csv(index=False).encode("utf-8"),
        file_name="ages_zone_brut.csv",
        mime="text/csv",
    )
