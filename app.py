import io
import math
import time
import csv
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


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace("\u202f", "").replace(" ", "").replace(",", ".")
        return float(x)
    except Exception:
        return np.nan


def safe_int(x):
    v = safe_float(x)
    if pd.isna(v):
        return np.nan
    return int(round(v))


def extract_insee_5digits(value: str) -> Optional[str]:
    """
    Extrait un code INSEE 5 chiffres depuis une valeur (ex: '22050', 'COM-22050', 'FR22050', etc.)
    """
    if value is None:
        return None
    s = str(value)
    # on cherche 5 chiffres consécutifs
    import re
    m = re.search(r"(\d{5})", s)
    if m:
        return m.group(1)
    return None


# -----------------------------
# Robust CSV reader (handles ; and quotes)
# -----------------------------
@st.cache_data(show_spinner=False)
def read_uploaded_csv_smart(file_bytes: bytes) -> Tuple[pd.DataFrame, Dict]:
    size_mb = round(len(file_bytes) / (1024 * 1024), 2)

    sample = file_bytes[:300_000].decode("utf-8", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except Exception:
        sep = ";"  # fallback FR

    t0 = time.time()
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        sep=sep,
        engine="python",
        quotechar='"',
        encoding="utf-8",
        on_bad_lines="skip",
        low_memory=False,
    )
    diag = {
        "format": "csv",
        "size_mb": size_mb,
        "sep_used": sep,
        "seconds": round(time.time() - t0, 2),
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
    }
    return df, diag


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


# -----------------------------
# Concordance handling
# -----------------------------
def guess_concordance_columns(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Devine une colonne "clé GEO" et une colonne "code INSEE".
    On reste permissif : l'utilisateur peut aussi choisir manuellement via selectbox.
    """
    lower = {c: str(c).lower() for c in cols}

    insee_candidates = []
    geo_candidates = []

    for c, cl in lower.items():
        if "insee" in cl and "code" in cl:
            insee_candidates.append(c)
        if "codgeo" in cl or ("code" in cl and "geo" in cl):
            insee_candidates.append(c)
        if cl.strip() in {"insee", "code_insee"}:
            insee_candidates.append(c)

        if cl.strip() in {"geo", "id_geo", "geo_code", "code_geo"}:
            geo_candidates.append(c)
        if "geo" in cl and "object" not in cl and "code" not in cl:
            geo_candidates.append(c)

    insee_col = insee_candidates[0] if insee_candidates else None
    geo_col = geo_candidates[0] if geo_candidates else None
    return geo_col, insee_col


def build_geo_to_insee_map(df_conc: pd.DataFrame, geo_col: str, insee_col: str) -> Dict[str, str]:
    tmp = df_conc[[geo_col, insee_col]].copy()
    tmp[geo_col] = tmp[geo_col].astype(str)
    tmp[insee_col] = tmp[insee_col].astype(str).str.extract(r"(\d{5})")[0]
    tmp = tmp.dropna()
    return dict(zip(tmp[geo_col].tolist(), tmp[insee_col].tolist()))


# -----------------------------
# INSEE age SDMX-like processing
# -----------------------------
def ensure_required_cols(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Colonnes manquantes dans le fichier âges: {missing}")


def normalize_codes_from_age_df(
    df_age: pd.DataFrame,
    geo_to_insee: Optional[Dict[str, str]] = None,
    geo_col: str = "GEO",
) -> pd.DataFrame:
    """
    Ajoute une colonne CODE_INSEE (5 chiffres).
    Priorité:
    1) concordance geo_to_insee si fournie
    2) extraction regex 5 chiffres depuis GEO
    """
    out = df_age.copy()
    out[geo_col] = out[geo_col].astype(str)

    if geo_to_insee:
        out["CODE_INSEE"] = out[geo_col].map(geo_to_insee)
    else:
        out["CODE_INSEE"] = out[geo_col].apply(extract_insee_5digits)

    return out


def parse_age_value(age_val: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Retourne (age_min, age_max) si détectable.
    Gère quelques formats:
      - "0-4" / "95+"
      - "0" (age exact)
      - codes type "Y0T4" / "Y95T99" / "Y_GE95" (heuristiques)
    Si non détectable => (None, None)
    """
    if age_val is None:
        return None, None
    s = str(age_val).strip()

    # 0-4
    if "-" in s:
        parts = s.split("-")
        try:
            a = float(parts[0])
            b = float(parts[1])
            return a, b
        except Exception:
            pass

    # 95+
    if s.endswith("+"):
        try:
            a = float(s.replace("+", ""))
            return a, None
        except Exception:
            pass

    # âge exact
    if s.isdigit():
        a = float(s)
        return a, a

    # heuristiques type SDMX
    # Ex: Y0T4, Y5T9, Y95T99, Y_GE95, Y_LT5...
    import re

    m = re.search(r"(\d{1,3})\s*T\s*(\d{1,3})", s.replace(" ", ""))
    if m:
        a = float(m.group(1))
        b = float(m.group(2))
        return a, b

    m = re.search(r"GE(\d{1,3})", s)
    if m:
        a = float(m.group(1))
        return a, None

    m = re.search(r"LT(\d{1,3})", s)
    if m:
        b = float(m.group(1)) - 1
        return 0.0, b

    # dernier essai: extraire 1 ou 2 nombres
    nums = re.findall(r"\d{1,3}", s)
    if len(nums) == 1:
        a = float(nums[0])
        return a, a
    if len(nums) >= 2:
        a = float(nums[0])
        b = float(nums[1])
        if a <= b:
            return a, b

    return None, None


def age_midpoint(age_min: Optional[float], age_max: Optional[float]) -> Optional[float]:
    if age_min is None and age_max is None:
        return None
    if age_min is not None and age_max is not None:
        return (age_min + age_max) / 2.0
    if age_min is not None and age_max is None:
        # 95+ => approx +2.5
        return age_min + 2.5
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


def compute_zone_age_indicators(
    df_age: pd.DataFrame,
    selected_insee: List[str],
    year_value: str,
    sex_mode: str,
) -> Tuple[pd.DataFrame, Optional[float], pd.DataFrame]:
    """
    Retourne:
      - buckets (table tranche -> effectif)
      - age_mean_approx
      - brut_agrege (age -> effectif)
    Hypothèses:
      - colonnes: AGE, CODE_INSEE, TIME_PERIOD, SEX, OBS_VALUE
    """
    required = ["AGE", "CODE_INSEE", "TIME_PERIOD", "OBS_VALUE"]
    ensure_required_cols(df_age, required)

    d = df_age.copy()
    d["CODE_INSEE"] = d["CODE_INSEE"].astype(str)
    d = d[d["CODE_INSEE"].isin(selected_insee)]

    d["TIME_PERIOD"] = d["TIME_PERIOD"].astype(str)
    d = d[d["TIME_PERIOD"] == str(year_value)]

    d["OBS_VALUE"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce").fillna(0.0)

    # SEX handling (optionnel)
    if "SEX" in d.columns and sex_mode != "Somme (tous sexes présents)":
        d["SEX"] = d["SEX"].astype(str)
        if sex_mode == "Total seulement (T si présent)":
            if (d["SEX"] == "T").any():
                d = d[d["SEX"] == "T"]
            else:
                # fallback: somme si pas de T
                pass
        elif sex_mode == "Hommes seulement (M si présent)":
            if (d["SEX"] == "M").any():
                d = d[d["SEX"] == "M"]
        elif sex_mode == "Femmes seulement (F si présent)":
            if (d["SEX"] == "F").any():
                d = d[d["SEX"] == "F"]

    # Agrégation par AGE
    age_tot = d.groupby("AGE", as_index=False)["OBS_VALUE"].sum().rename(columns={"OBS_VALUE": "effectif"})
    age_tot["effectif"] = age_tot["effectif"].apply(safe_int).fillna(0).astype(int)

    # Parse âges -> midpoints
    mins, maxs, mids, buckets = [], [], [], []
    for a in age_tot["AGE"].tolist():
        amin, amax = parse_age_value(a)
        mid = age_midpoint(amin, amax)
        mins.append(amin)
        maxs.append(amax)
        mids.append(mid)
        buckets.append(bucket_from_mid(mid))

    age_tot["age_min"] = mins
    age_tot["age_max"] = maxs
    age_tot["age_mid"] = mids
    age_tot["bucket"] = buckets

    # Buckets
    bucket_df = age_tot.groupby("bucket", as_index=False)["effectif"].sum().sort_values("bucket")

    # Age mean approx
    valid = age_tot.dropna(subset=["age_mid"]).copy()
    denom = valid["effectif"].sum()
    if denom <= 0:
        mean_age = None
    else:
        mean_age = float((valid["age_mid"] * valid["effectif"]).sum() / denom)

    return bucket_df, mean_age, age_tot


# -----------------------------
# UI
# -----------------------------
st.title("Zone de chalandise (carte + rayon) + Démographie depuis fichiers uploadés")

diag_mode = st.checkbox("🧪 Mode diagnostic", value=True)

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) Point + rayon")
    q = st.text_input("Adresse / code postal / ville", value="")
    radius_km = st.slider("Rayon (km)", 1, 80, 15, 1)

    if st.button("🔎 Rechercher le point"):
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
        st.info("➡️ Tape une ville/adresse puis clique **Rechercher le point**.")
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

    st.subheader("2) Upload fichiers INSEE")
    age_file = st.file_uploader("Fichier âges (CSV INSEE : AGE;GEO;...;OBS_VALUE)", type=["csv"])
    conc_file = st.file_uploader("Fichier de concordance (optionnel, CSV)", type=["csv"])

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

    st.subheader("Analyses âges (INSEE upload)")
    if age_file is None:
        st.info("➡️ Upload ton fichier âges CSV pour continuer.")
        st.stop()

    # Read age CSV
    try:
        age_bytes = age_file.getvalue()
        df_age, age_diag = read_uploaded_csv_smart(age_bytes)
    except Exception as e:
        st.error("Impossible de lire le CSV âges.")
        st.exception(e)
        st.stop()

    # Basic required columns in this INSEE format
    # (Based on your header: AGE;GEO;GEO_OBJECT;RP_MEASURE;SEX;TIME_PERIOD;OBS_VALUE)
    # We will require at least AGE, GEO, TIME_PERIOD, OBS_VALUE.
    if diag_mode:
        st.markdown("### Diagnostic âges (lecture)")
        st.json(
            {
                "filename": age_file.name,
                **{k: v for k, v in age_diag.items() if k != "columns"},
                "columns_count": len(age_diag["columns"]),
                "columns": age_diag["columns"][:60],
            }
        )

    # Concordance (optional)
    geo_to_insee = None
    geo_col_conc = None
    insee_col_conc = None

    if conc_file is not None:
        try:
            conc_bytes = conc_file.getvalue()
            df_conc, conc_diag = read_uploaded_csv_smart(conc_bytes)
            if diag_mode:
                st.markdown("### Diagnostic concordance (lecture)")
                st.json(
                    {
                        "filename": conc_file.name,
                        **{k: v for k, v in conc_diag.items() if k != "columns"},
                        "columns_count": len(conc_diag["columns"]),
                        "columns": conc_diag["columns"][:60],
                    }
                )

            # let user choose columns
            geo_guess, insee_guess = guess_concordance_columns(df_conc.columns.tolist())
            geo_col_conc = st.selectbox("Concordance: colonne GEO (clé)", df_conc.columns.tolist(), index=(df_conc.columns.tolist().index(geo_guess) if geo_guess in df_conc.columns else 0))
            insee_col_conc = st.selectbox("Concordance: colonne code INSEE", df_conc.columns.tolist(), index=(df_conc.columns.tolist().index(insee_guess) if insee_guess in df_conc.columns else 0))

            if st.button("Construire la table de concordance"):
                geo_to_insee = build_geo_to_insee_map(df_conc, geo_col_conc, insee_col_conc)
                st.session_state["geo_to_insee"] = geo_to_insee
        except Exception as e:
            st.warning("Concordance uploadée mais non exploitable (on continue sans).")
            if diag_mode:
                st.exception(e)

    if "geo_to_insee" in st.session_state:
        geo_to_insee = st.session_state["geo_to_insee"]

    # Normalize to CODE_INSEE
    # Identify GEO column in age file
    geo_col_age = "GEO" if "GEO" in df_age.columns else None
    if geo_col_age is None:
        # fallback: find first column that looks like GEO
        for c in df_age.columns:
            if str(c).strip().upper() == "GEO":
                geo_col_age = c
                break
    if geo_col_age is None:
        st.error("Je ne trouve pas la colonne `GEO` dans le fichier âges.")
        st.stop()

    df_age2 = normalize_codes_from_age_df(df_age, geo_to_insee=geo_to_insee, geo_col=geo_col_age)

    # Diagnostics for mapping success
    mapped_ok = df_age2["CODE_INSEE"].notna().mean()
    if diag_mode:
        st.markdown("### Diagnostic mapping GEO → CODE_INSEE")
        st.write(f"Taux de lignes avec CODE_INSEE détecté: **{mapped_ok*100:.1f}%**")
        st.write("Exemples GEO → CODE_INSEE (10):")
        ex = df_age2[[geo_col_age, "CODE_INSEE"]].dropna().head(10)
        st.dataframe(ex, width="stretch", height=230)

    if mapped_ok < 0.5:
        st.warning(
            "Le mapping GEO → code INSEE marche mal (<50%). "
            "Dans ce cas, le fichier de concordance est probablement nécessaire (ou la colonne GEO n'est pas un code commune)."
        )

    # Choose year
    if "TIME_PERIOD" not in df_age2.columns:
        st.error("Colonne TIME_PERIOD absente du fichier âges.")
        st.stop()

    years = sorted(df_age2["TIME_PERIOD"].dropna().astype(str).unique().tolist())
    if not years:
        st.error("Aucune valeur TIME_PERIOD exploitable.")
        st.stop()

    year_choice = st.selectbox("Année / TIME_PERIOD", options=years, index=len(years)-1)

    sex_mode = st.selectbox(
        "Sexe (si colonne SEX présente)",
        options=[
            "Somme (tous sexes présents)",
            "Total seulement (T si présent)",
            "Hommes seulement (M si présent)",
            "Femmes seulement (F si présent)",
        ],
        index=0,
    )

    # Compute
    try:
        with st.spinner("Calcul des indicateurs zone (filtrage + agrégation)..."):
            buckets, mean_age, age_tot = compute_zone_age_indicators(
                df_age2, selected_insee=sel, year_value=year_choice, sex_mode=sex_mode
            )
    except Exception as e:
        st.error("Erreur pendant le calcul (format AGE/SEX/TIME_PERIOD inattendu ?).")
        st.exception(e)
        st.stop()

    # Display results
    st.markdown("### Structure d’âge (zone) – grands groupes")
    st.dataframe(buckets, width="stretch", hide_index=True)

    st.markdown("### Âge moyen approximatif (zone)")
    st.metric("Âge moyen approx", f"{mean_age:.1f} ans" if mean_age is not None else "N/A")

    st.markdown("### Détail brut agrégé (zone) – par modalité AGE")
    st.dataframe(age_tot.sort_values("effectif", ascending=False).head(200), width="stretch", height=340)

    st.download_button(
        "⬇️ Télécharger CSV agrégé (AGE → effectif, zone)",
        data=age_tot.to_csv(index=False).encode("utf-8"),
        file_name=f"age_agrege_zone_{year_choice}.csv",
        mime="text/csv",
    )
