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

st.set_page_config(page_title="Zone + Démographie (upload INSEE)", layout="wide")

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
    if value is None:
        return None
    m = re.search(r"(\d{5})", str(value))
    return m.group(1) if m else None


def sort_years(vals: List[str]) -> List[str]:
    def k(v):
        s = str(v)
        return int(s) if s.isdigit() else s
    return sorted([str(v) for v in vals], key=k)


# =========================
# Robust CSV reader (python engine, no low_memory)
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
# Concordance helpers
# =========================
def guess_concordance_columns(cols: List[str]) -> Tuple[Optional[str], Optional[str]]:
    lower = {c: str(c).lower() for c in cols}
    geo_candidates, insee_candidates = [], []
    for c, cl in lower.items():
        if cl.strip() in {"geo", "id_geo", "geo_code", "code_geo"}:
            geo_candidates.append(c)
        if cl.strip() in {"codgeo", "code_insee", "insee"}:
            insee_candidates.append(c)
        if ("insee" in cl and "code" in cl) or ("code" in cl and ("geo" in cl or "géo" in cl)):
            insee_candidates.append(c)
    return (geo_candidates[0] if geo_candidates else None, insee_candidates[0] if insee_candidates else None)


def build_geo_to_insee_map(df_conc: pd.DataFrame, geo_col: str, insee_col: str) -> Dict[str, str]:
    tmp = df_conc[[geo_col, insee_col]].copy()
    tmp[geo_col] = tmp[geo_col].astype(str)
    tmp[insee_col] = tmp[insee_col].astype(str).apply(extract_insee_5digits)
    tmp = tmp.dropna()
    return dict(zip(tmp[geo_col].tolist(), tmp[insee_col].tolist()))


# =========================
# AGE -> bucket
# =========================
def parse_age_value(age_val: str) -> Tuple[Optional[float], Optional[float]]:
    if age_val is None:
        return None, None
    s = str(age_val).strip()

    if "-" in s:
        try:
            a, b = s.split("-")
            return float(a), float(b)
        except Exception:
            pass

    if s.endswith("+"):
        try:
            return float(s[:-1]), None
        except Exception:
            pass

    if s.isdigit():
        a = float(s)
        return a, a

    m = re.search(r"(\d{1,3})\s*T\s*(\d{1,3})", s.replace(" ", ""))
    if m:
        return float(m.group(1)), float(m.group(2))

    m = re.search(r"GE(\d{1,3})", s)
    if m:
        return float(m.group(1)), None

    nums = re.findall(r"\d{1,3}", s)
    if len(nums) == 1:
        a = float(nums[0])
        return a, a
    if len(nums) >= 2:
        a, b = float(nums[0]), float(nums[1])
        if a <= b:
            return a, b

    return None, None


def age_midpoint(age_min: Optional[float], age_max: Optional[float]) -> Optional[float]:
    if age_min is None and age_max is None:
        return None
    if age_min is not None and age_max is not None:
        return (age_min + age_max) / 2.0
    if age_min is not None and age_max is None:
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


@st.cache_data(show_spinner=False)
def build_age_maps(age_values: List[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
    bucket_map = {}
    mid_map = {}
    for a in age_values:
        amin, amax = parse_age_value(a)
        mid = age_midpoint(amin, amax)
        bucket_map[a] = bucket_from_mid(mid)
        if mid is not None and not (isinstance(mid, float) and np.isnan(mid)):
            mid_map[a] = float(mid)
    return bucket_map, mid_map


def normalize_codes(df_age: pd.DataFrame, geo_to_insee: Optional[Dict[str, str]]) -> pd.DataFrame:
    out = df_age.copy()
    out["GEO"] = out["GEO"].astype(str)
    if geo_to_insee:
        out["CODE_INSEE"] = out["GEO"].map(geo_to_insee)
    else:
        out["CODE_INSEE"] = out["GEO"].apply(extract_insee_5digits)
    return out


def apply_sex_filter(df: pd.DataFrame, sex_mode: str) -> pd.DataFrame:
    if "SEX" not in df.columns:
        return df
    d = df.copy()
    d["SEX"] = d["SEX"].astype(str)

    if sex_mode == "Total seulement (T si présent)":
        if (d["SEX"] == "T").any():
            return d[d["SEX"] == "T"]
        return d
    if sex_mode == "Hommes seulement (M si présent)":
        if (d["SEX"] == "M").any():
            return d[d["SEX"] == "M"]
        return d
    if sex_mode == "Femmes seulement (F si présent)":
        if (d["SEX"] == "F").any():
            return d[d["SEX"] == "F"]
        return d
    return d  # Somme


# =========================
# Timeseries
# =========================
def pop_timeseries_by_commune(df_zone: pd.DataFrame) -> pd.DataFrame:
    d = df_zone.copy()
    d["TIME_PERIOD"] = d["TIME_PERIOD"].astype(str)
    d["OBS_VALUE"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce").fillna(0.0)

    pop = (
        d.groupby(["TIME_PERIOD", "CODE_INSEE"], as_index=False)["OBS_VALUE"]
        .sum()
        .rename(columns={"OBS_VALUE": "population"})
    )
    pop["population"] = pop["population"].apply(safe_int).fillna(0).astype(int)
    return pop


def bucket_timeseries_zone(df_zone: pd.DataFrame, bucket_map: Dict[str, str]) -> pd.DataFrame:
    d = df_zone.copy()
    d["TIME_PERIOD"] = d["TIME_PERIOD"].astype(str)
    d["OBS_VALUE"] = pd.to_numeric(d["OBS_VALUE"], errors="coerce").fillna(0.0)
    d["bucket"] = d["AGE"].astype(str).map(bucket_map).fillna("Inconnu")

    g = (
        d.groupby(["TIME_PERIOD", "bucket"], as_index=False)["OBS_VALUE"]
        .sum()
        .rename(columns={"OBS_VALUE": "effectif"})
    )
    g["effectif"] = g["effectif"].apply(safe_int).fillna(0).astype(int)

    pivot = g.pivot(index="TIME_PERIOD", columns="bucket", values="effectif").fillna(0).astype(int)
    pivot = pivot.loc[sort_years(pivot.index.tolist())]
    return pivot


# =========================
# APP UI (with in-app traceback)
# =========================
try:
    st.title("Zone de chalandise + évolutions population & tranches d’âge (INSEE upload)")

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
                raise RuntimeError("Commune introuvable à partir du point.")
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
        st.info(f"Communes sélectionnées : **{len(st.session_state['selected_codes'])}**")

        st.subheader("2) Upload fichiers INSEE")
        age_file = st.file_uploader("Fichier âges (CSV INSEE : AGE;GEO;TIME_PERIOD;OBS_VALUE...)", type=["csv"])
        conc_file = st.file_uploader("Fichier de concordance (optionnel, CSV)", type=["csv"])

    with right:
        st.subheader("Carte")
        m = folium.Map(location=[center["lat"], center["lon"]], zoom_start=10, control_scale=True)
        folium.Marker([center["lat"], center["lon"]], tooltip="Centre", popup=center["label"]).add_to(m)
        folium.Circle([center["lat"], center["lon"]], radius=radius_km * 1000, fill=False).add_to(m)

        sample = in_radius.head(250)
        for _, r in sample.iterrows():
            folium.CircleMarker([r["lat"], r["lon"]], radius=4, tooltip=f"{r['nom']} ({r['code_insee']})").add_to(m)

        # ✅ width MUST be int here
        st_folium(m, width=900, height=420)

        sel = sorted(list(st.session_state["selected_codes"]))
        if not sel:
            st.warning("Sélectionne au moins une commune.")
            st.stop()

        if age_file is None:
            st.info("➡️ Upload ton fichier âges CSV pour continuer.")
            st.stop()

        st.subheader("Paramètres graphiques")
        top_n = st.slider("Nombre max de communes affichées (Graph 1)", 3, 60, 15, 1)
        sex_mode = st.selectbox(
            "Sexe (si colonne SEX présente)",
            options=[
                "Total seulement (T si présent)",
                "Somme (tous sexes présents)",
                "Hommes seulement (M si présent)",
                "Femmes seulement (F si présent)",
            ],
            index=0,
        )

        with st.spinner("Lecture CSV âges..."):
            df_raw, age_diag = read_uploaded_csv_smart(age_file.getvalue())

        needed = [c for c in ["AGE", "GEO", "TIME_PERIOD", "SEX", "OBS_VALUE"] if c in df_raw.columns]
        df = df_raw[needed].copy()
        del df_raw

        if not {"AGE", "GEO", "TIME_PERIOD", "OBS_VALUE"}.issubset(set(df.columns)):
            raise RuntimeError(f"Colonnes nécessaires absentes. Colonnes trouvées: {df.columns.tolist()}")

        if diag_mode:
            st.markdown("### Diagnostic lecture")
            st.json(
                {
                    "filename": age_file.name,
                    "size_mb": age_diag.get("size_mb"),
                    "sep_used": age_diag.get("sep_used"),
                    "encoding_used": age_diag.get("encoding_used"),
                    "seconds": age_diag.get("seconds"),
                    "rows": age_diag.get("rows"),
                    "cols": age_diag.get("cols"),
                    "kept_columns": needed,
                }
            )

        geo_to_insee = None
        if conc_file is not None:
            df_conc, conc_diag = read_uploaded_csv_smart(conc_file.getvalue())
            geo_guess, insee_guess = guess_concordance_columns(df_conc.columns.tolist())

            st.markdown("### Concordance (optionnelle)")
            geo_col_conc = st.selectbox(
                "Concordance: colonne GEO (clé)",
                df_conc.columns.tolist(),
                index=(df_conc.columns.tolist().index(geo_guess) if geo_guess in df_conc.columns else 0),
            )
            insee_col_conc = st.selectbox(
                "Concordance: colonne code INSEE",
                df_conc.columns.tolist(),
                index=(df_conc.columns.tolist().index(insee_guess) if insee_guess in df_conc.columns else 0),
            )

            if st.button("Construire / actualiser la concordance"):
                geo_to_insee = build_geo_to_insee_map(df_conc, geo_col_conc, insee_col_conc)
                st.session_state["geo_to_insee"] = geo_to_insee

            if "geo_to_insee" in st.session_state:
                geo_to_insee = st.session_state["geo_to_insee"]

        df = normalize_codes(df, geo_to_insee)
        df_zone = df[df["CODE_INSEE"].isin(sel)].copy()
        if df_zone.empty:
            raise RuntimeError("Après mapping + filtrage zone, aucune ligne ne correspond aux communes sélectionnées.")

        df_zone = apply_sex_filter(df_zone, sex_mode)

        years = sort_years(df_zone["TIME_PERIOD"].dropna().astype(str).unique().tolist())
        if not years:
            raise RuntimeError("Aucune année TIME_PERIOD exploitable après filtrage zone.")
        st.caption(f"Années disponibles (zone) : {years[0]} → {years[-1]} ({len(years)} points)")

        # AGE maps based on zone (fast)
        age_values = df_zone["AGE"].dropna().astype(str).unique().tolist()
        bucket_map, _mid_map = build_age_maps(age_values)

        # Name map for legend
        name_map = dict(zip(in_radius["code_insee"].astype(str).tolist(), in_radius["nom"].astype(str).tolist()))

        # -------- Graph 1
        st.markdown("## 1) Évolution de la population (communes + total zone)")
        pop_long = pop_timeseries_by_commune(df_zone)

        latest_year = years[-1]
        latest = pop_long[pop_long["TIME_PERIOD"] == latest_year].copy()
        top_codes = (
            latest.groupby("CODE_INSEE")["population"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.astype(str)
            .tolist()
        )

        pop_top = pop_long[pop_long["CODE_INSEE"].astype(str).isin(top_codes)].copy()
        pop_pivot = pop_top.pivot(index="TIME_PERIOD", columns="CODE_INSEE", values="population").fillna(0).astype(int)
        pop_pivot = pop_pivot.loc[sort_years(pop_pivot.index.tolist())]

        # rename cols
        newcols = {}
        used = set()
        for code in pop_pivot.columns.astype(str).tolist():
            nom = name_map.get(code, code)
            label = f"{nom} ({code})"
            if label in used:
                label = f"{label}*"
            used.add(label)
            newcols[code] = label
        pop_pivot.columns = [newcols.get(str(c), str(c)) for c in pop_pivot.columns]

        pop_total_zone = pop_long.groupby("TIME_PERIOD")["population"].sum()
        pop_total_zone = pop_total_zone.loc[sort_years(pop_total_zone.index.tolist())]
        pop_pivot["TOTAL ZONE"] = pop_total_zone.reindex(pop_pivot.index).fillna(0).astype(int)

        st.line_chart(pop_pivot, use_container_width=True)

        # -------- Graph 2
        st.markdown("## 2) Évolution de la population par tranches d’âge (zone)")
        bucket_pivot = bucket_timeseries_zone(df_zone, bucket_map)
        st.line_chart(bucket_pivot, use_container_width=True)

except Exception as e:
    st.error("Erreur dans l'app (trace ci-dessous) :")
    st.exception(e)
