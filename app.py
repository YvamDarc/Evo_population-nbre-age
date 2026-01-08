import io
import math
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium


# =========================
# Config
# =========================
st.set_page_config(page_title="Zone de chalandise – Démographie (INSEE)", layout="wide")

# Géocodage (Géoplateforme IGN)
GEOCODE_URL = "https://data.geopf.fr/geocodage/search/"

# Communes (Etalab)
GEO_COMMUNES_URL = "https://geo.api.gouv.fr/communes"

# INSEE - fichiers
INSEE_POP_HIST_XLSX_URL = "https://www.insee.fr/fr/statistiques/fichier/3698339/base-pop-historiques-1876-2023.xlsx"
INSEE_AGE_ZIP_URL = "https://www.insee.fr/fr/statistiques/fichier/1893204/pop-sexe-age-quinquennal6822.zip"

# Cache disque (Streamlit Cloud : /tmp est OK)
CACHE_DIR = Path("/tmp/demographie_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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


def safe_int(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace("\u202f", "").replace(" ", "").replace(",", ".")
        return int(float(x))
    except Exception:
        return np.nan


def http_get_cached(url: str, filename: str, max_age_days: int = 60, timeout: int = 180) -> bytes:
    """
    Télécharge et met en cache sur disque (/tmp) pour accélérer Streamlit Cloud.
    """
    path = CACHE_DIR / filename
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return path.read_bytes()

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "streamlit-demographie/1.0"})
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content


# =========================
# Géocodage & communes
# =========================
@dataclass
class GeocodeResult:
    label: str
    lat: float
    lon: float
    citycode: Optional[str] = None
    postcode: Optional[str] = None
    city: Optional[str] = None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode_search(q: str, limit: int = 8) -> List[GeocodeResult]:
    params = {"q": q, "limit": limit}
    data = requests.get(
        GEOCODE_URL, params=params, timeout=30, headers={"User-Agent": "streamlit-demographie/1.0"}
    ).json()
    feats = data.get("features", [])
    out: List[GeocodeResult] = []
    for f in feats:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates", None)
        if not coords or len(coords) < 2:
            continue
        lon, lat = coords[0], coords[1]
        out.append(
            GeocodeResult(
                label=str(props.get("label", "")),
                lat=float(lat),
                lon=float(lon),
                citycode=props.get("citycode"),
                postcode=props.get("postcode"),
                city=props.get("city"),
            )
        )
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geo_commune_by_latlon(lat: float, lon: float) -> Optional[dict]:
    params = {"lat": lat, "lon": lon, "fields": "nom,code,codeDepartement,codeRegion", "format": "json"}
    data = requests.get(
        GEO_COMMUNES_URL, params=params, timeout=30, headers={"User-Agent": "streamlit-demographie/1.0"}
    ).json()
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return None


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=True)
def geo_communes_dept(code_dept: str) -> pd.DataFrame:
    """
    Toutes les communes d'un département en GeoJSON (centres uniquement) -> plus léger que région.
    """
    params = {
        "codeDepartement": code_dept,
        "fields": "nom,code,codeDepartement,codesPostaux,population,centre",
        "format": "geojson",
        "geometry": "centre",
    }
    geojson = requests.get(
        GEO_COMMUNES_URL, params=params, timeout=60, headers={"User-Agent": "streamlit-demographie/1.0"}
    ).json()

    feats = geojson.get("features", [])
    rows = []
    for f in feats:
        props = f.get("properties", {}) or {}
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates", None)
        if not coords:
            continue
        lon, lat = coords[0], coords[1]
        rows.append(
            {
                "code_insee": props.get("code"),
                "nom": props.get("nom"),
                "code_dept": props.get("codeDepartement"),
                "codes_postaux": ", ".join(props.get("codesPostaux") or []),
                "population_geoapi": props.get("population"),
                "lat": lat,
                "lon": lon,
            }
        )
    df = pd.DataFrame(rows).dropna(subset=["code_insee", "lat", "lon"])
    return df


# =========================
# INSEE loaders (cachés)
# =========================
@st.cache_data(ttl=30 * 24 * 3600, show_spinner=True)
def load_insee_pop_history() -> pd.DataFrame:
    """
    Historique population communale (xlsx) -> format long:
    [code_insee, libelle, annee, population]
    """
    content = http_get_cached(INSEE_POP_HIST_XLSX_URL, "insee_pop_hist.xlsx", max_age_days=90, timeout=180)
    xls = pd.ExcelFile(io.BytesIO(content))

    sheet = xls.sheet_names[0]
    # heuristique: feuille contenant "comm" / "pop"
    for s in xls.sheet_names:
        sl = s.lower()
        if "comm" in sl or "pop" in sl:
            sheet = s
            break

    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    code_col = None
    lib_col = None
    for c in df.columns:
        cl = c.lower()
        if ("code" in cl and "geo" in cl) or cl in {"codgeo", "code"}:
            code_col = c
        if ("lib" in cl and "geo" in cl) or cl in {"libgeo", "libelle", "libellé", "nom"}:
            lib_col = c

    if code_col is None:
        code_col = df.columns[0]
    if lib_col is None:
        lib_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    year_cols = []
    for c in df.columns:
        try:
            y = int(str(c))
            if 1800 <= y <= 2100:
                year_cols.append(c)
        except Exception:
            pass

    base = df[[code_col, lib_col] + year_cols].copy()
    base.rename(columns={code_col: "code_insee", lib_col: "libelle"}, inplace=True)
    base["code_insee"] = base["code_insee"].astype(str).str.zfill(5)

    long = base.melt(id_vars=["code_insee", "libelle"], value_vars=year_cols, var_name="annee", value_name="population")
    long["annee"] = long["annee"].astype(int)
    long["population"] = long["population"].apply(safe_int)
    long = long.dropna(subset=["population"])
    return long


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=True)
def load_insee_age_structure() -> pd.DataFrame:
    """
    Zip INSEE âges (xlsx zippé) -> long:
    [code_insee, annee, age_groupe, total]
    (on ne force pas hommes/femmes car les fichiers varient)
    """
    zbytes = http_get_cached(INSEE_AGE_ZIP_URL, "insee_age.zip", max_age_days=90, timeout=240)

    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError("Aucun .xlsx trouvé dans le zip INSEE âge.")
        xlsx_bytes = zf.read(xlsx_names[0])

    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    rows = []

    for sheet in xls.sheet_names:
        sl = sheet.lower()
        # on prend feuilles communes si possible
        if "com" not in sl and "comm" not in sl:
            continue

        year = None
        for tok in sheet.replace("-", " ").replace("_", " ").split():
            if tok.isdigit() and len(tok) == 4:
                year = int(tok)
                break
        if year is None:
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]

        code_col = None
        for c in df.columns:
            cl = c.lower()
            if ("code" in cl and "geo" in cl) or cl in {"codgeo", "code"}:
                code_col = c
                break
        if code_col is None:
            continue

        df[code_col] = df[code_col].astype(str).str.zfill(5)

        # colonnes d'âges (0-4, 5-9, ..., 95+)
        age_cols = []
        for c in df.columns:
            s = str(c)
            if any(ch.isdigit() for ch in s) and ("-" in s or "+" in s):
                age_cols.append(c)
        if not age_cols:
            continue

        tmp = df[[code_col] + age_cols].copy()
        tmp.rename(columns={code_col: "code_insee"}, inplace=True)
        long = tmp.melt(id_vars=["code_insee"], var_name="age_groupe", value_name="total")
        long["annee"] = year
        long["total"] = long["total"].apply(safe_int)
        rows.append(long[["code_insee", "annee", "age_groupe", "total"]])

    if not rows:
        raise RuntimeError("Impossible d'extraire des feuilles 'communes' du fichier INSEE âge.")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["code_insee", "annee", "age_groupe"])
    return out


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


def aggregate_age_buckets(df_age: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe quinquennal en grands groupes:
    0-14, 15-29, 30-44, 45-59, 60-74, 75+
    """
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

    tmp = df_age.copy()
    tmp["mid"] = tmp["age_groupe"].apply(midpoint_from_age_group)
    tmp["bucket"] = tmp["mid"].apply(bucket)
    g = tmp.groupby("bucket", as_index=False)["total"].sum()
    return g


# =========================
# UI
# =========================
st.title("Zone de chalandise – communes + démographie (INSEE)")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) Définir la zone")
    q = st.text_input("Adresse / code postal / ville", value="22100 Dinan")
    radius_km = st.slider("Rayon (km)", min_value=1, max_value=80, value=15, step=1)

    if not q.strip():
        st.stop()

    with st.spinner("Géocodage..."):
        geo_results = geocode_search(q.strip(), limit=8)

    if not geo_results:
        st.warning("Aucun résultat de géocodage. Essaie une adresse plus précise.")
        st.stop()

    choice = st.selectbox(
        "Choisir le point de départ",
        options=list(range(len(geo_results))),
        format_func=lambda i: geo_results[i].label,
    )
    center = geo_results[choice]
    st.caption(f"Point : **{center.label}**  (lat={center.lat:.6f}, lon={center.lon:.6f})")

    with st.spinner("Détection de la commune (département) + chargement des communes..."):
        com = geo_commune_by_latlon(center.lat, center.lon)
        if not com:
            st.error("Impossible de déterminer la commune via geo.api.gouv.fr à partir du point.")
            st.stop()

        code_dept = com.get("codeDepartement")
        df_communes = geo_communes_dept(code_dept)

    # Filtre distance
    df_communes["dist_km"] = df_communes.apply(
        lambda r: haversine_km(center.lat, center.lon, r["lat"], r["lon"]), axis=1
    )
    in_radius = df_communes[df_communes["dist_km"] <= radius_km].copy()
    in_radius = in_radius.sort_values(["dist_km", "population_geoapi"], ascending=[True, False])

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
            "ajouter": st.column_config.CheckboxColumn("Ajouter à la liste"),
            "dist_km": st.column_config.NumberColumn("Distance (km)", format="%.1f"),
        },
        disabled=["code_insee", "nom", "codes_postaux", "code_dept", "population_geoapi", "dist_km"],
        key="editor_communes",
    )

    st.session_state["selected_codes"] = set(
        edited.loc[edited["ajouter"] == True, "code_insee"].astype(str).tolist()
    )
    st.info(f"Communes sélectionnées : **{len(st.session_state['selected_codes'])}**")

with right:
    st.subheader("2) Carte + analyses")

    # Carte
    m = folium.Map(location=[center.lat, center.lon], zoom_start=10, control_scale=True)
    folium.Marker([center.lat, center.lon], tooltip="Point centre", popup=center.label).add_to(m)
    folium.Circle([center.lat, center.lon], radius=radius_km * 1000, fill=False).add_to(m)

    # Marqueurs limités (perf)
    sample = in_radius.head(250)
    for _, r in sample.iterrows():
        tooltip = f"{r['nom']} ({r['code_insee']}) - {r['dist_km']:.1f} km"
        folium.CircleMarker([r["lat"], r["lon"]], radius=4, tooltip=tooltip).add_to(m)

    st_folium(m, use_container_width=True, height=420)

    sel = sorted(list(st.session_state["selected_codes"]))
    if not sel:
        st.warning("Sélectionne au moins une commune pour afficher les graphes.")
        st.stop()

    st.markdown("### 3) Charger les données INSEE (au clic)")
    if "insee_loaded" not in st.session_state:
        st.session_state["insee_loaded"] = False

    if st.button("Charger INSEE (population + âges)"):
        st.session_state["insee_loaded"] = True

    if not st.session_state["insee_loaded"]:
        st.info("Clique sur **Charger INSEE** pour lancer le téléchargement/parse (une seule fois, puis cache).")
        st.stop()

    with st.spinner("Chargement INSEE (cache disque + cache Streamlit)..."):
        pop_hist = load_insee_pop_history()
        age = load_insee_age_structure()

    # Population agrégée
    pop_sel = pop_hist[pop_hist["code_insee"].isin(sel)].copy()
    pop_zone = pop_sel.groupby("annee", as_index=False)["population"].sum().sort_values("annee")

    st.markdown("### Population totale – zone sélectionnée")
    st.line_chart(pop_zone.set_index("annee")["population"])

    # Âges
    age_sel = age[age["code_insee"].isin(sel)].dropna(subset=["total"]).copy()
    years = sorted(age_sel["annee"].dropna().unique().astype(int).tolist())
    if not years:
        st.error("Aucune donnée d'âge trouvée pour ces communes.")
        st.stop()

    y_min, y_max = min(years), max(years)
    y1, y2 = st.select_slider("Comparer 2 années (structure d'âge)", options=years, value=(y_min, y_max))

    def buckets_for_year(y: int) -> pd.DataFrame:
        dfy = age_sel[age_sel["annee"] == y].copy()
        dfy_zone = dfy.groupby("age_groupe", as_index=False)["total"].sum()
        return aggregate_age_buckets(dfy_zone)

    b1 = buckets_for_year(y1)
    b2 = buckets_for_year(y2)

    comp = pd.merge(b1, b2, on="bucket", how="outer", suffixes=(f"_{y1}", f"_{y2}")).fillna(0)
    comp["delta"] = comp[f"total_{y2}"] - comp[f"total_{y1}"]

    st.markdown("### Structure d’âge (grands groupes) – comparaison")
    st.dataframe(
        comp.rename(
            columns={
                "bucket": "Tranche",
                f"total_{y1}": f"Effectif {y1}",
                f"total_{y2}": f"Effectif {y2}",
                "delta": "Évolution (effectif)",
            }
        )[["Tranche", f"Effectif {y1}", f"Effectif {y2}", "Évolution (effectif)"]],
        use_container_width=True,
        hide_index=True,
    )

    # Parts (%)
    comp["part_" + str(y1)] = comp[f"total_{y1}"] / max(comp[f"total_{y1}"].sum(), 1)
    comp["part_" + str(y2)] = comp[f"total_{y2}"] / max(comp[f"total_{y2}"].sum(), 1)
    parts = comp[["Tranche", "part_" + str(y1), "part_" + str(y2)]].set_index("Tranche")
    st.markdown("### Parts (%) par tranche d’âge")
    st.line_chart(parts)

    # Âge moyen approximatif
    st.markdown("### Âge moyen approximatif (zone) – série")
    age_sel2 = age_sel.copy()
    age_sel2["mid"] = age_sel2["age_groupe"].apply(midpoint_from_age_group)
    age_sel2 = age_sel2.dropna(subset=["mid"])

    age_mean = (
        age_sel2.groupby("annee")
        .apply(lambda d: (d["mid"] * d["total"]).sum() / max(d["total"].sum(), 1))
        .reset_index(name="age_moyen_approx")
        .sort_values("annee")
    )

    st.line_chart(age_mean.set_index("annee")["age_moyen_approx"])

    # Export CSV
    export_pop = pop_zone.rename(columns={"annee": "year", "population": "value"})
    export_pop["type"] = "population_totale_zone"

    export_age = age_mean.rename(columns={"annee": "year", "age_moyen_approx": "value"})
    export_age["type"] = "age_moyen_approx_zone"

    export = pd.concat([export_pop[["type", "year", "value"]], export_age[["type", "year", "value"]]], ignore_index=True)

    st.download_button(
        "Télécharger CSV (indicateurs zone)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="indicateurs_zone.csv",
        mime="text/csv",
    )
