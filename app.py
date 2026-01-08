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

GEOCODE_URL = "https://data.geopf.fr/geocodage/search/"
GEO_COMMUNES_URL = "https://geo.api.gouv.fr/communes"

INSEE_POP_HIST_XLSX_URL = "https://www.insee.fr/fr/statistiques/fichier/3698339/base-pop-historiques-1876-2023.xlsx"
INSEE_AGE_ZIP_URL = "https://www.insee.fr/fr/statistiques/fichier/1893204/pop-sexe-age-quinquennal6822.zip"

CACHE_DIR = Path("/tmp/demographie_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UA = {"User-Agent": "streamlit-demographie/1.0"}


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


def http_get_cached(url: str, filename: str, max_age_days: int = 90, timeout: int = 240) -> bytes:
    path = CACHE_DIR / filename
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return path.read_bytes()

    r = requests.get(url, timeout=timeout, headers=UA)
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
    r = requests.get(GEOCODE_URL, params=params, timeout=25, headers=UA)
    r.raise_for_status()
    data = r.json()

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
        geom = f.get("geometry", {}) or {}
        coords = geom.get("coordinates", None)
        if not coords:
            continue
        lon, lat = coords[0], coords[1]
        rows.append(
            {
                "code_insee": str(props.get("code")).zfill(5) if props.get("code") else None,
                "nom": props.get("nom"),
                "code_dept": props.get("codeDepartement"),
                "codes_postaux": ", ".join(props.get("codesPostaux") or []),
                "population_geoapi": props.get("population"),
                "lat": float(lat),
                "lon": float(lon),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["code_insee", "lat", "lon"])
    return df


# =========================
# INSEE loaders (cachés)
# =========================
@st.cache_data(ttl=30 * 24 * 3600, show_spinner=True)
def load_insee_pop_history() -> pd.DataFrame:
    content = http_get_cached(INSEE_POP_HIST_XLSX_URL, "insee_pop_hist.xlsx")
    xls = pd.ExcelFile(io.BytesIO(content))

    sheet = xls.sheet_names[0]
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

    long = base.melt(
        id_vars=["code_insee", "libelle"],
        value_vars=year_cols,
        var_name="annee",
        value_name="population",
    )
    long["annee"] = long["annee"].astype(int)
    long["population"] = long["population"].apply(safe_int)
    return long.dropna(subset=["population"])


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=True)
def load_insee_age_structure() -> pd.DataFrame:
    zbytes = http_get_cached(INSEE_AGE_ZIP_URL, "insee_age.zip")

    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError("Aucun .xlsx trouvé dans le zip INSEE âge.")
        xlsx_bytes = zf.read(xlsx_names[0])

    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    rows = []

    for sheet in xls.sheet_names:
        sl = sheet.lower()
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
    return out.dropna(subset=["code_insee", "annee", "age_groupe"])


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
    return tmp.groupby("bucket", as_index=False)["total"].sum()


# =========================
# UI (IMPORTANT : zéro requête tant que pas de clic)
# =========================
st.title("Zone de chalandise – communes + démographie (INSEE)")

with st.expander("⚙️ Diagnostic rapide (si ça bloque)", expanded=False):
    st.write(
        "Si l'écran reste blanc sur Streamlit Cloud, c'est souvent parce que des requêtes réseau tournent au démarrage. "
        "Cette version ne lance rien tant que tu ne cliques pas sur **Rechercher**."
    )
    st.write(f"Cache disque: {CACHE_DIR}")

left, right = st.columns([0.42, 0.58], gap="large")

# ---- Paramètres utilisateur (sans requêtes)
with left:
    st.subheader("1) Définir la zone")

    q = st.text_input("Adresse / code postal / ville", value="", placeholder="Ex: 22100 Dinan ou 'Saint-Brieuc'")
    radius_km = st.slider("Rayon (km)", 1, 80, 15, 1)

    run_search = st.button("🔎 Rechercher (géocodage + communes)")

    if "search_done" not in st.session_state:
        st.session_state["search_done"] = False
    if "geo_results" not in st.session_state:
        st.session_state["geo_results"] = []
    if "in_radius" not in st.session_state:
        st.session_state["in_radius"] = None
    if "center" not in st.session_state:
        st.session_state["center"] = None

    if run_search:
        if not q.strip():
            st.warning("Tape une adresse/ville avant de cliquer sur Rechercher.")
        else:
            try:
                with st.spinner("Géocodage..."):
                    results = geocode_search(q.strip(), limit=8)
                if not results:
                    st.error("Aucun résultat de géocodage.")
                else:
                    st.session_state["geo_results"] = results
                    st.session_state["search_done"] = True
            except Exception as e:
                st.error(f"Erreur géocodage : {e}")

    # Si on n'a pas cliqué encore, on n'exécute rien.
    if not st.session_state["search_done"]:
        st.info("➡️ Renseigne une ville/adresse puis clique **Rechercher**.")
        st.stop()

    geo_results: List[GeocodeResult] = st.session_state["geo_results"]

    choice = st.selectbox(
        "Choisir le point de départ",
        options=list(range(len(geo_results))),
        format_func=lambda i: geo_results[i].label,
    )
    center = geo_results[choice]
    st.session_state["center"] = center

    build_zone = st.button("📍 Construire la liste des communes dans le rayon")

    if build_zone:
        try:
            with st.spinner("Détection département + récupération communes + filtre rayon..."):
                com = geo_commune_by_latlon(center.lat, center.lon)
                if not com:
                    raise RuntimeError("Impossible de déterminer la commune (lat/lon).")

                code_dept = com.get("codeDepartement")
                df_communes = geo_communes_dept(code_dept)

                df_communes["dist_km"] = df_communes.apply(
                    lambda r: haversine_km(center.lat, center.lon, r["lat"], r["lon"]), axis=1
                )
                in_radius = df_communes[df_communes["dist_km"] <= radius_km].copy()
                in_radius = in_radius.sort_values(["dist_km", "population_geoapi"], ascending=[True, False])

                st.session_state["in_radius"] = in_radius
        except Exception as e:
            st.error(f"Erreur construction zone : {e}")

    if st.session_state["in_radius"] is None:
        st.info("➡️ Clique **Construire la liste des communes**.")
        st.stop()

    in_radius = st.session_state["in_radius"]
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

# ---- Affichage carte + INSEE
with right:
    st.subheader("2) Carte + analyses")

    center = st.session_state["center"]
    in_radius = st.session_state["in_radius"]

    m = folium.Map(location=[center.lat, center.lon], zoom_start=10, control_scale=True)
    folium.Marker([center.lat, center.lon], tooltip="Point centre", popup=center.label).add_to(m)
    folium.Circle([center.lat, center.lon], radius=radius_km * 1000, fill=False).add_to(m)

    sample = in_radius.head(250)
    for _, r in sample.iterrows():
        tooltip = f"{r['nom']} ({r['code_insee']}) - {r['dist_km']:.1f} km"
        folium.CircleMarker([r["lat"], r["lon"]], radius=4, tooltip=tooltip).add_to(m)

    st_folium(m, width="stretch", height=420)

    sel = sorted(list(st.session_state["selected_codes"]))
    if not sel:
        st.warning("Sélectionne au moins une commune pour afficher les graphes.")
        st.stop()

    st.markdown("### 3) Charger les données INSEE (au clic)")
    if "insee_loaded" not in st.session_state:
        st.session_state["insee_loaded"] = False

    if st.button("📦 Charger INSEE (population + âges)"):
        st.session_state["insee_loaded"] = True

    if not st.session_state["insee_loaded"]:
        st.info("Clique **Charger INSEE**. (Le 1er chargement est long, ensuite cache.)")
        st.stop()

    try:
        with st.spinner("Chargement INSEE (cache disque + cache Streamlit)..."):
            pop_hist = load_insee_pop_history()
            age = load_insee_age_structure()
    except Exception as e:
        st.error(f"Erreur INSEE : {e}")
        st.stop()

    # Population zone
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
        width="stretch",
        hide_index=True,
    )

    # Parts
    comp["part_" + str(y1)] = comp[f"total_{y1}"] / max(comp[f"total_{y1}"].sum(), 1)
    comp["part_" + str(y2)] = comp[f"total_{y2}"] / max(comp[f"total_{y2}"].sum(), 1)
    parts = comp[["Tranche", "part_" + str(y1), "part_" + str(y2)]].set_index("Tranche")

    st.markdown("### Parts (%) par tranche d’âge")
    st.line_chart(parts)

    # Export CSV
    export_pop = pop_zone.rename(columns={"annee": "year", "population": "value"})
    export_pop["type"] = "population_totale_zone"

    export = export_pop[["type", "year", "value"]].copy()
    st.download_button(
        "Télécharger CSV (population zone)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="population_zone.csv",
        mime="text/csv",
    )
