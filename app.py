import io
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium


# =========================
# Config
# =========================
st.set_page_config(page_title="Zone de chalandise - Démographie (INSEE)", layout="wide")

# Géocodage (Géoplateforme / IGN)
GEOCODE_URL = "https://data.geopf.fr/geocodage/search/"

# Communes (Etalab)
GEO_COMMUNES_URL = "https://geo.api.gouv.fr/communes"

# INSEE - fichiers (liens directs vus dans les pages INSEE)
# Historique population communale 1876-2023 (xlsx)
INSEE_POP_HIST_XLSX_URL = "https://www.insee.fr/fr/statistiques/fichier/3698339/base-pop-historiques-1876-2023.xlsx"
# Sexe & âge quinquennal 1968-2022 (zip xlsx)
INSEE_AGE_ZIP_URL = "https://www.insee.fr/fr/statistiques/fichier/1893204/pop-sexe-age-quinquennal6822.zip"


# =========================
# Utils
# =========================
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance great-circle en km."""
    r = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def http_get_bytes(url: str, params: Optional[dict] = None, timeout: int = 30) -> bytes:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "streamlit-demographie/1.0"})
    r.raise_for_status()
    return r.content


def safe_int(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace("\u202f", "").replace(" ", "").replace(",", ".")
        return int(float(x))
    except Exception:
        return np.nan


# =========================
# Géocodage & communes
# =========================
@dataclass
class GeocodeResult:
    label: str
    lat: float
    lon: float
    citycode: Optional[str] = None  # code INSEE commune (souvent présent)
    postcode: Optional[str] = None
    city: Optional[str] = None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode_search(q: str, limit: int = 5) -> List[GeocodeResult]:
    params = {"q": q, "limit": limit}
    data = requests.get(GEOCODE_URL, params=params, timeout=30, headers={"User-Agent": "streamlit-demographie/1.0"}).json()
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
    # Renvoie la commune correspondant au point
    params = {"lat": lat, "lon": lon, "fields": "nom,code,codeDepartement,codeRegion", "format": "json"}
    data = requests.get(GEO_COMMUNES_URL, params=params, timeout=30, headers={"User-Agent": "streamlit-demographie/1.0"}).json()
    if isinstance(data, list) and len(data) > 0:
        return data[0]
    return None


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=True)
def geo_communes_region(code_region: str) -> pd.DataFrame:
    """
    Récupère toutes les communes d'une région (sans contour, juste centre)
    puis on filtrera par distance côté app.
    """
    params = {
        "codeRegion": code_region,
        "fields": "nom,code,codeDepartement,codesPostaux,population,centre",
        "format": "geojson",
        "geometry": "centre",
    }
    geojson = requests.get(GEO_COMMUNES_URL, params=params, timeout=60, headers={"User-Agent": "streamlit-demographie/1.0"}).json()
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
# INSEE data loaders
# =========================
@st.cache_data(ttl=30 * 24 * 3600, show_spinner=True)
def load_insee_pop_history() -> pd.DataFrame:
    """
    Charge l'historique des populations communales (xlsx).
    Sortie: long format [code_insee, libelle, annee, population]
    """
    content = http_get_bytes(INSEE_POP_HIST_XLSX_URL)
    xls = pd.ExcelFile(io.BytesIO(content))

    # Stratégie robuste: prendre la 1ère feuille "communes" si trouvable, sinon 1ère feuille.
    sheet = None
    for s in xls.sheet_names:
        sl = s.lower()
        if "comm" in sl or "populations communales" in sl or "pop" in sl:
            sheet = s
            break
    if sheet is None:
        sheet = xls.sheet_names[0]

    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    # On cherche code commune + libellé
    code_col = None
    lib_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in {"codgeo", "code", "code commune", "codgeo"} or "code" in cl and "geo" in cl:
            code_col = c
        if cl in {"libgeo", "libellé", "libelle", "nom"} or "lib" in cl and "geo" in cl:
            lib_col = c

    if code_col is None:
        # fallback: première colonne
        code_col = df.columns[0]
    if lib_col is None:
        # fallback: deuxième colonne si existe
        lib_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

    # Colonnes années = chiffres (1876..2023)
    year_cols = []
    for c in df.columns:
        try:
            y = int(str(c))
            if 1800 <= y <= 2100:
                year_cols.append(c)
        except Exception:
            pass

    if not year_cols:
        # dernier fallback: colonnes qui ressemblent à "P2023" etc.
        for c in df.columns:
            cl = str(c).lower()
            digits = "".join([ch for ch in cl if ch.isdigit()])
            if len(digits) == 4:
                y = int(digits)
                if 1800 <= y <= 2100:
                    year_cols.append(c)

    base = df[[code_col, lib_col] + year_cols].copy()
    base.rename(columns={code_col: "code_insee", lib_col: "libelle"}, inplace=True)
    base["code_insee"] = base["code_insee"].astype(str).str.zfill(5)

    long = base.melt(id_vars=["code_insee", "libelle"], value_vars=year_cols, var_name="annee", value_name="population")
    long["annee"] = long["annee"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    long["population"] = long["population"].apply(safe_int)
    long = long.dropna(subset=["population"])
    return long


@st.cache_data(ttl=30 * 24 * 3600, show_spinner=True)
def load_insee_age_structure() -> pd.DataFrame:
    """
    Charge le zip INSEE "pop-sexe-age-quinquennal..." (xlsx zippé).
    Sortie: long format [code_insee, annee, age_groupe, hommes, femmes, total]
    """
    zbytes = http_get_bytes(INSEE_AGE_ZIP_URL, timeout=120)

    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
        # on prend le 1er xlsx trouvé
        xlsx_names = [n for n in zf.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError("Aucun .xlsx trouvé dans le zip INSEE âge.")
        xlsx_bytes = zf.read(xlsx_names[0])

    xls = pd.ExcelFile(io.BytesIO(xlsx_bytes))
    rows = []

    for sheet in xls.sheet_names:
        sl = sheet.lower()

        # On ne garde que les feuilles communes (COM) si possible
        # (les noms varient, donc heuristique)
        if "com" not in sl and "comm" not in sl:
            continue

        # Essai d'extraire l'année du nom d'onglet
        year = None
        for tok in sheet.replace("-", " ").replace("_", " ").split():
            if tok.isdigit() and len(tok) == 4:
                year = int(tok)
                break
        if year is None:
            # si pas d'année dans le nom, on skip
            continue

        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]

        # Chercher code commune
        code_col = None
        for c in df.columns:
            cl = c.lower()
            if cl in {"codgeo", "codgeo", "code", "code commune"} or ("code" in cl and "geo" in cl):
                code_col = c
                break
        if code_col is None:
            continue

        df[code_col] = df[code_col].astype(str).str.zfill(5)

        # Colonnes d'âges : souvent "0-4", "5-9", ... "95+"
        # Et colonnes hommes/femmes ou total; on fait heuristique:
        # - si colonnes "Hommes ..." et "Femmes ..." on sépare
        cols = df.columns.tolist()

        # On cherche colonnes hommes/femmes par motif
        hommes_cols = [c for c in cols if "hom" in c.lower()]
        femmes_cols = [c for c in cols if "fem" in c.lower()]

        # Cas A: colonnes déjà par âge *et* sexe (hommes/femmes)
        if hommes_cols and femmes_cols:
            # On devine l'âge groupe depuis le libellé (dernier token)
            for hc in hommes_cols:
                age = hc.split()[-1]
                # trouver colonne femme correspondante qui finit pareil
                fc = None
                for cand in femmes_cols:
                    if cand.split()[-1] == age:
                        fc = cand
                        break
                if fc is None:
                    continue
                tmp = df[[code_col, hc, fc]].copy()
                tmp.rename(columns={code_col: "code_insee", hc: "hommes", fc: "femmes"}, inplace=True)
                tmp["annee"] = year
                tmp["age_groupe"] = age
                tmp["hommes"] = tmp["hommes"].apply(safe_int)
                tmp["femmes"] = tmp["femmes"].apply(safe_int)
                tmp["total"] = tmp[["hommes", "femmes"]].sum(axis=1, min_count=1)
                rows.append(tmp)

        else:
            # Cas B: colonnes par âge en total (pas de sexe)
            # On essaye de repérer colonnes d'âge via regex simple
            age_cols = []
            for c in cols:
                s = str(c)
                if any(ch.isdigit() for ch in s) and ("-" in s or "+" in s):
                    # ex: "0-4" / "95+"
                    age_cols.append(c)

            if not age_cols:
                continue

            tmp = df[[code_col] + age_cols].copy()
            tmp.rename(columns={code_col: "code_insee"}, inplace=True)
            long = tmp.melt(id_vars=["code_insee"], var_name="age_groupe", value_name="total")
            long["annee"] = year
            long["total"] = long["total"].apply(safe_int)
            long["hommes"] = np.nan
            long["femmes"] = np.nan
            rows.append(long[["code_insee", "annee", "age_groupe", "hommes", "femmes", "total"]])

    if not rows:
        raise RuntimeError("Impossible d'extraire des feuilles 'communes' du fichier INSEE âge.")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["code_insee", "annee", "age_groupe"])
    return out


def midpoint_from_age_group(age_groupe: str) -> Optional[float]:
    s = str(age_groupe).strip()
    if "+" in s:
        # ex "95+" -> approx 97.5 (arbitraire)
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


def aggregate_age_buckets(df_age_year: pd.DataFrame) -> pd.DataFrame:
    """
    Regroupe les tranches quinquennales en grands groupes:
    0-14, 15-29, 30-44, 45-59, 60-74, 75+
    """
    def bucket(mid):
        if mid is None or np.isnan(mid):
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

    tmp = df_age_year.copy()
    tmp["mid"] = tmp["age_groupe"].apply(midpoint_from_age_group)
    tmp["bucket"] = tmp["mid"].apply(bucket)
    g = tmp.groupby("bucket", as_index=False)["total"].sum()
    return g


# =========================
# UI
# =========================
st.title("Zone de chalandise – communes + démographie (INSEE)")

with st.expander("Sources (pour tracer/justifier au client)", expanded=False):
    st.markdown(
        "- Géocodage : service Géoplateforme IGN (endpoint `data.geopf.fr/geocodage/search`) \n"
        "- Communes : `geo.api.gouv.fr` (communes, centres) \n"
        "- Populations : INSEE *Historique des populations communales 1876–2023* (xlsx) \n"
        "- Âges : INSEE *Population selon sexe et âge quinquennal 1968–2022* (zip xlsx)\n"
    )

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("1) Définir la zone")
    q = st.text_input("Adresse / code postal / ville", value="22100 Dinan")
    radius_km = st.slider("Rayon (km)", min_value=1, max_value=80, value=15, step=1)

    geo_results = []
    if q.strip():
        with st.spinner("Géocodage..."):
            geo_results = geocode_search(q.strip(), limit=8)

    if not geo_results:
        st.warning("Aucun résultat de géocodage. Essaie une adresse plus précise.")
        st.stop()

    choice = st.selectbox("Choisir le point de départ", options=list(range(len(geo_results))),
                          format_func=lambda i: geo_results[i].label)

    center = geo_results[choice]
    st.caption(f"Point : **{center.label}**  (lat={center.lat:.6f}, lon={center.lon:.6f})")

    with st.spinner("Détection de la région + chargement des communes..."):
        com = geo_commune_by_latlon(center.lat, center.lon)
        if not com:
            st.error("Impossible de déterminer la commune via geo.api.gouv.fr à partir du point.")
            st.stop()

        code_region = com.get("codeRegion")
        df_communes = geo_communes_region(code_region)

    # Filtre distance
    df_communes["dist_km"] = df_communes.apply(
        lambda r: haversine_km(center.lat, center.lon, r["lat"], r["lon"]), axis=1
    )
    in_radius = df_communes[df_communes["dist_km"] <= radius_km].copy()
    in_radius = in_radius.sort_values(["dist_km", "population_geoapi"], ascending=[True, False])

    st.write(f"Communes dans {radius_km} km : **{len(in_radius):,}**")

    if "selected_codes" not in st.session_state:
        st.session_state["selected_codes"] = set()

    # Tableau sélectionnable
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
        key="editor_communes"
    )

    # Mise à jour sélection
    new_selected = set(edited.loc[edited["ajouter"] == True, "code_insee"].astype(str).tolist())
    st.session_state["selected_codes"] = new_selected

    st.info(f"Communes sélectionnées : **{len(st.session_state['selected_codes'])}**")

with right:
    st.subheader("2) Carte + analyses démographiques")

    # Carte
    m = folium.Map(location=[center.lat, center.lon], zoom_start=10, control_scale=True)
    folium.Marker([center.lat, center.lon], tooltip="Point centre", popup=center.label).add_to(m)
    folium.Circle([center.lat, center.lon], radius=radius_km * 1000, fill=False).add_to(m)

    # Marqueurs (limités pour perf)
    sample = in_radius.head(250)
    for _, r in sample.iterrows():
        tooltip = f"{r['nom']} ({r['code_insee']}) - {r['dist_km']:.1f} km"
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=4,
            tooltip=tooltip
        ).add_to(m)

    st_folium(m, use_container_width=True, height=420)

    sel = sorted(list(st.session_state["selected_codes"]))
    if not sel:
        st.warning("Sélectionne au moins une commune pour afficher les graphes.")
        st.stop()

    # Load INSEE data
    with st.spinner("Chargement INSEE (pop + âges) ... (cache ensuite)"):
        pop_hist = load_insee_pop_history()
        age = load_insee_age_structure()

    # Filtre communes
    pop_sel = pop_hist[pop_hist["code_insee"].isin(sel)].copy()
    age_sel = age[age["code_insee"].isin(sel)].copy()

    # Population agrégée sur la zone
    pop_zone = pop_sel.groupby("annee", as_index=False)["population"].sum().sort_values("annee")

    st.markdown("### Population totale – zone sélectionnée")
    st.line_chart(pop_zone.set_index("annee")["population"])

    # Âges : années dispo
    years = sorted(age_sel["annee"].dropna().unique().astype(int).tolist())
    if years:
        y_min, y_max = min(years), max(years)
        st.caption(f"Données âges disponibles sur la sélection : {y_min} → {y_max}")
        y1, y2 = st.select_slider("Comparer 2 années (structure d'âge)", options=years, value=(y_min, y_max))
    else:
        st.error("Aucune donnée d'âge trouvée pour ces communes (cas rare).")
        st.stop()

    def age_bucket_for_year(y: int) -> pd.DataFrame:
        dfy = age_sel[age_sel["annee"] == y].copy()
        dfy = dfy.dropna(subset=["total"])
        dfy_zone = dfy.groupby(["age_groupe"], as_index=False)["total"].sum()
        return aggregate_age_buckets(dfy_zone)

    b1 = age_bucket_for_year(y1)
    b2 = age_bucket_for_year(y2)

    # Mise en forme comparative
    comp = pd.merge(b1, b2, on="bucket", how="outer", suffixes=(f"_{y1}", f"_{y2}")).fillna(0)
    comp["delta"] = comp[f"total_{y2}"] - comp[f"total_{y1}"]
    comp["part_"+str(y1)] = comp[f"total_{y1}"] / max(comp[f"total_{y1}"].sum(), 1)
    comp["part_"+str(y2)] = comp[f"total_{y2}"] / max(comp[f"total_{y2}"].sum(), 1)

    st.markdown("### Structure d’âge (grands groupes) – comparaison")
    st.dataframe(
        comp.rename(columns={
            f"total_{y1}": f"Effectif {y1}",
            f"total_{y2}": f"Effectif {y2}",
            "bucket": "Tranche",
            "delta": "Évolution (effectif)"
        })[["Tranche", f"Effectif {y1}", f"Effectif {y2}", "Évolution (effectif)"]],
        use_container_width=True,
        hide_index=True
    )

    # Parts chart
    parts = comp[["bucket", "part_"+str(y1), "part_"+str(y2)]].copy()
    parts = parts.set_index("bucket")
    st.markdown("### Parts (%) par tranche d’âge")
    st.line_chart(parts)

    # Âge moyen approximatif (par milieux de classes)
    st.markdown("### Âge moyen approximatif (zone) – série")
    age_sel2 = age_sel.dropna(subset=["total"]).copy()
    age_sel2["mid"] = age_sel2["age_groupe"].apply(midpoint_from_age_group)
    age_sel2 = age_sel2.dropna(subset=["mid"])
    # Moyenne pondérée par année
    age_mean = age_sel2.groupby("annee").apply(
        lambda d: (d["mid"] * d["total"]).sum() / max(d["total"].sum(), 1)
    ).reset_index(name="age_moyen_approx").sort_values("annee")

    st.line_chart(age_mean.set_index("annee")["age_moyen_approx"])

    # Export CSV
    st.markdown("### Export")
    export_pop = pop_zone.copy()
    export_pop["type"] = "population_totale_zone"
    export_age_mean = age_mean.copy()
    export_age_mean["type"] = "age_moyen_approx_zone"

    export = pd.concat(
        [
            export_pop.rename(columns={"annee": "year", "population": "value"})[["type", "year", "value"]],
            export_age_mean.rename(columns={"annee": "year", "age_moyen_approx": "value"})[["type", "year", "value"]],
        ],
        ignore_index=True
    )
    st.download_button(
        "Télécharger CSV (indicateurs zone)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="indicateurs_zone.csv",
        mime="text/csv",
    )
