import io
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="INSEE brut par commune", layout="wide")

UA = {"User-Agent": "insee-brut/1.0"}

# -----------------------
# 1) Sélection commune (geo.api.gouv.fr)
# -----------------------
@st.cache_data(ttl=24*3600)
def search_communes(q: str):
    # recherche par nom ou code postal, renvoie une liste de communes
    url = "https://geo.api.gouv.fr/communes"
    params = {"nom": q, "fields": "nom,code,codeDepartement", "boost": "population", "limit": 15}
    r = requests.get(url, params=params, timeout=20, headers=UA)
    r.raise_for_status()
    return r.json()

# -----------------------
# 2) Population (2012-2024) via OFGL (API Opendatasoft)
# -----------------------
@st.cache_data(ttl=6*3600)
def ofgl_population_series(code_insee: str) -> pd.DataFrame:
    """
    Dataset OFGL: populations-ofgl-communes (2012-2024)
    API Opendatasoft v2.1.
    On récupère uniquement les lignes de la commune => léger.
    """
    base = "https://data.ofgl.fr/api/explore/v2.1/catalog/datasets/populations-ofgl-communes/records"
    # on ne connait pas à 100% le nom exact du champ code INSEE, donc on essaie plusieurs "where"
    wheres = [
        f"codgeo='{code_insee}'",
        f"code_insee='{code_insee}'",
        f"code='{code_insee}'",
        f"insee='{code_insee}'",
        f"codgeo='{code_insee}'",
    ]

    last_err = None
    for w in wheres:
        try:
            params = {"where": w, "limit": 100, "order_by": "annee"}
            r = requests.get(base, params=params, timeout=25, headers=UA)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if results:
                df = pd.DataFrame(results)
                # on tente de normaliser colonnes usuelles
                # (selon les portails, ça peut être "annee" + "population")
                return df
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Aucune donnée retournée pour {code_insee} (erreur: {last_err})")

# -----------------------
# 3) Âge (1 année) via data.gouv (fichier xlsx) - filtrage commune
# -----------------------
CACHE_DIR = Path("/tmp/insee_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AGE_XLSX_URL = None  # <- tu colleras ici l'URL "Télécharger" du xlsx depuis data.gouv

def download_cached(url: str, filename: str, max_age_days: int = 60) -> bytes:
    path = CACHE_DIR / filename
    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return path.read_bytes()
    r = requests.get(url, timeout=60, headers=UA)
    r.raise_for_status()
    path.write_bytes(r.content)
    return r.content

@st.cache_data(ttl=7*24*3600)
def age_structure_one_year(code_insee: str) -> pd.DataFrame:
    """
    Lit un fichier xlsx (population par âges) et filtre la commune.
    IMPORTANT: il faut renseigner AGE_XLSX_URL (lien direct du xlsx).
    """
    if not AGE_XLSX_URL:
        raise RuntimeError("AGE_XLSX_URL non renseignée : colle le lien direct du fichier xlsx data.gouv dans le code.")

    content = download_cached(AGE_XLSX_URL, "age_2020_tranches.xlsx", max_age_days=120)
    df = pd.read_excel(io.BytesIO(content))

    # Le dataset indique un champ "INSEE Code géographique" (et nom de commune, etc.)
    # On fait une recherche flexible
    possible_cols = [c for c in df.columns if "code" in str(c).lower() and "géo" in str(c).lower() or "insee" in str(c).lower()]
    if not possible_cols:
        possible_cols = df.columns.tolist()

    # test de colonnes courantes
    code_cols_try = ["INSEE Code géographique", "CODGEO", "codgeo", "Code", "code_insee"]
    code_col = next((c for c in code_cols_try if c in df.columns), None)
    if code_col is None:
        # fallback: première colonne candidate
        code_col = possible_cols[0]

    df[code_col] = df[code_col].astype(str).str.zfill(5)
    out = df[df[code_col] == code_insee].copy()
    return out

# -----------------------
# UI
# -----------------------
st.title("Récupération INSEE “brut” par commune")

q = st.text_input("Commune (nom) ou ville", value="Dinan")
if not q.strip():
    st.stop()

with st.spinner("Recherche commune..."):
    communes = search_communes(q.strip())

if not communes:
    st.warning("Aucune commune trouvée.")
    st.stop()

options = {f"{c['nom']} ({c['code']})": c["code"] for c in communes}
label = st.selectbox("Sélection", list(options.keys()))
code_insee = options[label]

st.success(f"Commune sélectionnée : {label}")

tab1, tab2 = st.tabs(["Population (OFGL 2012-2024)", "Âges (dataset data.gouv)"])

with tab1:
    st.subheader("Population – série (données brutes)")
    try:
        df_pop = ofgl_population_series(code_insee)
        st.caption("Affichage tel que renvoyé par l’API (brut).")
        st.dataframe(df_pop, width="stretch", height=380)

        st.download_button(
            "Télécharger CSV (brut)",
            data=df_pop.to_csv(index=False).encode("utf-8"),
            file_name=f"population_ofgl_{code_insee}.csv",
            mime="text/csv",
        )
    except Exception as e:
        st.error(str(e))
        st.info("Astuce : clique sur 'Télécharger CSV' quand ça marche pour voir les colonnes exactes.")

with tab2:
    st.subheader("Structure d’âge – brut (une année)")
    st.warning(
        "Ici on lit un fichier XLSX data.gouv : c’est rapide si le lien est bon + cache /tmp, "
        "mais il faut coller le lien direct du fichier dans AGE_XLSX_URL."
    )
    if st.button("Charger âges (brut) pour cette commune"):
        try:
            df_age = age_structure_one_year(code_insee)
            st.dataframe(df_age, width="stretch", height=380)
            st.download_button(
                "Télécharger CSV âges (brut)",
                data=df_age.to_csv(index=False).encode("utf-8"),
                file_name=f"ages_{code_insee}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(str(e))
