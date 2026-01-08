import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="INSEE brut par commune (data.gouv)", layout="wide")

UA = {"User-Agent": "insee-brut/1.0"}
CACHE_DIR = Path("/tmp/insee_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ✅ Ton URL
AGE_URL = "https://www.data.gouv.fr/api/1/datasets/r/7e3937bb-e815-4e58-92d4-d7ef478abcc0"


# -----------------------------
# Helpers: HTTP + diagnostics
# -----------------------------
def http_get_with_diagnostics(
    url: str,
    timeout: int = 180,
    stream: bool = True,
) -> Tuple[bytes, Dict]:
    """
    Télécharge une ressource en suivant les redirects et renvoie:
    - bytes du contenu
    - dict diagnostic (status, redirects, headers, size, timings)
    """
    diag = {
        "requested_url": url,
        "final_url": None,
        "status_code": None,
        "content_type": None,
        "content_length_header": None,
        "downloaded_bytes": None,
        "elapsed_seconds": None,
        "redirect_chain": [],
        "response_headers_sample": {},
    }

    t0 = time.time()
    r = requests.get(url, timeout=timeout, headers=UA, allow_redirects=True, stream=stream)
    elapsed = time.time() - t0

    diag["elapsed_seconds"] = round(elapsed, 3)
    diag["status_code"] = r.status_code
    diag["final_url"] = r.url
    diag["content_type"] = r.headers.get("Content-Type")
    diag["content_length_header"] = r.headers.get("Content-Length")

    # Redirect chain
    if r.history:
        for h in r.history:
            diag["redirect_chain"].append({"status": h.status_code, "url": h.url})
        diag["redirect_chain"].append({"status": r.status_code, "url": r.url})
    else:
        diag["redirect_chain"] = [{"status": r.status_code, "url": r.url}]

    # Sample headers (pour debug sans spam)
    keep = ["Content-Type", "Content-Length", "Content-Disposition", "Cache-Control", "ETag", "Last-Modified"]
    diag["response_headers_sample"] = {k: r.headers.get(k) for k in keep if r.headers.get(k) is not None}

    r.raise_for_status()

    # Si c'est JSON (parfois l'API renvoie un JSON qui contient un lien), on gère
    ctype = (r.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        data = r.json()
        # Tentatives de trouver une URL dans la réponse JSON (selon structures possibles)
        possible = []
        if isinstance(data, dict):
            for key in ["url", "download_url", "latest", "href"]:
                if key in data and isinstance(data[key], str):
                    possible.append(data[key])
            # Structures imbriquées fréquentes
            for path in [("resource", "url"), ("resource", "download_url"), ("data", "url")]:
                cur = data
                ok = True
                for p in path:
                    if isinstance(cur, dict) and p in cur:
                        cur = cur[p]
                    else:
                        ok = False
                        break
                if ok and isinstance(cur, str):
                    possible.append(cur)

        possible = [u for u in possible if u.startswith("http")]
        if possible:
            # On retélécharge le binaire via l'URL trouvée
            url2 = possible[0]
            diag["json_followup_url"] = url2
            t1 = time.time()
            r2 = requests.get(url2, timeout=timeout, headers=UA, allow_redirects=True, stream=stream)
            diag["json_followup_elapsed_seconds"] = round(time.time() - t1, 3)
            diag["json_followup_final_url"] = r2.url
            diag["json_followup_status"] = r2.status_code
            r2.raise_for_status()
            content = r2.content
            diag["downloaded_bytes"] = len(content)
            return content, diag
        else:
            raise RuntimeError("Réponse JSON reçue mais impossible d’en extraire une URL de téléchargement.")

    # Sinon: binaire direct
    content = r.content
    diag["downloaded_bytes"] = len(content)
    return content, diag


def download_to_cache(url: str, filename: str, max_age_days: int = 120) -> Tuple[Path, Dict]:
    """
    Télécharge vers /tmp si absent ou trop vieux.
    Retourne:
      - path fichier
      - diag téléchargement (ou diag "cache hit")
    """
    path = CACHE_DIR / filename

    if path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return path, {
                "cache": "HIT",
                "path": str(path),
                "age_days": round(age_days, 2),
                "size_bytes": path.stat().st_size,
            }

    content, diag = http_get_with_diagnostics(url)
    path.write_bytes(content)
    diag.update({"cache": "MISS->SAVED", "path": str(path)})
    return path, diag


# -----------------------------
# Commune selection (geo.api.gouv.fr)
# -----------------------------
@st.cache_data(ttl=24 * 3600)
def search_communes(query: str) -> List[dict]:
    url = "https://geo.api.gouv.fr/communes"
    params = {"nom": query, "fields": "nom,code,codeDepartement", "boost": "population", "limit": 20}
    r = requests.get(url, params=params, timeout=20, headers=UA)
    r.raise_for_status()
    return r.json()


# -----------------------------
# INSEE "âge" loader (XLSX)
# -----------------------------
@st.cache_data(ttl=7 * 24 * 3600, show_spinner=True)
def load_age_xlsx(path: str) -> pd.DataFrame:
    # lecture Excel (une seule fois grâce au cache)
    # engine openpyxl auto via pandas
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_code_insee_column(df: pd.DataFrame) -> str:
    """
    Détection robuste de la colonne code INSEE.
    """
    candidates = []
    for c in df.columns:
        cl = str(c).lower()
        if "cod" in cl and ("geo" in cl or "géo" in cl or "insee" in cl):
            candidates.append(c)
    # fallback usuels
    for name in ["CODGEO", "Code", "code", "code_insee", "INSEE Code géographique", "Code géographique"]:
        if name in df.columns:
            return name
    if candidates:
        return candidates[0]
    # dernier recours: première colonne
    return df.columns[0]


# -----------------------------
# UI
# -----------------------------
st.title("Structure d’âge – INSEE brut par commune (via data.gouv)")

with st.expander("🧪 Diagnostic (à ouvrir en cas de lenteur/erreur)", expanded=False):
    st.write("Cette app télécharge un fichier (~38 Mo) une seule fois, le met en cache /tmp, puis filtre la commune.")
    st.write(f"Cache dir: `{CACHE_DIR}`")
    st.write(f"URL utilisée: `{AGE_URL}`")

colL, colR = st.columns([0.45, 0.55], gap="large")

with colL:
    st.subheader("1) Choisir la commune")
    q = st.text_input("Nom de commune", value="Dinan")
    if not q.strip():
        st.stop()

    communes = []
    try:
        with st.spinner("Recherche commune..."):
            communes = search_communes(q.strip())
    except Exception as e:
        st.error(f"Erreur geo.api.gouv.fr : {e}")
        st.stop()

    if not communes:
        st.warning("Aucune commune trouvée.")
        st.stop()

    options = {f"{c['nom']} ({c['code']})": c["code"] for c in communes}
    label = st.selectbox("Sélection", list(options.keys()))
    code_insee = options[label]
    st.success(f"Commune sélectionnée : {label}")

    st.subheader("2) Télécharger + lire le fichier âges")
    run = st.button("📦 Charger les données âges (brut)")

    show_diag = st.checkbox("Afficher le diagnostic détaillé", value=True)

with colR:
    st.subheader("Résultat")

    if not run:
        st.info("Clique **Charger les données âges** pour lancer le téléchargement/lecture (une seule fois puis cache).")
        st.stop()

    # 1) download (cached)
    try:
        path, diag = download_to_cache(AGE_URL, filename="ages_communes.xlsx", max_age_days=120)
    except Exception as e:
        st.error("Échec téléchargement.")
        st.exception(e)
        st.stop()

    if show_diag:
        st.markdown("### Diagnostic téléchargement / cache")
        st.json(diag)

    # 2) read (cached)
    try:
        df = load_age_xlsx(str(path))
    except Exception as e:
        st.error("Téléchargement OK mais lecture Excel KO (format inattendu ou fichier non-xlsx).")
        st.exception(e)
        # petit indice utile
        st.write("Astuce: regarde `content_type` dans le diagnostic, si c’est du HTML/JSON au lieu d’un XLSX.")
        st.stop()

    # 3) detect code column + filter
    code_col = detect_code_insee_column(df)
    df[code_col] = df[code_col].astype(str).str.zfill(5)

    out = df[df[code_col] == code_insee].copy()

    st.markdown("### Données brutes (ligne(s) de la commune)")
    st.caption(f"Colonne code INSEE détectée : `{code_col}` | lignes trouvées : {len(out)}")
    st.dataframe(out, width="stretch", height=420)

    if len(out) == 0:
        st.warning(
            "Aucune ligne trouvée pour ce code INSEE dans le fichier.\n"
            "Ca arrive si le fichier n’est pas au niveau 'communes' (ou si la colonne code n’est pas détectée correctement)."
        )

    # 4) export brut
    st.download_button(
        "⬇️ Télécharger CSV (brut) pour cette commune",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name=f"ages_brut_{code_insee}.csv",
        mime="text/csv",
    )

    # 5) aperçu colonnes (utile pour comprendre le fichier)
    with st.expander("Voir les colonnes du fichier (pour debug)", expanded=False):
        st.write(df.columns.tolist())
