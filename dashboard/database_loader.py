
import streamlit as st
import gdown

from config import DB_PATH


def ensure_database_exists():
    if DB_PATH.exists() and DB_PATH.stat().st_size > 0:
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    file_id = st.secrets.get("DB_FILE_ID")

    if not file_id:
        raise RuntimeError("DB_FILE_ID absent des Secrets Streamlit.")

    url = f"https://drive.google.com/uc?id={file_id}"

    downloaded_file = gdown.download(
        url,
        str(DB_PATH),
        quiet=False
    )

    if not downloaded_file or not DB_PATH.exists():
        raise RuntimeError("Impossible de télécharger la base DuckDB.")

    if DB_PATH.stat().st_size == 0:
        raise RuntimeError("La base DuckDB téléchargée est vide.")
