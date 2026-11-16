
import os
import gdown
from pathlib import Path

from config import DB_PATH


def ensure_database_exists():

    if DB_PATH.exists():
        return

    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    file_id = os.environ.get("DB_FILE_ID")

    if not file_id:
        raise RuntimeError(
            "Missing DB_FILE_ID configuration."
        )

    url = f"https://drive.google.com/uc?id={file_id}"

    gdown.download(
        url,
        str(DB_PATH),
        quiet=False
    )

    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        raise RuntimeError(
            "Database download failed."
        )
