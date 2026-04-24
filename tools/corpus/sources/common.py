from __future__ import annotations

import json
import shutil
import ssl
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen


ANC_INSECURE_HOSTS = {"anc.org", "www.anc.org"}
MANIFEST_NAME = "corpus_manifest.json"


def _ssl_context(url: str) -> ssl.SSLContext | None:
    host = (urlparse(url).hostname or "").lower()
    if host in ANC_INSECURE_HOSTS:
        # The ANC site currently serves an expired TLS certificate. We only
        # relax verification for these official hosts so corpus automation can
        # still work without requiring manual downloads.
        return ssl._create_unverified_context()  # noqa: SLF001
    return None


def fetch_bytes(url: str, *, retries: int = 3, delay_seconds: float = 1.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            context = _ssl_context(url)
            with urlopen(url, context=context, timeout=60) as response:  # noqa: S310
                return response.read()
        except (HTTPError, URLError, TimeoutError, ssl.SSLError) as exc:
            last_error = exc
            if attempt + 1 == retries:
                break
            time.sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def download_file(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(fetch_bytes(url))
    return target


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_text_file(src: Path, dst: Path) -> bool:
    text = src.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return True


def extract_archive(archive_path: Path, output_dir: Path) -> Path:
    reset_dir(output_dir)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(output_dir)
        return output_dir
    if archive_path.suffix == ".tgz" or archive_path.name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(output_dir)
        return output_dir
    raise ValueError(f"Unsupported archive format: {archive_path}")


def stage_archive(url: str, archive_dir: Path, filename: str) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / filename
    if not archive_path.exists():
        download_file(url, archive_path)
    return archive_path


def with_temporary_dir() -> tempfile.TemporaryDirectory[str]:
    return tempfile.TemporaryDirectory(prefix="decipher_corpus_")


def manifest_path(corpus_dir: Path) -> Path:
    return corpus_dir / MANIFEST_NAME


def load_manifest(corpus_dir: Path) -> dict:
    path = manifest_path(corpus_dir)
    if not path.exists():
        return {"sources": []}
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(corpus_dir: Path, *, language: str, source_entry: dict) -> Path:
    path = manifest_path(corpus_dir)
    payload = load_manifest(corpus_dir)
    payload["language"] = language
    payload.setdefault("sources", [])
    payload["sources"] = [entry for entry in payload["sources"] if entry.get("name") != source_entry.get("name")]
    payload["sources"].append(source_entry)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
