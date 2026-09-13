"""Start the HealthBandhu web app on Google Colab (or any Linux machine) with a public link.

From a notebook cell:
    import runpy
    runpy.run_path("scripts/colab_launch.py")["launch"](port=8501, tunnel=True)

From a terminal:
    python scripts/colab_launch.py              # Streamlit + Cloudflare tunnel
    python scripts/colab_launch.py --no-tunnel  # local only

Fixes over the old launcher: it waits for Streamlit to report healthy instead of
sleeping 8 seconds, downloads cloudflared only once, never blocks forever on the
tunnel output, and writes logs to tools/ for troubleshooting.
"""

from __future__ import annotations

import argparse
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
TUNNEL_PATTERN = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-lines:])
    except OSError:
        return ""


def ensure_requirements() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Installing app requirements ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(ROOT / "requirements.txt")],
                       check=True)


def is_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def wait_until_healthy(port: int, process: subprocess.Popen, timeout: float = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_healthy(port):
            return True
        if process.poll() is not None:
            return False
        time.sleep(1.5)
    return False


def start_streamlit(port: int) -> subprocess.Popen:
    log = open(TOOLS / "streamlit.log", "w")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "App.py"),
         "--server.port", str(port), "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def ensure_cloudflared() -> Path:
    found = shutil.which("cloudflared")
    if found:
        return Path(found)
    target = TOOLS / "cloudflared"
    if not target.exists():
        print("      Downloading cloudflared (one time) ...")
        urllib.request.urlretrieve(CLOUDFLARED_URL, target)
        target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return target


def start_tunnel(binary: Path, port: int, timeout: float = 90):
    log_path = TOOLS / "cloudflared.log"
    log = open(log_path, "w")
    process = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        match = TUNNEL_PATTERN.search(log_path.read_text(errors="ignore"))
        if match:
            return process, match.group(0)
        if process.poll() is not None:
            break
        time.sleep(1)
    return process, None


def launch(port: int = 8501, tunnel: bool = True):
    TOOLS.mkdir(exist_ok=True)
    ensure_requirements()
    if not (ROOT / "models" / "healthbandhu_symptoms.json").exists():
        print("Note: no trained models in models/. The app will open on its setup instructions.")

    print(f"[1/3] Starting Streamlit on port {port} ...")
    if is_healthy(port):
        print("      Already running, reusing it.")
    else:
        app = start_streamlit(port)
        if not wait_until_healthy(port, app):
            print("Streamlit did not start. Last log lines:")
            print(_tail(TOOLS / "streamlit.log"))
            return None
    local_url = f"http://localhost:{port}"
    print(f"      Local URL : {local_url}")
    if not tunnel:
        return local_url

    print("[2/3] Preparing cloudflared ...")
    binary = ensure_cloudflared()
    print("[3/3] Opening public tunnel ...")
    _, public_url = start_tunnel(binary, port)
    print("=" * 60)
    if public_url:
        print(f"HealthBandhu is live: {public_url}")
        print("The link works while this runtime stays connected.")
    else:
        print("Tunnel URL not found yet. Last log lines:")
        print(_tail(TOOLS / "cloudflared.log"))
    print("=" * 60)
    return public_url


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Launch the HealthBandhu Streamlit app.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--no-tunnel", action="store_true", help="start locally without a public link")
    args = parser.parse_args(argv)
    launch(port=args.port, tunnel=not args.no_tunnel)


if __name__ == "__main__":
    main()
