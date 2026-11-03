"""
PREDATOR NET — Unified Desktop Cockpit Launcher.

Starts the Python AI Network Defense Engine on port 8765 and launches
the Electron Desktop GUI application with GPU acceleration.
"""

import os
import sys
import time
import socket
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_port(port: int, timeout: int = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.5)
    return False

def main():
    print("=" * 70)
    print("  PREDATOR-NET — AI Neural Firewall & Antivirus Desktop Cockpit")
    print("=" * 70)

    # 1. Clean ambient environment to ensure Electron launches as a true native desktop window
    clean_env = os.environ.copy()
    clean_env.pop("ELECTRON_RUN_AS_NODE", None)
    clean_env.pop("ELECTRON_NO_ATTACH_CONSOLE", None)

    # 2. Start Python AI Backend Service if not already running
    backend_proc = None
    if not is_port_in_use(8765):
        print("[1/3] Starting Python AI Engine (FastAPI + WebSockets + Sniffer)...")
        server_script = PROJECT_ROOT / "src" / "predator_net" / "service" / "server.py"
        backend_proc = subprocess.Popen(
            [sys.executable, str(server_script)],
            cwd=str(PROJECT_ROOT),
        )
        if not wait_for_port(8765, timeout=12):
            print("[-] Warning: Backend service took longer than expected to bind port 8765.")
        else:
            print("[+] Python AI Engine successfully online on http://127.0.0.1:8765")
            print("[+] Target Victim Server online on http://0.0.0.0:8080 (Ready for k6 attacks)")
    else:
        print("[+] Python AI Engine is already active on port 8765.")

    # 3. Check if desktop-app build exists
    dist_dir = PROJECT_ROOT / "desktop-app" / "dist"
    if not dist_dir.exists():
        print("[2/3] Building Desktop GUI bundle...")
        subprocess.run(["npm", "run", "build"], cwd=str(PROJECT_ROOT / "desktop-app"), shell=True)
    else:
        print("[2/3] Verified Desktop GUI bundle.")

    # 4. Launch Native Electron Desktop GUI Window
    print("[3/3] Opening PREDATOR-NET Native Desktop Application Window...")
    try:
        gui_proc = subprocess.run(
            "npm run electron",
            cwd=str(PROJECT_ROOT / "desktop-app"),
            env=clean_env,
            shell=True,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if backend_proc:
            print("\nClosing background AI engine...")
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=3)
            except Exception:
                backend_proc.kill()
        print("Desktop application closed. Perimeter secured.")

if __name__ == "__main__":
    main()
