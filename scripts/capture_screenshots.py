"""
PREDATOR NET — Automated High-Resolution Screenshot Capture Utility
Uses Playwright Chromium to capture screenshots of the Desktop Cockpit UI
for the repository README and technical documentation.
"""

import os
import sys
import time
import socket
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "assets" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_port(port: int, timeout: int = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(0.4)
    return False

def main():
    print("[*] Starting backend service & static web server...")
    backend_proc = None
    web_proc = None

    try:
        if not is_port_in_use(8765):
            server_script = PROJECT_ROOT / "src" / "predator_net" / "service" / "server.py"
            backend_proc = subprocess.Popen(
                [sys.executable, str(server_script)],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not wait_for_port(8765, timeout=12):
                print("[-] Warning: Backend took longer than expected to bind port 8765.")
            else:
                print("[+] Python AI Engine listening on :8765")

        # Start static server for desktop-app/dist on port 4173
        dist_dir = PROJECT_ROOT / "desktop-app" / "dist"
        if not dist_dir.exists():
            print("[*] Building desktop-app bundle...")
            subprocess.run(["npm", "run", "build"], cwd=str(PROJECT_ROOT / "desktop-app"), shell=True)

        web_proc = subprocess.Popen(
            [sys.executable, "-m", "http.server", "4173", "--directory", str(dist_dir)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_for_port(4173, timeout=10):
            raise RuntimeError("Static web server failed to start on :4173")
        print("[+] Web UI server listening on :4173")

        with sync_playwright() as p:
            print("[*] Launching Chromium (1440x900 viewport)...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                device_scale_factor=2,  # Crisp retina 2x capture
            )
            page = context.new_page()

            # 1. Main Dashboard (Neural Defense 2D)
            print("[*] Capturing 01: Main Neural Defense Cockpit...")
            page.goto("http://127.0.0.1:4173/", wait_until="networkidle")
            time.sleep(3.5)  # Allow WebSocket connection & canvas simulation pulses to render
            page.screenshot(path=str(SCREENSHOTS_DIR / "01_neural_defense_dashboard.png"), full_page=False)

            # 2. Packets & Network Page
            print("[*] Capturing 02: Packets & Network Inspector...")
            page.locator("button:has-text('Packets & Network')").click()
            time.sleep(2.0)
            # Click on first packet row to show the deep feature dissection modal
            first_packet = page.locator("table tbody tr").first
            if first_packet.count() > 0:
                first_packet.click()
                time.sleep(1.0)
            page.screenshot(path=str(SCREENSHOTS_DIR / "02_packet_inspector.png"), full_page=False)

            # Close modal if opened
            close_btn = page.locator("button:has-text('Close')")
            if close_btn.count() > 0:
                close_btn.click()
                time.sleep(0.5)

            # 3. Algorithm & Math Specification Page
            print("[*] Capturing 03: Algorithm & Math Specifications...")
            page.locator("button:has-text('Algorithm & Math')").click()
            time.sleep(2.0)
            page.screenshot(path=str(SCREENSHOTS_DIR / "03_algorithm_mathematics.png"), full_page=False)

            # 4. Attack Hub (k6)
            print("[*] Capturing 04: Attack Hub (k6 Flood Demonstration)...")
            page.locator("button:has-text('Attack Hub (k6)')").click()
            time.sleep(2.0)
            page.screenshot(path=str(SCREENSHOTS_DIR / "04_k6_attack_hub.png"), full_page=False)

            # 5. Live Firewall & Threat Mitigation
            print("[*] Capturing 05: Live Firewall & Defense Monitor...")
            page.locator("button:has-text('Live Firewall')").click()
            time.sleep(2.0)
            page.screenshot(path=str(SCREENSHOTS_DIR / "05_live_firewall_mitigation.png"), full_page=False)

            # 6. System Operations View
            print("[*] Capturing 06: System Operations Telemetry...")
            page.locator("button:has-text('System Operations')").click()
            time.sleep(2.0)
            page.screenshot(path=str(SCREENSHOTS_DIR / "06_system_operations.png"), full_page=False)

            browser.close()
            print("[+] All screenshots successfully captured in assets/screenshots/")

    finally:
        if web_proc:
            web_proc.terminate()
        if backend_proc:
            backend_proc.terminate()

if __name__ == "__main__":
    main()
