"""
PREDATOR NET — Live Victim Target Server for Attack Demonstration.

Runs a lightweight HTTP server on port 8080 (0.0.0.0) designed specifically
to receive network traffic from external PCs (e.g., k6 scripts, curl, siege, wrk)
or local test attacks, calculating packet flow metrics and forwarding them
instantly into the PREDATOR live detection pipeline.
"""

import time
import socket
import logging
import threading
from typing import Callable, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler

logger = logging.getLogger("predator_net.victim")


class AttackTargetHandler(BaseHTTPRequestHandler):
    """Handles incoming HTTP traffic and reports flows to PREDATOR engine."""

    # Reference to callback provided by VictimServer
    flow_callback: Optional[Callable] = None
    blacklist_checker: Optional[Callable[[str], bool]] = None
    total_requests: int = 0
    recent_hit_times: list = []

    def log_message(self, format, *args):
        """Suppress default HTTP stderr logs to keep console clean."""
        pass

    def _is_client_blocked(self, client_ip: str) -> bool:
        if AttackTargetHandler.blacklist_checker is not None:
            try:
                return bool(AttackTargetHandler.blacklist_checker(client_ip))
            except Exception as e:
                logger.error(f"Error checking blacklist for {client_ip}: {e}")
        return False

    def _record_hit(self, is_blocked: bool = False, is_harmful_trigger: bool = False):
        now = time.time()
        AttackTargetHandler.total_requests += 1
        AttackTargetHandler.recent_hit_times.append(now)

        # Keep last 500 hit timestamps for rate computation
        if len(AttackTargetHandler.recent_hit_times) > 500:
            AttackTargetHandler.recent_hit_times = AttackTargetHandler.recent_hit_times[-500:]

        # Calculate instantaneous requests per second
        recent = [t for t in AttackTargetHandler.recent_hit_times if now - t <= 1.0]
        reqs_per_sec = len(recent)

        # Extract client details
        client_ip = self.client_address[0]
        client_port = self.client_address[1]
        server_ip = self.server.server_address[0]
        if server_ip in ["0.0.0.0", ""]:
            server_ip = "127.0.0.1"
        server_port = self.server.server_address[1]

        content_len = 0
        if "Content-Length" in self.headers:
            try:
                content_len = int(self.headers["Content-Length"])
            except ValueError:
                content_len = 0

        pkt_len = 120 + content_len

        # If callback registered, feed into flow pipeline
        if AttackTargetHandler.flow_callback is not None:
            AttackTargetHandler.flow_callback(
                src_ip=client_ip,
                src_port=client_port,
                dst_ip=server_ip,
                dst_port=server_port,
                protocol="TCP",
                length=pkt_len,
                reqs_per_sec=reqs_per_sec,
                path=self.path,
                method=self.command,
                is_blocked=is_blocked,
                is_harmful_trigger=is_harmful_trigger,
            )

    def do_GET(self):
        client_ip = self.client_address[0]
        is_harmful = any(kw in self.path.lower() for kw in ["harmful", "malware", "phish", "exploit", "attack", "eicar", "threat"])

        if self._is_client_blocked(client_ip):
            self._record_hit(is_blocked=True, is_harmful_trigger=is_harmful)
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Firewall-Status", "BLOCKED")
            self.send_header("X-Firewall-Mitigation", "PREDATOR-NET-ACTIVE-DEFENSE")
            self.end_headers()
            body = b'{"status": "BLOCKED", "error": "Access Denied by PREDATOR-NET ANN Firewall", "mitigated": true, "code": 403}\n'
            self.wfile.write(body)
            return

        self._record_hit(is_blocked=False, is_harmful_trigger=is_harmful)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if is_harmful:
            body = b'{"status": "WARNING", "alert": "Malicious payload simulated and captured by PREDATOR-NET"}\n'
        else:
            body = b'{"status": "protected_by_predator_net", "victim_active": true}\n'
        self.wfile.write(body)

    def do_POST(self):
        client_ip = self.client_address[0]
        is_harmful = any(kw in self.path.lower() for kw in ["harmful", "malware", "phish", "exploit", "attack", "eicar", "threat"])

        if self._is_client_blocked(client_ip):
            self._record_hit(is_blocked=True, is_harmful_trigger=is_harmful)
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("X-Firewall-Status", "BLOCKED")
            self.send_header("X-Firewall-Mitigation", "PREDATOR-NET-ACTIVE-DEFENSE")
            self.end_headers()
            body = b'{"status": "BLOCKED", "error": "Access Denied by PREDATOR-NET ANN Firewall", "mitigated": true, "code": 403}\n'
            self.wfile.write(body)
            return

        self._record_hit(is_blocked=False, is_harmful_trigger=is_harmful)
        # Read body if present
        if "Content-Length" in self.headers:
            try:
                length = int(self.headers["Content-Length"])
                self.rfile.read(min(length, 65536))
            except Exception:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        body = b'{"status": "received", "firewall": "predator_monitoring"}\n'
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()


class VictimServer:
    """Multi-threaded target server to catch attacks from other PCs."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, on_flow_packet: Optional[Callable] = None):
        self.host = host
        self.port = port
        self.on_flow_packet = on_flow_packet
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.is_running = False

    def set_blacklist_checker(self, checker: Callable[[str], bool]) -> None:
        """Register a dynamic firewall blacklist checker function."""
        AttackTargetHandler.blacklist_checker = checker

    def start(self) -> None:
        if self.is_running:
            return

        AttackTargetHandler.flow_callback = self.on_flow_packet
        try:
            self.server = HTTPServer((self.host, self.port), AttackTargetHandler)
            self.is_running = True
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            logger.info(f"Target Victim Server started on http://{self.host}:{self.port} (Ready for k6 attacks)")
        except Exception as e:
            logger.error(f"Failed to bind Victim Server to port {self.port}: {e}")

    def stop(self) -> None:
        if not self.is_running or not self.server:
            return
        self.is_running = False
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass
        logger.info("Target Victim Server stopped.")

    @staticmethod
    def get_local_lan_ip() -> str:
        """Find the local machine's IP address on the physical LAN so other PCs can connect."""
        import psutil
        try:
            # 1. Prioritize Wi-Fi or Wireless interface
            addrs = psutil.net_if_addrs()
            for name, addr_list in addrs.items():
                if any(w in name.lower() for w in ["wi-fi", "wireless", "wlan"]):
                    for a in addr_list:
                        if a.family == socket.AF_INET and not a.address.startswith("169.254."):
                            return a.address

            # 2. Check Ethernet physical interfaces
            for name, addr_list in addrs.items():
                if "ethernet" in name.lower() and not any(v in name.lower() for v in ["virtual", "vbox", "vmware"]):
                    for a in addr_list:
                        if a.family == socket.AF_INET and not a.address.startswith("169.254.") and not a.address.startswith("192.168.56."):
                            return a.address

            # 3. Fallback to outbound socket connection
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

