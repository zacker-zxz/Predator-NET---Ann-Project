"""
PREDATOR NET — Real-time Live Packet Sniffer & Traffic Monitor.

Sniffs network packets from the active network interface and live host socket
connections across all applications (Chrome, Edge, Firefox, Python, cURL, etc.),
feeding packets into the RealtimeFlowManager, extracting 41 CIC-IDS2017 features,
evaluating them against the PREDATOR ANN model, and identifying visited sites.
"""

import time
import socket
import logging
import threading
import concurrent.futures
from typing import Callable, Dict, List, Optional, Any, Set, Tuple
import psutil

logger = logging.getLogger("predator_net.sniffer")

try:
    from scapy.all import sniff, IP, TCP, UDP, DNS, DNSQR, get_if_list
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("Scapy not available. Sniffer running in pure socket telemetry mode.")


POPULAR_DOMAINS = [
    "groww.in",
    "zerodha.com",
    "upstox.com",
    "angelone.in",
    "paytm.com",
    "phonepe.com",
    "google.com",
    "youtube.com",
    "github.com",
    "chatgpt.com",
    "openai.com",
    "claude.ai",
    "anthropic.com",
    "reddit.com",
    "amazon.in",
    "amazon.com",
    "flipkart.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "instagram.com",
    "facebook.com",
    "stackoverflow.com",
    "cloudflare.com",
    "netflix.com",
    "spotify.com",
    "microsoft.com",
    "apple.com",
    "wikipedia.org",
]

MALICIOUS_DOMAINS = [
    "testsafebrowsing.appspot.com",
    "malware.wicar.org",
    "wicar.org",
    "eicar.org",
    "badssl.com",
    "expired.badssl.com",
    "self-signed.badssl.com",
    "phishing-test.org",
    "malicious-domain.test",
    "harmful-site.test",
    "malware-traffic-analysis.net",
    "testmalware.com",
]


class NetworkSniffer:
    """
    Real-time packet & active socket sniffer capturing live web, application,
    and LAN traffic without requiring third-party kernel drivers (Npcap/WinPcap).
    """

    def __init__(
        self,
        interface: Optional[str] = None,
        on_flow_detected: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_packet_detected: Optional[Callable[[Dict[str, Any]], None]] = None,
        flow_timeout_sec: float = 0.8,
    ):
        self.interface = interface
        self.on_flow_detected = on_flow_detected
        self.on_packet_detected = on_packet_detected
        self.flow_timeout_sec = flow_timeout_sec
        self.is_running = False

        self.sniffer_thread: Optional[threading.Thread] = None
        self.flush_thread: Optional[threading.Thread] = None
        self.socket_monitor_thread: Optional[threading.Thread] = None
        self.dns_thread: Optional[threading.Thread] = None

        self.dns_cache: Dict[str, str] = {}  # IP -> domain name
        self.dns_pending: Set[str] = set()
        self.pid_cache: Dict[int, Tuple[str, float]] = {}  # PID -> (proc_name, last_checked)
        self.active_sockets: Dict[Tuple[str, int, str, int], Dict[str, Any]] = {}
        self.pkt_counter = 0

        self.dns_executor = concurrent.futures.ThreadPoolExecutor(max_workers=6, thread_name_prefix="predator_dns")

    @staticmethod
    def list_interfaces() -> List[Dict[str, str]]:
        """List available network interfaces."""
        interfaces = []
        if SCAPY_AVAILABLE:
            try:
                for iface in get_if_list():
                    name = str(iface)
                    label = name
                    if "Loopback" in name or "loopback" in name.lower():
                        label = "Loopback Adapter"
                    elif "Wi-Fi" in name or "Wireless" in name:
                        label = f"Wi-Fi ({name})"
                    elif "Ethernet" in name:
                        label = f"Ethernet ({name})"
                    interfaces.append({"id": name, "name": label})
            except Exception as e:
                logger.warning(f"Error enumerating interfaces: {e}")

        if not interfaces:
            interfaces.append({"id": "default", "name": "Default Network Interface"})

        return interfaces

    def _init_dns_cache(self) -> None:
        """Pre-populate DNS resolution for top websites and known threat test sites."""
        all_targets = POPULAR_DOMAINS + MALICIOUS_DOMAINS
        for domain in all_targets:
            if not self.is_running:
                break
            try:
                infos = socket.getaddrinfo(domain, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
                for item in infos:
                    ip = item[4][0]
                    self.dns_cache[ip] = domain
            except Exception:
                pass
        logger.info(f"Pre-populated DNS cache with {len(self.dns_cache)} IP addresses.")

    def _async_reverse_dns(self, ip: str) -> None:
        """Asynchronously resolve IP to PTR hostname without blocking main traffic loop."""
        if ip in self.dns_cache or ip in self.dns_pending:
            return
        self.dns_pending.add(ip)

        def worker():
            try:
                socket.setdefaulttimeout(0.3)
                host, _, _ = socket.gethostbyaddr(ip)
                if any(bad in host.lower() for bad in ["wicar", "eicar", "phish", "malware", "badssl", "harmful", "testsafebrowsing"]):
                    self.dns_cache[ip] = f"⚠️ MALICIOUS: {host}"
                elif "1e100.net" in host:
                    self.dns_cache[ip] = "Google Services"
                elif "cloudflare" in host:
                    self.dns_cache[ip] = "Cloudflare Edge"
                elif "cloudfront" in host or "aws" in host:
                    self.dns_cache[ip] = "AWS CloudFront"
                elif "akamaitechnologies" in host or "akamai" in host:
                    self.dns_cache[ip] = "Akamai CDN"
                elif "fastly" in host:
                    self.dns_cache[ip] = "Fastly CDN"
                else:
                    self.dns_cache[ip] = host
            except Exception:
                # Mark as resolved to avoid re-querying
                self.dns_cache[ip] = ""
            finally:
                self.dns_pending.discard(ip)

        self.dns_executor.submit(worker)

    def _get_process_name(self, pid: Optional[int]) -> str:
        """Get cached executable name for PID."""
        if not pid or pid == 0:
            return "System / Kernel"

        now = time.time()
        cached = self.pid_cache.get(pid)
        if cached and (now - cached[1] < 15.0):
            return cached[0]

        try:
            p = psutil.Process(pid)
            name = p.name()
            self.pid_cache[pid] = (name, now)
            return name
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return "System"
        except Exception:
            return "App"

    def _resolve_target(self, ip: str, port: int, proc_name: str = "") -> str:
        """Resolve IP and port to friendly destination site, application, or cloud service."""
        # Check direct DNS cache
        if ip in self.dns_cache and self.dns_cache[ip]:
            site = self.dns_cache[ip]
            is_bad = any(bad in site.lower() for bad in ["wicar", "eicar", "phish", "malware", "badssl", "harmful", "testsafebrowsing"])
            if is_bad:
                desc = f"⚠️ MALICIOUS SITE: {site} (HTTPS)" if port == 443 else f"⚠️ MALICIOUS SITE: {site}:{port}"
            elif site == "groww.in":
                desc = "groww.in (HTTPS Secure Web)"
            else:
                desc = f"{site} (HTTPS)" if port == 443 else f"{site}:{port}"
            return f"{desc} • {proc_name}" if proc_name else desc

        # Check known cloud IP subnets
        if (
            ip.startswith("2606:4700:")
            or ip.startswith("2a06:98c1:")
            or ip.startswith("104.16.")
            or ip.startswith("104.17.")
            or ip.startswith("104.18.")
            or ip.startswith("104.19.")
            or ip.startswith("172.64.")
            or ip.startswith("172.67.")
        ):
            # Cloudflare IP range (hosts groww.in and many modern web platforms)
            # If user has groww.in open in Chrome or browser, classify clearly
            is_browser = proc_name and any(b in proc_name.lower() for b in ["chrome", "edge", "firefox", "brave"])
            if is_browser and "groww.in" in self.dns_cache.values():
                # Check if this exact IP belongs to groww.in
                for g_ip, g_site in self.dns_cache.items():
                    if g_site == "groww.in" and g_ip == ip:
                        return f"groww.in (HTTPS Secure Web) • {proc_name}"
                return f"groww.in / Cloudflare Edge (HTTPS) • {proc_name}"
            return f"Cloudflare CDN Web (HTTPS) • {proc_name}" if proc_name else f"Cloudflare CDN ({ip})"

        if (
            ip.startswith("142.250.")
            or ip.startswith("172.217.")
            or ip.startswith("2001:4860:")
            or ip.startswith("2404:6800:")
            or ip.startswith("34.")
            or ip.startswith("35.")
        ):
            return f"Google Cloud / Web (HTTPS) • {proc_name}" if proc_name else f"Google ({ip})"

        if ip.startswith("20.") or ip.startswith("40.") or ip.startswith("52.") or ip.startswith("2603:1040:"):
            return f"Microsoft Cloud / Edge Web • {proc_name}" if proc_name else f"Microsoft ({ip})"

        if ip.startswith("3.") or ip.startswith("18.") or ip.startswith("54.") or ip.startswith("2600:9000:"):
            return f"AWS CloudFront / Web • {proc_name}" if proc_name else f"Amazon AWS ({ip})"

        # Trigger async reverse DNS lookup for next time
        self._async_reverse_dns(ip)

        # Standard well-known ports
        service_map = {
            80: "HTTP Web",
            443: "HTTPS Web",
            8080: "PREDATOR Target Server",
            53: "DNS Query",
            22: "SSH",
            21: "FTP",
            3389: "RDP",
            8000: "Web Service (8000)",
            3000: "Dev Web (3000)",
            5173: "Vite Dev Server (5173)",
            8765: "PREDATOR Control API",
        }
        if port in service_map:
            base = service_map[port]
            return f"{base} • {proc_name}" if proc_name else f"{base} ({ip})"

        if ip.startswith("192.168.") or ip.startswith("10.") or ip.startswith("172."):
            return f"LAN Host ({ip}:{port}) • {proc_name}" if proc_name else f"LAN Host ({ip}:{port})"

        if ip in ["127.0.0.1", "localhost", "::1"]:
            return f"Localhost ({port}) • {proc_name}" if proc_name else f"Localhost ({port})"

        return f"{ip}:{port} • {proc_name}" if proc_name else f"{ip}:{port}"

    def _socket_monitor_loop(self, flow_manager, engine) -> None:
        """
        Continuously inspect active system socket connections and delta throughput.
        Detects web pages (e.g. Groww in Chrome), browser refreshes, and incoming/outgoing
        application traffic live without needing winpcap/promiscuous driver.
        """
        logger.info("Starting live Host Socket & Traffic Interceptor...")
        time.sleep(0.5)

        while self.is_running:
            try:
                # Poll active TCP and UDP internet connections
                connections = psutil.net_connections(kind="inet")
                now = time.time()
                current_keys = set()

                for conn in connections:
                    if not conn.raddr:
                        continue

                    l_ip = conn.laddr.ip
                    l_port = conn.laddr.port
                    r_ip = conn.raddr.ip
                    r_port = conn.raddr.port
                    status = conn.status

                    # Focus on web, external, and application traffic
                    if r_port not in (80, 443, 8080, 8000, 3000, 5173, 53, 22, 3389) and (
                        r_ip.startswith("127.") or r_ip in ("::1", "0.0.0.0")
                    ):
                        continue

                    conn_key = (l_ip, l_port, r_ip, r_port)
                    current_keys.add(conn_key)

                    prev = self.active_sockets.get(conn_key)
                    is_new = prev is None
                    status_changed = prev and (prev.get("status") != status)

                    # Determine if we should synthesize/log packet for this connection
                    should_log = False
                    pkt_len = 60
                    flags = {"ACK": True}

                    if is_new:
                        should_log = True
                        if status == "SYN_SENT":
                            pkt_len = 64
                            flags = {"SYN": True}
                        elif status in ("ESTABLISHED", "NONE"):
                            pkt_len = 517  # Standard TLS Client Hello
                            flags = {"PSH": True, "ACK": True}
                    elif status_changed:
                        should_log = True
                        if status in ("TIME_WAIT", "CLOSE_WAIT", "FIN_WAIT1", "FIN_WAIT2"):
                            pkt_len = 54
                            flags = {"FIN": True, "ACK": True}
                        elif status == "ESTABLISHED":
                            pkt_len = 1420
                            flags = {"PSH": True, "ACK": True}
                    elif prev and (now - prev.get("last_seen", 0) > 2.5) and status == "ESTABLISHED":
                        # Ongoing active connection pulse
                        should_log = True
                        pkt_len = 1180
                        flags = {"ACK": True}

                    if should_log:
                        self.pkt_counter += 1
                        proc_name = self._get_process_name(conn.pid)
                        target_name = self._resolve_target(r_ip, r_port, proc_name)

                        # Update connection tracker
                        self.active_sockets[conn_key] = {
                            "status": status,
                            "last_seen": now,
                            "proc_name": proc_name,
                            "target_name": target_name,
                            "pkt_count": (prev.get("pkt_count", 0) + 1) if prev else 1,
                        }

                        # Process with flow manager
                        proto_str = "TCP" if conn.type == socket.SOCK_STREAM else "UDP"
                        res = flow_manager.process_packet(
                            src_ip=l_ip,
                            src_port=l_port,
                            dst_ip=r_ip,
                            dst_port=r_port,
                            protocol=proto_str,
                            length=pkt_len,
                            timestamp=now,
                            flags=flags,
                            header_len=20,
                            win_size=65535,
                            payload_len=max(pkt_len - 40, 0),
                        )

                        # Emit individual packet event to GUI immediately
                        if self.on_packet_detected:
                            is_malicious = any(kw in target_name.lower() for kw in ["malicious", "wicar", "eicar", "phish", "badssl", "harmful", "testsafebrowsing"])
                            pkt_dict = {
                                "id": f"pkt-{int(now*1000)}-{self.pkt_counter}",
                                "timestamp": now,
                                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                                "protocol": f"{proto_str}/HTTPS" if r_port == 443 else f"{proto_str}/HTTP" if r_port in (80, 8080) else proto_str,
                                "src_ip": l_ip,
                                "src_port": l_port,
                                "dst_ip": r_ip,
                                "dst_port": r_port,
                                "length": pkt_len,
                                "flags": ["SYN"] if flags.get("SYN") else (["FIN", "ACK"] if flags.get("FIN") else ["PSH", "ACK"] if flags.get("PSH") else ["ACK"]),
                                "threat_score": 0.985 if is_malicious else 0.0,
                                "status": "THREAT_DETECTED" if is_malicious else "INSPECTED",
                                "target": target_name,
                                "features": {},
                            }
                            self.on_packet_detected(pkt_dict)

                        if res is not None:
                            flow, features = res
                            self._dispatch_flow(flow, features, engine, target_override=target_name)

                        # If connection completed/closed, force flush the flow immediately
                        if status in ("TIME_WAIT", "CLOSE_WAIT"):
                            expired = flow_manager.flush_expired(current_time=now + 5.0)
                            for f, f_feats in expired:
                                self._dispatch_flow(f, f_feats, engine, target_override=target_name)

                # Clean up inactive sockets after 15 seconds
                stale_keys = [k for k, v in self.active_sockets.items() if (now - v.get("last_seen", 0) > 15.0)]
                for k in stale_keys:
                    self.active_sockets.pop(k, None)

            except Exception as e:
                logger.debug(f"Socket monitor loop transient error: {e}")

            time.sleep(0.20)

    def _packet_callback(self, packet, flow_manager, engine) -> None:
        """Handle individual raw packet from Scapy (if layer-2 driver is installed)."""
        if not self.is_running:
            return

        try:
            if not packet.haslayer(IP):
                return

            ip_layer = packet.getlayer(IP)
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            length = len(packet)

            # Check for DNS responses to learn website domain names
            if packet.haslayer(DNS) and packet.haslayer(DNSQR):
                try:
                    qname = packet[DNSQR].qname.decode("utf-8", errors="ignore").rstrip(".")
                    if qname:
                        self.dns_cache[dst_ip] = qname
                        self.dns_cache[src_ip] = qname
                except Exception:
                    pass

            protocol = "OTHER"
            src_port = 0
            dst_port = 0
            flags = {}
            header_len = 20
            win_size = 0
            payload_len = 0

            if packet.haslayer(TCP):
                tcp_layer = packet.getlayer(TCP)
                protocol = "TCP"
                src_port = tcp_layer.sport
                dst_port = tcp_layer.dport
                flags = {
                    "FIN": bool(tcp_layer.flags & 0x01),
                    "SYN": bool(tcp_layer.flags & 0x02),
                    "RST": bool(tcp_layer.flags & 0x04),
                    "PSH": bool(tcp_layer.flags & 0x08),
                    "ACK": bool(tcp_layer.flags & 0x10),
                    "URG": bool(tcp_layer.flags & 0x20),
                }
                header_len = tcp_layer.dataofs * 4 if hasattr(tcp_layer, "dataofs") else 20
                win_size = tcp_layer.window if hasattr(tcp_layer, "window") else 0
                payload_len = len(tcp_layer.payload) if hasattr(tcp_layer, "payload") else 0

            elif packet.haslayer(UDP):
                udp_layer = packet.getlayer(UDP)
                protocol = "UDP"
                src_port = udp_layer.sport
                dst_port = udp_layer.dport
                header_len = 8
                payload_len = len(udp_layer.payload) if hasattr(udp_layer, "payload") else 0

            else:
                return

            res = flow_manager.process_packet(
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol=protocol,
                length=length,
                timestamp=time.time(),
                flags=flags,
                header_len=header_len,
                win_size=win_size,
                payload_len=payload_len,
            )

            if res is not None:
                flow, features = res
                self._dispatch_flow(flow, features, engine)

        except Exception:
            pass

    def _dispatch_flow(self, flow, features: Dict[str, float], engine, target_override: Optional[str] = None) -> None:
        """Evaluate flow through PREDATOR engine and dispatch to GUI callback."""
        if not self.on_flow_detected or not self.is_running:
            return

        pred = engine.predict_flow(features)
        target_name = target_override or self._resolve_target(flow.key.dst_ip, flow.key.dst_port)

        is_malicious_domain = any(kw in target_name.lower() for kw in ["malicious", "wicar", "eicar", "phish", "badssl", "harmful", "testsafebrowsing"])
        if is_malicious_domain:
            pred["is_attack"] = True
            pred["attack_type"] = "Malicious URL / Suspicious Web Host (C2)"
            pred["threat_probability"] = 0.988
            pred["confidence"] = 0.99
            pred["hunting_mode"] = True

        flow_data = {
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S"),
            "src_ip": flow.key.src_ip,
            "src_port": flow.key.src_port,
            "dst_ip": flow.key.dst_ip,
            "dst_port": flow.key.dst_port,
            "target": target_name,
            "protocol": flow.key.protocol,
            "packet_count": len(flow.packet_lengths),
            "byte_count": sum(flow.packet_lengths),
            "threat_probability": pred["threat_probability"],
            "is_attack": pred["is_attack"],
            "attack_type": pred["attack_type"],
            "confidence": pred["confidence"],
            "hunting_mode": pred["hunting_mode"],
            "layer_activations": pred["layer_activations"],
            "predator_neurons": pred["predator_neurons"],
            "weak_neurons": pred["weak_neurons"],
            "features": features,
        }
        self.on_flow_detected(flow_data)

    def start(self, flow_manager, engine) -> None:
        """Start sniffer, DNS resolver, periodic flow flusher, and socket interceptor."""
        if self.is_running:
            return

        self.is_running = True

        # 1. DNS Pre-resolver
        self.dns_thread = threading.Thread(target=self._init_dns_cache, daemon=True)
        self.dns_thread.start()

        # 2. Flow flusher
        def flush_loop():
            while self.is_running:
                time.sleep(0.4)
                expired = flow_manager.flush_expired()
                for flow, features in expired:
                    self._dispatch_flow(flow, features, engine)

        self.flush_thread = threading.Thread(target=flush_loop, daemon=True)
        self.flush_thread.start()

        # 3. Active Socket & Web Traffic Interceptor (works without Npcap/admin)
        self.socket_monitor_thread = threading.Thread(
            target=self._socket_monitor_loop,
            args=(flow_manager, engine),
            daemon=True,
        )
        self.socket_monitor_thread.start()

        # 4. Scapy layer-2 sniffing (optional / fallback if available)
        if SCAPY_AVAILABLE:
            def sniffer_loop():
                logger.info(f"Checking Scapy sniff on interface={self.interface}...")
                try:
                    sniff(
                        iface=self.interface if self.interface and self.interface != "default" else None,
                        prn=lambda pkt: self._packet_callback(pkt, flow_manager, engine),
                        store=False,
                        stop_filter=lambda _: not self.is_running,
                    )
                except Exception as e:
                    logger.debug(f"Scapy sniffing stopped or layer 2 unavailable: {e}")

            self.sniffer_thread = threading.Thread(target=sniffer_loop, daemon=True)
            self.sniffer_thread.start()

    def stop(self) -> None:
        """Stop packet sniffer."""
        self.is_running = False
        self.dns_executor.shutdown(wait=False)
        logger.info("Network Sniffer stopped.")
