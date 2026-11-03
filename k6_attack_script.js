// ============================================================================
// PREDATOR-NET — Cross-Machine Attack Demonstration Script (k6)
// ============================================================================
// Run this script from ANOTHER PC / Laptop on the same Wi-Fi / Local Network:
//
// 1. Install k6 on the attacker machine (if needed):
//    Windows: winget install k6   OR   choco install k6
//    Mac:     brew install k6
//    Linux:   sudo apt install k6
//
// 2. Ensure TARGET_IP below matches your Host laptop's Wi-Fi IP (shown in GUI header)
// 3. Execute:
//    k6 run k6_attack_script.js
// ============================================================================

import http from 'k6/http';
import { check, sleep } from 'k6';

// Host IP address on your Wi-Fi LAN (shown in GUI header):
const TARGET_IP = '10.5.128.179';
const TARGET_PORT = '8080';

export const options = {
  stages: [
    { duration: '5s', target: 20 },   // Stage 1: Initial Reconnaissance & probe
    { duration: '45s', target: 80 },  // Stage 2: Sustained Volumetric HTTP Flood attack
    { duration: '10s', target: 120 }, // Stage 3: High-concurrency peak burst
    { duration: '5s', target: 0 },    // Stage 4: Ramp down
  ],
  thresholds: {
    // Before blocking: 100% requests succeed
    // After clicking "Block IP in Firewall": requests fail with 403 Forbidden!
    http_req_failed: ['rate<0.5'],
  },
};

export default function () {
  const url = `http://${TARGET_IP}:${TARGET_PORT}/api/demonstration-attack`;
  
  const payload = JSON.stringify({
    source: 'External_Attacker_Laptop_k6',
    attack_type: 'Volumetric_HTTP_Flood',
    iteration: __ITER,
    virtual_user: __VU,
    timestamp: Date.now(),
    payload_buffer: 'X'.repeat(512),
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'k6-Volumetric-DDoS-Tool/2.0 (High-Concurrency Attack)',
      'X-Attack-Vector': 'Volumetric-HTTP-Flood',
    },
  };

  const res = http.post(url, payload, params);

  check(res, {
    'request allowed (pre-mitigation 200 OK)': (r) => r.status === 200,
    'firewall active block (403 Forbidden)': (r) => r.status === 403,
  });

  // Rapid request pacing to maintain high throughput (>100 req/s)
  sleep(0.02);
}
