const BASE_URL = 'http://127.0.0.1:8765';

export async function fetchStatus() {
  try {
    const res = await fetch(`${BASE_URL}/api/status`);
    return await res.json();
  } catch (e) {
    return null;
  }
}

export async function fetchSystemTelemetry() {
  try {
    const res = await fetch(`${BASE_URL}/api/system-telemetry`);
    return await res.json();
  } catch (e) {
    return null;
  }
}

export async function fetchPackets() {
  try {
    const res = await fetch(`${BASE_URL}/api/packets`);
    return await res.json();
  } catch (e) {
    return { packets: [], total_packets: 0 };
  }
}

export async function fetchInterfaces() {
  try {
    const res = await fetch(`${BASE_URL}/api/interfaces`);
    return await res.json();
  } catch (e) {
    return [];
  }
}

export async function setSensitivity(threshold: number) {
  const res = await fetch(`${BASE_URL}/api/sensitivity`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ threshold }),
  });
  return await res.json();
}

export async function blockIP(ip: string) {
  const res = await fetch(`${BASE_URL}/api/block-ip`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ip }),
  });
  return await res.json();
}

export async function unblockIP(ip: string) {
  const res = await fetch(`${BASE_URL}/api/unblock-ip`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ip }),
  });
  return await res.json();
}

export async function simulateAttack(attack_type: string, intensity: number = 50) {
  const res = await fetch(`${BASE_URL}/api/simulate-attack`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ attack_type, intensity }),
  });
  return await res.json();
}

export async function fetchK6Script() {
  const res = await fetch(`${BASE_URL}/api/k6-script`);
  return await res.json();
}

export async function fetchHistory() {
  try {
    const res = await fetch(`${BASE_URL}/api/history`);
    return await res.json();
  } catch (e) {
    return { recent_flows: [], recent_packets: [], active_alerts: [], blocked_ips: [] };
  }
}
