// Web Audio API Synthesizer for PREDATOR-NET
// Plain, calm, subtle acoustic cues (non-annoying, low-frequency, throttled)

let audioCtx: AudioContext | null = null;

// Read persistence from localStorage if available
function getInitialSoundEnabled(): boolean {
  try {
    const val = localStorage.getItem('predator_sound_enabled');
    return val !== null ? val === 'true' : true;
  } catch {
    return true;
  }
}

function getInitialSoundVolume(): number {
  try {
    const val = localStorage.getItem('predator_sound_volume');
    return val !== null ? Math.min(1, Math.max(0, parseFloat(val))) : 0.40;
  } catch {
    return 0.40;
  }
}

let soundEnabled = getInitialSoundEnabled();
let soundVolume = getInitialSoundVolume();

// Throttling timestamps to prevent rapid-fire stuttering or annoying audio spam
let lastNeuronTickTime = 0;
let lastThreatAlertTime = 0;
let lastBlockSuccessTime = 0;

function getAudioContext(): AudioContext | null {
  if (!soundEnabled) return null;
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
}

export function setSoundEnabled(enabled: boolean) {
  soundEnabled = enabled;
  try {
    localStorage.setItem('predator_sound_enabled', String(enabled));
  } catch {}
}

export function isSoundEnabled(): boolean {
  return soundEnabled;
}

export function setSoundVolume(volume: number) {
  soundVolume = Math.min(1, Math.max(0, volume));
  try {
    localStorage.setItem('predator_sound_volume', String(soundVolume));
  } catch {}
}

export function getSoundVolume(): number {
  return soundVolume;
}

/**
 * Plain and simple neuron hit / flow inspection tap:
 * Replaces harsh 880Hz electronic chirp with an ultra-subtle, warm, low-frequency acoustic haptic tap
 * (160Hz -> 85Hz, 18ms duration, soft low-pass filter, throttled to 160ms interval).
 */
export function playTactileBeep() {
  playNeuronTick();
}

export function playNeuronTick() {
  if (!soundEnabled || soundVolume <= 0) return;
  const now = performance.now();
  if (now - lastNeuronTickTime < 160) {
    return; // Throttle to prevent machine-gun audio spam during heavy traffic bursts
  }
  lastNeuronTickTime = now;

  const ctx = getAudioContext();
  if (!ctx) return;

  try {
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    // Warm, rounded low-pass filter removes any sharp click/chirp
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(450, t);

    // Muted low-frequency pitch drop: 160Hz down to 80Hz (wood/haptic acoustic tap)
    osc.type = 'sine';
    osc.frequency.setValueAtTime(160, t);
    osc.frequency.exponentialRampToValueAtTime(80, t + 0.022);

    // Ultra-soft volume envelope (very low gain, quick exponential decay)
    const peakGain = 0.018 * soundVolume;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.linearRampToValueAtTime(peakGain, t + 0.003);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.022);

    osc.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc.start(t);
    osc.stop(t + 0.025);
  } catch (e) {
    // Ignore audio context errors
  }
}

/**
 * Plain and simple threat alert tone:
 * Replaces screaming square/sawtooth alarm with a discreet, mellow two-tone enterprise notification
 * (Dual sine 320Hz -> 250Hz, warm envelope, throttled to 1000ms).
 */
export function playThreatAlert() {
  if (!soundEnabled || soundVolume <= 0) return;
  const now = performance.now();
  if (now - lastThreatAlertTime < 1000) {
    return; // Don't spam alert sound during continuous floods
  }
  lastThreatAlertTime = now;

  const ctx = getAudioContext();
  if (!ctx) return;

  try {
    const t = ctx.currentTime;
    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gain = ctx.createGain();
    const filter = ctx.createBiquadFilter();

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(900, t);

    // Muted low-mid two-tone prompt
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(329.63, t); // E4
    osc1.frequency.setValueAtTime(261.63, t + 0.09); // C4

    osc2.type = 'triangle';
    osc2.frequency.setValueAtTime(164.81, t); // E3 sub-harmonic
    osc2.frequency.setValueAtTime(130.81, t + 0.09);

    const peakGain = 0.045 * soundVolume;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.linearRampToValueAtTime(peakGain, t + 0.01);
    gain.gain.setValueAtTime(peakGain * 0.8, t + 0.09);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.22);

    osc1.connect(filter);
    osc2.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    osc1.start(t);
    osc2.start(t);
    osc1.stop(t + 0.23);
    osc2.stop(t + 0.23);
  } catch (e) {}
}

/**
 * Plain and simple block confirmation tone:
 * Gentle, clean ascending chime (F4 -> A4) fading out softly.
 */
export function playBlockSuccess() {
  if (!soundEnabled || soundVolume <= 0) return;
  const now = performance.now();
  if (now - lastBlockSuccessTime < 250) return;
  lastBlockSuccessTime = now;

  const ctx = getAudioContext();
  if (!ctx) return;

  try {
    const t = ctx.currentTime;
    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gain = ctx.createGain();

    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(349.23, t); // F4
    osc1.frequency.exponentialRampToValueAtTime(440.00, t + 0.08); // A4

    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(523.25, t + 0.05); // C5 accent

    const peakGain = 0.035 * soundVolume;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.linearRampToValueAtTime(peakGain, t + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(ctx.destination);

    osc1.start(t);
    osc2.start(t + 0.05);
    osc1.stop(t + 0.19);
    osc2.stop(t + 0.19);
  } catch (e) {}
}
