/**
 * Behavioral biometrics collection SDK (plan 3.2).
 *
 * Collects timing/shape statistics only:
 *   - keystroke: press duration and inter-key interval distributions
 *   - mouse: pointer velocity and direction entropy
 *   - touch: pressure, swipe speed and angle
 *
 * It never records typed text, key codes, element values or raw pointer paths.
 * Only per-session aggregates are retained and sent.
 *
 * Usage:
 *   import { hashSubjectId, createCollector } from "./behavioral-biometrics.js";
 *   const subjectRef = await hashSubjectId(customerId, window.__FRAUD_SALT);
 *   const collector = createCollector({ subjectRef, consent: userConsent });
 *   collector.attach();
 *   // at payment time:
 *   await collector.flush("/api/fraud/behavior");
 *
 * `hashSubjectId` requires a secure context (HTTPS or localhost) because it
 * uses Web Crypto. The salt must come from the server, never be hard-coded.
 */

const SCHEMA = "behavioral.session.v1";
const SDK_VERSION = "1.0.0";

const MAX_SAMPLES = 5000;
const MAX_ANGLE_BINS = 8;

function mean(values) {
  if (values.length === 0) return 0;
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

function stddev(values) {
  if (values.length === 0) return 0;
  const mu = mean(values);
  let variance = 0;
  for (const value of values) variance += (value - mu) ** 2;
  return Math.sqrt(variance / values.length);
}

function boundedPush(values, value, limit = MAX_SAMPLES) {
  values.push(value);
  if (values.length > limit) values.shift();
}

/** Shannon entropy (bits) of movement direction, 0 = perfectly straight. */
function directionEntropy(angles) {
  if (angles.length === 0) return 0;
  const bins = new Array(MAX_ANGLE_BINS).fill(0);
  for (const angle of angles) {
    const normalized = (angle + Math.PI) / (2 * Math.PI);
    const bin = Math.min(MAX_ANGLE_BINS - 1, Math.floor(normalized * MAX_ANGLE_BINS));
    bins[bin] += 1;
  }
  let entropy = 0;
  for (const count of bins) {
    if (count === 0) continue;
    const p = count / angles.length;
    entropy -= p * Math.log2(p);
  }
  return entropy;
}

function normalizeAngle(angle) {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}

/** Salted SHA-256 pseudonym; the raw customer id never leaves the device. */
export async function hashSubjectId(subjectId, salt = "") {
  if (!subjectId) throw new Error("subjectId is required");
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto unavailable; behavioral biometrics require a secure context");
  }
  const bytes = new TextEncoder().encode(`${salt}:${subjectId}`);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  const hex = Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
  return `sha256:${hex}`;
}

function emptyCollector(subjectRef, sessionId) {
  const payload = () => ({
    schema: SCHEMA,
    sdk_version: SDK_VERSION,
    subject_ref: subjectRef || "",
    session_id: sessionId,
    collected_at: new Date().toISOString(),
    consent: { behavioral_biometrics: false },
    keystroke: { key_hold_mean_ms: 0, key_hold_std_ms: 0, inter_key_ms: 0, samples: 0 },
    mouse: {
      velocity_px_s: 0,
      entropy: 0,
      clicks: 0,
      click_x_mean_norm: 0,
      click_y_mean_norm: 0,
      samples: 0,
    },
    touch: { pressure_mean: 0, speed_px_s: 0, swipe_angle_deg: 0, samples: 0 },
  });
  return { attach() {}, detach() {}, snapshot: payload, flush: async () => false };
}

export function createCollector({
  subjectRef,
  sessionId = globalThis.crypto?.randomUUID?.() || `sess-${Date.now()}`,
  consent = true,
} = {}) {
  if (!consent) return emptyCollector(subjectRef, sessionId);
  if (!subjectRef || !subjectRef.startsWith("sha256:")) {
    throw new Error("subjectRef must be the salted hash returned by hashSubjectId()");
  }

  let attached = false;
  const holdDurations = [];
  const interKeyIntervals = [];
  const pointerSpeeds = [];
  const pointerAngles = [];
  const touchPressures = [];
  const touchSpeeds = [];
  const touchAngles = [];
  const clickXs = [];
  const clickYs = [];
  const pressed = new Map();
  let lastKeyDownAt = null;
  let lastPointer = null;
  let lastTouch = null;
  let target = null;

  function onKeyDown(event) {
    const t = event.timeStamp;
    if (pressed.has(event.code)) return;
    pressed.set(event.code, t);
    if (lastKeyDownAt !== null && t >= lastKeyDownAt) {
      boundedPush(interKeyIntervals, t - lastKeyDownAt);
    }
    lastKeyDownAt = t;
  }

  function onKeyUp(event) {
    const start = pressed.get(event.code);
    if (start === undefined) return;
    pressed.delete(event.code);
    const duration = event.timeStamp - start;
    if (duration >= 0) boundedPush(holdDurations, duration);
  }

  function onPointerMove(event) {
    const point = { t: event.timeStamp, x: event.clientX, y: event.clientY };
    if (lastPointer) {
      const dt = point.t - lastPointer.t;
      const dx = point.x - lastPointer.x;
      const dy = point.y - lastPointer.y;
      const distance = Math.hypot(dx, dy);
      if (dt > 0) boundedPush(pointerSpeeds, (distance / dt) * 1000);
      if (distance > 0) boundedPush(pointerAngles, normalizeAngle(Math.atan2(dy, dx)));
    }
    lastPointer = point;
  }

  function onClick(event) {
    // Normalised viewport coordinates only; never element content or DOM paths.
    const width = globalThis.innerWidth || target?.clientWidth || 1;
    const height = globalThis.innerHeight || target?.clientHeight || 1;
    const clamp = (value) => Math.min(1, Math.max(0, value));
    boundedPush(clickXs, clamp(event.clientX / width));
    boundedPush(clickYs, clamp(event.clientY / height));
  }

  function onTouchMove(event) {
    const touch = event.touches?.[0];
    if (!touch) return;
    const point = {
      t: event.timeStamp,
      x: touch.clientX,
      y: touch.clientY,
      pressure: typeof touch.force === "number" ? touch.force : 0,
    };
    boundedPush(touchPressures, point.pressure);
    if (lastTouch) {
      const dt = point.t - lastTouch.t;
      const dx = point.x - lastTouch.x;
      const dy = point.y - lastTouch.y;
      const distance = Math.hypot(dx, dy);
      if (dt > 0) boundedPush(touchSpeeds, (distance / dt) * 1000);
      if (distance > 0) boundedPush(touchAngles, normalizeAngle(Math.atan2(dy, dx)));
    }
    lastTouch = point;
  }

  function snapshot() {
    return {
      schema: SCHEMA,
      sdk_version: SDK_VERSION,
      subject_ref: subjectRef,
      session_id: sessionId,
      collected_at: new Date().toISOString(),
      consent: { behavioral_biometrics: true },
      keystroke: {
        key_hold_mean_ms: mean(holdDurations),
        key_hold_std_ms: stddev(holdDurations),
        inter_key_ms: mean(interKeyIntervals),
        samples: holdDurations.length,
      },
      mouse: {
        velocity_px_s: mean(pointerSpeeds),
        entropy: directionEntropy(pointerAngles),
        clicks: clickXs.length,
        click_x_mean_norm: mean(clickXs),
        click_y_mean_norm: mean(clickYs),
        samples: pointerSpeeds.length,
      },
      touch: {
        pressure_mean: mean(touchPressures),
        speed_px_s: mean(touchSpeeds),
        swipe_angle_deg: touchAngles.length ? (mean(touchAngles) * 180) / Math.PI : 0,
        samples: touchSpeeds.length,
      },
    };
  }

  function attach(attachTarget = globalThis) {
    if (attached) return;
    target = attachTarget;
    target.addEventListener("keydown", onKeyDown, { passive: true });
    target.addEventListener("keyup", onKeyUp, { passive: true });
    target.addEventListener("mousemove", onPointerMove, { passive: true });
    target.addEventListener("click", onClick, { passive: true });
    target.addEventListener("touchmove", onTouchMove, { passive: true });
    attached = true;
  }

  function detach() {
    if (!attached || !target) return;
    target.removeEventListener("keydown", onKeyDown);
    target.removeEventListener("keyup", onKeyUp);
    target.removeEventListener("mousemove", onPointerMove);
    target.removeEventListener("click", onClick);
    target.removeEventListener("touchmove", onTouchMove);
    attached = false;
  }

  async function flush(endpoint) {
    if (!endpoint) return false;
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(snapshot()),
      keepalive: true,
    });
    return response.ok;
  }

  return { attach, detach, snapshot, flush };
}
