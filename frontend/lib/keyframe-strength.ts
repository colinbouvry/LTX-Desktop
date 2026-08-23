/** Full i2v lock. Matches the backend KeyframeInput default; the UI owns the value. */
export const DEFAULT_KEYFRAME_STRENGTH = 1

/** Persist, restore, and HTTP all go through this so a bad float cannot 422 or fail project parse. */
export function clampKeyframeStrength(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) return DEFAULT_KEYFRAME_STRENGTH
  return Math.min(1, Math.max(0, value))
}
