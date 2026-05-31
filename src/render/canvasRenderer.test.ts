import { describe, expect, test } from 'vitest';
import { CanvasRenderer, computeTile } from './canvasRenderer';
import { PacmanEnvironment, createDefaultEnv } from '../env/environment';

// Minimal CanvasRenderingContext2D stand-in: records fillRect calls (the per-frame
// black clear + wall fills) so we can detect whether draw() actually repainted.
const makeCtx = (clientWidth = 600) => {
  const calls = { fillRect: 0 };
  const canvas = { width: 0, height: 0, parentElement: { clientWidth } };
  const ctx = {
    canvas,
    fillStyle: '', strokeStyle: '', lineWidth: 0,
    fillRect: () => { calls.fillRect += 1; },
    beginPath: () => {}, arc: () => {}, fill: () => {}, moveTo: () => {},
    lineTo: () => {}, stroke: () => {}, quadraticCurveTo: () => {}, closePath: () => {},
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
};

describe('computeTile (D6.6)', () => {
  test('scales container width by the 0.5625 height-fit factor', () => {
    // (200-20)/10 * 0.5625 = 18 * 0.5625 = 10.125 → floor 10
    expect(computeTile(10, 200)).toBe(10);
    // (600-20)/28 * 0.5625 = 20.714… * 0.5625 = 11.65… → floor 11
    expect(computeTile(28, 600)).toBe(11);
  });

  test('clamps to a 6px floor for tiny containers', () => {
    expect(computeTile(100, 50)).toBe(6);
    expect(computeTile(28, 0)).toBe(6);
  });
});

describe('CanvasRenderer.draw', () => {
  test('paints on first draw and sizes the canvas', () => {
    const { ctx, calls } = makeCtx();
    const env = createDefaultEnv();
    new CanvasRenderer(ctx).draw(env, false);
    expect(calls.fillRect).toBeGreaterThan(0);
    expect(ctx.canvas.width).toBeGreaterThan(0);
    expect(ctx.canvas.height).toBeGreaterThan(0);
  });

  // D6.1: toggling showHeatmap on an otherwise-unchanged frame must repaint.
  // Pre-fix the hash omitted showHeatmap, so the second draw would be skipped.
  test('repaints when showHeatmap toggles even if game state is unchanged (D6.1)', () => {
    const { ctx, calls } = makeCtx();
    const env = createDefaultEnv();
    const r = new CanvasRenderer(ctx);
    r.draw(env, false);              // first paint
    const afterFirst = calls.fillRect;
    r.draw(env, true);               // only showHeatmap changed
    expect(calls.fillRect).toBeGreaterThan(afterFirst);
  });

  // D7.4: the Q-value overlay tints open tiles and repaints when values change
  // (qSig is part of the render-skip hash).
  test('renders a Q-value overlay and repaints when values change (D7.4)', () => {
    const { ctx, calls } = makeCtx();
    const env = createDefaultEnv();
    const { width, height } = env.world;
    const mk = (val: number): (number | null)[][] =>
      Array.from({ length: height }, () => Array.from({ length: width }, () => val as number | null));
    const r = new CanvasRenderer(ctx);

    r.draw(env, false, mk(0.5));
    const afterFirst = calls.fillRect;
    expect(afterFirst).toBeGreaterThan(0);

    r.draw(env, false, mk(0.9)); // different values → qSig changes → repaint
    expect(calls.fillRect).toBeGreaterThan(afterFirst);
  });

  // D6.9: rendering must not assume a single Pac-Man — numPacmen can be 1–4.
  test('draws all Pac-Men without throwing at numPacmen=4 (D6.9)', () => {
    const { ctx, calls } = makeCtx();
    const env = new PacmanEnvironment();
    env.setParams({ numPacmen: 4, numGhosts: 2 });
    env.reset(42);
    const r = new CanvasRenderer(ctx);
    expect(() => r.draw(env, false)).not.toThrow();
    expect(calls.fillRect).toBeGreaterThan(0);
    expect(env.getPacmen()).toHaveLength(4);
  });
});
