import { describe, expect, test } from 'vitest';
import { CanvasRenderer, computeTile } from './canvasRenderer';
import { PacmanEnvironment, createDefaultEnv } from '../env/environment';
import { toAction } from '../engine/types';

// Minimal CanvasRenderingContext2D stand-in: records fillRect calls (the per-frame
// black clear + wall fills) so we can detect whether draw() actually repainted.
const makeCtx = (clientWidth = 600, clientHeight = 0) => {
  const calls = { fillRect: 0 };
  const canvas = { width: 0, height: 0, parentElement: { clientWidth, clientHeight } };
  const ctx = {
    canvas,
    fillStyle: '', strokeStyle: '', lineWidth: 0,
    fillRect: () => { calls.fillRect += 1; },
    beginPath: () => {}, arc: () => {}, fill: () => {}, moveTo: () => {},
    lineTo: () => {}, stroke: () => {}, quadraticCurveTo: () => {}, closePath: () => {},
  };
  return { ctx: ctx as unknown as CanvasRenderingContext2D, calls };
};

describe('computeTile (D6.6, D6.7)', () => {
  test('fits both axes — tile is the min of the width-fit and height-fit (D6.7)', () => {
    // 28×31 maze in an 820×440 box: byWidth=(820-20)/28=28.57, byHeight=(440-20)/31=13.5
    // → height-constrained → floor 13
    expect(computeTile(28, 31, 820, 440)).toBe(13);
    // 10×10 maze in a tall 220×800 box: byWidth=20, byHeight=78 → width-constrained → 20
    expect(computeTile(10, 10, 220, 800)).toBe(20);
  });

  test('falls back to width-only fit when container height is unknown (D6.7)', () => {
    // height=0 → byHeight ignored → (600-20)/28 = 20.71 → floor 20
    expect(computeTile(28, 31, 600, 0)).toBe(20);
  });

  test('clamps to a 6px floor for tiny containers', () => {
    expect(computeTile(100, 100, 50, 50)).toBe(6);
    expect(computeTile(28, 31, 0, 0)).toBe(6);
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

  // D6.10: the persisted renderer must recompute its tile when the maze's column
  // count changes, not keep a tile sized for the previous maze.
  test('recomputes tile size when the maze width changes (D6.10)', () => {
    const { ctx } = makeCtx(600);
    const r = new CanvasRenderer(ctx);
    const env = new PacmanEnvironment();
    env.setParams({ mazeId: 'pacman-classic' }); // 28 columns
    env.reset(42);
    r.draw(env, false);
    const wide = env.world.width;
    const tileWide = ctx.canvas.width / wide;
    expect(tileWide).toBe(computeTile(wide, env.world.height, 600, 0));

    env.setParams({ mazeId: 'corridors' }); // 17 columns, same container
    env.reset(42);
    r.draw(env, false);
    const narrow = env.world.width;
    const tileNarrow = ctx.canvas.width / narrow;
    // Recomputed for the new width (fewer columns → larger tile), not stale.
    expect(tileNarrow).toBe(computeTile(narrow, env.world.height, 600, 0));
    expect(tileNarrow).not.toBe(tileWide);
  });

  // D6.7: when the container height is the binding constraint (wide, short box),
  // the tile is sized to fit the height so the maze doesn't overflow vertically.
  test('sizes the tile to the container height when height is the constraint (D6.7)', () => {
    const { ctx } = makeCtx(2000, 400); // very wide, short container
    const r = new CanvasRenderer(ctx);
    const env = new PacmanEnvironment();
    env.setParams({ mazeId: 'pacman-classic' });
    env.reset(42);
    r.draw(env, false);
    const { width, height } = env.world;
    const tile = ctx.canvas.width / width;
    expect(tile).toBe(computeTile(width, height, 2000, 400));
    // height-bound: byHeight=(400-20)/height < byWidth=(2000-20)/width
    expect(tile).toBe(Math.max(6, Math.floor((400 - 20) / height)));
  });

  // H12 guard: pacmen can be momentarily empty during env.reset(); draw must
  // bail rather than throw inside the React effect.
  test('does not throw when pacmen is transiently empty (H12)', () => {
    const { ctx } = makeCtx();
    const env = createDefaultEnv();
    (env as unknown as { pacmen: unknown[] }).pacmen = [];
    const r = new CanvasRenderer(ctx);
    expect(() => r.draw(env, false)).not.toThrow();
  });

  // D6.13: pac 0's mouth faces its travel direction. We can't read canvas angles
  // through the stub, but exercising all four headings covers DIR_ANGLE and the
  // getPacLastDir integration without throwing.
  test('faces pac 0 mouth by direction without throwing (D6.13)', () => {
    const { ctx } = makeCtx();
    const env = new PacmanEnvironment();
    env.setParams({ numGhosts: 0 });
    env.reset(42);
    const r = new CanvasRenderer(ctx);
    for (const a of [0, 1, 2, 3].map(toAction)) {
      env.step(a);
      expect(() => r.draw(env, false)).not.toThrow();
    }
  });
});
