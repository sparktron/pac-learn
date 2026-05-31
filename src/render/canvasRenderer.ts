import type { PacmanEnvironment } from '../env/environment';
import { MAZES } from '../mazes/mazes';

const GHOST_COLORS = ['#ef4444', '#f472b6', '#38bdf8', '#fb923c', '#a78bfa', '#34d399'];
const EDIBLE_COLOR = '#93c5fd';
const EDIBLE_FLASH_COLOR = '#ffffff';
const PAC_COLOR = '#facc15';

/**
 * Tile size (px) for a maze of the given column count inside a container of the
 * given pixel width. Subtracts 20px of padding, scales by 0.5625 (a height-fit
 * heuristic for the project's ~16:9 maze proportions), and clamps to a 6px floor
 * so tiny containers stay legible. Pure + exported so it can be unit-tested.
 */
export const computeTile = (width: number, containerWidth: number): number =>
  Math.max(6, Math.floor(((containerWidth - 20) / width) * 0.5625));

export class CanvasRenderer {
  private frameCount = 0;
  private lastHash = '';
  private lastContainerWidth = 0;

  constructor(private ctx: CanvasRenderingContext2D, private tile = 0) {}

  /**
   * @param qOverlay  Optional per-tile value grid (`number | null`, indexed
   *   [y][x]); when provided, each open tile is tinted by its normalized value
   *   (blue = low, green = high). Used for the "Q-Values" view.
   */
  draw(env: PacmanEnvironment, showHeatmap: boolean, qOverlay?: (number | null)[][]): void {
    this.frameCount = (this.frameCount + 1) % 3600;

    // Transient state during env.reset() can leave pacmen empty for a frame;
    // guard so the renderer doesn't throw inside a React effect and kill
    // further redraws.
    const pacmen = env.getPacmen();
    if (pacmen.length === 0) return;

    // Recompute tile when the container width changes (sidebar collapse,
    // window resize). Computing once was enough to avoid layout thrash but
    // froze the canvas at its first-paint size — invalidate on width drift
    // greater than 1 px.
    const { width, height, pellets, powerPellets, heatmap, isWall } = env.world;
    const containerWidth = this.ctx.canvas.parentElement?.clientWidth ?? width * 20;
    if (this.tile === 0 || Math.abs(containerWidth - this.lastContainerWidth) > 1) {
      this.tile = computeTile(width, containerWidth);
      this.lastContainerWidth = containerWidth;
      this.lastHash = ''; // force a full redraw at the new size
    }

    // Normalize the Q-value overlay once (min/max over finite values) and build
    // a cheap signature so the hash repaints when Q-values shift between frames.
    let qMin = Infinity;
    let qMax = -Infinity;
    let qSum = 0;
    if (qOverlay) {
      for (const row of qOverlay) {
        for (const v of row) {
          if (v === null || !Number.isFinite(v)) continue;
          if (v < qMin) qMin = v;
          if (v > qMax) qMax = v;
          qSum += v;
        }
      }
    }
    const qSpan = Math.max(0.0001, qMax - qMin);

    // Compute hash of game state; skip render if unchanged. D6.1: showHeatmap is
    // a render-relevant input, so a toggle on an otherwise-static frame must
    // repaint. D6.9: hash every Pac-Man's position, not just pacmen[0].
    // qOverlay: include a coarse value signature so a learning Q-table repaints.
    const pacHash = pacmen.map((p) => `${p.pos.x},${p.pos.y}`).join('/');
    const qSig = qOverlay ? `q${qSum.toFixed(1)}` : '';
    const hash = `${env.stepCount}:${env.pelletsLeft}:${showHeatmap ? 1 : 0}:${qSig}:${env.ghosts.map((g) => `${g.pos.x},${g.pos.y}`).join('|')}:${pacHash}`;
    if (hash === this.lastHash && this.frameCount % 4 !== 1) {
      return;
    }
    this.lastHash = hash;

    // D6.3: assigning canvas.width/height clears the canvas and forces a reflow;
    // only do it when the pixel size actually changes (we clear explicitly below).
    const canvasW = width * this.tile;
    const canvasH = height * this.tile;
    if (this.ctx.canvas.width !== canvasW) this.ctx.canvas.width = canvasW;
    if (this.ctx.canvas.height !== canvasH) this.ctx.canvas.height = canvasH;
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

    // Get maze wall color
    const maze = MAZES.find((m) => m.id === env.params.mazeId);
    const wallColor = maze?.wallColor ?? '#1e3a8a';

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (isWall(x, y)) {
          this.drawWall(x, y, wallColor, isWall);
          continue;
        }
        if (showHeatmap) {
          const h = Math.min(1, heatmap[y][x]);
          if (h > 0.01) {
            this.ctx.fillStyle = `rgba(239, 68, 68, ${h})`;
            this.ctx.fillRect(x * this.tile, y * this.tile, this.tile, this.tile);
          }
        }
        if (qOverlay) {
          const v = qOverlay[y]?.[x];
          if (v !== null && v !== undefined && Number.isFinite(v)) {
            // Normalized 0..1 → blue (low value) to green (high value).
            const t = (v - qMin) / qSpan;
            const g = Math.round(70 + t * 170);
            const b = Math.round(210 - t * 160);
            this.ctx.fillStyle = `rgba(40, ${g}, ${b}, 0.5)`;
            this.ctx.fillRect(x * this.tile, y * this.tile, this.tile, this.tile);
          }
        }
        if (pellets[y][x]) {
          this.ctx.fillStyle = '#fde68a';
          this.ctx.beginPath();
          this.ctx.arc(x * this.tile + this.tile / 2, y * this.tile + this.tile / 2, 2, 0, Math.PI * 2);
          this.ctx.fill();
        }
        if (powerPellets[y][x]) {
          this.drawPowerPellet(x, y);
        }
      }
    }

    env.ghosts.forEach((g, i) => {
      if (g.edibleTimer > 0) {
        // Flash white when edible timer is running low
        const flashing = g.edibleTimer < 5 && this.frameCount % 4 < 2;
        this.ctx.fillStyle = flashing ? EDIBLE_FLASH_COLOR : EDIBLE_COLOR;
      } else {
        this.ctx.fillStyle = GHOST_COLORS[i % GHOST_COLORS.length];
      }
      this.drawGhost(g.pos.x, g.pos.y);
    });

    // D6.9: draw every Pac-Man (numPacmen can be 1–4), not just pacmen[0].
    this.ctx.fillStyle = PAC_COLOR;
    // Animated mouth: oscillates from 0.1 to 0.45 radians (shared across pacs).
    const mouthAngle = 0.1 + 0.35 * Math.abs(Math.sin(this.frameCount * 0.15));
    for (const p of pacmen) this.drawPac(p.pos.x, p.pos.y, mouthAngle);
  }

  private drawPac(px: number, py: number, mouthAngle: number): void {
    const cx = px * this.tile + this.tile / 2;
    const cy = py * this.tile + this.tile / 2;
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, this.tile * 0.4, mouthAngle, Math.PI * 2 - mouthAngle);
    this.ctx.lineTo(cx, cy);
    this.ctx.fill();
  }

  private drawWall(x: number, y: number, color: string, isWall: (x: number, y: number) => boolean): void {
    const cx = x * this.tile + this.tile / 2;
    const cy = y * this.tile + this.tile / 2;
    const half = this.tile / 2;

    // Fill the wall tile with a darker shade
    this.ctx.fillStyle = '#0a0a0a';
    this.ctx.fillRect(x * this.tile, y * this.tile, this.tile, this.tile);

    // Draw colored border segments only on edges adjacent to open space
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();

    if (!isWall(x, y - 1)) { this.ctx.moveTo(cx - half, cy - half); this.ctx.lineTo(cx + half, cy - half); }
    if (!isWall(x, y + 1)) { this.ctx.moveTo(cx - half, cy + half); this.ctx.lineTo(cx + half, cy + half); }
    if (!isWall(x - 1, y)) { this.ctx.moveTo(cx - half, cy - half); this.ctx.lineTo(cx - half, cy + half); }
    if (!isWall(x + 1, y)) { this.ctx.moveTo(cx + half, cy - half); this.ctx.lineTo(cx + half, cy + half); }

    this.ctx.stroke();
  }

  private drawPowerPellet(x: number, y: number): void {
    const cx = x * this.tile + this.tile / 2;
    const cy = y * this.tile + this.tile / 2;
    // Pulsing glow effect
    const pulse = 0.7 + 0.3 * Math.sin(this.frameCount * 0.15);
    const radius = this.tile * 0.35 * pulse;

    // Outer glow
    this.ctx.fillStyle = `rgba(249, 115, 22, 0.3)`;
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, radius + 3, 0, Math.PI * 2);
    this.ctx.fill();

    // Inner pellet
    this.ctx.fillStyle = '#f97316';
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    this.ctx.fill();
  }

  private drawGhost(gx: number, gy: number): void {
    const cx = gx * this.tile + this.tile / 2;
    const cy = gy * this.tile + this.tile / 2;
    const r = this.tile * 0.4;

    // Ghost body: semicircle top + wavy bottom
    this.ctx.beginPath();
    this.ctx.arc(cx, cy - r * 0.1, r, Math.PI, 0, false);
    // Wavy bottom
    const bottom = cy + r * 0.7;
    const left = cx - r;
    const right = cx + r;
    this.ctx.lineTo(right, bottom);
    const waves = 3;
    const waveW = (right - left) / waves;
    for (let i = waves; i > 0; i--) {
      const wx = left + i * waveW;
      this.ctx.quadraticCurveTo(wx - waveW * 0.25, bottom + r * 0.3, wx - waveW * 0.5, bottom);
      this.ctx.quadraticCurveTo(wx - waveW * 0.75, bottom - r * 0.3, wx - waveW, bottom);
    }
    this.ctx.closePath();
    this.ctx.fill();

    // Eyes
    this.ctx.fillStyle = '#fff';
    this.ctx.beginPath();
    this.ctx.arc(cx - r * 0.3, cy - r * 0.2, r * 0.22, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.beginPath();
    this.ctx.arc(cx + r * 0.3, cy - r * 0.2, r * 0.22, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.fillStyle = '#111';
    this.ctx.beginPath();
    this.ctx.arc(cx - r * 0.25, cy - r * 0.15, r * 0.1, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.beginPath();
    this.ctx.arc(cx + r * 0.35, cy - r * 0.15, r * 0.1, 0, Math.PI * 2);
    this.ctx.fill();
  }
}
