import type { PacmanEnvironment } from '../env/environment';
import { MAZES } from '../mazes/mazes';

const GHOST_COLORS = ['#ef4444', '#f472b6', '#38bdf8', '#fb923c', '#a78bfa', '#34d399'];
const EDIBLE_COLOR = '#93c5fd';
const EDIBLE_FLASH_COLOR = '#ffffff';

export class CanvasRenderer {
  private frameCount = 0;

  constructor(private ctx: CanvasRenderingContext2D, private tile = 24) {}

  draw(env: PacmanEnvironment, showHeatmap: boolean): void {
    this.frameCount += 1;
    const { width, height, pellets, powerPellets, heatmap, isWall } = env.world;
    this.ctx.canvas.width = width * this.tile;
    this.ctx.canvas.height = height * this.tile;
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

    // Get maze wall color
    const maze = MAZES.find((m) => m.id === env.params.mazeId);
    const wallColor = maze?.wallColor ?? '#1e3a8a';

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (isWall(x, y)) {
          this.drawWall(x, y, wallColor, width, height, isWall);
          continue;
        }
        if (showHeatmap) {
          const h = Math.min(1, heatmap[y][x]);
          if (h > 0.01) {
            this.ctx.fillStyle = `rgba(239, 68, 68, ${h})`;
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

    const p = env.getPacmen()[0];
    this.ctx.fillStyle = '#facc15';
    this.ctx.beginPath();
    this.ctx.arc(p.pos.x * this.tile + this.tile / 2, p.pos.y * this.tile + this.tile / 2, this.tile * 0.4, 0.2, Math.PI * 1.8);
    this.ctx.lineTo(p.pos.x * this.tile + this.tile / 2, p.pos.y * this.tile + this.tile / 2);
    this.ctx.fill();
  }

  private drawWall(x: number, y: number, color: string, _w: number, _h: number, isWall: (x: number, y: number) => boolean): void {
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
