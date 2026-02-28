import type { PacmanEnvironment } from '../env/environment';

export class CanvasRenderer {
  constructor(private ctx: CanvasRenderingContext2D, private tile = 24) {}

  draw(env: PacmanEnvironment, showHeatmap: boolean): void {
    const { width, height, pellets, powerPellets, heatmap, isWall } = env.world;
    this.ctx.canvas.width = width * this.tile;
    this.ctx.canvas.height = height * this.tile;
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if (isWall(x, y)) {
          this.ctx.fillStyle = '#1e3a8a';
          this.ctx.fillRect(x * this.tile, y * this.tile, this.tile, this.tile);
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
          this.ctx.arc(x * this.tile + this.tile / 2, y * this.tile + this.tile / 2, 3, 0, Math.PI * 2);
          this.ctx.fill();
        }
        if (powerPellets[y][x]) {
          this.ctx.fillStyle = '#f97316';
          this.ctx.beginPath();
          this.ctx.arc(x * this.tile + this.tile / 2, y * this.tile + this.tile / 2, 6, 0, Math.PI * 2);
          this.ctx.fill();
        }
      }
    }

    env.ghosts.forEach((g) => {
      this.ctx.fillStyle = g.edibleTimer > 0 ? '#93c5fd' : '#ef4444';
      this.ctx.beginPath();
      this.ctx.arc(g.pos.x * this.tile + this.tile / 2, g.pos.y * this.tile + this.tile / 2, this.tile * 0.4, 0, Math.PI * 2);
      this.ctx.fill();
    });

    const p = env.getPacmen()[0];
    this.ctx.fillStyle = '#facc15';
    this.ctx.beginPath();
    this.ctx.arc(p.pos.x * this.tile + this.tile / 2, p.pos.y * this.tile + this.tile / 2, this.tile * 0.4, 0.2, Math.PI * 1.8);
    this.ctx.lineTo(p.pos.x * this.tile + this.tile / 2, p.pos.y * this.tile + this.tile / 2);
    this.ctx.fill();
  }
}
