import { useEffect, useRef, useState, FC } from 'react';

type LineChartProps = {
  values: number[];
  height?: number;
  color?: string;
  label?: string;
  xLabel?: string;
  yLabel?: string;
};

const hexToRgb = (hex: string): string => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  if (!result) return '0,0,0';
  return `${parseInt(result[1], 16)},${parseInt(result[2], 16)},${parseInt(result[3], 16)}`;
};

export const LineChart: FC<LineChartProps> = (props) => {
  const { values, height = 80, color = '#22c55e', label, xLabel, yLabel } = props;
  const ref = useRef<HTMLCanvasElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(220);

  useEffect(() => {
    const el = wrapperRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      if (w > 0) setWidth(Math.floor(w));
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, width, height);

    if (values.length < 2) {
      if (label) {
        ctx.fillStyle = '#9ca3af';
        ctx.font = '11px sans-serif';
        ctx.fillText(label, 10, 20);
      }
      return;
    }

    const padding = { left: 40, right: 10, top: 20, bottom: 30 };
    const gw = width - padding.left - padding.right;
    const gh = height - padding.top - padding.bottom;

    // Bin data into pixel columns so any data density renders correctly.
    // Each bin holds min/max/avg of all values mapping to that pixel column.
    const numBins = Math.max(2, gw);
    type Bin = { min: number; max: number; avg: number };
    const bins: Bin[] = [];
    for (let b = 0; b < numBins; b++) {
      const lo = Math.floor((b / numBins) * values.length);
      const hi = Math.floor(((b + 1) / numBins) * values.length);
      const slice = values.slice(lo, Math.max(hi, lo + 1));
      const mn = Math.min(...slice);
      const mx = Math.max(...slice);
      bins.push({ min: mn, max: mx, avg: slice.reduce((a, v) => a + v, 0) / slice.length });
    }

    const globalMin = Math.min(...bins.map(b => b.min));
    const globalMax = Math.max(...bins.map(b => b.max));
    const span = Math.max(1, globalMax - globalMin);

    const toY = (v: number) => (height - padding.bottom) - ((v - globalMin) / span) * gh;
    const toX = (b: number) => (b / (numBins - 1)) * gw + padding.left;

    // Axes
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, height - padding.bottom);
    ctx.lineTo(width - padding.right, height - padding.bottom);
    ctx.stroke();

    const rgb = hexToRgb(color);

    // When dense (more data than pixels): draw min–max band, then avg line.
    // When sparse (fewer data than pixels): just draw filled area + line as before.
    const isDense = values.length > numBins;

    if (isDense) {
      // Min–max band
      ctx.fillStyle = `rgba(${rgb}, 0.18)`;
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(bins[0].max));
      for (let b = 1; b < numBins; b++) ctx.lineTo(toX(b), toY(bins[b].max));
      for (let b = numBins - 1; b >= 0; b--) ctx.lineTo(toX(b), toY(bins[b].min));
      ctx.closePath();
      ctx.fill();

      // Average line
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      bins.forEach((bin, b) => {
        if (b === 0) ctx.moveTo(toX(b), toY(bin.avg));
        else ctx.lineTo(toX(b), toY(bin.avg));
      });
      ctx.stroke();
    } else {
      // Sparse: filled area + line
      ctx.fillStyle = `rgba(${rgb}, 0.3)`;
      ctx.beginPath();
      ctx.moveTo(toX(0), height - padding.bottom);
      bins.forEach((bin, b) => ctx.lineTo(toX(b), toY(bin.avg)));
      ctx.lineTo(toX(numBins - 1), height - padding.bottom);
      ctx.fill();

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      bins.forEach((bin, b) => {
        if (b === 0) ctx.moveTo(toX(b), toY(bin.avg));
        else ctx.lineTo(toX(b), toY(bin.avg));
      });
      ctx.stroke();
    }

    // Y-axis labels
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(globalMax.toFixed(1), padding.left - 5, padding.top + 10);
    ctx.fillText(globalMin.toFixed(1), padding.left - 5, height - padding.bottom + 5);

    if (xLabel) {
      ctx.textAlign = 'center';
      ctx.fillText(xLabel, width / 2, height - 5);
    }

    if (label) {
      ctx.fillStyle = '#e5e7eb';
      ctx.font = 'bold 12px sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(label, padding.left, padding.top - 5);
    }

    if (yLabel) {
      ctx.save();
      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px sans-serif';
      ctx.translate(12, height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center';
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
    }
  }, [values, width, height, color, label, xLabel, yLabel]);

  return (
    <div ref={wrapperRef} style={{ width: '100%', margin: '8px 0' }}>
      <canvas ref={ref} style={{ border: '1px solid #374151', display: 'block', width: '100%' }} />
    </div>
  );
};
