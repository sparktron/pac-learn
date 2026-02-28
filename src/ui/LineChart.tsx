import { useEffect, useRef } from 'react';

interface Props { values: number[]; width?: number; height?: number; color?: string; }

export const LineChart = ({ values, width = 220, height = 80, color = '#22c55e' }: Props): JSX.Element => {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    c.width = width;
    c.height = height;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#111827';
    ctx.fillRect(0, 0, width, height);
    if (values.length < 2) return;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(1, max - min);
    ctx.strokeStyle = color;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * (width - 8) + 4;
      const y = height - ((v - min) / span) * (height - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [values, width, height, color]);
  return <canvas ref={ref} style={{ border: '1px solid #374151' }} />;
};
