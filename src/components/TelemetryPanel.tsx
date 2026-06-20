import { useMemo } from 'react';
import { movingAverage, buildSparkPath, computeDelta, fmtNum } from '../uiHelpers';

type TimeRange = 120 | 500 | 0;

type ChartCardProps = {
  title: string;
  color: string;
  value: string;
  values: number[];
  gradId: string;
};

const ChartCard = ({ title, color, value, values, gradId }: ChartCardProps): JSX.Element => {
  const { line, fill } = buildSparkPath(values);
  const delta = computeDelta(values);
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <div className="chart-bullet" style={{ background: color }} />
        <span className="chart-title">{title}</span>
        <div className="chart-header-spacer" />
        <span className="chart-value">{value}</span>
        {delta.dir !== 'flat' && (
          <span className={`chart-delta ${delta.dir}`}>
            {delta.dir === 'up' ? '▲' : '▼'} {fmtNum(delta.pct, 1)}%
          </span>
        )}
      </div>
      <div className="chart-sparkline">
        <svg viewBox="0 0 400 90" preserveAspectRatio="none">
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={color} stopOpacity="0.28" />
              <stop offset="100%" stopColor={color} stopOpacity="0.02" />
            </linearGradient>
          </defs>
          <line x1="0" y1="22" x2="400" y2="22" stroke="#1f2330" strokeWidth="1" />
          <line x1="0" y1="45" x2="400" y2="45" stroke="#1f2330" strokeWidth="1" />
          <line x1="0" y1="68" x2="400" y2="68" stroke="#1f2330" strokeWidth="1" />
          {fill && <path d={fill} fill={`url(#${gradId})`} />}
          {line && <path d={line} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />}
        </svg>
      </div>
    </div>
  );
};

export interface TelemetryPanelProps {
  scores: number[];
  lengths: number[];
  epsilons: number[];
  curEpsilon: number;
  timeRange: TimeRange;
  setTimeRange: (r: TimeRange) => void;
}

/**
 * Telemetry column (A5 slice 4a): the four training sparklines + the time-range
 * selector. Pure presentation — it derives the 20-ep moving average and the
 * time-window slice from the raw stat arrays it's handed. The arrays are mutated
 * in place by the trainer, so memoization is keyed on `.length` (the only thing
 * that actually moves), matching the original App behavior.
 */
export function TelemetryPanel({
  scores, lengths, epsilons, curEpsilon, timeRange, setTimeRange,
}: TelemetryPanelProps): JSX.Element {
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const movAvg = useMemo(() => movingAverage(scores, 20), [scores.length]);
  const slice = (vals: number[]): number[] => (timeRange === 0 ? vals : vals.slice(-timeRange));

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Telemetry</span>
        <div className="panel-header-spacer" />
        <div className="time-pills" role="group" aria-label="Time range">
          {([120, 500, 0] as const).map((r) => (
            <button
              key={r}
              className={`time-pill${timeRange === r ? ' active' : ''}`}
              onClick={() => setTimeRange(r)}
              aria-pressed={timeRange === r}
            >
              {r === 0 ? 'All' : `${r} ep`}
            </button>
          ))}
        </div>
      </div>

      <div className="chart-stack">
        <ChartCard
          title="Episode Score"
          color="#22c55e"
          gradId="grad-score"
          values={slice(scores)}
          value={scores.length > 0 ? fmtNum(scores[scores.length - 1], 0) : '—'}
        />
        <ChartCard
          title="Episode Length"
          color="#a78bfa"
          gradId="grad-length"
          values={slice(lengths)}
          value={lengths.length > 0 ? fmtNum(lengths[lengths.length - 1], 0) : '—'}
        />
        <ChartCard
          title="Score Moving Avg (20 ep)"
          color="#3b82f6"
          gradId="grad-mavg"
          values={slice(movAvg)}
          value={movAvg.length > 0 ? fmtNum(movAvg[movAvg.length - 1], 1) : '—'}
        />
        <ChartCard
          title="ε Exploration"
          color="#f59e0b"
          gradId="grad-eps"
          values={slice(epsilons)}
          value={fmtNum(curEpsilon, 3)}
        />
      </div>
    </div>
  );
}
