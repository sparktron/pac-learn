// @vitest-environment jsdom
import { afterEach, describe, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { TelemetryPanel, type TelemetryPanelProps } from './TelemetryPanel';

afterEach(cleanup);

const baseProps: TelemetryPanelProps = {
  scores: [10, 20, 30],
  lengths: [5, 6, 7],
  epsilons: [0.5, 0.4, 0.3],
  curEpsilon: 0.3,
  timeRange: 120,
  setTimeRange: () => {},
};

describe('TelemetryPanel', () => {
  test('renders the four sparkline cards', () => {
    render(<TelemetryPanel {...baseProps} />);
    for (const title of ['Episode Score', 'Episode Length', 'Score Moving Avg (20 ep)', 'ε Exploration']) {
      expect(screen.getByText(title)).toBeInTheDocument();
    }
  });

  test('shows the current ε and latest score', () => {
    render(<TelemetryPanel {...baseProps} />);
    expect(screen.getByText('0.300')).toBeInTheDocument(); // curEpsilon, fmtNum(_, 3)
    expect(screen.getByText('30')).toBeInTheDocument();     // last score
  });

  test('time-range pills reflect and change the selection', () => {
    const setTimeRange = vi.fn();
    render(<TelemetryPanel {...baseProps} setTimeRange={setTimeRange} />);
    expect(screen.getByRole('button', { name: '120 ep' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'All' }));
    expect(setTimeRange).toHaveBeenCalledWith(0);
  });
});
