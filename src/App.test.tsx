// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import App from './App';
import { OBSERVATION_KEY_VERSION } from './env/observation';

// First component-test harness for the UI (ROADMAP A5 groundwork). These are
// behavior smoke tests: they pin the wiring the audit fixed (algorithm selector
// D7.8, view-mode pills, tabs, maze list incl. the A3 vertical-tunnel maze) so
// the upcoming App.tsx decomposition can refactor against a safety net.

beforeEach(() => {
  // jsdom has no 2D canvas; the renderer already guards a null context, but stub
  // it explicitly so a future unguarded path can't throw inside an effect.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as any;
  // Neutralize the two background drivers so tests are deterministic and don't
  // leak timers / update state after assertions:
  //   - requestAnimationFrame backs the training loop
  //   - setInterval backs the AI-watch loop (active by default: mode 'ai', idle)
  vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0);
  vi.spyOn(window, 'setInterval').mockReturnValue(0 as unknown as ReturnType<typeof setInterval>);
  localStorage.clear();
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('App (smoke)', () => {
  test('renders the three panels and brand', () => {
    render(<App />);
    expect(screen.getByText('Pac Learn')).toBeInTheDocument();
    expect(screen.getByText('Configuration')).toBeInTheDocument();
    expect(screen.getByText('Telemetry')).toBeInTheDocument();
  });

  test('algorithm selector defaults to tabular, switches to linear, and persists', () => {
    render(<App />);
    const algo = screen.getByLabelText('Algorithm') as HTMLSelectElement;
    expect(algo.value).toBe('tabular');
    fireEvent.change(algo, { target: { value: 'linear' } });
    expect(algo.value).toBe('linear');
    expect(localStorage.getItem('pac-learn-algorithm')).toBe('linear');
  });

  test('maze selector lists the static mazes including the A3 vertical-tunnel maze', () => {
    render(<App />);
    const maze = screen.getByLabelText('Maze') as HTMLSelectElement;
    const options = within(maze).getAllByRole('option').map((o) => o.textContent);
    expect(options).toEqual(expect.arrayContaining(['Pac-Man Classic', 'Vertical Loop']));
  });

  test('view-mode pills switch the active view (Q-Values)', () => {
    render(<App />);
    const qBtn = screen.getByRole('button', { name: 'Q-Values' });
    expect(qBtn).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(qBtn);
    expect(qBtn).toHaveAttribute('aria-pressed', 'true');
  });

  test('config tabs switch to Tuning and reveal reward fields', () => {
    render(<App />);
    // Environment tab is active by default; the reward "Preset" field lives on Tuning.
    expect(screen.queryByLabelText('pelletReward')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Tuning' }));
    expect(screen.getByLabelText('pelletReward')).toBeInTheDocument();
  });

  test('Training button toggles to Pause without throwing', () => {
    render(<App />);
    const trainBtn = screen.getByRole('button', { name: /▶ Training/ });
    fireEvent.click(trainBtn);
    // startTraining ran (rAF stubbed) and flipped the control label.
    expect(screen.getByRole('button', { name: /⏸ Pause/ })).toBeInTheDocument();
  });

  test('loading a policy synchronizes numGhosts without discarding the Q-table', async () => {
    const { container } = render(<App />);
    const policy = {
      algorithm: 'qlearning',
      mazeId: 'pacman-classic',
      timestamp: '2026-07-21T00:00:00.000Z',
      numGhostsEncoded: 3,
      observationKeyVersion: OBSERVATION_KEY_VERSION,
      hyper: {
        alpha: 0.1, gamma: 0.99, epsilon: 0.2,
        epsilonDecay: 0.999997, epsilonMin: 0.2,
      },
      qTable: { 'v9:0:0:0:0:0:0:-1:4:2': [1, 2, 3, 4] },
      visitTable: { 'v9:0:0:0:0:0:0:-1:4:2': [1, 1, 1, 1] },
    };
    const file = new File([JSON.stringify(policy)], 'policy.json', { type: 'application/json' });
    Object.defineProperty(file, 'text', { value: async () => JSON.stringify(policy) });
    const input = container.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input).not.toBeNull();

    fireEvent.change(input!, { target: { files: [file] } });

    await waitFor(() => expect(screen.getByLabelText('numGhosts')).toHaveValue(3));
    // A loaded table pins its training ghost count, so changing away from 3 now
    // requires confirmation. A discarded table would allow this silently.
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    fireEvent.change(screen.getByLabelText('numGhosts'), { target: { value: '2' } });
    expect(confirm).toHaveBeenCalled();
    expect(screen.getByLabelText('numGhosts')).toHaveValue(3);
  });
});
