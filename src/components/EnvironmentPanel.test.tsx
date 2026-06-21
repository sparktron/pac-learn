// @vitest-environment jsdom
import { createRef } from 'react';
import { afterEach, describe, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { createDefaultEnv } from '../env/environment';
import { EnvironmentPanel, type EnvironmentPanelProps } from './EnvironmentPanel';

afterEach(cleanup);

const renderPanel = (over: Partial<EnvironmentPanelProps> = {}) => {
  const props: EnvironmentPanelProps = {
    canvasRef: createRef<HTMLCanvasElement>(),
    mazeBodyRef: createRef<HTMLDivElement>(),
    env: createDefaultEnv(),
    viewMode: 'live',
    setViewMode: vi.fn(),
    episodeCount: 3,
    scatterPhase: false,
    numGhosts: 2,
    maxEpisodeSteps: 1000,
    pacScore: 42,
    ghostsEatenCombo: 1,
    ...over,
  };
  return { ...render(<EnvironmentPanel {...props} />), props };
};

describe('EnvironmentPanel', () => {
  test('renders title, active view pill, and the score', () => {
    renderPanel();
    expect(screen.getByText('Environment')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Live' })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByText('42')).toBeInTheDocument(); // pacScore in the stat strip
  });

  test('view-mode pills call setViewMode', () => {
    const { props } = renderPanel();
    fireEvent.click(screen.getByRole('button', { name: 'Q-Values' }));
    expect(props.setViewMode).toHaveBeenCalledWith('qvalues');
  });

  test('shows the SCATTER phase chip when scattering with ghosts present', () => {
    const { container } = renderPanel({ scatterPhase: true, numGhosts: 2 });
    expect(container.querySelector('.hud-top-left')?.textContent).toContain('SCATTER');
  });

  test('hides the phase chip entirely when there are no ghosts', () => {
    const { container } = renderPanel({ numGhosts: 0 });
    const chip = container.querySelector('.hud-top-left')?.textContent ?? '';
    expect(chip).not.toContain('SCATTER');
    expect(chip).not.toContain('CHASE');
  });
});
