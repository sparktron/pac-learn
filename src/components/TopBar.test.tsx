// @vitest-environment jsdom
import { afterEach, describe, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { TopBar, type TopBarProps } from './TopBar';

afterEach(cleanup);

const baseProps: TopBarProps = {
  version: '9.9.9',
  isTraining: false,
  episodeCount: 1234,
  avgScore: 42.5,
  bestScore: 99,
  curEpsilon: 0.321,
  onReset: () => {},
  onToggleTraining: () => {},
};

describe('TopBar', () => {
  test('renders brand, version, and key stats', () => {
    render(<TopBar {...baseProps} />);
    expect(screen.getByText('Pac Learn')).toBeInTheDocument();
    expect(screen.getByText('v9.9.9')).toBeInTheDocument();
    expect(screen.getByText('1,234')).toBeInTheDocument(); // episodeCount, toLocaleString
    expect(screen.getByText('42.5')).toBeInTheDocument();  // avgScore, fmtNum(_, 1)
    expect(screen.getByText('99')).toBeInTheDocument();    // bestScore, fmtNum(_, 0)
    expect(screen.getByText('0.321')).toBeInTheDocument(); // curEpsilon, fmtNum(_, 3)
  });

  test('shows Idle when not training and the episode count when training', () => {
    const { rerender } = render(<TopBar {...baseProps} isTraining={false} />);
    expect(screen.getByText('Idle')).toBeInTheDocument();
    rerender(<TopBar {...baseProps} isTraining />);
    expect(screen.getByText('Training · ep 1,234')).toBeInTheDocument();
  });

  test('Reset button fires onReset', () => {
    const onReset = vi.fn();
    render(<TopBar {...baseProps} onReset={onReset} />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    expect(onReset).toHaveBeenCalledOnce();
  });

  test('both training toggles fire onToggleTraining', () => {
    const onToggleTraining = vi.fn();
    render(<TopBar {...baseProps} onToggleTraining={onToggleTraining} />);
    fireEvent.click(screen.getByRole('button', { name: /Resume/ }));
    fireEvent.click(screen.getByRole('button', { name: /▶ Training/ }));
    expect(onToggleTraining).toHaveBeenCalledTimes(2);
  });
});
