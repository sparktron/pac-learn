// @vitest-environment jsdom
import { afterEach, describe, expect, test, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { createDefaultEnv } from '../env/environment';
import { QLearningAgent } from '../rl/qlearning';
import { ConfigurationPanel, type ConfigurationPanelProps } from './ConfigurationPanel';

afterEach(cleanup);

const makeProps = (over: Partial<ConfigurationPanelProps> = {}): ConfigurationPanelProps => ({
  rewardPreset: 'default',
  onSaveParams: vi.fn(),
  activeTab: 'environment',
  setActiveTab: vi.fn(),
  mode: 'ai',
  setMode: vi.fn(),
  algorithm: 'tabular',
  changeAlgorithm: vi.fn(),
  params: createDefaultEnv().params,
  setParams: vi.fn(),
  changeNumGhosts: vi.fn(),
  ghostAIType: 'classic',
  setGhostAIType: vi.fn(),
  viewMode: 'live',
  setViewMode: vi.fn(),
  setGhostPersonality: vi.fn(),
  setRewardPreset: vi.fn(),
  setReward: vi.fn(),
  agent: new QLearningAgent({ alpha: 0.1, gamma: 0.99, epsilon: 0.5, epsilonDecay: 0.999, epsilonMin: 0.2 }),
  requestRender: vi.fn(),
  trainingSpeed: 'normal',
  updateTrainingSpeed: vi.fn(),
  stepsPerFrame: 1,
  setStepsPerFrame: vi.fn(),
  renderEveryNSteps: 1,
  setRenderEveryNSteps: vi.fn(),
  seed: 7,
  setSeed: vi.fn(),
  onResetQ: vi.fn(),
  onSavePolicy: vi.fn(),
  onLoadPolicy: vi.fn(),
  ...over,
});

describe('ConfigurationPanel', () => {
  test('environment tab renders mode/algorithm/maze selects', () => {
    render(<ConfigurationPanel {...makeProps()} />);
    expect(screen.getByLabelText('Mode')).toBeInTheDocument();
    expect((screen.getByLabelText('Algorithm') as HTMLSelectElement).value).toBe('tabular');
    expect(screen.getByLabelText('Maze')).toBeInTheDocument();
  });

  test('algorithm select calls changeAlgorithm', () => {
    const changeAlgorithm = vi.fn();
    render(<ConfigurationPanel {...makeProps({ changeAlgorithm })} />);
    fireEvent.change(screen.getByLabelText('Algorithm'), { target: { value: 'linear' } });
    expect(changeAlgorithm).toHaveBeenCalledWith('linear');
  });

  test('tab buttons call setActiveTab', () => {
    const setActiveTab = vi.fn();
    render(<ConfigurationPanel {...makeProps({ setActiveTab })} />);
    fireEvent.click(screen.getByRole('tab', { name: 'Tuning' }));
    expect(setActiveTab).toHaveBeenCalledWith('tuning');
  });

  test('tuning tab shows reward + learning fields and edits route through setReward', () => {
    const setReward = vi.fn();
    render(<ConfigurationPanel {...makeProps({ activeTab: 'tuning', setReward })} />);
    expect(screen.getByLabelText('epsilon')).toBeInTheDocument(); // learning field
    fireEvent.change(screen.getByLabelText('pelletReward'), { target: { value: '7' } });
    expect(setReward).toHaveBeenCalledWith('pelletReward', '7');
  });

  test('runtime tab seed input calls setSeed', () => {
    const setSeed = vi.fn();
    render(<ConfigurationPanel {...makeProps({ activeTab: 'runtime', setSeed })} />);
    fireEvent.change(screen.getByLabelText('seed'), { target: { value: '99' } });
    expect(setSeed).toHaveBeenCalled();
  });

  test('footer buttons fire their handlers', () => {
    const onResetQ = vi.fn();
    const onSavePolicy = vi.fn();
    const onSaveParams = vi.fn();
    render(<ConfigurationPanel {...makeProps({ onResetQ, onSavePolicy, onSaveParams })} />);
    fireEvent.click(screen.getByRole('button', { name: 'Reset Q' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save policy' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save params' }));
    expect(onResetQ).toHaveBeenCalledOnce();
    expect(onSavePolicy).toHaveBeenCalledOnce();
    expect(onSaveParams).toHaveBeenCalledOnce();
  });
});
