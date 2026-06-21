import { type Dispatch, type SetStateAction } from 'react';
import { type EnvParams } from '../env/environment';
import type { QLearningAgent } from '../rl/qlearning';
import type { LinearQLearningAgent } from '../rl/linearQlearning';
import type { GhostAIType } from '../ghosts/ghostAi';
import type { TrainingSpeed } from '../hooks/useTrainingLoop';
import { trainingSpeedOptions } from '../hooks/useTrainingLoop';
import { MAZES } from '../mazes/mazes';
import { REWARD_PRESETS as rewardPresets } from '../rl/rewardPresets';
import { safeNum } from '../uiHelpers';

export type Algorithm = 'tabular' | 'linear';
type Mode = 'human' | 'ai';
type ViewMode = 'live' | 'heatmap' | 'qvalues';
type ActiveTab = 'environment' | 'tuning' | 'runtime';

const ghostAITypes: GhostAIType[] = ['classic', 'heatmap', 'hybrid'];

// ── Small field controls (used only by this panel) ─────────────

type ToggleProps = { checked: boolean; onChange: (v: boolean) => void; label: string; sublabel?: string; id: string };

const Toggle = ({ checked, onChange, label, sublabel, id }: ToggleProps): JSX.Element => (
  <div className="toggle-row">
    <div className="toggle-labels">
      <label className="toggle-label-text" htmlFor={id}>{label}</label>
      {sublabel && <span className="toggle-sublabel">{sublabel}</span>}
    </div>
    <button
      id={id}
      role="switch"
      aria-checked={checked}
      className={`toggle-switch${checked ? ' on' : ''}`}
      onClick={() => onChange(!checked)}
    >
      <span className="toggle-knob" />
    </button>
  </div>
);

type FieldProps = { label: string; unit?: string; htmlFor?: string; children: React.ReactNode };

const Field = ({ label, unit, htmlFor, children }: FieldProps): JSX.Element => (
  <div className="field">
    <div className="field-label">
      <label htmlFor={htmlFor}>{label}</label>
      {unit && <span className="field-unit">{unit}</span>}
    </div>
    {children}
  </div>
);

// Static documentation block — no props, pulled out to keep the panel readable.
const VariableReference = (): JSX.Element => (
  <div className="config-section howto-section">
    <div className="howto-header">
      <span className="howto-header-title">Variable Reference</span>
      <span className="howto-header-sub">how each setting shapes agent behavior</span>
    </div>

    <div className="howto-group">
      <div className="howto-group-label">
        <span className="howto-group-bar" />
        Rewards
      </div>
      <dl className="howto-list">
        <div className="howto-item">
          <dt>pelletReward</dt>
          <dd>Points earned each time Pac-Man eats a regular dot. Raising this relative to other rewards trains the agent to prioritize pellet collection as its main objective. Too high and it may rush into danger; too low and it may ignore pellets entirely.</dd>
        </div>
        <div className="howto-item">
          <dt>powerPelletReward</dt>
          <dd>Points for eating a large power pellet. Increasing this relative to <code>pelletReward</code> teaches the agent to actively seek out power-ups before cleaning the board. Only relevant when power pellets are enabled.</dd>
        </div>
        <div className="howto-item">
          <dt>ghostEatReward</dt>
          <dd>Points for eating a frightened ghost while powered up. Very high values shift the entire strategy toward ghost-hunting — the agent will use power pellets aggressively and linger in dangerous zones to chase ghosts.</dd>
        </div>
        <div className="howto-item">
          <dt>winBonus</dt>
          <dd>One-time reward when every pellet is cleared. A large bonus pushes the agent toward completion rather than survival. Low values mean the agent treats clearing the board as just one of many possible goals.</dd>
        </div>
        <div className="howto-item">
          <dt>deathPenalty</dt>
          <dd>Negative reward applied when Pac-Man is caught by a ghost. More negative values create a more cautious, ghost-avoidant agent. If set too large, the agent may freeze in safe corners rather than collect pellets.</dd>
        </div>
        <div className="howto-item">
          <dt>stepPenalty</dt>
          <dd>Small negative reward applied on every step. Discourages aimless wandering by making episodes with fewer wasted moves more valuable. If too large, the agent rushes and takes unnecessary risks just to end the episode quickly.</dd>
        </div>
        <div className="howto-item">
          <dt>survivalReward</dt>
          <dd>Small positive reward added every step the agent stays alive. Counteracts <code>stepPenalty</code> and rewards longevity. The "survival" preset raises this to 0.2 — roughly 10× the default — making staying alive the dominant objective over collecting pellets.</dd>
        </div>
      </dl>
    </div>

    <div className="howto-group">
      <div className="howto-group-label">
        <span className="howto-group-bar" />
        Learning
      </div>
      <dl className="howto-list">
        <div className="howto-item">
          <dt>epsilon <span className="howto-greek">ε</span></dt>
          <dd>Exploration rate. At 1.0 the agent acts randomly (pure exploration); at 0.0 it always exploits its current best-known Q-values (pure exploitation). Training starts near 1.0 and decays over episodes as the agent builds knowledge.</dd>
        </div>
        <div className="howto-item">
          <dt>epsilonDecay</dt>
          <dd>Multiplied against <code>epsilon</code> at the end of every episode. Values close to 1.0 (e.g. 0.999) preserve exploration longer. Values further from 1.0 (e.g. 0.99) shift to exploitation faster — useful for simpler mazes where the agent learns quickly.</dd>
        </div>
        <div className="howto-item">
          <dt>alpha <span className="howto-greek">α</span></dt>
          <dd>Q-table learning rate. Controls how much each new experience updates stored values. High α (near 1.0) means recent experiences overwrite old ones rapidly, useful when the environment is changing. Low α produces slow, stable convergence.</dd>
        </div>
        <div className="howto-item">
          <dt>gamma <span className="howto-greek">γ</span></dt>
          <dd>Discount factor for future rewards. Values near 1.0 make the agent plan many steps ahead, valuing long-term outcomes. Lower values produce myopic behavior focused only on immediate rewards. At γ=0.95 the effective planning horizon is ~14 steps — too short for 1000-step episodes. 0.99 (horizon ~69 steps) is recommended.</dd>
        </div>
        <div className="howto-item">
          <dt>heatmapLearningRate</dt>
          <dd>How strongly each ghost visit marks a tile on the danger heatmap. Higher values create a more reactive but noisier map. Only affects agents using Heatmap or Hybrid ghost AI modes — has no effect on Classic.</dd>
        </div>
        <div className="howto-item">
          <dt>heatmapDecayRate</dt>
          <dd>Multiplier applied to every heatmap cell each step, causing old danger markings to fade. Values close to 1.0 give the map long memory. Lower values make the map forget danger zones quickly, suited to environments where ghosts move unpredictably.</dd>
        </div>
      </dl>
    </div>
  </div>
);

export interface ConfigurationPanelProps {
  rewardPreset: string;
  onSaveParams: () => void;

  activeTab: ActiveTab;
  setActiveTab: (t: ActiveTab) => void;

  // Environment tab
  mode: Mode;
  setMode: (m: Mode) => void;
  algorithm: Algorithm;
  changeAlgorithm: (a: Algorithm) => void;
  params: EnvParams;
  setParams: Dispatch<SetStateAction<EnvParams>>;
  changeNumGhosts: (n: number) => void;
  ghostAIType: GhostAIType;
  setGhostAIType: (t: GhostAIType) => void;
  viewMode: ViewMode;
  setViewMode: (v: ViewMode) => void;
  setGhostPersonality: (i: number, value: string) => void;

  // Tuning tab
  setRewardPreset: (p: string) => void;
  setReward: (key: keyof EnvParams['reward'], rawValue: string) => void;
  agent: QLearningAgent | LinearQLearningAgent;
  requestRender: () => void;

  // Runtime tab
  trainingSpeed: TrainingSpeed;
  updateTrainingSpeed: (s: TrainingSpeed) => void;
  stepsPerFrame: number;
  setStepsPerFrame: Dispatch<SetStateAction<number>>;
  renderEveryNSteps: number;
  setRenderEveryNSteps: Dispatch<SetStateAction<number>>;
  seed: number;
  setSeed: Dispatch<SetStateAction<number>>;

  // Footer (logic-bearing handlers stay in App)
  onResetQ: () => void;
  onSavePolicy: () => void;
  onLoadPolicy: (file: File) => void | Promise<void>;
}

/**
 * Configuration column (A5 slice 4c): the preset chip + save-params button, the
 * three-tab body (Environment / Tuning / Runtime), and the save/load/reset
 * footer. Mostly presentation — pure field edits go straight through setParams,
 * but logic-bearing actions (save params, reset-Q, load policy) are delegated to
 * App via callbacks, since they touch the agent/env/trainer it owns.
 */
export function ConfigurationPanel(props: ConfigurationPanelProps): JSX.Element {
  const {
    rewardPreset, onSaveParams, activeTab, setActiveTab,
    mode, setMode, algorithm, changeAlgorithm, params, setParams, changeNumGhosts,
    ghostAIType, setGhostAIType, viewMode, setViewMode, setGhostPersonality,
    setRewardPreset, setReward, agent, requestRender,
    trainingSpeed, updateTrainingSpeed, stepsPerFrame, setStepsPerFrame,
    renderEveryNSteps, setRenderEveryNSteps, seed, setSeed,
    onResetQ, onSavePolicy, onLoadPolicy,
  } = props;

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Configuration</span>
        <div className="panel-header-spacer" />
        <div className="preset-chip">
          <span className="preset-chip-label">Preset ·</span>
          <span className="preset-chip-value">{rewardPreset}</span>
        </div>
        <button className="icon-btn" aria-label="Save params" title="Save params" onClick={onSaveParams}>↓</button>
      </div>

      {/* Tab bar. No field counts — they were hard-coded and silently
          lied whenever a Field was added or removed. */}
      <div className="tab-bar" role="tablist">
        {([
          ['environment', 'Environment'],
          ['tuning',      'Tuning'],
          ['runtime',     'Runtime'],
        ] as [ActiveTab, string][]).map(([id, label]) => (
          <button
            key={id}
            role="tab"
            aria-selected={activeTab === id}
            className={`tab-btn${activeTab === id ? ' active' : ''}`}
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Scrollable config body */}
      <div className="config-body" role="tabpanel">

        {/* ENVIRONMENT TAB */}
        {activeTab === 'environment' && (
          <>
            <div className="config-section">
              <div className="section-heading">Mode &amp; Maze</div>
              <div className="field-grid">
                <Field label="Mode" htmlFor="cfg-mode">
                  <select id="cfg-mode" className="field-select" value={mode}
                    onChange={(e) => setMode(e.target.value as Mode)}>
                    <option value="human">Human</option>
                    <option value="ai">AI controlled</option>
                  </select>
                </Field>
                <Field label="Algorithm" htmlFor="cfg-algo">
                  <select id="cfg-algo" className="field-select" value={algorithm}
                    onChange={(e) => changeAlgorithm(e.target.value as Algorithm)}>
                    <option value="tabular">Tabular Q</option>
                    <option value="linear">Linear FA</option>
                  </select>
                </Field>
                <Field label="Maze" htmlFor="cfg-maze">
                  <select id="cfg-maze" className="field-select" value={params.mazeId}
                    onChange={(e) => setParams((p) => ({ ...p, mazeId: e.target.value }))}>
                    {MAZES.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
                  </select>
                </Field>
                <Field label="numGhosts" unit="int" htmlFor="cfg-nghosts">
                  <input id="cfg-nghosts" className="field-input" type="number"
                    value={params.numGhosts} min={1} max={6} step={1}
                    onChange={(e) => changeNumGhosts(safeNum(e.target.value, params.numGhosts))} />
                </Field>
                <Field label="Ghost AI" htmlFor="cfg-ghostai">
                  <select id="cfg-ghostai" className="field-select" value={ghostAIType}
                    onChange={(e) => setGhostAIType(e.target.value as GhostAIType)}>
                    {ghostAITypes.map((t) => <option key={t}>{t}</option>)}
                  </select>
                </Field>
                <Field label="Capture rules" htmlFor="cfg-capture">
                  <select id="cfg-capture" className="field-select" value={params.captureRules}
                    onChange={(e) => setParams((p) => ({ ...p, captureRules: e.target.value as 'touch' | 'tile' }))}>
                    <option value="touch">touch</option>
                    <option value="tile">tile</option>
                  </select>
                </Field>
                <Field label="ghostSpeed" htmlFor="cfg-gspeed">
                  <input id="cfg-gspeed" className="field-input" type="number"
                    value={params.ghostSpeed} min={0.2} max={3} step={0.05}
                    onChange={(e) => setParams((p) => ({ ...p, ghostSpeed: safeNum(e.target.value, p.ghostSpeed) }))} />
                </Field>
                <Field label="pacmanSpeed" htmlFor="cfg-pspeed">
                  <input id="cfg-pspeed" className="field-input" type="number"
                    value={params.pacmanSpeed} min={0.2} max={3} step={0.05}
                    onChange={(e) => setParams((p) => ({ ...p, pacmanSpeed: safeNum(e.target.value, p.pacmanSpeed) }))} />
                </Field>
                <Field label="chaseDuration" unit="steps" htmlFor="cfg-chase">
                  <input id="cfg-chase" className="field-input" type="number"
                    value={params.chaseDuration} min={1} max={5000} step={10}
                    onChange={(e) => setParams((p) => ({ ...p, chaseDuration: safeNum(e.target.value, p.chaseDuration) }))} />
                </Field>
                <Field label="scatterDuration" unit="steps" htmlFor="cfg-scatter">
                  <input id="cfg-scatter" className="field-input" type="number"
                    value={params.scatterDuration} min={1} max={5000} step={10}
                    onChange={(e) => setParams((p) => ({ ...p, scatterDuration: safeNum(e.target.value, p.scatterDuration) }))} />
                </Field>
              </div>
            </div>

            <div className="config-section">
              <div className="section-heading">Toggles</div>
              <Toggle id="tog-pp" label="Enable power pellets" sublabel="grant temporary ghost-eating window"
                checked={params.enablePowerPellets}
                onChange={(v) => setParams((p) => ({ ...p, enablePowerPellets: v }))} />
              <Toggle id="tog-elroy" label="Cruise Elroy" sublabel="Blinky speeds up as pellets clear"
                checked={params.elroyEnabled}
                onChange={(v) => setParams((p) => ({ ...p, elroyEnabled: v }))} />
              <Toggle id="tog-hm" label="Show ghost heatmap" sublabel="visualize danger overlay"
                checked={viewMode === 'heatmap'}
                onChange={(v) => setViewMode(v ? 'heatmap' : 'live')} />
            </div>

            {/* A2: per-ghost targeting personality. 'auto' = default id%4. */}
            <div className="config-section">
              <div className="section-heading">Ghost Personalities</div>
              <div className="field-grid">
                {Array.from({ length: params.numGhosts }, (_, i) => (
                  <Field key={i} label={`Ghost ${i}`} htmlFor={`cfg-gp-${i}`}>
                    <select id={`cfg-gp-${i}`} className="field-select"
                      value={params.ghostPersonalities[i] === undefined ? 'auto' : String(params.ghostPersonalities[i])}
                      onChange={(e) => setGhostPersonality(i, e.target.value)}>
                      <option value="auto">auto (id%4)</option>
                      <option value="0">Blinky · chase</option>
                      <option value="1">Pinky · ambush</option>
                      <option value="2">Inky · flank</option>
                      <option value="3">Clyde · skittish</option>
                    </select>
                  </Field>
                ))}
              </div>
            </div>
          </>
        )}

        {/* TUNING TAB (Rewards + Learning combined) */}
        {activeTab === 'tuning' && (
          <>
            {/* ── Rewards ───────────────────────────────── */}
            <div className="config-section">
              <div className="section-heading">Rewards</div>
              <div className="field-grid" style={{ marginBottom: 10 }}>
                <Field label="Preset" htmlFor="cfg-rpreset">
                  <select id="cfg-rpreset" className="field-select" value={rewardPreset}
                    onChange={(e) => {
                      const preset = e.target.value;
                      setRewardPreset(preset);
                      // N16: spread-clone the preset so params.reward is a fresh
                      // object — NOT a direct reference to the rewardPresets entry.
                      // A direct reference would be mutated by any code that writes
                      // params.reward[key], silently corrupting the preset for the
                      // session. 'custom' has no entry → guard prevents a setParams.
                      if (preset in rewardPresets) setParams((p) => ({ ...p, reward: { ...rewardPresets[preset] } }));
                    }}>
                    {rewardPreset === 'custom' && <option value="custom">custom</option>}
                    {Object.keys(rewardPresets).map((p) => <option key={p} value={p}>{p}</option>)}
                  </select>
                </Field>
              </div>
              <div className="field-grid">
                <Field label="pelletReward" htmlFor="cfg-pr">
                  <input id="cfg-pr" className="field-input" type="number"
                    value={params.reward.pelletReward} step={1}
                    onChange={(e) => setReward('pelletReward', e.target.value)} />
                </Field>
                <Field label="powerPelletReward" htmlFor="cfg-ppr">
                  <input id="cfg-ppr" className="field-input" type="number"
                    value={params.reward.powerPelletReward} step={1}
                    onChange={(e) => setReward('powerPelletReward', e.target.value)} />
                </Field>
                <Field label="ghostEatReward" htmlFor="cfg-ger">
                  <input id="cfg-ger" className="field-input" type="number"
                    value={params.reward.ghostEatReward} step={1}
                    onChange={(e) => setReward('ghostEatReward', e.target.value)} />
                </Field>
                <Field label="winBonus" htmlFor="cfg-wb">
                  <input id="cfg-wb" className="field-input" type="number"
                    value={params.reward.winBonus} step={10}
                    onChange={(e) => setReward('winBonus', e.target.value)} />
                </Field>
                <Field label="deathPenalty" htmlFor="cfg-dp">
                  <input id="cfg-dp" className="field-input" type="number"
                    value={params.reward.deathPenalty} step={1}
                    onChange={(e) => setReward('deathPenalty', e.target.value)} />
                </Field>
                <Field label="stepPenalty" htmlFor="cfg-sp">
                  <input id="cfg-sp" className="field-input" type="number"
                    value={params.reward.stepPenalty} step={0.01}
                    onChange={(e) => setReward('stepPenalty', e.target.value)} />
                </Field>
                <Field label="survivalReward" htmlFor="cfg-sr">
                  <input id="cfg-sr" className="field-input" type="number"
                    value={params.reward.survivalReward} step={0.01}
                    onChange={(e) => setReward('survivalReward', e.target.value)} />
                </Field>
                <Field label="reversePenalty" htmlFor="cfg-rp">
                  <input id="cfg-rp" className="field-input" type="number"
                    value={params.reward.reversePenalty} step={0.5}
                    onChange={(e) => setReward('reversePenalty', e.target.value)} />
                </Field>
              </div>
            </div>

            {/* ── Learning ──────────────────────────────── */}
            <div className="config-section">
              <div className="section-heading">Learning</div>
              <div className="field-grid">
                <Field label="epsilon" unit="ε" htmlFor="cfg-eps">
                  <input id="cfg-eps" className="field-input" type="number"
                    value={agent.hyper.epsilon} min={0} max={1} step={0.01}
                    onChange={(e) => { agent.hyper.epsilon = safeNum(e.target.value, agent.hyper.epsilon); requestRender(); }} />
                </Field>
                <Field label="epsilonDecay" htmlFor="cfg-epsd">
                  <input id="cfg-epsd" className="field-input" type="number"
                    value={agent.hyper.epsilonDecay} min={0.9} max={1} step={0.0001}
                    onChange={(e) => { agent.hyper.epsilonDecay = safeNum(e.target.value, agent.hyper.epsilonDecay); requestRender(); }} />
                </Field>
                <Field label="alpha" unit="α" htmlFor="cfg-alpha">
                  <input id="cfg-alpha" className="field-input" type="number"
                    value={agent.hyper.alpha} min={0} max={1} step={0.01}
                    onChange={(e) => { agent.hyper.alpha = safeNum(e.target.value, agent.hyper.alpha); requestRender(); }} />
                </Field>
                <Field label="gamma" unit="γ" htmlFor="cfg-gamma">
                  <input id="cfg-gamma" className="field-input" type="number"
                    value={agent.hyper.gamma} min={0} max={1} step={0.01}
                    onChange={(e) => { agent.hyper.gamma = safeNum(e.target.value, agent.hyper.gamma); requestRender(); }} />
                </Field>
                <Field label="heatmapLearningRate" htmlFor="cfg-hlr">
                  <input id="cfg-hlr" className="field-input" type="number"
                    value={params.heatmapLearningRate} min={0.001} max={1} step={0.01}
                    onChange={(e) => setParams((p) => ({ ...p, heatmapLearningRate: safeNum(e.target.value, p.heatmapLearningRate) }))} />
                </Field>
                <Field label="heatmapDecayRate" htmlFor="cfg-hdr">
                  <input id="cfg-hdr" className="field-input" type="number"
                    value={params.heatmapDecayRate} min={0.9} max={1} step={0.001}
                    onChange={(e) => setParams((p) => ({ ...p, heatmapDecayRate: safeNum(e.target.value, p.heatmapDecayRate) }))} />
                </Field>
              </div>
            </div>

            <VariableReference />
          </>
        )}

        {/* RUNTIME TAB */}
        {activeTab === 'runtime' && (
          <div className="config-section">
            <div className="speed-row">
              <div className="speed-row-label">Training speed</div>
              <div className="segmented" role="group" aria-label="Training speed">
                {trainingSpeedOptions.map((s) => (
                  <button
                    key={s}
                    className={`segmented-btn${trainingSpeed === s ? ' active' : ''}`}
                    onClick={() => updateTrainingSpeed(s)}
                    aria-pressed={trainingSpeed === s}
                  >
                    {s.charAt(0).toUpperCase() + s.slice(1)}
                  </button>
                ))}
              </div>
            </div>
            <div className="field-grid">
              <Field label="steps/frame" unit="int" htmlFor="cfg-spf">
                <input id="cfg-spf" className="field-input" type="number"
                  value={stepsPerFrame} min={1} max={5000} step={1}
                  onChange={(e) => setStepsPerFrame((prev) => safeNum(e.target.value, prev))} />
              </Field>
              <Field label="renderEveryN" unit="int" htmlFor="cfg-ren">
                <input id="cfg-ren" className="field-input" type="number"
                  value={renderEveryNSteps} min={1} max={1000} step={1}
                  onChange={(e) => setRenderEveryNSteps((prev) => safeNum(e.target.value, prev))} />
              </Field>
              <Field label="maxEpisodeSteps" unit="int" htmlFor="cfg-mes">
                <input id="cfg-mes" className="field-input" type="number"
                  value={params.maxEpisodeSteps} min={20} max={10000} step={10}
                  onChange={(e) => setParams((p) => ({ ...p, maxEpisodeSteps: safeNum(e.target.value, p.maxEpisodeSteps) }))} />
              </Field>
              <Field label="seed" unit="int" htmlFor="cfg-seed">
                <input id="cfg-seed" className="field-input" type="number"
                  value={seed} min={0} max={999999} step={1}
                  onChange={(e) => setSeed((prev) => safeNum(e.target.value, prev))} />
              </Field>
            </div>
          </div>
        )}
      </div>

      {/* Sticky footer */}
      <div className="config-footer">
        <button className="footer-btn" onClick={onResetQ}>Reset Q</button>
        <button className="footer-btn" onClick={onSavePolicy}>Save policy</button>
        <label className="footer-btn" style={{ cursor: 'pointer' }}>
          Load
          <input hidden type="file" accept="application/json" onChange={async (e) => {
            const file = e.target.files?.[0];
            if (file) await onLoadPolicy(file);
            e.target.value = ''; // allow re-selecting the same file after a fix
          }} />
        </label>
      </div>
    </div>
  );
}
