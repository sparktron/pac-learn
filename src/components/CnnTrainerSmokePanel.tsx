import { useState } from 'react';
import type { TensorBackend } from '../rl/tfRuntime';
import type { CnnTrainerSmokeResult } from '../rl/cnnTrainerSmoke';

const hours = (value: number): string => (value < 1 ? `${(value * 60).toFixed(1)} min` : `${value.toFixed(1)} h`);

/**
 * Development-only control for the T6 end-to-end trainer smoke. The portable
 * benchmark next door measures updates; this measures the whole loop, because
 * action selection runs `trainEvery` times more often and was never timed.
 */
export function CnnTrainerSmokePanel(): JSX.Element {
  const [backend, setBackend] = useState<TensorBackend>('webgl');
  const [totalSteps, setTotalSteps] = useState(256);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<string | null>(null);
  const [result, setResult] = useState<CnnTrainerSmokeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (): Promise<void> => {
    setRunning(true);
    setResult(null);
    setError(null);
    setProgress('starting…');
    try {
      const { runCnnTrainerSmoke } = await import('../rl/cnnTrainerSmoke');
      setResult(await runCnnTrainerSmoke({
        backend,
        totalSteps,
        onProgress: (done, total) => setProgress(`step ${done}/${total}`),
      }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
      setProgress(null);
    }
  };

  return (
    <div className="panel" data-testid="cnn-trainer-smoke">
      <div className="panel-header"><span className="panel-title">T6 End-to-End Trainer Smoke</span></div>
      <div className="panel-content">
        <label>
          Backend
          <select
            value={backend}
            onChange={(event) => setBackend(event.target.value as TensorBackend)}
            disabled={running}
          >
            <option value="webgl">WebGL</option>
            <option value="webgpu">WebGPU (experimental training)</option>
            <option value="cpu">CPU (portable baseline)</option>
            <option value="wasm">WASM</option>
          </select>
        </label>
        <label>
          Steps
          <input
            type="number"
            min={128}
            max={2_048}
            step={128}
            value={totalSteps}
            onChange={(event) => setTotalSteps(
              Math.min(2_048, Math.max(128, Number(event.target.value) || 128)),
            )}
            disabled={running}
          />
        </label>
        <button onClick={run} disabled={running}>
          {running ? `Running… ${progress ?? ''}` : 'Run end-to-end smoke'}
        </button>
        <p>
          Measures the real loop (encode → select → step → observe → periodic learn) and
          attributes wall time per phase. The fixed step budget keeps this diagnostic bounded;
          the gate projection uses the completed episodes&apos; mean length.
        </p>
        {result && (
          <>
            <p data-testid="smoke-headline">
              backend={result.backend}; {result.steps} steps / {result.updates} batch-{result.batchSize} updates
              {' '}across {result.episodes} completed episodes in {result.wallClockSec.toFixed(1)}s ={' '}
              <strong>{result.stepsPerSec.toFixed(1)} steps/sec</strong>
            </p>
            <table>
              <thead>
                <tr><th>Phase</th><th>Per op (ms)</th><th>Share of wall time</th></tr>
              </thead>
              <tbody>
                <tr>
                  <td>State encoding (current + next)</td>
                  <td>{result.encodingMs.toFixed(3)}</td>
                  <td>{(result.encodingShare * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                  <td>Environment + legal actions</td>
                  <td>{result.envStepMs.toFixed(3)}</td>
                  <td>{(result.envShare * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                  <td><strong>Action selection</strong> (every step)</td>
                  <td><strong>{result.actMs.toFixed(2)}</strong></td>
                  <td><strong>{(result.actShare * 100).toFixed(1)}%</strong></td>
                </tr>
                <tr>
                  <td>— of which forward + upload</td>
                  <td>{result.actForwardMs.toFixed(2)}</td>
                  <td>—</td>
                </tr>
                <tr>
                  <td>— of which GPU→CPU readback</td>
                  <td>{result.actReadbackMs.toFixed(2)}</td>
                  <td>—</td>
                </tr>
                <tr>
                  <td>Replay insertion</td>
                  <td>{result.replayMs.toFixed(3)}</td>
                  <td>{(result.replayShare * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                  <td>Batch-{result.batchSize} update (every {result.trainEvery} steps)</td>
                  <td>
                    {result.updates === 0
                      ? 'not measured'
                      : `${result.updateMs.toFixed(1)}${result.updateTimingIsCold ? ` ⚠ cold, n=${result.updates}` : ''}`}
                  </td>
                  <td>{result.updates === 0 ? '—' : `${(result.updateShare * 100).toFixed(1)}%`}</td>
                </tr>
              </tbody>
            </table>
            {result.updateProfile && (
              <p data-testid="smoke-update-profile">
                Profiled update: kernels {result.updateProfile.kernelMs.toFixed(1)} ms;
                {' '}scalar loss readback {result.updateProfile.readbackMs.toFixed(1)} ms;
                {' '}profile wall {result.updateProfile.wallMs.toFixed(1)} ms;
                {' '}loss {result.updateProfile.loss.toFixed(5)}.
              </p>
            )}
            <p data-testid="smoke-projection">
              Projected 2k-episode gate: {result.projection.steps.toLocaleString()} steps,{' '}
              {result.projection.updates.toLocaleString()} updates →{' '}
              encoding {hours(result.projection.encodingHours)} + env{' '}
              {hours(result.projection.envHours)} + inference{' '}
              {hours(result.projection.inferenceHours)} + replay{' '}
              {hours(result.projection.replayHours)} + update{' '}
              {hours(result.projection.updateHours)} ={' '}
              <strong>{hours(result.projection.totalHours)}</strong>.
              {' '}Without the per-step readback:{' '}
              <strong>{hours(result.projection.totalHoursWithoutReadback)}</strong>.
            </p>
            {result.updateTimingIsCold && result.updates > 0 && (
              <p role="alert">
                ⚠ Only {result.updates} update ran, so the update figure is first-use shader
                compilation, not throughput — the portable benchmark measured 6.3s cold against
                ~120ms warmed. The update term in the projection above is unusable; raise
                episodes until at least 5 updates run.
              </p>
            )}
            <p>
              Step count comes from this run&apos;s mean episode length
              ({(result.projection.steps / 2_000).toFixed(0)} steps/ep). An untrained policy
              dies early, so a real curve is longer and these totals are a floor.
            </p>
            <p>
              tensors={result.tensorMemory.numTensors}; bytes={result.tensorMemory.numBytes.toLocaleString()}
            </p>
          </>
        )}
        {error && <p role="alert">{backend} smoke failed: {error}</p>}
      </div>
    </div>
  );
}
