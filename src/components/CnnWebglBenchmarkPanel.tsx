import { useState } from 'react';
import type {
  PortableCnnBackend,
  PortableCnnBenchmarkResult,
} from '../rl/cnnWebglBenchmark';

/** Development-only control for the portable T6 throughput diagnostic. */
export function CnnWebglBenchmarkPanel(): JSX.Element {
  const [backend, setBackend] = useState<PortableCnnBackend>('webgl');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<PortableCnnBenchmarkResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (): Promise<void> => {
    setRunning(true);
    setResult(null);
    setError(null);
    try {
      const { runPortableCnnBenchmark } = await import('../rl/cnnWebglBenchmark');
      setResult(await runPortableCnnBenchmark({ backend }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="panel" data-testid="cnn-webgl-benchmark">
      <div className="panel-header"><span className="panel-title">T6 Portable CNN Benchmark</span></div>
      <div className="panel-content">
        <label>
          Backend
          <select
            value={backend}
            onChange={(event) => setBackend(event.target.value as PortableCnnBackend)}
            disabled={running}
          >
            <option value="webgl">WebGL</option>
            <option value="webgpu">WebGPU (experimental training)</option>
          </select>
        </label>
        <button onClick={run} disabled={running}>
          {running ? 'Running cold + warmed updates…' : 'Run 30 warmed updates per batch'}
        </button>
        <p>Reload the page before a run when first-update latency must include fresh shader compilation.</p>
        {result && (
          <>
            <p>
              backend={result.backend}; renderer={result.device.renderer}; vendor={result.device.vendor}
              {result.device.isSoftwareRenderer === true && ' ⚠ software renderer detected'}
              {result.device.isSoftwareRenderer === false && ' (hardware acceleration verified)'}
              {result.device.isSoftwareRenderer === null && ' (hardware acceleration could not be verified)'}
            </p>
            <table>
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Cold ms</th>
                  <th>Updates/s</th>
                  <th>Samples/s</th>
                  <th>Profile kernel/readback ms</th>
                  <th>Slowest kernels</th>
                </tr>
              </thead>
              <tbody>
                {result.batches.map((batch) => (
                  <tr key={batch.batchSize}>
                    <td>{batch.batchSize}</td>
                    <td>{batch.coldLatencyMs.toFixed(1)}</td>
                    <td>{batch.updatesPerSec.toFixed(2)}</td>
                    <td>{batch.samplesPerSec.toFixed(1)}</td>
                    <td>{batch.profile.kernelMs.toFixed(1)} / {batch.profile.readbackMs.toFixed(1)}</td>
                    <td>
                      {[...batch.profile.kernels]
                        .filter((kernel) => kernel.timeMs !== null)
                        .sort((left, right) => (right.timeMs ?? 0) - (left.timeMs ?? 0))
                        .slice(0, 3)
                        .map((kernel) => `${kernel.name} ${kernel.timeMs?.toFixed(1)}ms`)
                        .join(', ')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
        {error && <p role="alert">{backend} update failed: {error}</p>}
      </div>
    </div>
  );
}
