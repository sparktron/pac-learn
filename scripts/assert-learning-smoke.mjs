import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const [outDir] = process.argv.slice(2);
if (!outDir) {
  throw new Error('usage: node scripts/assert-learning-smoke.mjs <output-dir>');
}

const runDir = (name) => join(outDir, name);
const read = (name, file) => readFileSync(join(runDir(name), file), 'utf8');

const evalsA = read('run-a', 'evals.csv');
const evalsB = read('run-b', 'evals.csv');
if (evalsA !== evalsB) {
  throw new Error('fixed-seed smoke is not reproducible: evals.csv differs between runs');
}

const summary = (name) => {
  const data = JSON.parse(read(name, 'summary.json'));
  delete data.elapsedSec;
  return JSON.stringify(data);
};
if (summary('run-a') !== summary('run-b')) {
  throw new Error('fixed-seed smoke is not reproducible: summary differs between runs');
}

const [header, ...lines] = evalsA.trim().split('\n');
const columns = header.split(',');
const column = (name) => {
  const index = columns.indexOf(name);
  if (index < 0) throw new Error(`evals.csv is missing ${name}`);
  return index;
};
const winsColumn = column('wins');
const winRateColumn = column('winRate');
const pelletP5Column = column('pl_p5');
const panelColumn = column('panel');
const rows = lines.map((line) => line.split(','));

if (rows.length !== 4) {
  throw new Error(`expected one final evaluation for each of four panels, got ${rows.length}`);
}
if (new Set(rows.map((row) => row[panelColumn])).size !== 4) {
  throw new Error('evaluation panels are not disjoint');
}

const winRates = rows.map((row) => Number(row[winRateColumn]));
const wins = rows.reduce((total, row) => total + Number(row[winsColumn]), 0);
const pelletP5s = rows.map((row) => Number(row[pelletP5Column]));
const meanWinRate = winRates.reduce((total, value) => total + value, 0) / winRates.length;
const worstPanelWinRate = Math.min(...winRates);
const worstPelletP5 = Math.max(...pelletP5s);

// Calibrated on 2026-07-29 with the promoted T7 linear baseline at 2,000
// episodes: 67/200 wins (33.5% mean), 20.0% worst panel, and pl_p5=0 on all
// panels. Floors deliberately leave room for an intentional, measured tuning
// change while failing fast on a broken RNG/key/reward/evaluation path.
if (wins < 60) throw new Error(`training smoke regressed: wins ${wins} < 60`);
if (meanWinRate < 0.30) {
  throw new Error(`training smoke regressed: mean win rate ${(meanWinRate * 100).toFixed(1)}% < 30.0%`);
}
if (worstPanelWinRate < 0.18) {
  throw new Error(`training smoke regressed: worst-panel win rate ${(worstPanelWinRate * 100).toFixed(1)}% < 18.0%`);
}
// Lower pellets-left percentiles are better. Zero means every 50-game panel
// includes enough wins to place a clear at its fifth percentile.
if (worstPelletP5 > 0) {
  throw new Error(`training smoke regressed: worst pl_p5 ${worstPelletP5} > 0`);
}

console.log(
  `[learning-smoke] reproducible; wins=${wins}/200 mean=${(meanWinRate * 100).toFixed(1)}% ` +
  `worstPanel=${(worstPanelWinRate * 100).toFixed(1)}% maxPlP5=${worstPelletP5}`,
);
