export class SeededRng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  next(): number {
    this.state += 0x6d2b79f5;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  // Returns an integer in [0, maxExclusive). Guards against non-positive
  // bounds, which would otherwise yield 0 or a negative index — JS array
  // access then returns undefined rather than throwing, masking the bug.
  int(maxExclusive: number): number {
    if (!(maxExclusive > 0)) return 0; // also catches NaN
    return Math.floor(this.next() * maxExclusive);
  }
}
