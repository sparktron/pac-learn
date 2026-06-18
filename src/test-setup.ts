// Vitest global setup. Registers @testing-library/jest-dom matchers
// (toBeInTheDocument, toHaveAttribute, …) against Vitest's expect for the
// component tests. Safe in the node environment too — it only extends expect.
import '@testing-library/jest-dom/vitest';
