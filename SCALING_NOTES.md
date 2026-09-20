# Scaling notes: pairwise embedding comparison loop

Context: `main.py` computes embeddings for `documents` with `all-MiniLM-L6-v2`
(384 dimensions per embedding), then compares every pair of documents with a
nested loop that prints a per-dimension difference and a running total for
each pair.

## Measured baseline

- n = 30 documents → 435 pairs (n·(n−1)/2)
- Measured loop time (`stopwatch`, excludes model load/encode): **9.71s**

## Why this loop is O(n²) — and effectively O(n²·d²)

- Pair count grows as `n(n-1)/2`, so doubling `n` roughly quadruples the work.
- Inside each pair, `sum_difference` is recomputed from scratch on *every*
  iteration of the 384-step inner loop, instead of once per pair. That makes
  the per-pair cost O(d²) (d = 384) rather than O(d).
- Each pair also emits `384 × 2 = 768` `print()` calls. Console I/O ends up
  dominating wall-clock time well before the arithmetic does.

## Extrapolated estimates (scaling by pair count from the n=30 baseline)

| n (documents) | pairs | est. loop time |
|---|---|---|
| 30 | 435 | 9.71s (measured) |
| 500 | 124,750 | ~2,784s (~46 min) |
| 100,000 | ~5.0 billion | ~111.6M sec (~3.5 years) |

These are optimistic — they assume constant per-print throughput. In
practice, print-call overhead grows the effective cost further at scale:

- n = 500 → ~95.8 million print statements
- n = 100,000 → ~3.84 trillion print statements (not practically runnable)

Embedding-generation time (`model.encode(documents)`) is separate from the
timed loop and isn't included above; it becomes non-trivial at 100k+ documents
but is dwarfed by the O(n²) comparison cost regardless.

## Takeaway

The current design (per-dimension printing, O(d²) redundant sum) is fine for
demos at n≈30 but breaks down fast. To compare hundreds/thousands of
documents in practice:

- Drop the per-dimension `print` calls (or gate them behind a debug flag).
- Compute each pair's total difference once (don't recompute the sum per
  dimension iteration).
- Vectorize with NumPy or `scipy.spatial.distance.pdist` instead of nested
  Python loops — turns this into a handful of array operations instead of
  billions of scalar ones.
