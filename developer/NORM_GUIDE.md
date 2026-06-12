# Normalization guide for LCP-family distances

This note documents officially supported normalization choices for the LCP
distance family in Sequenzo, and clarifies the difference between design-level
parameters and runtime safety checks.

## Norm codes (C++ kernel)

| Code | Name       | Generic formula role                          |
|------|------------|-----------------------------------------------|
| 0    | none       | Return raw distance                           |
| 1    | maxlength  | Scale by max sequence length / duration       |
| 2    | gmean      | Geometric-mean style rescaling                |
| 3    | maxdist    | Divide by a method-specific upper bound       |
| 4    | YujianBo   | Yujian–Bo normalized dissimilarity            |

Invalid codes (`< 0` or `> 4`) are rejected in C++ constructors.

## Method-specific support

### Position-wise LCP / RLCP

- Supported in C++: `none`, `maxlength`, `gmean`, `maxdist`, and `YujianBo`.
- `norm="auto"` (Python) → `gmean`.
- `maxdist` and `YujianBo` are bounded rescalings.
- `maxlength` is accepted as a generic rescaling, but it is not guaranteed to
  lie in `[0, 1]` for all inputs.
- Input contract: full position-wise sequences of equal length. State codes
  (including `0`) are ordinary states, not padding sentinels.

### LCPspell / RLCPspell

- Supported in C++: `none` (0) and `maxdist` (3) only.
- `maxdist` uses a **conservative method-specific upper bound**:

  ```
  d_max = (n_s + m_s) + lambda * min(n_s, m_s)
  ```

  provided `duration_ref` (tau) is at least the largest spell duration allowed
  under the observation design. This bound prioritizes simplicity and stability
  over tightness.

- `gmean`, `maxlength`, and `YujianBo` are rejected at the C++ layer because
  they are not part of the LCPspell definition.

#### `duration_ref`: design-level vs runtime guard

**Design-level (Python wrapper / documentation):**

- `duration_ref` should be fixed before pairwise computation, preferably from
  the observation design (default: total observation window `T`).
- It is a reference scale for expressing spell durations as proportions, not a
  dataset-tuning knob.

**Runtime guard (C++ constructor):**

- When `norm=maxdist` and `expcost > 0`, C++ verifies that the supplied
  `duration_ref` is at least the largest **active spell duration in the
  current input arrays**.
- This is sample-level admissibility only. It does **not** replace the
  design-level requirement that `duration_ref` be chosen from the study design.

### LCPmst / RLCPmst

- Supported: `none`, `gmean`, `maxdist`, `YujianBo`
- **Not supported:** `maxlength` (rejected in C++). Raw MST distances are
  duration totals; dividing by `max(Tx, Ty)` is not a standard bounded
  normalization for this method.
- Normalized outputs are checked to lie in `[0, 1]` up to floating-point noise;
  values outside that range raise an error rather than being silently clipped.

### LCPprod / RLCPprod

- **Raw only:** `norm=none` (C++ and Python wrapper).
- Product-duration weighting yields a squared-duration raw dissimilarity.
  No method-specific upper bound is defined, so normalized dissimilarities are
  not supported.

## `refseqS` conventions (C++ kernel)

| `refseqS`   | Meaning                                              |
|-------------|------------------------------------------------------|
| `[-1, -1]`  | Full pairwise matrix; `compute_refseq_distances` invalid |
| `[k, k]`    | Single reference row, **1-based** index `k`          |
| `[a, b]`    | Subset mode: rows `0..a-1` vs `a..b-1`, **0-based half-open**, requires `0 < a < b == n_rows` |

Reversed ranges such as `[3, 2]` are rejected.

## Spell / DSS input contract

Spell-based kernels (`LCPspell`, `LCPmst`, `LCPprod`) expect **canonical**
representations:

- Adjacent active spells must have distinct states (DSS / collapsed spells).
- Active spell durations must be strictly positive.
- `totaldur[i]` must equal the sum of active spell durations for row `i`.

The Python `get_distance_matrix` wrapper builds these arrays from `SequenceData`.
Direct C++ callers must supply the same representation.

## Low-level C++ kernels vs public Python API

`get_distance_matrix()` is the supported public entry point. The pybind11 distance
classes are low-level kernels: they validate structural invariants (indices,
normalization codes, reference ranges, spell shape) but array arguments remain
subject to pybind11's documented dtype conversion behaviour (for example,
`py::array::forcecast` on position-wise `LCPdistance` sequences).
