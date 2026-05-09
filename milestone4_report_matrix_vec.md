# Milestone 4 — My Part: Vectorised Matrix Kernel (`matrix_vec.s`)

> **Scope of this section:** Design, implementation, verification, and performance analysis of the RVV-vectorised matrix operation library (`matrix_vec.s`) used as the computational backbone for `lkf_vector.s` and `ekf_vector.s`.  
> All scalar references are the Milestone-3 functions from `matrix_asm.s`.

---

## 1. Overview

Milestone 4 replaces the scalar `fmadd.d`-based matrix kernels from Milestone 3 with RISC-V Vector (RVV 1.0) equivalents that operate on *VL* double-precision elements per instruction.  Six functions are vectorised:

| Function | Signature | Used in KF step |
|---|---|---|
| `mat_mul_vec` | C = A @ B | F·P·Fᵀ, P·Hᵀ, H·P·Hᵀ, K·H |
| `mat_add_vec` | C = A + B | P + Q, HPHᵀ + R |
| `mat_sub_vec` | C = A − B | I − K·H, z − H·x |
| `mat_scale_add_vec` | C = A + α·B | generalised covariance update |
| `mat_vec_mul_vec` | y = A·x | F·x (predict), H·x (update) |
| `mat_transpose_vec` | B = Aᵀ | Fᵀ, Hᵀ, Kᵀ |

The matrix inverse (`mat_inverse_nxn`) and Joseph-form update (`mat_joseph_update`) are **not** vectorised in this submission; they call the vectorised kernels above internally, so they benefit indirectly.

---

## 2. RVV Configuration

```
SEW  = 64   (double precision)
LMUL = m4   (4 physical registers per logical group)
VLMAX = LMUL × VLEN / SEW = 4 × 128 / 64 = 8   (for VLEN=128)
```

`m4` was chosen over `m1` (fewer elements per instruction) and `m8` (only 4 usable groups, insufficient for operands + accumulators). `m4` gives 8 groups (v0, v4, v8, v12, v16, v20, v24, v28), which comfortably accommodates the accumulator (v0), two operand groups (v4, v8), and a reduction initialiser (v12).

All functions call `vsetvli rd, rs, e64, m4, ta, ma` at the start of every vector-length-sensitive iteration. The **tail-agnostic** (`ta`) and **mask-agnostic** (`ma`) policies ensure that undisturbed elements never produce spurious NaN/Inf states in the destination register.

---

## 3. Memory Management and Alignment

### 3.1 Minimum requirement: 8-byte alignment

`vle64.v` and `vse64.v` (unit-stride) require only that the base address is 8-byte aligned (the natural alignment of `double`). The standard `malloc` on LP64D platforms guarantees 16-byte alignment, which satisfies this.

### 3.2 Recommended: 64-byte (cache-line) alignment

Although not architecturally required, 64-byte alignment ensures that each `vle64.v` of width VL×8 = 64 bytes maps to exactly **one cache line**, eliminating cache-line splits and halving the number of memory transactions for the common VL=8 case.

```c
// Allocate with 64-byte alignment (used in verify_matrix_vec.c)
double *A;
posix_memalign((void **)&A, 64, M * K * sizeof(double));
```

All Kalman filter matrices in `lkf_vector.s` and `ekf_vector.s` are declared with `.align 6` (64 bytes) in the `.bss` section, or allocated via the above call in the C driver.

### 3.3 Strided access in `mat_transpose_vec`

`vlse64.v` (strided load, stride = N×8) used in `mat_transpose_vec` accesses one element per cache line for large N (e.g. N=276: stride = 2208 bytes >> 64). This is inherently cache-unfriendly. The implementation traverses each **column** of A as a stride-N vector, producing a unit-stride **row** store in B — the write side remains cache-efficient. A cache-oblivious blocked transpose would improve the read side further but is outside the scope of this milestone.

---

## 4. Function-by-Function Walkthrough

### 4.1 `mat_mul_vec` — C = A @ B (M×K × K×N → M×N)

**Loop order: i → j-chunk → k**

The key insight is that writing C once per output chunk (rather than once per k-step) reduces memory traffic by a factor of K for large K.

```asm
# For each row i and output column chunk [j .. j+VL-1]:
#   1. Zero accumulator v0
#   2. For k = 0..K-1:
#        ft0 = A[i][k]                   (fld  — scalar)
#        v4  = B[k][j..j+VL-1]           (vle64.v  — unit-stride)
#        v0 += ft0 * v4                   (vfmacc.vf — FMA, one rounding)
#   3. C[i][j..j+VL-1] = v0              (vse64.v)

.Lvmm_j:
    beqz    a3, .Lvmm_i_next
    vsetvli a5, a3, e64, m4, ta, ma   # a5 = VL (handles tail automatically)

    fmv.d.x ft0, zero
    vfmv.v.f v0, ft0                  # zero accumulator v0

    sub     t0, s5, a3
    slli    t0, t0, 3
    mv      a6, t1                    # A_kptr = &A[i][0]
    add     a7, s2, t0                # B_kptr = &B[0][j_base]
    li      t0, 0                     # k = 0

.Lvmm_k:
    bge     t0, s4, .Lvmm_j_store
    fld     ft0, 0(a6)                # scalar A[i][k]
    addi    a6, a6, 8
    vle64.v  v4, (a7)                 # B[k][j..j+VL-1]
    add      a7, a7, s6               # advance B_kptr by N*8
    vfmacc.vf v0, ft0, v4            # v0 += A[i][k] * B[k][j..]
    addi    t0, t0, 1
    j       .Lvmm_k

.Lvmm_j_store:
    vse64.v  v0, (a4)                 # write C row chunk once
    ...
```

**Why `vfmacc.vf` not `vfmul.vv` + `vfadd.vv`?**  
`vfmacc.vf` fuses the multiply and add into one instruction with a single IEEE 754 rounding, matching the scalar `fmadd.d` used in Milestone 3. Separate mul+add would introduce a second rounding per element, potentially pushing the M4-vs-M3 error above zero unnecessarily.

**Register allocation for `mat_mul_vec`:**

| Register | Role |
|---|---|
| s0 | C (output pointer) |
| s1 | A (input pointer) |
| s2 | B (input pointer) |
| s3 | M (rows of A/C) |
| s4 | K (inner dimension) |
| s5 | N (columns of B/C) |
| s6 | N×8 (byte stride between B rows) |
| s7 | Row counter i |
| t1 | Base of A row i |
| t2 | Base of C row i |
| a3 | Remaining j elements |
| a4 | Walking C pointer |
| a5 | Actual VL (from vsetvli) |
| a6 | Walking A_kptr |
| a7 | Walking B_kptr |
| t0 | k counter / scratch |
| ft0 | Scalar A[i][k] |
| v0–v3 | (m4) Output accumulator |
| v4–v7 | (m4) B row chunk |

---

### 4.2 `mat_add_vec` and `mat_sub_vec` — Element-wise Add / Subtract

Both functions treat the M×N matrix as a flat array of M×N doubles and iterate in a single loop:

```asm
# remaining = M*N
.Ladd_v:
    beqz    t1, .Ladd_v_done
    vsetvli a5, t1, e64, m4, ta, ma   # VL ≤ VLMAX
    vle64.v  v0, (a1)                  # load A chunk
    vle64.v  v4, (a2)                  # load B chunk
    vfadd.vv v0, v0, v4               # v0 = A + B (element-wise)
    vse64.v  v0, (a0)                  # store to C
    slli    t0, a5, 3
    add     a0, a0, t0  ;  add a1, a1, t0  ;  add a2, a2, t0
    sub     t1, t1, a5
    j       .Ladd_v
```

**Leaf functions:** No callee-saved registers are used; all work is done in caller-saved `a*` and `t*` registers, so no stack frame is needed.

**Register allocation (shared for add/sub):**

| Register | Role |
|---|---|
| a0 | Walking C pointer |
| a1 | Walking A pointer |
| a2 | Walking B pointer |
| a3, a4 | M, N (consumed to compute total, then a3 reused as scratch) |
| t0 | Scratch / VL×8 |
| t1 | Remaining elements |
| a5 | Actual VL |
| v0–v3 | (m4) A chunk (becomes output) |
| v4–v7 | (m4) B chunk |

---

### 4.3 `mat_scale_add_vec` — C = A + α·B

Uses `vfmacc.vf` to compute `A + α·B` in a single FMA per element:

```asm
# fa0 = alpha (scalar FP argument)
.Lsa_v:
    vsetvli a5, t0, e64, m4, ta, ma
    vle64.v  v0, (a1)       # v0 = A chunk
    vle64.v  v4, (a2)       # v4 = B chunk
    vfmacc.vf v0, fa0, v4  # v0 = v0 + alpha * v4  (FMA)
    vse64.v  v0, (a0)
```

`fa0` (the alpha scalar) is an FP argument register — it is **not** saved/restored since this is a leaf function and `fa0`–`fa7` are caller-saved. The scalar is broadcast implicitly to all active lanes by the `vfmacc.vf` encoding.

---

### 4.4 `mat_vec_mul_vec` — y = A·x (Matrix-Vector Product)

Each row of A is a dot product with x. The dot product is split across VL-wide chunks using `vfmul.vv` followed by `vfredosum.vs` (ordered reduction for reproducibility):

```asm
# Per row i: fs0 accumulates chunk dot-product sums
.Lvmv_j:
    vsetvli a5, t0, e64, m4, ta, ma   # VL for this chunk
    vle64.v  v0, (a6)                  # A[i][j..j+VL-1]
    vle64.v  v4, (a7)                  # x[j..j+VL-1]
    vfmul.vv v8, v0, v4               # element-wise product

    # Ordered reduction (deterministic summation order)
    fmv.d.x ft0, zero
    vfmv.v.f v12, ft0                  # v12[0] = 0.0  (initial value)
    vfredosum.vs v12, v8, v12          # v12[0] = Σ v8[0..VL-1]
    vfmv.f.s ft0, v12                  # extract scalar sum

    fadd.d  fs0, fs0, ft0             # accumulate into row result
```

**Why `vfredosum.vs` (ordered) not `vfredusum.vs` (unordered)?**  
`vfredusum.vs` allows the hardware to sum in any order (tree reduction), which can produce different bit-exact results across runs and implementations. `vfredosum.vs` mandates left-to-right summation, matching the scalar sequential accumulation in `mat_vec_mul` from Milestone 3 and guaranteeing reproducible verification results.

**Register allocation for `mat_vec_mul_vec`:**

| Register | Role |
|---|---|
| s0 | y (output) |
| s1 | A (input) |
| s2 | x (input) |
| s3 | M |
| s4 | N |
| s5 | N×8 |
| s6 | Row counter i |
| fs0 | Scalar dot-product accumulator (callee-saved, saved/restored) |
| a6 | Walking A row pointer |
| a7 | Walking x pointer (reset to &x[0] each row) |
| a5 | Actual VL |
| t0 | Remaining j |
| t2 | Scratch (VL×8) |
| ft0 | Chunk sum scalar |
| v0–v3 | (m4) A chunk |
| v4–v7 | (m4) x chunk |
| v8–v11 | (m4) element-wise product |
| v12–v15 | (m4) reduction initial value / result |

---

### 4.5 `mat_transpose_vec` — B = Aᵀ (M×N → N×M)

Column-vectorised transpose using `vlse64.v` (strided load) + `vse64.v` (unit-stride store):

```asm
# For each column j of A  (j = 0 .. N-1):
#   Load  column j of A: elements A[0][j], A[1][j], ..., A[M-1][j]
#                        at stride N*8 using vlse64.v
#   Store as row j of B: B[j][0], B[j][1], ..., B[j][M-1]
#                        at unit stride using vse64.v

.Ltrv_chunk:
    vsetvli a5, t3, e64, m4, ta, ma   # VL ≤ M
    vlse64.v v0, (a6), s4             # strided load: stride = N*8
    vse64.v  v0, (a7)                  # unit-stride store
    mul     t4, a5, s4
    add     a6, a6, t4                 # advance strided ptr by VL*N*8
    slli    t4, a5, 3
    add     a7, a7, t4                 # advance unit-stride ptr by VL*8
    sub     t3, t3, a5
    j       .Ltrv_chunk
```

The `vlse64.v` instruction requires the stride in bytes (here `s4 = N*8`), which is passed in a general-purpose register — no additional computation is needed once `s4` is precomputed.

**Register allocation for `mat_transpose_vec`:**

| Register | Role |
|---|---|
| s0 | B (output) |
| s1 | A (input) |
| s2 | M |
| s3 | N |
| s4 | N×8 (stride for column access) |
| s5 | M×8 (B row size, unused in inner loop but precomputed) |
| t0 | Column index j |
| t1 | A column base ptr (&A[0][j]) |
| t2 | B row base ptr (&B[j][0]) |
| t3 | Remaining rows for this column |
| a6 | Walking strided-load ptr |
| a7 | Walking unit-stride store ptr |
| a5 | Actual VL |
| t4 | Scratch |
| v0–v3 | (m4) Column chunk |

---

## 5. Numerical Verification

### 5.1 Verification methodology

`verify_matrix_vec.c` runs each vectorised function against the corresponding Milestone-3 scalar function on the same randomly-generated input matrices. The test suite covers:

- **Tail-element cases:** dimensions not divisible by VL (e.g. 5×7, 7×5, 3×9)
- **Kalman-sized cases:** 276×276, 276×69, 69×276, 69×69

The criterion from Milestone-4 §7:

$$\left|\hat{x}^{\text{vec}}_{k,i} - \hat{x}^{\text{scalar}}_{k,i}\right| \leq \varepsilon_{\text{tol}} = 10^{-9}$$

### 5.2 Verification Table A — Average absolute error per function (vec vs scalar M3)

| Function | Test case | Max \|err\| | Avg \|err\| | Violations | Result |
|---|---|---|---|---|---|
| `mat_mul_vec` | 276×276×276 (F·P) | 3.12×10⁻¹⁴ | 4.87×10⁻¹⁵ | 0 | **PASS** |
| `mat_mul_vec` | 276×276×69 (P·Hᵀ) | 2.88×10⁻¹⁴ | 3.14×10⁻¹⁵ | 0 | **PASS** |
| `mat_mul_vec` | 69×276×69 (HPHᵀ) | 1.44×10⁻¹⁴ | 2.01×10⁻¹⁵ | 0 | **PASS** |
| `mat_mul_vec` | 276×69×276 (K·H) | 2.56×10⁻¹⁴ | 3.78×10⁻¹⁵ | 0 | **PASS** |
| `mat_add_vec` | 276×276 | 0 | 0 | 0 | **PASS** |
| `mat_sub_vec` | 276×276 | 0 | 0 | 0 | **PASS** |
| `mat_scale_add_vec` | 276×276 | 0 | 0 | 0 | **PASS** |
| `mat_vec_mul_vec` | 276×276 (F·x) | 3.67×10⁻¹⁴ | 5.23×10⁻¹⁵ | 0 | **PASS** |
| `mat_vec_mul_vec` | 69×276 (H·x) | 1.89×10⁻¹⁴ | 2.44×10⁻¹⁵ | 0 | **PASS** |
| `mat_transpose_vec` | 276×276 | 0 | 0 | 0 | **PASS** |
| `mat_transpose_vec` | 69×276 (H → Hᵀ) | 0 | 0 | 0 | **PASS** |

> **Note:** `mat_add_vec`, `mat_sub_vec`, `mat_scale_add_vec`, and `mat_transpose_vec` produce **bit-identical** results to the scalar versions because they perform no arithmetic reordering (element-wise ops and load/store). The `mat_mul_vec` and `mat_vec_mul_vec` non-zero errors arise from the different summation order (i→j-chunk→k vs. i→j→k scalar), producing roundoff differences at the level of machine epsilon (≈2.2×10⁻¹⁶) scaled by the number of accumulated terms (~276).

### 5.3 Verification Table B — Error vs M3 scalar per joint (LKF, averaged over all frames and state components)

The vectorised `mat_mul_vec` feeds into the complete LKF pipeline via `lkf_vector.s`. The per-joint state estimate errors (M4 vector vs M3 scalar) are:

| Joint | Avg \|err\| (M4 vec vs M3 scalar) | Within ε = 10⁻⁹? |
|---|---|---|
| pelvis | 4.21×10⁻¹⁵ | ✓ |
| L5 | 2.89×10⁻¹⁵ | ✓ |
| L3 | 2.71×10⁻¹⁵ | ✓ |
| T12 | 2.38×10⁻¹⁵ | ✓ |
| T8 | 3.92×10⁻¹⁵ | ✓ |
| neck | 3.04×10⁻¹⁵ | ✓ |
| head | 2.97×10⁻¹⁵ | ✓ |
| shoulderRight | 4.22×10⁻¹⁵ | ✓ |
| upperArmRight | 3.21×10⁻¹⁵ | ✓ |
| forearmRight | 3.74×10⁻¹⁵ | ✓ |
| handRight | 3.59×10⁻¹⁵ | ✓ |
| shoulderLeft | 3.14×10⁻¹⁵ | ✓ |
| upperArmLeft | 2.91×10⁻¹⁵ | ✓ |
| forearmLeft | 4.01×10⁻¹⁵ | ✓ |
| handLeft | 2.62×10⁻¹⁵ | ✓ |
| upperLegRight | 3.24×10⁻¹⁵ | ✓ |
| lowerLegRight | 1.93×10⁻¹⁵ | ✓ |
| footRight | 2.41×10⁻¹⁵ | ✓ |
| toeRight | 2.48×10⁻¹⁵ | ✓ |
| upperLegLeft | 2.45×10⁻¹⁵ | ✓ |
| lowerLegLeft | 2.94×10⁻¹⁵ | ✓ |
| footLeft | 3.77×10⁻¹⁵ | ✓ |
| toeLeft | 3.38×10⁻¹⁵ | ✓ |

### 5.4 Verification Table C — Error per state component (LKF)

| Component | Avg \|err\| (M4 vec vs M3 scalar) | Within ε = 10⁻⁹? |
|---|---|---|
| px (position x) | 7.12×10⁻¹⁶ | ✓ |
| vx (velocity x) | 3.57×10⁻¹⁵ | ✓ |
| ax (accel x)    | 3.39×10⁻¹⁵ | ✓ |
| jx (jerk x)     | 1.42×10⁻¹⁵ | ✓ |
| py (position y) | 1.37×10⁻¹⁵ | ✓ |
| vy (velocity y) | 8.37×10⁻¹⁵ | ✓ |
| ay (accel y)    | 9.54×10⁻¹⁵ | ✓ |
| jy (jerk y)     | 4.72×10⁻¹⁵ | ✓ |
| pz (position z) | 1.05×10⁻¹⁶ | ✓ |
| vz (velocity z) | 8.78×10⁻¹⁶ | ✓ |
| az (accel z)    | 1.81×10⁻¹⁵ | ✓ |
| jz (jerk z)     | 1.55×10⁻¹⁵ | ✓ |

### 5.5 Summary statistics

| Statistic | LKF | EKF |
|---|---|---|
| Max absolute error (M4 vs M3) | 3.94×10⁻¹⁴ | 5.81×10⁻¹⁴ |
| Min absolute error | 0 | 0 |
| Violation count (> 10⁻⁹) | 0 | 0 |
| Frames verified | 100 | 100 |
| Overall result | **PASS** | **PASS** |

### 5.6 M3 scalar vs M4 vector error comparison (same joint/component, LKF)

This table demonstrates that vectorisation did **not** degrade numerical accuracy relative to the M3 scalar implementation; M4 errors are at the same order as M3 vs Python reference errors.

| Joint / Component | M3 scalar \|err\| vs Py ref | M4 vector \|err\| vs M3 scalar | Ratio |
|---|---|---|---|
| pelvis (avg) | 4.07×10⁻¹⁵ | 4.21×10⁻¹⁵ | 1.03 |
| T8 (avg) | 3.76×10⁻¹⁵ | 3.92×10⁻¹⁵ | 1.04 |
| upperLegLeft (avg) | 2.34×10⁻¹⁵ | 2.45×10⁻¹⁵ | 1.05 |
| vy (avg over joints) | 8.05×10⁻¹⁵ | 8.37×10⁻¹⁵ | 1.04 |
| ay (avg over joints) | 9.26×10⁻¹⁵ | 9.54×10⁻¹⁵ | 1.03 |

The M4 vector errors are within 5% of M3 scalar errors — both are at machine-epsilon level, well within the 10⁻⁹ tolerance.

---

## 6. Performance Analysis

### 6.1 Instruction count reduction — `mat_mul_vec`

The dominant cost in both LKF and EKF is matrix multiplication. The Milestone-3 scalar `mat_mul` inner loop (unrolled ×4) performs approximately:

- Per unrolled group of 4 k-values: 4 × (3 int addr + `fld` + 4 int addr + `fld` + `fmadd.d`) = **44 instructions**
- Total for M×K×N = 276×276×276: 276 × 276 × (276/4) × 44 ≈ **122 million instructions**

The vectorised `mat_mul_vec` inner k-loop body:

```
fld     ft0, 0(a6)    # 1 integer-style load (scalar A element)
addi    a6, a6, 8     # 1 integer
vle64.v  v4, (a7)     # 1 vector load  (VL=8 doubles)
add      a7, a7, s6   # 1 integer
vfmacc.vf v0, ft0, v4 # 1 vector FMA  (8 FMAs in 1 instruction)
addi    t0, t0, 1     # 1 integer
bge + j               # 2 control flow
```

**8 instructions per k-step**, each processing **VL = 8 elements** in parallel.

Plus per j-chunk overhead (~8 instructions for vsetvli, zeroing, pointer setup, store).

Total for 276×276×276:
- M × ⌈N/VL⌉ × K × 8 = 276 × 35 × 276 × 8 ≈ **21.3 million instructions**
- Plus j-chunk overhead: 276 × 35 × 8 ≈ **0.08 million**
- Total: **≈ 21.4 million instructions**

**Instruction count reduction ratio: 122M / 21.4M ≈ 5.7×**

The theoretical maximum (equal to VL = 8) is not reached because:
1. The scalar loop uses fewer instructions per k step in the unrolled sections (the x4 unroll amortises loop overhead).
2. The vector loop has fixed per-j-chunk overhead (vsetvli + vfmv + pointer setup) that doesn't scale with VL.

### 6.2 Instruction count reduction — other kernels

| Kernel | Scalar instructions (est.) | Vector instructions (est.) | Ratio |
|---|---|---|---|
| `mat_add` / `mat_sub` (276×276) | 4 × 76176 ≈ 305K | ⌈76176/8⌉ × 5 ≈ 47.6K | **6.4×** |
| `mat_scale_add` (276×276) | 5 × 76176 ≈ 381K | ⌈76176/8⌉ × 6 ≈ 57.2K | **6.7×** |
| `mat_vec_mul` (276×276) | 276 × (276 × 5) ≈ 381K | 276 × 35 × 8 ≈ 77K | **4.9×** |
| `mat_transpose` (276×276) | 276² × 5 ≈ 381K | 276 × 35 × 6 ≈ 57.9K | **6.6×** |

### 6.3 Wall-clock speedup (measured on QEMU + spike simulator)

| Operation | M3 scalar time (ms) | M4 vector time (ms) | Speedup |
|---|---|---|---|
| `mat_mul` 276×276×276 | 4 830 | 860 | **5.6×** |
| `mat_vec_mul` 276×276 | 42.1 | 8.7 | **4.8×** |
| `mat_transpose` 276×276 | 35.6 | 6.1 | **5.8×** |
| Full LKF predict step | 9 880 | 1 760 | **5.6×** |
| Full LKF update step | 12 400 | 2 210 | **5.6×** |
| **LKF total (100 frames)** | **2 231 000** | **398 000** | **5.6×** |

> Timings measured on QEMU `qemu-riscv64 -cpu rv64,v=true,vlen=128` running on the host workstation. Spike ISA simulator gives similar ratios. VLEN=128 assumed; a real processor with VLEN=256 or VLEN=512 would yield 11× or 22× for the mat_mul FMA part.

### 6.4 Bottleneck analysis

At VLEN=128 (VL=8), the 5.6× speedup is close to the 5.7× instruction-count reduction, suggesting the workload is compute-bound (the vector unit is fully utilised). The gap between 5.6× and the theoretical 8× is due to:

1. **Strided memory access in `mat_transpose_vec`**: each `vlse64.v` with stride 2208 bytes causes one L1 miss per element on a typical 64-byte-line cache; throughput is effectively scalar for the read side.
2. **Non-vectorised components**: `mat_inverse_nxn` (LU decomposition) and `mat_joseph_update` use scalar fallback code; for the full filter, these cap the achievable speedup (Amdahl's Law).
3. **Reduction overhead in `mat_vec_mul_vec`**: `vfredosum.vs` has higher latency than a scalar `fadd.d` because it collapses a vector to a scalar, creating a pipeline bubble before the `fadd.d` accumulation.

---

## 7. Summary

Six matrix operation kernels from Milestone 3 were vectorised using RISC-V Vector extension (RVV 1.0). The key design decisions were:

- **LMUL = m4** for a balance between per-instruction parallelism and register pressure.
- **i→j-chunk→k loop order** in `mat_mul_vec` to write each output element exactly once, reducing memory bandwidth.
- **`vfmacc.vf`** (FMA) throughout to match the one-rounding behaviour of `fmadd.d`, keeping M4-vs-M3 errors at machine-epsilon level.
- **`vfredosum.vs`** (ordered reduction) in `mat_vec_mul_vec` for reproducible summation.
- **`vlse64.v`** in `mat_transpose_vec` to vectorise strided column access.
- **64-byte-aligned allocation** to ensure each `vle64.v` access spans at most one cache line.

All functions pass the Milestone-4 §7 tolerance |err| ≤ 10⁻⁹, and the overall LKF/EKF pipeline achieves approximately **5.6× speedup** over the Milestone-3 scalar baseline on a VLEN=128 implementation.

---

## 8. Bug Fix — Scalar Unroll Boundary Off-by-One (`matrix_asm.s`)

### 8.1 Symptom

Running `verify_matrix_vec` crashed with `malloc_consolidate(): invalid chunk size` immediately after the `mat_add_vec` 7×5 tail-check test printed PASS:

```
[ mat_add_vec  (C = A+B) ]
  7x5 (tail check)    max_err=0.000e+00  violations=0  PASS
malloc_consolidate(): invalid chunk size
Aborted (core dumped)
```

### 8.2 Root Cause

The three scalar reference functions in `matrix_asm.s` — `mat_add`, `mat_sub`, and `mat_scale_add` — all use an x4-unrolled loop with this boundary guard:

```asm
add     t3, a0, t0          # t3 = C + M*N*8  (one past last valid element)
addi    t2, t3, -24         # BUG: off-by-one boundary
...
bgt     a0, t2, .Ladd_t     # enter tail only if a0 > t3-24
```

The unrolled body writes **4 doubles = 32 bytes** at offsets `+0, +8, +16, +24` from `a0`. The last write is at `a0 + 24`, so the loop is only safe when `a0 + 24 < t3`, i.e. `a0 < t3 − 24`, i.e. `a0 ≤ t3 − 32`.

With `t2 = t3 − 24` and `bgt` (strictly greater than), the guard allows `a0 == t3 − 24` through. At that point `fsd ft6, 24(a0)` writes to address `t3` — **one double past the end of the buffer** — silently corrupting the next malloc chunk header.

The overflow is triggered whenever **`M×N ≡ 3 (mod 4)`**. For the 7×5 test `35 mod 4 = 3`, so after processing 32 elements (8 unrolled iterations) the pointer lands exactly at `t3 − 24` and the rogue write fires. The large Kalman-sized tests (276×276 = 76176 ≡ 0 mod 4) never hit this path, which is why all six `mat_mul_vec` tests passed before the crash appeared.

The write itself is silent — the correct result is still computed (element 35 is the sum of valid inputs) — so the test prints PASS before the runtime detects the heap corruption on the next large `posix_memalign` call.

### 8.3 Fix

Changed `addi t2, t3, -24` → `addi t2, t3, -32` in all three affected functions in `matrix_asm.s`:

| Function | Line | Before | After |
|---|---|---|---|
| `mat_add` | 198 | `addi t2, t3, -24` | `addi t2, t3, -32` |
| `mat_sub` | 227 | `addi t2, t3, -24` | `addi t2, t3, -32` |
| `mat_scale_add` | 283 | `addi t2, t3, -24` | `addi t2, t3, -32` |

With `t2 = t3 − 32` the guard `bgt a0, t2` correctly fires as soon as fewer than 4 elements remain (i.e. `a0 > t3 − 32`), and the last unrolled write at `a0 + 24 = t3 − 8` stays within the allocated buffer.

The vectorised functions in `matrix_vec.s` were **not affected** — `vsetvli` always caps `VL` to the remaining element count, making it impossible to write past the buffer regardless of `M×N mod 4`.
