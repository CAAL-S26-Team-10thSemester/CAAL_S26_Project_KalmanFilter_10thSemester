/*
 * matrix_vec.h  —  Declarations for RISC-V RVV vectorised matrix functions
 *                  implemented in matrix_vec.s  (Kalman Filter Milestone-4)
 *
 * All functions have identical signatures to their Milestone-3 scalar
 * counterparts in matrix_asm.h; the _vec suffix distinguishes them so that
 * both implementations can coexist in the same binary during cross-verification.
 *
 * ABI: RV64GCV  (LP64D + V extension, RVV 1.0)
 * All matrices: row-major, float64.  Element [i][j] of (m×n) matrix at p:
 *     p + (i*n + j) * sizeof(double)
 *
 * Memory alignment requirements:
 *   All pointer arguments MUST be 8-byte aligned (required by vle64.v/vse64.v).
 *   64-byte (cache-line) alignment is recommended for peak throughput — use
 *   posix_memalign(&ptr, 64, bytes) or __attribute__((aligned(64))).
 *
 * Numerical tolerance (Milestone-4 §7):
 *   |x_vec[k,i] - x_scalar[k,i]| <= epsilon_tol = 1e-9
 */

#ifndef MATRIX_VEC_H
#define MATRIX_VEC_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Matrix×Matrix product ------------------------------------------- */

/**
 * mat_mul_vec  —  C = A @ B   (M×K) × (K×N) → (M×N)
 *
 * Vectorised loop order: i → j-chunk (width VL) → k.
 * Each j-chunk accumulates Σ_k A[i][k] · B[k][j..j+VL-1] using vfmacc.vf
 * (one FMA rounding) then writes C once — no per-k write-back to C.
 *
 * LMUL=m4, SEW=64.  VLMAX=8 with VLEN=128; scales with larger VLEN.
 */
void mat_mul_vec(double *C,
                 const double *A,
                 const double *B,
                 int M, int K, int N);

/* ---- Element-wise operations ----------------------------------------- */

/**
 * mat_add_vec  —  C = A + B  (element-wise, M×N)
 * Flat-array vectorisation: treats M×N matrix as M*N contiguous doubles.
 */
void mat_add_vec(double *C,
                 const double *A,
                 const double *B,
                 int M, int N);

/**
 * mat_sub_vec  —  C = A - B  (element-wise, M×N)
 */
void mat_sub_vec(double *C,
                 const double *A,
                 const double *B,
                 int M, int N);

/**
 * mat_scale_add_vec  —  C = A + alpha * B  (element-wise, M×N)
 * Uses vfmacc.vf (FMA) for α·B + A in a single pass — no intermediate store.
 *
 * @param alpha  scalar multiplier passed in fa0
 */
void mat_scale_add_vec(double *C,
                       const double *A,
                       const double *B,
                       double alpha,
                       int M, int N);

/* ---- Matrix×Vector product ------------------------------------------- */

/**
 * mat_vec_mul_vec  —  y = A · x  (M×N matrix times N-vector)
 *
 * Dot-product per row using vfmul.vv + vfredosum.vs (ordered reduction for
 * deterministic summation across runs).  Chunk sums accumulated in scalar fs0.
 */
void mat_vec_mul_vec(double *y,
                     const double *A,
                     const double *x,
                     int M, int N);

/* ---- Matrix transpose ------------------------------------------------ */

/**
 * mat_transpose_vec  —  B = A^T   (A is M×N → B is N×M)
 *
 * Vectorised by column of A using vlse64.v (strided load, stride = N*8)
 * to read each column as a vector, then vse64.v (unit-stride store) to write
 * as the corresponding row of B.  Requires v extension strided-load support.
 */
void mat_transpose_vec(double *B,
                       const double *A,
                       int M, int N);

/* ---- Dimension constants (match matrix_asm.h) ------------------------ */

#ifndef NUM_JOINTS
#define NUM_JOINTS           23
#define STATE_DIM            12        /* per joint */
#define MEAS_DIM              3        /* per joint */
#define TOTAL_STATE_DIM     (NUM_JOINTS * STATE_DIM)   /* 276 */
#define TOTAL_MEAS_DIM      (NUM_JOINTS * MEAS_DIM)    /* 69  */
#endif

/* Milestone-4 §7 numerical tolerance */
#define EPSILON_VEC_TOL  1e-9

/* Recommended alignment for cache-line performance (bytes) */
#define MATRIX_ALIGN  64

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_VEC_H */
