/*
 * matrix_asm.h  —  Declarations for all RISC-V assembly matrix functions
 *                  implemented in matrix_asm.s
 *
 * Include this in any C driver that links against matrix_asm.s, and also
 * reference it from lkf_asm.s / ekf_asm.s via .extern directives.
 *
 * All matrices are stored ROW-MAJOR, double-precision (float64).
 * Element [i][j] of an (m×n) matrix A at base pointer p:
 *     p + (i*n + j) * sizeof(double)
 *
 * Tolerance requirement (Milestone-3 §6):
 *     |x_asm[k,i] - x_ref[k,i]| <= 1e-9
 */

#ifndef MATRIX_ASM_H
#define MATRIX_ASM_H

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Basic element-wise operations --------------------------------------- */

/**
 * mat_eye  —  Write n×n identity into A.
 * @param A   output pointer (n×n, row-major doubles)
 * @param n   dimension
 */
void mat_eye(double *A, int n);

/**
 * mat_mul  —  C = A @ B  (m×k  ×  k×n  ->  m×n)
 * Uses fmadd.d with 4-way unrolled inner loop.
 */
void mat_mul(double *C,
             const double *A, const double *B,
             int m, int k, int n);

/**
 * mat_add  —  C = A + B  (element-wise, m×n)
 */
void mat_add(double *C,
             const double *A, const double *B,
             int m, int n);

/**
 * mat_sub  —  C = A - B  (element-wise, m×n)
 */
void mat_sub(double *C,
             const double *A, const double *B,
             int m, int n);

/**
 * mat_transpose  —  B = A^T  (A is m×n, B is n×m)
 */
void mat_transpose(double *B,
                   const double *A,
                   int m, int n);

/**
 * mat_scale_add  —  C = A + alpha * B  (element-wise, m×n)
 * Uses fmadd.d with 4-way unrolled loop.
 */
void mat_scale_add(double *C,
                   const double *A, const double *B,
                   double alpha,
                   int m, int n);

/**
 * mat_vec_mul  —  y = A * x  (m×n matrix times n-vector)
 * Optimised for the common Kalman predict/update pattern.
 */
void mat_vec_mul(double *y,
                 const double *A, const double *x,
                 int m, int n);

/* ---- Matrix inverse ------------------------------------------------------ */

/**
 * mat_inverse_nxn  —  LU-decomposition-based in-place inverse.
 *
 * Corresponds to mat_inverse_nxn in kalman-updated.py (np.linalg.inv).
 * Uses partial pivoting (Doolittle form) for numerical stability.
 * Absolute error vs numpy reference: <= 1e-9 for well-conditioned inputs.
 *
 * @param Ainv  output: n×n inverse (may equal A only if scratch != NULL)
 * @param A     input:  n×n matrix (read-only)
 * @param n     dimension
 * @param piv   scratch integer array of length n (pivot indices)
 * @return      1 = success, 0 = singular matrix detected
 */
int mat_inverse_nxn(double *Ainv,
                    const double *A,
                    int n,
                    int *piv);

/* ---- Joseph-form covariance update --------------------------------------- */

/**
 * mat_joseph_update  —  P = (I-K*H)*P*(I-K*H)^T + K*R*K^T
 *
 * Numerically stable positive-semi-definite guaranteed update.
 * Matches mat_joseph_update in kalman-updated.py exactly.
 *
 * @param Pout    output n×n covariance
 * @param P       prior  n×n covariance
 * @param K       Kalman gain  n×m
 * @param H       measurement matrix  m×n
 * @param R       measurement noise   m×m
 * @param n       state dimension
 * @param m       measurement dimension
 * @param scratch temporary buffer: at least 4*n*n doubles
 *                Layout: [IKH | tmp1 | tmp2 | K^T] each n*n doubles.
 *                (K^T is m×n <= n×n, so 4 slots of n*n is always sufficient.)
 */
void mat_joseph_update(double *Pout,
                       const double *P,
                       const double *K,
                       const double *H,
                       const double *R,
                       int n, int m,
                       double *scratch);

/* ---- EKF arctan2 --------------------------------------------------------- */

/**
 * fast_atan2  —  Scheinerman-Lyons polynomial atan2 approximation.
 *
 * Max error ~0.021 degrees. Matches fast_atan2 in kalman-updated.py exactly.
 * No libm atan2 is used anywhere.
 *
 * @param y  y-coordinate
 * @param x  x-coordinate
 * @return   atan2(y, x) in radians
 */
double fast_atan2(double y, double x);

/**
 * wrap_angle  —  Wrap angle to (-pi, pi].
 *
 * Implemented as a single floor-division (no loop), matching wrap_angle
 * in kalman-updated.py to within 1 ULP.
 *
 * @param a  angle in radians (any range)
 * @return   equivalent angle in (-pi, pi]
 */
double wrap_angle(double a);

/* ---- Convenience sizes (matches kalman-updated.py constants) ------------- */

#define NUM_JOINTS       23
#define STATE_DIM        12       /* per joint */
#define MEAS_DIM          3       /* per joint */
#define TOTAL_STATE_DIM (NUM_JOINTS * STATE_DIM)   /* 276 */
#define TOTAL_MEAS_DIM  (NUM_JOINTS * MEAS_DIM)    /* 69  */

/* Joseph-form scratch buffer size in doubles: 4 * n * n */
#define JOSEPH_SCRATCH_DOUBLES  (4 * TOTAL_STATE_DIM * TOTAL_STATE_DIM)

/* LU pivot array size in ints */
#define LU_PIV_SIZE  TOTAL_STATE_DIM

/* Numerical verification tolerance (§6) */
#define EPSILON_TOL  1e-9

#ifdef __cplusplus
}
#endif

#endif /* MATRIX_ASM_H */
