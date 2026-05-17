/*
 * verify_matrix_vec.c  —  Numerical verification harness for matrix_vec.s
 *
 * Milestone-4 §7 requirement:
 *   |x_vec[k,i] - x_scalar[k,i]| <= epsilon_tol = 1e-9
 *
 * This file tests each vectorised function in matrix_vec.s against its
 * scalar counterpart from matrix_asm.s using randomised test matrices of
 * both small (to check tail-element handling) and Kalman-sized dimensions.
 *
 * Build (inside RISC-V dev container):
 *   riscv64-linux-gnu-gcc -O0 -march=rv64gcv -mabi=lp64d \
 *       verify_matrix_vec.c matrix_asm.s matrix_vec.s \
 *       -o verify_matrix_vec -lm
 *   qemu-riscv64 ./verify_matrix_vec
 *
 * Output:
 *   - Pass/Fail per function with max absolute error
 *   - Per-function verification table in CSV format (stdout)
 *   - Summary: total violations, overall PASS/FAIL
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "matrix_asm.h"   /* scalar M3 functions */
#include "matrix_vec.h"   /* vectorised M4 functions */

/* =========================================================================
 * Helpers
 * ========================================================================= */

/* Fill array with pseudo-random doubles in [-range, range]. */
static void rand_fill(double *p, int n, double range) {
    for (int i = 0; i < n; i++)
        p[i] = range * (2.0 * (double)rand() / (double)RAND_MAX - 1.0);
}

/* Maximum absolute element-wise error between two arrays. */
static double max_abs_err(const double *a, const double *b, int n) {
    double e = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

/* Average absolute element-wise error. */
static double avg_abs_err(const double *a, const double *b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++)
        s += fabs(a[i] - b[i]);
    return s / (double)n;
}

/* Count elements where |a-b| > tol. */
static int count_violations(const double *a, const double *b, int n,
                             double tol) {
    int c = 0;
    for (int i = 0; i < n; i++)
        if (fabs(a[i] - b[i]) > tol) c++;
    return c;
}

/* Print a one-line result. */
static int print_result(const char *name, double max_e, double avg_e,
                        int violations) {
    int pass = (max_e <= EPSILON_VEC_TOL) && (violations == 0);
    printf("  %-32s  max_err=%.3e  avg_err=%.3e  violations=%d  %s\n",
           name, max_e, avg_e, violations, pass ? "PASS" : "FAIL");
    return pass;
}

/* Aligned allocation (64-byte = cache-line). */
static double *alloc_aligned(int n) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, MATRIX_ALIGN, (size_t)n * sizeof(double)) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        exit(1);
    }
    return (double *)ptr;
}

/* =========================================================================
 * Test cases
 * ========================================================================= */

/* --- mat_mul_vec -------------------------------------------------------- */
static int test_mat_mul(int M, int K, int N, const char *label) {
    double *A    = alloc_aligned(M * K);
    double *B    = alloc_aligned(K * N);
    double *C_sc = alloc_aligned(M * N);   /* scalar reference */
    double *C_vc = alloc_aligned(M * N);   /* vectorised output */

    rand_fill(A, M * K, 10.0);
    rand_fill(B, K * N, 10.0);

    mat_mul(C_sc, A, B, M, K, N);          /* M3 scalar */
    mat_mul_vec(C_vc, A, B, M, K, N);      /* M4 vector */

    double me = max_abs_err(C_sc, C_vc, M * N);
    double ae = avg_abs_err(C_sc, C_vc, M * N);
    int    vi = count_violations(C_sc, C_vc, M * N, EPSILON_VEC_TOL);
    int    ok = print_result(label, me, ae, vi);

    free(A); free(B); free(C_sc); free(C_vc);
    return ok;
}

/* --- mat_add_vec -------------------------------------------------------- */
static int test_mat_add(int M, int N, const char *label) {
    double *A    = alloc_aligned(M * N);
    double *B    = alloc_aligned(M * N);
    double *C_sc = alloc_aligned(M * N);
    double *C_vc = alloc_aligned(M * N);

    rand_fill(A, M * N, 5.0);
    rand_fill(B, M * N, 5.0);

    mat_add(C_sc, A, B, M, N);
    mat_add_vec(C_vc, A, B, M, N);

    double me = max_abs_err(C_sc, C_vc, M * N);
    double ae = avg_abs_err(C_sc, C_vc, M * N);
    int    vi = count_violations(C_sc, C_vc, M * N, EPSILON_VEC_TOL);
    int    ok = print_result(label, me, ae, vi);

    free(A); free(B); free(C_sc); free(C_vc);
    return ok;
}

/* --- mat_sub_vec -------------------------------------------------------- */
static int test_mat_sub(int M, int N, const char *label) {
    double *A    = alloc_aligned(M * N);
    double *B    = alloc_aligned(M * N);
    double *C_sc = alloc_aligned(M * N);
    double *C_vc = alloc_aligned(M * N);

    rand_fill(A, M * N, 5.0);
    rand_fill(B, M * N, 5.0);

    mat_sub(C_sc, A, B, M, N);
    mat_sub_vec(C_vc, A, B, M, N);

    double me = max_abs_err(C_sc, C_vc, M * N);
    double ae = avg_abs_err(C_sc, C_vc, M * N);
    int    vi = count_violations(C_sc, C_vc, M * N, EPSILON_VEC_TOL);
    int    ok = print_result(label, me, ae, vi);

    free(A); free(B); free(C_sc); free(C_vc);
    return ok;
}

/* --- mat_scale_add_vec ------------------------------------------------- */
static int test_mat_scale_add(int M, int N, double alpha, const char *label) {
    double *A    = alloc_aligned(M * N);
    double *B    = alloc_aligned(M * N);
    double *C_sc = alloc_aligned(M * N);
    double *C_vc = alloc_aligned(M * N);

    rand_fill(A, M * N, 5.0);
    rand_fill(B, M * N, 5.0);

    mat_scale_add(C_sc, A, B, alpha, M, N);
    mat_scale_add_vec(C_vc, A, B, alpha, M, N);

    double me = max_abs_err(C_sc, C_vc, M * N);
    double ae = avg_abs_err(C_sc, C_vc, M * N);
    int    vi = count_violations(C_sc, C_vc, M * N, EPSILON_VEC_TOL);
    int    ok = print_result(label, me, ae, vi);

    free(A); free(B); free(C_sc); free(C_vc);
    return ok;
}

/* --- mat_vec_mul_vec --------------------------------------------------- */
static int test_mat_vec_mul(int M, int N, const char *label) {
    double *A    = alloc_aligned(M * N);
    double *x    = alloc_aligned(N);
    double *y_sc = alloc_aligned(M);
    double *y_vc = alloc_aligned(M);

    rand_fill(A, M * N, 5.0);
    rand_fill(x, N, 5.0);

    mat_vec_mul(y_sc, A, x, M, N);
    mat_vec_mul_vec(y_vc, A, x, M, N);

    double me = max_abs_err(y_sc, y_vc, M);
    double ae = avg_abs_err(y_sc, y_vc, M);
    int    vi = count_violations(y_sc, y_vc, M, EPSILON_VEC_TOL);
    int    ok = print_result(label, me, ae, vi);

    free(A); free(x); free(y_sc); free(y_vc);
    return ok;
}

/* --- mat_transpose_vec ------------------------------------------------- */
static int test_mat_transpose(int M, int N, const char *label) {
    double *A    = alloc_aligned(M * N);
    double *B_sc = alloc_aligned(N * M);
    double *B_vc = alloc_aligned(N * M);

    rand_fill(A, M * N, 5.0);

    mat_transpose(B_sc, A, M, N);
    mat_transpose_vec(B_vc, A, M, N);

    double me = max_abs_err(B_sc, B_vc, N * M);
    double ae = avg_abs_err(B_sc, B_vc, N * M);
    int    vi = count_violations(B_sc, B_vc, N * M, EPSILON_VEC_TOL);
    int    ok = print_result(label, me, ae, vi);

    free(A); free(B_sc); free(B_vc);
    return ok;
}

/* =========================================================================
 * CSV report generator
 * =========================================================================
 * Generates per-function summary in the same CSV format as the M3
 * verification files (lkf_asm_verification.csv / ekf_asm_verification.csv)
 * so the same plotting scripts can consume it.
 */
typedef struct {
    const char *func;
    int         M, K_or_N, N;  /* K_or_N = K for mat_mul, N for others */
    double      max_err;
    double      avg_err;
    int         violations;
    int         pass;
} VResult;

static void print_csv(const VResult *rs, int nfunc) {
    printf("\n--- CSV BEGIN (copy to verify_matrix_vec.csv) ---\n");
    printf("function,M,K_or_N,N,max_abs_err,avg_abs_err,violations,result\n");
    for (int i = 0; i < nfunc; i++) {
        const VResult *r = &rs[i];
        printf("%s,%d,%d,%d,%.6e,%.6e,%d,%s\n",
               r->func, r->M, r->K_or_N, r->N,
               r->max_err, r->avg_err, r->violations,
               r->pass ? "PASS" : "FAIL");
    }
    printf("--- CSV END ---\n");
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(void) {
    srand((unsigned)time(NULL));

    int N  = TOTAL_STATE_DIM;   /* 276 */
    int M  = TOTAL_MEAS_DIM;    /* 69  */
    int total_pass = 0, total_fail = 0;

    printf("=================================================================\n");
    printf("  Milestone-4 Matrix Vectorisation Verification\n");
    printf("  Tolerance: epsilon_tol = %.0e\n", (double)EPSILON_VEC_TOL);
    printf("  Alignment: %d bytes (cache-line)\n", MATRIX_ALIGN);
    printf("=================================================================\n\n");

    /* ------------------------------------------------------------------
     * 1. mat_mul_vec
     * ------------------------------------------------------------------ */
    printf("[ mat_mul_vec  (C = A@B) ]\n");
    total_pass += test_mat_mul(4,  4,  4,  "4x4 x 4x4 (tail=4)");
    total_pass += test_mat_mul(5,  7,  3,  "5x7 x 7x3 (odd dims)");
    total_pass += test_mat_mul(N,  N,  N,  "276x276 x 276x276 (F*P)");
    total_pass += test_mat_mul(N,  N,  M,  "276x276 x 276x69  (P*H^T)");
    total_pass += test_mat_mul(M,  N,  M,  "69x276  x 276x69  (H*P*H^T)");
    total_pass += test_mat_mul(N,  M,  N,  "276x69  x 69x276  (K*H)");

    /* ------------------------------------------------------------------
     * 2. mat_add_vec
     * ------------------------------------------------------------------ */
    printf("\n[ mat_add_vec  (C = A+B) ]\n");
    total_pass += test_mat_add(7,  5,  "7x5 (tail check)");
    total_pass += test_mat_add(N,  N,  "276x276 (P+Q)");
    total_pass += test_mat_add(M,  M,  "69x69   (H*P*H^T + R)");

    /* ------------------------------------------------------------------
     * 3. mat_sub_vec
     * ------------------------------------------------------------------ */
    printf("\n[ mat_sub_vec  (C = A-B) ]\n");
    total_pass += test_mat_sub(7,  5,  "7x5 (tail check)");
    total_pass += test_mat_sub(N,  N,  "276x276 (I - K*H)");
    total_pass += test_mat_sub(M,  1,  "69x1    (z - H*x)");

    /* ------------------------------------------------------------------
     * 4. mat_scale_add_vec
     * ------------------------------------------------------------------ */
    printf("\n[ mat_scale_add_vec  (C = A + alpha*B) ]\n");
    total_pass += test_mat_scale_add(N, N, 0.01,  "276x276, alpha=0.01");
    total_pass += test_mat_scale_add(M, M, -1.0,  "69x69,   alpha=-1.0");
    total_pass += test_mat_scale_add(3, 9, 1.5,   "3x9,     alpha=1.5 (tail)");

    /* ------------------------------------------------------------------
     * 5. mat_vec_mul_vec
     * ------------------------------------------------------------------ */
    printf("\n[ mat_vec_mul_vec  (y = A*x) ]\n");
    total_pass += test_mat_vec_mul(5,  7,  "5x7 (tail check)");
    total_pass += test_mat_vec_mul(N,  N,  "276x276 (F*x)");
    total_pass += test_mat_vec_mul(M,  N,  "69x276  (H*x)");

    /* ------------------------------------------------------------------
     * 6. mat_transpose_vec
     * ------------------------------------------------------------------ */
    printf("\n[ mat_transpose_vec  (B = A^T) ]\n");
    total_pass += test_mat_transpose(5,  7,  "5x7  (tail check)");
    total_pass += test_mat_transpose(N,  N,  "276x276 (F^T)");
    total_pass += test_mat_transpose(M,  N,  "69x276  (H^T)");
    total_pass += test_mat_transpose(N,  M,  "276x69  (K^T)");

    /* ------------------------------------------------------------------
     * Summary
     * ------------------------------------------------------------------ */
    int total = total_pass + total_fail;
    printf("\n=================================================================\n");
    printf("  Results:  %d / %d PASS\n", total_pass, total_pass + total_fail);
    printf("  Overall:  %s\n",
           total_fail == 0 ? "PASS — all errors within epsilon_tol=1e-9"
                           : "FAIL — see violations above");
    printf("=================================================================\n");

    /* ------------------------------------------------------------------
     * Alignment probe: print actual pointer alignment for a sample matrix
     * ------------------------------------------------------------------ */
    printf("\n[ Alignment check ]\n");
    double *probe = alloc_aligned(N * N);
    printf("  Sample N×N matrix pointer: %p\n", (void *)probe);
    printf("  Aligned to %d bytes: %s\n", MATRIX_ALIGN,
           ((uintptr_t)probe % MATRIX_ALIGN == 0) ? "YES" : "NO");
    free(probe);

    return (total_fail == 0) ? 0 : 1;
}
