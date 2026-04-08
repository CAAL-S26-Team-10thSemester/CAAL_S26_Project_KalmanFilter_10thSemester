/*
 * lkf_harness.c  --  Drive lkf_asm.s and verify against Python reference output
 * =============================================================================
 * Build:
 *   riscv64-unknown-linux-gnu-gcc -O2 -march=rv64gc -mabi=lp64d \
 *       lkf_harness.c lkf_asm.s -o lkf_asm -lm
 *
 * Run:
 *   ./lkf_asm  <noisy_csv>  <reference_lkf_csv>
 *
 * Output:
 *   lkf_asm_results.csv    -- estimated state (N × 276, same format as Python)
 *   lkf_verification.csv   -- per-joint / per-state error statistics
 *
 * Numerical contract (§6 of milestone spec):
 *   |x_asm[k,i] - x_ref[k,i]| <= EPSILON_TOL = 1e-9  for all k, i
 *
 * Memory layout (all double, heap-allocated):
 *   F        : 276×276  = 608,448 bytes
 *   Q        : 276×276  = 608,448 bytes
 *   H        :  69×276  = 152,064 bytes
 *   R_cart   :  69× 69  =  38,088 bytes
 *   x        : 276             =   2,208 bytes
 *   P        : 276×276  = 608,448 bytes
 *   K_buf    : 276× 69  = 152,064 bytes
 *   S_buf    :  69× 69  =  38,088 bytes
 *   PHt_buf  : 276× 69  = 152,064 bytes
 *   y_buf    :  69             =     552 bytes
 *   Sinv_buf :  69× 69  =  38,088 bytes
 *   IKH_buf  : 2×276×276= 1,216,896 bytes  (Joseph form, two 276² blocks)
 *   tmp276   : 276             =   2,208 bytes
 *   tmpNN    : 276×276  = 608,448 bytes
 *   pivot    :  69 ints        =     276 bytes
 *
 * Total heap: ~4.2 MB
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- Constants ---- */
#define NUM_JOINTS   23
#define STATE_DIM    12
#define MEAS_DIM      3
#define N_STATE     276
#define N_MEAS       69
#define DT    (1.0 / 30.0)
#define EPSILON_TOL  1e-9

/* ---- Assembly prototypes ---- */
extern void lkf_init_F(double *F, double dt);
extern void lkf_init_Q(double *Q);
extern void lkf_init_H(double *H);
extern void lkf_init_R_cart(double *R);
extern void lkf_init_state(double *x, int joint, double px, double py, double pz);
extern void lkf_predict(double *x, double *P,
                        const double *F, const double *Q,
                        int N, double *tmp276, double *tmpNN);
extern void lkf_update(double *x, double *P,
                       const double *H, const double *R_cart,
                       const double *meas_cart,
                       double *K_buf, double *S_buf, double *PHt_buf,
                       /* stack: */
                       double *y_buf, double *Sinv_buf,
                       double *IKH_buf,
                       int N_joints, int *pivot_buf);
extern int  lkf_lu_inverse(double *A, double *Ainv, int n, int *pivot);

/* ---- CSV helpers (same as ekf_harness.c) ---- */
static double **alloc_matrix(int rows, int cols) {
    double *data = (double *)calloc((size_t)rows * cols, sizeof(double));
    double **m   = (double **)malloc(rows * sizeof(double *));
    if (!data || !m) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < rows; i++) m[i] = data + (size_t)i * cols;
    return m;
}

static double **load_noisy_csv(const char *path, int *n_frames_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int rows = 0;
    char buf[65536];
    int has_header = 0;
    while (fgets(buf, sizeof(buf), f)) {
        if (rows == 0) {
            double dummy;
            if (sscanf(buf, "%lf", &dummy) != 1) has_header = 1;
        }
        rows++;
    }
    if (has_header) rows--;
    rewind(f);
    if (has_header) fgets(buf, sizeof(buf), f);
    int cols = NUM_JOINTS * 3;
    double **m = alloc_matrix(rows, cols);
    for (int r = 0; r < rows; r++) {
        if (!fgets(buf, sizeof(buf), f)) break;
        char *p = buf;
        for (int c = 0; c < cols; c++) {
            m[r][c] = strtod(p, &p);
            if (*p == ',') p++;
        }
    }
    fclose(f);
    *n_frames_out = rows;
    return m;
}

static double **load_ref_csv(const char *path, int *n_frames_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    char buf[65536];
    fgets(buf, sizeof(buf), f);      /* skip header */
    int rows = 0;
    long pos = ftell(f);
    while (fgets(buf, sizeof(buf), f)) rows++;
    fseek(f, pos, SEEK_SET);
    double **m = alloc_matrix(rows, N_STATE);
    for (int r = 0; r < rows; r++) {
        if (!fgets(buf, sizeof(buf), f)) break;
        char *p = buf;
        strtod(p, &p); if (*p == ',') p++;   /* skip frame index */
        for (int c = 0; c < N_STATE; c++) {
            m[r][c] = strtod(p, &p);
            if (*p == ',') p++;
        }
    }
    fclose(f);
    *n_frames_out = rows;
    return m;
}

/* ---- Main ---- */
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <noisy_csv> <ref_lkf_csv>\n", argv[0]);
        return 1;
    }

    int N_frames_noisy, N_frames_ref;
    double **noisy = load_noisy_csv(argv[1], &N_frames_noisy);
    double **ref   = load_ref_csv(argv[2], &N_frames_ref);
    int N = N_frames_noisy < N_frames_ref ? N_frames_noisy : N_frames_ref;
    printf("[INFO] Loaded %d frames (noisy=%d, ref=%d)\n",
           N, N_frames_noisy, N_frames_ref);

    /* ---- Allocate buffers ---- */
    double *F        = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    double *Q        = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    double *H        = (double *)calloc(N_MEAS  * N_STATE, sizeof(double));
    double *R_cart   = (double *)calloc(N_MEAS  * N_MEAS,  sizeof(double));
    double *x        = (double *)calloc(N_STATE,            sizeof(double));
    double *P        = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    double *K_buf    = (double *)calloc(N_STATE * N_MEAS,  sizeof(double));
    double *S_buf    = (double *)calloc(N_MEAS  * N_MEAS,  sizeof(double));
    double *PHt_buf  = (double *)calloc(N_STATE * N_MEAS,  sizeof(double));
    double *y_buf    = (double *)calloc(N_MEAS,             sizeof(double));
    double *Sinv_buf = (double *)calloc(N_MEAS  * N_MEAS,  sizeof(double));
    /* 2 × 276×276 for Joseph form: (IKH) and (IKH*P) */
    double *IKH_buf  = (double *)calloc(2 * N_STATE * N_STATE, sizeof(double));
    double *tmp276   = (double *)calloc(N_STATE,            sizeof(double));
    double *tmpNN    = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    int    *pivot    = (int    *)calloc(N_MEAS,             sizeof(int));

    if (!F||!Q||!H||!R_cart||!x||!P||!K_buf||!S_buf||!PHt_buf||!y_buf||
        !Sinv_buf||!IKH_buf||!tmp276||!tmpNN||!pivot) {
        fprintf(stderr, "OOM allocating LKF buffers\n"); return 1;
    }

    /* ---- Initialise matrices (assembly) ---- */
    lkf_init_F(F, DT);
    lkf_init_Q(Q);
    lkf_init_H(H);
    lkf_init_R_cart(R_cart);

    /* P = I */
    for (int i = 0; i < N_STATE; i++) P[i * N_STATE + i] = 1.0;

    /* Initialise state from first frame */
    for (int j = 0; j < NUM_JOINTS; j++) {
        double px = noisy[0][j * 3 + 0];
        double py = noisy[0][j * 3 + 1];
        double pz = noisy[0][j * 3 + 2];
        lkf_init_state(x, j, px, py, pz);
    }

    /* ---- Result storage ---- */
    double **results = alloc_matrix(N, N_STATE);

    printf("[INFO] Running LKF assembly on %d frames (276-D global state)...\n", N);

    for (int frame = 0; frame < N; frame++) {
        if (frame > 0) {
            lkf_predict(x, P, F, Q, N_STATE, tmp276, tmpNN);
        }

        lkf_update(x, P, H, R_cart,
                   noisy[frame],       /* meas_cart */
                   K_buf, S_buf, PHt_buf,
                   /* stack args: */
                   y_buf, Sinv_buf, IKH_buf,
                   NUM_JOINTS, pivot);

        memcpy(results[frame], x, N_STATE * sizeof(double));

        if ((frame + 1) % 200 == 0 || frame == N - 1)
            printf("\r  LKF-ASM: %d/%d", frame + 1, N);
    }
    printf("\n");

    /* ---- Save results CSV ---- */
    {
        FILE *out = fopen("lkf_asm_results.csv", "w");
        if (!out) { fprintf(stderr, "Cannot write lkf_asm_results.csv\n"); return 1; }
        fprintf(out, "frame");
        for (int j = 0; j < NUM_JOINTS; j++)
            for (int ax = 0; ax < 3; ax++) {
                const char *axc = (ax==0)?"x":(ax==1)?"y":"z";
                fprintf(out, ",j%d_p%s,j%d_v%s,j%d_a%s,j%d_j%s",
                        j, axc, j, axc, j, axc, j, axc);
            }
        fprintf(out, "\n");
        for (int fr = 0; fr < N; fr++) {
            fprintf(out, "%d", fr);
            for (int c = 0; c < N_STATE; c++)
                fprintf(out, ",%.15g", results[fr][c]);
            fprintf(out, "\n");
        }
        fclose(out);
        printf("✅ Saved: lkf_asm_results.csv\n");
    }

    /* ---- Numerical verification ---- */
    double avg_err_joint[NUM_JOINTS];
    double avg_err_state[N_STATE];
    double max_err = 0.0, min_err = 1e300;
    int    violations = 0;

    memset(avg_err_joint, 0, sizeof(avg_err_joint));
    memset(avg_err_state, 0, sizeof(avg_err_state));

    for (int fr = 0; fr < N; fr++) {
        for (int i = 0; i < N_STATE; i++) {
            double err = fabs(results[fr][i] - ref[fr][i]);
            avg_err_state[i]              += err;
            avg_err_joint[i / STATE_DIM]  += err;
            if (err > max_err) max_err = err;
            if (err < min_err) min_err = err;
            if (err > EPSILON_TOL) violations++;
        }
    }
    for (int i = 0; i < N_STATE; i++)
        avg_err_state[i] /= (double)N;
    for (int j = 0; j < NUM_JOINTS; j++)
        avg_err_joint[j] /= (double)(N * STATE_DIM);

    printf("\n── Numerical Verification (LKF) ────────────────────────────\n");
    printf("  Tolerance (ε_tol)   : %.1e\n", EPSILON_TOL);
    printf("  Max |error|         : %.3e\n", max_err);
    printf("  Min |error|         : %.3e\n", min_err);
    printf("  Violations (> ε_tol): %d / %d\n", violations, N * N_STATE);
    printf("\n  Avg error per joint:\n");

    const char *joint_names[] = {
        "pelvis","L5","L3","T12","T8","neck","head",
        "shoulderRight","upperArmRight","forearmRight","handRight",
        "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
        "upperLegRight","lowerLegRight","footRight","toeRight",
        "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
    };
    for (int j = 0; j < NUM_JOINTS; j++)
        printf("    %-18s: %.4e\n", joint_names[j], avg_err_joint[j]);

    {
        FILE *v = fopen("lkf_verification.csv", "w");
        fprintf(v, "joint,avg_err_over_states\n");
        for (int j = 0; j < NUM_JOINTS; j++)
            fprintf(v, "%s,%.15g\n", joint_names[j], avg_err_joint[j]);
        fprintf(v, "\nstate_idx,avg_err_over_joints\n");
        for (int i = 0; i < N_STATE; i++)
            fprintf(v, "%d,%.15g\n", i, avg_err_state[i]);
        fprintf(v, "\nmax_err,%.15g\nmin_err,%.15g\nviolations,%d\n",
                max_err, min_err, violations);
        fclose(v);
        printf("✅ Saved: lkf_verification.csv\n");
    }

    printf("\n[%s] LKF assembly verification complete.\n",
           violations == 0 ? "PASS ✅" : "FAIL ❌");

    /* ---- Cleanup ---- */
    free(F); free(Q); free(H); free(R_cart);
    free(x); free(P); free(K_buf); free(S_buf);
    free(PHt_buf); free(y_buf); free(Sinv_buf);
    free(IKH_buf); free(tmp276); free(tmpNN); free(pivot);
    free(results[0]); free(results);
    free(noisy[0]); free(noisy);
    free(ref[0]); free(ref);

    return (violations == 0) ? 0 : 1;
}
