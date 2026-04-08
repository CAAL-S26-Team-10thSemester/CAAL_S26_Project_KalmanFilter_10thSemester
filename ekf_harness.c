/*
 * ekf_harness.c  --  Drive ekf_asm.s and verify against Python reference output
 * =============================================================================
 * Build:
 *   riscv64-unknown-linux-gnu-gcc -O2 -march=rv64gc -mabi=lp64d \
 *       ekf_harness.c ekf_asm.s -o ekf_asm -lm
 *
 * Run:
 *   ./ekf_asm  <noisy_csv>  <reference_ekf_csv>
 *
 * Output:
 *   ekf_asm_results.csv    -- estimated state (N × 276, same format as Python)
 *   ekf_verification.csv   -- per-joint / per-state error statistics
 *
 * Numerical contract (§6 of milestone spec):
 *   |x_asm[k,i] - x_ref[k,i]| <= EPSILON_TOL = 1e-9  for all k, i
 *
 * Memory layout (all double, heap-allocated):
 *   F       : 276×276 = 608,448 bytes
 *   Q       : 276×276 = 608,448 bytes
 *   R_sph   :  69× 69 =  38,088 bytes
 *   x       : 276          = 2,208 bytes
 *   P       : 276×276 = 608,448 bytes
 *   K_buf   : 276× 69 = 152,064 bytes
 *   S_buf   :  69× 69 =  38,088 bytes
 *   h_pred  :  69          =    552 bytes
 *   h_meas  :  69          =    552 bytes
 *   nu_buf  :  69          =    552 bytes
 *   z_tmp   : 276          =  2,208 bytes
 *   Hk_buf  :  69×276 = 152,064 bytes
 *   PHkt_buf: 276× 69 = 152,064 bytes
 *   IKH_buf : 2×276×276 = 1,216,896 bytes   (Joseph form needs 2× the 276² block)
 *   pivot   :  69 ints    =    276 bytes
 *   tmp276  : 276          =  2,208 bytes
 *   tmpNN   : 276×276 = 608,448 bytes
 *
 * Total heap: ~4.5 MB — well within RISC-V Linux defaults.
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ---- Constants (must match ekf_asm.s and Python) ---- */
#define NUM_JOINTS       23
#define STATE_DIM        12
#define MEAS_DIM          3
#define N_STATE         276      /* NUM_JOINTS * STATE_DIM */
#define N_MEAS           69      /* NUM_JOINTS * MEAS_DIM  */
#define DT    (1.0 / 30.0)
#define EPSILON_TOL      1e-9

/* ---- Assembly function prototypes ---- */
extern void   ekf_init_F(double *F, double dt);
extern void   ekf_init_Q(double *Q);
extern void   ekf_init_R_sph(double *R);
extern void   ekf_init_state(double *x, int joint, double px, double py, double pz);
extern void   ekf_predict(double *x, double *P,
                          const double *F, const double *Q,
                          int N, double *tmp276, double *tmpNN);
extern void   ekf_compute_h(const double *x, double *h_out);
extern int    ekf_lu_inverse(double *A, double *Ainv, int n, int *pivot);
extern void   ekf_update(double *x, double *P,
                         const double *R_sph,
                         const double *meas_cart,
                         double *K_buf, double *S_buf,
                         double *h_pred_buf, double *h_meas_buf,
                         /* stack args: */
                         double *nu_buf, double *z_tmp,
                         double *Hk_buf, double *PHkt_buf,
                         double *IKH_buf,
                         int N_joints, int *pivot_buf);

/* ---- CSV helpers ---- */
static int count_csv_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int rows = 0;
    char buf[65536];
    while (fgets(buf, sizeof(buf), f)) rows++;
    fclose(f);
    return rows - 1; /* subtract header */
}

static double **alloc_matrix(int rows, int cols) {
    double *data = (double *)calloc((size_t)rows * cols, sizeof(double));
    double **m   = (double **)malloc(rows * sizeof(double *));
    if (!data || !m) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < rows; i++) m[i] = data + (size_t)i * cols;
    return m;
}

/* Load noisy CSV: shape (N_frames, NUM_JOINTS, 3).
   Returns flat (N_frames, NUM_JOINTS*3) array, row-major. */
static double **load_noisy_csv(const char *path, int *n_frames_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    /* Count rows */
    int rows = 0;
    char buf[65536];
    int has_header = 0;
    while (fgets(buf, sizeof(buf), f)) {
        if (rows == 0) {
            /* Try to parse first field as double */
            double dummy;
            if (sscanf(buf, "%lf", &dummy) != 1) { has_header = 1; }
        }
        rows++;
    }
    if (has_header) rows--;
    rewind(f);
    if (has_header) fgets(buf, sizeof(buf), f); /* skip header */

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

/* Load reference EKF CSV (Python output): shape (N_frames, N_STATE=276).
   First column is frame index, then 276 state columns. */
static double **load_ref_csv(const char *path, int *n_frames_out) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }

    /* Skip header */
    char buf[65536];
    fgets(buf, sizeof(buf), f);

    /* Count rows */
    int rows = 0;
    long pos = ftell(f);
    while (fgets(buf, sizeof(buf), f)) rows++;
    fseek(f, pos, SEEK_SET);

    double **m = alloc_matrix(rows, N_STATE);
    for (int r = 0; r < rows; r++) {
        if (!fgets(buf, sizeof(buf), f)) break;
        char *p = buf;
        /* Skip frame index */
        strtod(p, &p); if (*p == ',') p++;
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
        fprintf(stderr, "Usage: %s <noisy_csv> <ref_ekf_csv>\n", argv[0]);
        return 1;
    }

    int N_frames_noisy, N_frames_ref;
    double **noisy = load_noisy_csv(argv[1], &N_frames_noisy);
    double **ref   = load_ref_csv(argv[2], &N_frames_ref);
    int N = N_frames_noisy < N_frames_ref ? N_frames_noisy : N_frames_ref;
    printf("[INFO] Loaded %d frames (noisy=%d, ref=%d)\n",
           N, N_frames_noisy, N_frames_ref);

    /* ---- Allocate all EKF buffers ---- */
    double *F        = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    double *Q        = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    double *R_sph    = (double *)calloc(N_MEAS  * N_MEAS,  sizeof(double));
    double *x        = (double *)calloc(N_STATE,            sizeof(double));
    double *P        = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    double *K_buf    = (double *)calloc(N_STATE * N_MEAS,  sizeof(double));
    double *S_buf    = (double *)calloc(N_MEAS  * N_MEAS,  sizeof(double));
    double *h_pred   = (double *)calloc(N_MEAS,             sizeof(double));
    double *h_meas   = (double *)calloc(N_MEAS,             sizeof(double));
    double *nu_buf   = (double *)calloc(N_MEAS,             sizeof(double));
    double *z_tmp    = (double *)calloc(N_STATE,            sizeof(double));
    double *Hk_buf   = (double *)calloc(N_MEAS  * N_STATE, sizeof(double));
    double *PHkt_buf = (double *)calloc(N_STATE * N_MEAS,  sizeof(double));
    /* IKH_buf needs 2 × 276×276 doubles for Joseph form (IKH and IKH*P) */
    double *IKH_buf  = (double *)calloc(2 * N_STATE * N_STATE, sizeof(double));
    double *tmp276   = (double *)calloc(N_STATE,            sizeof(double));
    double *tmpNN    = (double *)calloc(N_STATE * N_STATE, sizeof(double));
    int    *pivot    = (int    *)calloc(N_MEAS,             sizeof(int));

    if (!F||!Q||!R_sph||!x||!P||!K_buf||!S_buf||!h_pred||!h_meas||
        !nu_buf||!z_tmp||!Hk_buf||!PHkt_buf||!IKH_buf||!tmp276||!tmpNN||!pivot) {
        fprintf(stderr, "OOM allocating EKF buffers\n"); return 1;
    }

    /* ---- Initialise matrices (assembly) ---- */
    ekf_init_F(F, DT);
    ekf_init_Q(Q);
    ekf_init_R_sph(R_sph);

    /* P = I */
    for (int i = 0; i < N_STATE; i++) P[i * N_STATE + i] = 1.0;

    /* Initialise joint states from first frame */
    for (int j = 0; j < NUM_JOINTS; j++) {
        double px = noisy[0][j * 3 + 0];
        double py = noisy[0][j * 3 + 1];
        double pz = noisy[0][j * 3 + 2];
        ekf_init_state(x, j, px, py, pz);
    }

    /* ---- Result storage ---- */
    double **results = alloc_matrix(N, N_STATE);

    printf("[INFO] Running EKF assembly on %d frames (276-D global state)...\n", N);

    for (int frame = 0; frame < N; frame++) {
        /* Predict (skip for frame 0) */
        if (frame > 0) {
            ekf_predict(x, P, F, Q, N_STATE, tmp276, tmpNN);
        }

        /* Update */
        ekf_update(x, P, R_sph,
                   noisy[frame],     /* meas_cart: NUM_JOINTS*3 doubles */
                   K_buf, S_buf, h_pred, h_meas,
                   /* stack args: */
                   nu_buf, z_tmp, Hk_buf, PHkt_buf, IKH_buf,
                   NUM_JOINTS, pivot);

        /* Store result */
        memcpy(results[frame], x, N_STATE * sizeof(double));

        if ((frame + 1) % 200 == 0 || frame == N - 1)
            printf("\r  EKF-ASM: %d/%d", frame + 1, N);
    }
    printf("\n");

    /* ---- Save assembly results CSV ---- */
    {
        FILE *out = fopen("ekf_asm_results.csv", "w");
        if (!out) { fprintf(stderr, "Cannot write ekf_asm_results.csv\n"); return 1; }
        /* Header */
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
        printf("✅ Saved: ekf_asm_results.csv\n");
    }

    /* ---- Numerical Verification ---- */
    /* Per frame, per state: |asm - ref| */
    double avg_err_joint[NUM_JOINTS];   /* avg over states and frames */
    double avg_err_state[N_STATE];      /* avg over joints and frames */
    double max_err = 0.0, min_err = 1e300;
    int    violations = 0;

    memset(avg_err_joint, 0, sizeof(avg_err_joint));
    memset(avg_err_state, 0, sizeof(avg_err_state));

    for (int fr = 0; fr < N; fr++) {
        for (int i = 0; i < N_STATE; i++) {
            double err = fabs(results[fr][i] - ref[fr][i]);
            avg_err_state[i] += err;
            avg_err_joint[i / STATE_DIM] += err;
            if (err > max_err) max_err = err;
            if (err < min_err) min_err = err;
            if (err > EPSILON_TOL) violations++;
        }
    }
    /* Normalise */
    for (int i = 0; i < N_STATE; i++)
        avg_err_state[i] /= (double)N;
    for (int j = 0; j < NUM_JOINTS; j++)
        avg_err_joint[j] /= (double)(N * STATE_DIM);

    /* Print summary */
    printf("\n── Numerical Verification ──────────────────────────────────\n");
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

    /* Save verification CSV */
    {
        FILE *v = fopen("ekf_verification.csv", "w");
        fprintf(v, "joint,avg_err_over_states\n");
        for (int j = 0; j < NUM_JOINTS; j++)
            fprintf(v, "%s,%.15g\n", joint_names[j], avg_err_joint[j]);
        fprintf(v, "\nstate_idx,avg_err_over_joints\n");
        for (int i = 0; i < N_STATE; i++)
            fprintf(v, "%d,%.15g\n", i, avg_err_state[i]);
        fprintf(v, "\nmax_err,%.15g\nmin_err,%.15g\nviolations,%d\n",
                max_err, min_err, violations);
        fclose(v);
        printf("✅ Saved: ekf_verification.csv\n");
    }

    printf("\n[%s] EKF assembly verification complete.\n",
           violations == 0 ? "PASS ✅" : "FAIL ❌");

    /* ---- Cleanup ---- */
    free(F); free(Q); free(R_sph); free(x); free(P);
    free(K_buf); free(S_buf); free(h_pred); free(h_meas);
    free(nu_buf); free(z_tmp); free(Hk_buf); free(PHkt_buf);
    free(IKH_buf); free(tmp276); free(tmpNN); free(pivot);
    free(results[0]); free(results);
    free(noisy[0]); free(noisy);
    free(ref[0]); free(ref);

    return (violations == 0) ? 0 : 1;
}
