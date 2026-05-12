/*
 * verify_ekf_vector.c  —  Milestone-4 Verification Harness (Vectorised EKF)
 * ==========================================================================
 * Compares the RVV-vectorised EKF (ekf_vector.s) against the Python reference
 * (ekf_results.csv) using the same §6 methodology as Milestone-3.
 *
 * Also measures wall-clock time so speedup vs scalar can be computed.
 *
 * Build (in the devcontainer):
 *   riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O0 -g -Wall \
 *       -c ekf_vector.s -o ekf_vector.o
 *   riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O0 -g -Wall \
 *       -c matrix_vec.s -o matrix_vec.o
 *   riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O0 -g -Wall \
 *       -c ekf_utils_vector.s -o ekf_utils_vector.o
 *   riscv64-linux-gnu-gcc -march=rv64imfd -mabi=lp64d -O0 -g -Wall \
 *       -c lkf_asm.s -o lkf_asm.o
 *   riscv64-linux-gnu-gcc -march=rv64imfd -mabi=lp64d -O0 -g -Wall \
 *       -c matrix_asm.s -o matrix_asm.o
 *   riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O0 -g -Wall \
 *       verify_ekf_vector.c ekf_vector.o ekf_utils_vector.o \
 *       matrix_vec.o lkf_asm.o matrix_asm.o \
 *       -o verify_ekf_vector -lm -static
 *
 * Run:
 *   qemu-riscv64 -cpu rv64,v=true,vlen=128 \
 *       ./verify_ekf_vector "<noisy_csv>" "<ref_ekf_csv>" [max_frames]
 *
 * ==========================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- dimensions (match kalman-updated.py) -------------------------------- */
#define NUM_JOINTS   23
#define STATE_DIM    12
#define MEAS_DIM      3
#define N_STATE     276
#define N_MEAS       69
#define DT          (1.0 / 30.0)
#define EPSILON_TOL  1e-6       /* relaxed: vectorised FMA has slightly different rounding */
#define DEFAULT_MAX_FRAMES 100
#define PROGRESS_INTERVAL  10

static const char *JOINT_NAMES[NUM_JOINTS] = {
    "pelvis","L5","L3","T12","T8","neck","head",
    "shoulderRight","upperArmRight","forearmRight","handRight",
    "shoulderLeft","upperArmLeft","forearmLeft","handLeft",
    "upperLegRight","lowerLegRight","footRight","toeRight",
    "upperLegLeft","lowerLegLeft","footLeft","toeLeft"
};
static const char *COMP_NAMES[STATE_DIM] = {
    "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
};

/* ---- vectorised EKF assembly function declarations ----------------------- */
extern size_t ekf_vec_sizeof(void);
extern void   ekf_vec_init              (void *ekf, double dt);
extern void   ekf_vec_set_initial_state (void *ekf, int joint, const double *pos3);
extern void   ekf_vec_predict           (void *ekf);
extern void   ekf_vec_update            (void *ekf, const double *z_cart_flat);
extern void   ekf_vec_get_positions     (const void *ekf, double *pos_out);
extern void   ekf_vec_get_full_state    (const void *ekf, double *out);

/* ---- timing helper ------------------------------------------------------- */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* ---- helpers ------------------------------------------------------------- */
static double **alloc_matrix(int rows, int cols)
{
    double *data = (double *)calloc((size_t)rows * cols, sizeof(double));
    double **m   = (double **)malloc((size_t)rows * sizeof(double *));
    if (!data || !m) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < rows; i++) m[i] = data + (size_t)i * cols;
    return m;
}

/* Load noisy CSV (N_frames × (NUM_JOINTS*MEAS_DIM)) ----------------------- */
static double **load_noisy_csv(const char *path, int *n)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
    char buf[65536];
    int hdr = 0;
    if (fgets(buf, sizeof(buf), f)) {
        double d;
        hdr = (sscanf(buf, "%lf", &d) != 1);
    }
    rewind(f);
    int rows = 0;
    while (fgets(buf, sizeof(buf), f)) rows++;
    if (hdr) rows--;
    rewind(f);
    if (hdr) fgets(buf, sizeof(buf), f);

    int cols = NUM_JOINTS * MEAS_DIM;
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
    *n = rows;
    return m;
}

/* Load reference CSV (N_frames × N_STATE, first col = frame index) -------- */
static double **load_ref_csv(const char *path, int *n)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); exit(1); }
    char buf[65536];
    fgets(buf, sizeof(buf), f);          /* skip header */

    int rows = 0;
    long pos = ftell(f);
    while (fgets(buf, sizeof(buf), f)) rows++;
    fseek(f, pos, SEEK_SET);

    double **m = alloc_matrix(rows, N_STATE);
    for (int r = 0; r < rows; r++) {
        if (!fgets(buf, sizeof(buf), f)) break;
        char *p = buf;
        strtod(p, &p);          /* skip frame index */
        if (*p == ',') p++;
        for (int c = 0; c < N_STATE; c++) {
            m[r][c] = strtod(p, &p);
            if (*p == ',') p++;
        }
    }
    fclose(f);
    *n = rows;
    return m;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <noisy_csv> <ref_ekf_csv> [max_frames]\n",
            argv[0]);
        return 1;
    }

    int max_frames = DEFAULT_MAX_FRAMES;
    if (argc >= 4) {
        max_frames = atoi(argv[3]);
        if (max_frames <= 0) max_frames = DEFAULT_MAX_FRAMES;
    }

    /* ---- Load inputs ------------------------------------------------------ */
    printf("[INFO] Loading noisy CSV:     %s\n", argv[1]);
    int nn, nr;
    double **noisy = load_noisy_csv(argv[1], &nn);

    printf("[INFO] Loading reference CSV: %s\n", argv[2]);
    double **ref   = load_ref_csv(argv[2], &nr);

    int N = nn < nr ? nn : nr;
    if (N > max_frames) N = max_frames;
    printf("[INFO] Frames: %d  (noisy=%d, ref=%d, cap=%d)\n", N, nn, nr, max_frames);

    /* ---- Allocate and initialise vectorised EKF --------------------------- */
    size_t sz = ekf_vec_sizeof();
    printf("[INFO] EKF-VEC struct: %zu bytes (%.2f MB)\n",
           sz, (double)sz / (1024.0 * 1024.0));

    /* Use aligned allocation for cache-line efficiency with RVV */
    void *ekf = NULL;
    if (posix_memalign(&ekf, 64, sz) != 0 || !ekf) {
        fprintf(stderr, "OOM (posix_memalign)\n");
        return 1;
    }
    memset(ekf, 0, sz);

    ekf_vec_init(ekf, DT);
    for (int j = 0; j < NUM_JOINTS; j++) {
        double p3[3] = {
            noisy[0][j * MEAS_DIM],
            noisy[0][j * MEAS_DIM + 1],
            noisy[0][j * MEAS_DIM + 2]
        };
        ekf_vec_set_initial_state(ekf, j, p3);
    }

    /* ---- Allocate result storage ----------------------------------------- */
    double **results = alloc_matrix(N, N_STATE);
    double z[N_MEAS];

    /* ---- Main filter loop ------------------------------------------------- */
    printf("[INFO] Running EKF-VEC (RVV vectorised) on %d frames...\n", N);
    double t_start = now_sec();

    for (int fr = 0; fr < N; fr++) {
        if (fr > 0) ekf_vec_predict(ekf);

        for (int j = 0; j < NUM_JOINTS; j++) {
            z[j * MEAS_DIM    ] = noisy[fr][j * MEAS_DIM    ];
            z[j * MEAS_DIM + 1] = noisy[fr][j * MEAS_DIM + 1];
            z[j * MEAS_DIM + 2] = noisy[fr][j * MEAS_DIM + 2];
        }
        ekf_vec_update(ekf, z);
        ekf_vec_get_full_state(ekf, results[fr]);

        if ((fr + 1) % PROGRESS_INTERVAL == 0 || fr == N - 1)
            printf("\r  EKF-VEC: %d/%d   ", fr + 1, N);
    }
    double elapsed = now_sec() - t_start;
    printf("\n[INFO] EKF-VEC loop done.  Total time: %.3f s  (%.1f ms/frame)\n",
           elapsed, elapsed * 1000.0 / N);

    /* ---- Save results CSV ------------------------------------------------- */
    {
        FILE *out = fopen("ekf_vec_results.csv", "w");
        fprintf(out, "frame");
        for (int j = 0; j < NUM_JOINTS; j++)
            for (int ax = 0; ax < MEAS_DIM; ax++) {
                const char *a = (ax == 0) ? "x" : (ax == 1) ? "y" : "z";
                fprintf(out, ",j%d_p%s,j%d_v%s,j%d_a%s,j%d_j%s",
                        j, a, j, a, j, a, j, a);
            }
        fprintf(out, "\n");
        for (int fr = 0; fr < N; fr++) {
            fprintf(out, "%d", fr);
            for (int c = 0; c < N_STATE; c++)
                fprintf(out, ",%.15g", results[fr][c]);
            fprintf(out, "\n");
        }
        fclose(out);
        printf("[OK] Saved: ekf_vec_results.csv\n");
    }

    /* =========================================================================
     * §6-style Numerical Verification (vector vs Python reference)
     * ========================================================================= */

    /* Accumulators */
    double avg_joint[NUM_JOINTS];
    double avg_comp[STATE_DIM];
    double max_err = 0.0, min_err = 1e300;
    long violations = 0;

    memset(avg_joint, 0, sizeof(avg_joint));
    memset(avg_comp,  0, sizeof(avg_comp));

    for (int fr = 0; fr < N; fr++) {
        for (int j = 0; j < NUM_JOINTS; j++) {
            for (int c = 0; c < STATE_DIM; c++) {
                double err = fabs(results[fr][j * STATE_DIM + c]
                                - ref[fr][j * STATE_DIM + c]);
                avg_joint[j] += err;
                avg_comp[c]  += err;
                if (err > max_err) max_err = err;
                if (err < min_err) min_err = err;
                if (err > EPSILON_TOL) violations++;
            }
        }
    }

    /* Normalise */
    double nj = (double)N * STATE_DIM;
    double nc = (double)N * NUM_JOINTS;
    for (int j = 0; j < NUM_JOINTS; j++) avg_joint[j] /= nj;
    for (int c = 0; c < STATE_DIM;  c++) avg_comp[c]  /= nc;

    /* ---- Print summary --------------------------------------------------- */
    printf("\n==================================================================\n");
    printf("  Milestone-4 Numerical Verification  (EKF-VEC, RVV vectorised)\n");
    printf("==================================================================\n");
    printf("  Tolerance          : %.1e\n",  EPSILON_TOL);
    printf("  Frames compared    : %d\n",   N);
    printf("  Total comparisons  : %ld\n",  (long)N * N_STATE);
    printf("  Global max |error| : %.6e\n", max_err);
    printf("  Global min |error| : %.6e\n", min_err);
    printf("  Violations > eps   : %ld\n\n", violations);
    printf("  Wall-clock time    : %.3f s  (%.1f ms/frame)\n\n",
           elapsed, elapsed * 1000.0 / N);

    printf("  ── Table A: Avg |error| per joint ─────────────────────────\n");
    printf("  %-18s  %14s\n", "Joint", "Avg |error|");
    for (int j = 0; j < NUM_JOINTS; j++)
        printf("  %-18s  %14.6e\n", JOINT_NAMES[j], avg_joint[j]);

    printf("\n  ── Table B: Avg |error| per state component ────────────────\n");
    printf("  %-10s  %14s\n", "Component", "Avg |error|");
    for (int c = 0; c < STATE_DIM; c++)
        printf("  %-10s  %14.6e\n", COMP_NAMES[c], avg_comp[c]);

    printf("\n  [%s] EKF-VEC Milestone-4 Verification\n",
           violations == 0 ? "PASS \xe2\x9c\x93" : "FAIL \xe2\x9c\x97");
    printf("==================================================================\n\n");

    /* ---- Save verification CSV ------------------------------------------- */
    {
        FILE *v = fopen("ekf_vec_verification.csv", "w");
        fprintf(v, "section,name,avg_abs_error\n");
        for (int j = 0; j < NUM_JOINTS; j++)
            fprintf(v, "joint,%s,%.15g\n",     JOINT_NAMES[j], avg_joint[j]);
        for (int c = 0; c < STATE_DIM;  c++)
            fprintf(v, "component,%s,%.15g\n", COMP_NAMES[c],  avg_comp[c]);
        fprintf(v, "\nsummary,max_err,%.15g\n",   max_err);
        fprintf(v, "summary,min_err,%.15g\n",     min_err);
        fprintf(v, "summary,violations,%ld\n",    violations);
        fprintf(v, "summary,frames,%d\n",         N);
        fprintf(v, "summary,wall_time_s,%.6f\n",  elapsed);
        fprintf(v, "summary,ms_per_frame,%.3f\n", elapsed * 1000.0 / N);
        fprintf(v, "summary,result,%s\n",
                violations == 0 ? "PASS" : "FAIL");
        fclose(v);
        printf("[OK] Saved: ekf_vec_verification.csv\n");
    }

    free(ekf);
    free(results[0]); free(results);
    free(noisy[0]);   free(noisy);
    free(ref[0]);     free(ref);
    return violations == 0 ? 0 : 1;
}
