/*
 * perf_compare.c  —  Milestone-4 Performance Analysis
 * =====================================================================
 * Runs both M3 (scalar) and M4 (vector) LKF and EKF side-by-side,
 * measures wall-clock time, and prints speedup tables.
 *
 * Build (in the devcontainer):
 *   riscv64-linux-gnu-gcc -march=rv64gcv -mabi=lp64d -O0 -g -Wall \
 *       perf_compare.c \
 *       lkf_vector.o ekf_vector.o ekf_utils_vector.o matrix_vec.o \
 *       lkf_asm.o ekf_asm.o matrix_asm.o \
 *       -o perf_compare -lm -static
 *
 * Run:
 *   qemu-riscv64 -cpu rv64,v=true,vlen=128 \
 *       ./perf_compare "<noisy_csv>" [max_frames]
 *
 * =====================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- dimensions ---------------------------------------------------- */
#define NUM_JOINTS   23
#define STATE_DIM    12
#define MEAS_DIM      3
#define N_STATE     276
#define N_MEAS       69
#define DT          (1.0 / 30.0)
#define DEFAULT_MAX_FRAMES 50

/* ---- M3 scalar LKF declarations ------------------------------------ */
extern size_t lkf_sizeof(void);
extern void   lkf_init              (void *lkf, double dt);
extern void   lkf_set_initial_state (void *lkf, int joint, const double *pos3);
extern void   lkf_predict           (void *lkf);
extern void   lkf_update            (void *lkf, const double *z_flat);
extern void   lkf_get_full_state    (const void *lkf, double *out);

/* ---- M4 vector LKF declarations ------------------------------------ */
extern size_t lkf_vec_sizeof(void);
extern void   lkf_vec_init              (void *lkf, double dt);
extern void   lkf_vec_set_initial_state (void *lkf, int joint, const double *pos3);
extern void   lkf_vec_predict           (void *lkf);
extern void   lkf_vec_update            (void *lkf, const double *z_flat);
extern void   lkf_vec_get_full_state    (const void *lkf, double *out);

/* ---- M3 scalar EKF declarations ------------------------------------ */
extern size_t ekf_sizeof(void);
extern void   ekf_init              (void *ekf, double dt);
extern void   ekf_set_initial_state (void *ekf, int joint, const double *pos3);
extern void   ekf_predict           (void *ekf);
extern void   ekf_update            (void *ekf, const double *z_cart_flat);
extern void   ekf_get_full_state    (const void *ekf, double *out);

/* ---- M4 vector EKF declarations ------------------------------------ */
extern size_t ekf_vec_sizeof(void);
extern void   ekf_vec_init              (void *ekf, double dt);
extern void   ekf_vec_set_initial_state (void *ekf, int joint, const double *pos3);
extern void   ekf_vec_predict           (void *ekf);
extern void   ekf_vec_update            (void *ekf, const double *z_cart_flat);
extern void   ekf_vec_get_full_state    (const void *ekf, double *out);

/* ---- timing helper ------------------------------------------------- */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* ---- CSV loader ---------------------------------------------------- */
static double **alloc_matrix(int rows, int cols)
{
    double *data = (double *)calloc((size_t)rows * cols, sizeof(double));
    double **m   = (double **)malloc((size_t)rows * sizeof(double *));
    if (!data || !m) { fprintf(stderr, "OOM\n"); exit(1); }
    for (int i = 0; i < rows; i++) m[i] = data + (size_t)i * cols;
    return m;
}

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

/* =========================================================================
 * Run filter and return elapsed time
 * ========================================================================= */

typedef struct {
    size_t (*fn_sizeof)(void);
    void   (*fn_init)(void *, double);
    void   (*fn_set_initial)(void *, int, const double *);
    void   (*fn_predict)(void *);
    void   (*fn_update)(void *, const double *);
    void   (*fn_get_state)(const void *, double *);
} FilterVtable;

static double run_filter(const FilterVtable *vt, double **noisy, int N,
                         const char *label)
{
    size_t sz = vt->fn_sizeof();
    void *filt = NULL;
    if (posix_memalign(&filt, 64, sz) != 0 || !filt) {
        fprintf(stderr, "OOM for %s\n", label);
        exit(1);
    }
    memset(filt, 0, sz);

    vt->fn_init(filt, DT);
    for (int j = 0; j < NUM_JOINTS; j++) {
        double p3[3] = {
            noisy[0][j * MEAS_DIM],
            noisy[0][j * MEAS_DIM + 1],
            noisy[0][j * MEAS_DIM + 2]
        };
        vt->fn_set_initial(filt, j, p3);
    }

    double state[N_STATE];
    double z[N_MEAS];

    printf("  Running %-20s (%d frames)...", label, N);
    fflush(stdout);
    double t0 = now_sec();

    for (int fr = 0; fr < N; fr++) {
        if (fr > 0) vt->fn_predict(filt);
        for (int j = 0; j < NUM_JOINTS; j++) {
            z[j * MEAS_DIM    ] = noisy[fr][j * MEAS_DIM    ];
            z[j * MEAS_DIM + 1] = noisy[fr][j * MEAS_DIM + 1];
            z[j * MEAS_DIM + 2] = noisy[fr][j * MEAS_DIM + 2];
        }
        vt->fn_update(filt, z);
        vt->fn_get_state(filt, state);
    }

    double elapsed = now_sec() - t0;
    printf(" %.3f s  (%.1f ms/frame)\n", elapsed, elapsed * 1000.0 / N);

    free(filt);
    return elapsed;
}

/* =========================================================================
 * main
 * ========================================================================= */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <noisy_csv> [max_frames]\n", argv[0]);
        return 1;
    }

    int max_frames = DEFAULT_MAX_FRAMES;
    if (argc >= 3) {
        max_frames = atoi(argv[2]);
        if (max_frames <= 0) max_frames = DEFAULT_MAX_FRAMES;
    }

    printf("==================================================================\n");
    printf("  Milestone-4 Performance Analysis  (M3 Scalar vs M4 Vector)\n");
    printf("==================================================================\n\n");

    int nn;
    double **noisy = load_noisy_csv(argv[1], &nn);
    int N = nn < max_frames ? nn : max_frames;
    printf("[INFO] Frames: %d (available=%d, cap=%d)\n\n", N, nn, max_frames);

    /* ---- LKF comparison ------------------------------------------------ */
    printf("── LKF Performance ─────────────────────────────────────────────\n");

    FilterVtable lkf_scalar = {
        .fn_sizeof      = lkf_sizeof,
        .fn_init        = lkf_init,
        .fn_set_initial = lkf_set_initial_state,
        .fn_predict     = lkf_predict,
        .fn_update      = lkf_update,
        .fn_get_state   = lkf_get_full_state
    };
    FilterVtable lkf_vector = {
        .fn_sizeof      = lkf_vec_sizeof,
        .fn_init        = lkf_vec_init,
        .fn_set_initial = lkf_vec_set_initial_state,
        .fn_predict     = lkf_vec_predict,
        .fn_update      = lkf_vec_update,
        .fn_get_state   = lkf_vec_get_full_state
    };

    double t_lkf_scalar = run_filter(&lkf_scalar, noisy, N, "LKF-M3 (scalar)");
    double t_lkf_vector = run_filter(&lkf_vector, noisy, N, "LKF-M4 (vector)");
    double lkf_speedup = t_lkf_scalar / t_lkf_vector;

    printf("\n  LKF Speedup: %.2fx  (scalar=%.3fs, vector=%.3fs)\n\n",
           lkf_speedup, t_lkf_scalar, t_lkf_vector);

    /* ---- EKF comparison ------------------------------------------------ */
    printf("── EKF Performance ─────────────────────────────────────────────\n");

    FilterVtable ekf_scalar = {
        .fn_sizeof      = ekf_sizeof,
        .fn_init        = ekf_init,
        .fn_set_initial = ekf_set_initial_state,
        .fn_predict     = ekf_predict,
        .fn_update      = ekf_update,
        .fn_get_state   = ekf_get_full_state
    };
    FilterVtable ekf_vector = {
        .fn_sizeof      = ekf_vec_sizeof,
        .fn_init        = ekf_vec_init,
        .fn_set_initial = ekf_vec_set_initial_state,
        .fn_predict     = ekf_vec_predict,
        .fn_update      = ekf_vec_update,
        .fn_get_state   = ekf_vec_get_full_state
    };

    double t_ekf_scalar = run_filter(&ekf_scalar, noisy, N, "EKF-M3 (scalar)");
    double t_ekf_vector = run_filter(&ekf_vector, noisy, N, "EKF-M4 (vector)");
    double ekf_speedup = t_ekf_scalar / t_ekf_vector;

    printf("\n  EKF Speedup: %.2fx  (scalar=%.3fs, vector=%.3fs)\n\n",
           ekf_speedup, t_ekf_scalar, t_ekf_vector);

    /* ---- Summary table ------------------------------------------------- */
    printf("==================================================================\n");
    printf("  Performance Summary Table\n");
    printf("==================================================================\n");
    printf("  %-20s  %12s  %12s  %8s\n",
           "Filter", "M3 (s)", "M4 (s)", "Speedup");
    printf("  %-20s  %12.3f  %12.3f  %7.2fx\n",
           "LKF", t_lkf_scalar, t_lkf_vector, lkf_speedup);
    printf("  %-20s  %12.3f  %12.3f  %7.2fx\n",
           "EKF", t_ekf_scalar, t_ekf_vector, ekf_speedup);
    printf("  %-20s  %12.3f  %12.3f  %7.2fx\n",
           "Total",
           t_lkf_scalar + t_ekf_scalar,
           t_lkf_vector + t_ekf_vector,
           (t_lkf_scalar + t_ekf_scalar) / (t_lkf_vector + t_ekf_vector));
    printf("==================================================================\n");
    printf("  Frames: %d  |  VLEN=128  |  QEMU emulated\n", N);
    printf("==================================================================\n\n");

    /* ---- Save to CSV --------------------------------------------------- */
    {
        FILE *out = fopen("perf_analysis.csv", "w");
        fprintf(out, "filter,m3_time_s,m4_time_s,speedup,frames,ms_per_frame_m3,ms_per_frame_m4\n");
        fprintf(out, "LKF,%.6f,%.6f,%.4f,%d,%.3f,%.3f\n",
                t_lkf_scalar, t_lkf_vector, lkf_speedup, N,
                t_lkf_scalar * 1000.0 / N, t_lkf_vector * 1000.0 / N);
        fprintf(out, "EKF,%.6f,%.6f,%.4f,%d,%.3f,%.3f\n",
                t_ekf_scalar, t_ekf_vector, ekf_speedup, N,
                t_ekf_scalar * 1000.0 / N, t_ekf_vector * 1000.0 / N);
        fprintf(out, "Total,%.6f,%.6f,%.4f,%d,%.3f,%.3f\n",
                t_lkf_scalar + t_ekf_scalar,
                t_lkf_vector + t_ekf_vector,
                (t_lkf_scalar + t_ekf_scalar) / (t_lkf_vector + t_ekf_vector),
                N,
                (t_lkf_scalar + t_ekf_scalar) * 1000.0 / N,
                (t_lkf_vector + t_ekf_vector) * 1000.0 / N);
        fclose(out);
        printf("[OK] Saved: perf_analysis.csv\n");
    }

    free(noisy[0]); free(noisy);
    return 0;
}
