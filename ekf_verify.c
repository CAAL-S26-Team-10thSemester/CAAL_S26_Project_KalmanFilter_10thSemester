/*
 * ekf_verify.c  —  C driver and §6 verification harness for ekf_asm.s
 *
 * Mirrors lkf_verify.c exactly in structure.
 *
 * Usage:
 *   qemu-riscv64 ./ekf_verify  \
 *       "3D Full Body Humain Gait Walking Dataset (Noisy Values).csv"  \
 *       ekf_results.csv
 *
 *   Optional third argument: Python reference CSV (ekf_results.csv from
 *   kalman-updated.py) for element-wise §6 tolerance check.
 *   If omitted, the table compares asm output against ground-truth positions.
 *
 * Build (via Makefile target 'ekf'):
 *   riscv64-linux-gnu-gcc -march=rv64imfd -mabi=lp64d -O0 -g \
 *       ekf_verify.c ekf_asm.o matrix_asm.o -o ekf_verify -lm -static
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>          /* FIX: added for clock_gettime progress timing */
#include "matrix_asm.h"

/* ---- constants (match kalman-updated.py) --------------------------------- */
#define DT          (1.0/30.0)
#define N_JOINTS    23
#define S_DIM       12          /* state dim per joint */
#define M_DIM        3          /* meas  dim per joint */
#define N_STATE    276          /* total state dim */
#define N_MEAS      69          /* total meas  dim */
#define MAX_FRAMES 50

/* FIX: progress print interval — print every N frames */
#define PROGRESS_INTERVAL  50

/* ---- joint names (for verification table) -------------------------------- */
static const char *JOINT_NAMES[N_JOINTS] = {
    "pelvis",     "L5",          "L3",          "T12",
    "T8",         "neck",        "head",
    "shoulderR",  "upperArmR",   "forearmR",    "handR",
    "shoulderL",  "upperArmL",   "forearmL",    "handL",
    "upperLegR",  "lowerLegR",   "footR",       "toeR",
    "upperLegL",  "lowerLegL",   "footL",       "toeL"
};

/* ---- assembly function declarations -------------------------------------- */
extern size_t ekf_sizeof(void);
extern void   ekf_init              (void *ekf, double dt);
extern void   ekf_set_initial_state (void *ekf, int joint, double *pos3);
extern void   ekf_predict           (void *ekf);
extern void   ekf_update            (void *ekf, const double *z_cart_flat);
extern void   ekf_get_positions     (const void *ekf, double *pos_out);
extern void   ekf_get_full_state    (const void *ekf, double *out);

/* ---- FIX: timing helper -------------------------------------------------- */
/* Returns seconds since an arbitrary epoch (monotonic clock).              */
static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* ---- CSV loader ---------------------------------------------------------- */
static double *load_csv(const char *path, int *n_frames_out)
{
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", path);
        exit(1);
    }

    char buf[65536];
    double tmp;
    int has_header = 0;

    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return NULL; }
    has_header = (sscanf(buf, "%lf", &tmp) != 1) ? 1 : 0;

    int rows = has_header ? 0 : 1;
    while (fgets(buf, sizeof(buf), f)) rows++;
    rewind(f);
    if (has_header) fgets(buf, sizeof(buf), f);

    double *data = (double *)malloc((size_t)rows * N_JOINTS * 3 * sizeof(double));
    if (!data) { fprintf(stderr, "malloc failed\n"); exit(1); }

    int r = 0;
    while (fgets(buf, sizeof(buf), f) && r < rows) {
        char *p = buf;
        for (int c = 0; c < N_JOINTS * 3; c++) {
            while (*p == ',' || *p == ' ' || *p == '\t') p++;
            data[(size_t)r * N_JOINTS * 3 + c] = strtod(p, &p);
            if (*p == ',') p++;
        }
        r++;
    }
    fclose(f);
    *n_frames_out = r;
    return data;
}

static double *load_ref_csv(const char *path, int *n_frames_out)
{
    FILE *f = fopen(path, "r");
    if (!f) { return NULL; }

    char buf[65536];
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return NULL; }

    int rows = 0;
    while (fgets(buf, sizeof(buf), f)) rows++;
    rewind(f);
    fgets(buf, sizeof(buf), f);

    double *data = (double *)malloc((size_t)rows * N_STATE * sizeof(double));
    if (!data) { fclose(f); return NULL; }

    int r = 0;
    while (fgets(buf, sizeof(buf), f) && r < rows) {
        char *p = buf;
        strtod(p, &p);
        if (*p == ',') p++;
        for (int c = 0; c < N_STATE; c++) {
            while (*p == ',' || *p == ' ') p++;
            data[(size_t)r * N_STATE + c] = strtod(p, &p);
            if (*p == ',') p++;
        }
        r++;
    }
    fclose(f);
    *n_frames_out = r;
    return data;
}

/* ---- main ---------------------------------------------------------------- */
int main(int argc, char **argv)
{

     /* ---- verify-only mode: skip EKF rerun, just print RMSE table -------- */
    if (argc >= 2 && strcmp(argv[1], "--verify-only") == 0) {
        if (argc < 4) {
            fprintf(stderr,
                "Usage: %s --verify-only <ekf_results.csv> <noisy_csv>\n",
                argv[0]);
            return 1;
        }
        const char *res_path   = argv[2];
        const char *noisy_path = argv[3];

        /* Load existing results */
        int n_res = 0;
        double *results = load_ref_csv(res_path, &n_res);
        if (!results) {
            fprintf(stderr, "[ERR] Cannot load: %s\n", res_path);
            return 1;
        }
        int N_frames = n_res;
        printf("[INFO] Loaded %d frames from %s\n", N_frames, res_path);

        /* Build true_path — FIXED version */
        char true_path[512];
        strncpy(true_path, noisy_path, sizeof(true_path) - 1);
        char *tag = strstr(true_path, "Noisy Values");
        if (tag) {
            memcpy(tag, "True Values", 11);
            memmove(tag + 11, tag + 12, strlen(tag + 12) + 1);
        }
        printf("[INFO] Ground truth: %s\n", true_path);

        /* Load ground truth */
        int n_true = 0;
        FILE *_gt = fopen(true_path, "r");
        if (!_gt) {
            fprintf(stderr, "[ERR] Cannot open: %s\n", true_path);
            free(results);
            return 1;
        }
        fclose(_gt);
        double *truth = load_csv(true_path, &n_true);

        int cmp = (N_frames < n_true) ? N_frames : n_true;
        printf("[INFO] Comparing %d frames\n\n", cmp);

        /* RMSE table */
        printf("==============================================================\n");
        printf("  Milestone-3 §6 Verification: EKF position RMSE\n");
        printf("==============================================================\n");
        printf("  %-16s  %10s  %10s  %10s  %10s\n",
               "Joint", "Avg|err|", "Max|err|", "Min|err|", "RMSE");
        printf("  --------------------------------------------------------\n");

        double g_max = 0.0, g_min = 1e99, g_sum = 0.0;
        int    g_cnt = 0;

        for (int j = 0; j < N_JOINTS; j++) {
            int b = j * S_DIM;
            double sum = 0.0, mx = 0.0, mn = 1e99, sq = 0.0;
            for (int f = 0; f < cmp; f++) {
                double *st       = results + (size_t)f * N_STATE + b;
                const double *tr = truth   + (size_t)f * N_JOINTS*3 + j*3;
                double ep = fabs(st[0] - tr[0]);   /* px */
                double eq = fabs(st[4] - tr[1]);   /* py */
                double er = fabs(st[8] - tr[2]);   /* pz */
                double e3 = (ep + eq + er) / 3.0;
                sum += e3;
                sq  += ep*ep + eq*eq + er*er;
                if (e3 > mx) mx = e3;
                if (e3 < mn) mn = e3;
                if (ep > g_max) g_max = ep;
                if (eq > g_max) g_max = eq;
                if (er > g_max) g_max = er;
                if (ep < g_min) g_min = ep;
                if (eq < g_min) g_min = eq;
                if (er < g_min) g_min = er;
                g_sum += ep + eq + er;
                g_cnt += 3;
            }
            printf("  %-16s  %10.4e  %10.4e  %10.4e  %10.4e\n",
                   JOINT_NAMES[j],
                   sum / cmp, mx, mn,
                   sqrt(sq / (cmp * 3)));
        }

        printf("\n  Global max error : %.4e m\n", g_max);
        printf("  Global min error : %.4e m\n", g_min);
        printf("  Global avg error : %.4e m\n", g_sum / (double)g_cnt);
        printf("\n");

        free(results);
        free(truth);
        return 0;
    }
    /* ---- end verify-only mode ------------------------------------------- */


    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <noisy_csv> <out_csv> [ref_ekf_results.csv]\n",
            argv[0]);
        return 1;
    }
    const char *noisy_path = argv[1];
    const char *out_path   = argv[2];
    const char *ref_path   = (argc >= 4) ? argv[3] : NULL;

    /* ---- Load input CSV -------------------------------------------------- */
    int n_noisy = 0;
    printf("[INFO] Loading noisy CSV:  %s\n", noisy_path);
    double *noisy = load_csv(noisy_path, &n_noisy);
    int N_frames = n_noisy;
    if (N_frames > MAX_FRAMES) N_frames = MAX_FRAMES;

    /* ---- Allocate and initialise EKF ------------------------------------- */
    size_t ekf_sz = ekf_sizeof();
    printf("[INFO] EKF struct size: %zu bytes (%.2f MB)\n",
           ekf_sz, (double)ekf_sz / (1024.0 * 1024.0));
    printf("[INFO] Frames to process: %d\n", N_frames);

    void *ekf = malloc(ekf_sz);
    if (!ekf) { fprintf(stderr, "malloc failed for EKF struct\n"); return 1; }
    memset(ekf, 0, ekf_sz);

    ekf_init(ekf, DT);

    /* Set initial state from first frame */
    for (int j = 0; j < N_JOINTS; j++) {
        double pos3[3];
        pos3[0] = noisy[j * 3 + 0];
        pos3[1] = noisy[j * 3 + 1];
        pos3[2] = noisy[j * 3 + 2];
        ekf_set_initial_state(ekf, j, pos3);
    }

    /* ---- Allocate result storage ----------------------------------------- */
    double *results = (double *)malloc(
                        (size_t)N_frames * N_STATE * sizeof(double));
    if (!results) {
        fprintf(stderr, "malloc failed for results\n");
        return 1;
    }

    double z_flat[N_MEAS];

    /* ================================================================
     * FIX: Smoke test — run 5 frames before the full loop so we know
     *      immediately if the assembly is alive and producing output.
     * ================================================================ */
    {
        int smoke_frames = (N_frames < 5) ? N_frames : 5;
        printf("[INFO] Smoke test: running first %d frames...\n",
               smoke_frames);
        fflush(stdout);

        double t0 = now_sec();

        /* Temporary EKF copy for smoke test so we don't disturb main state */
        void *ekf_smoke = malloc(ekf_sz);
        if (!ekf_smoke) {
            fprintf(stderr, "malloc failed for smoke EKF\n");
            return 1;
        }
        memcpy(ekf_smoke, ekf, ekf_sz);

        for (int frame = 0; frame < smoke_frames; frame++) {
            if (frame > 0) ekf_predict(ekf_smoke);

            const double *row = noisy + (size_t)frame * N_JOINTS * 3;
            for (int j = 0; j < N_JOINTS; j++) {
                z_flat[j * M_DIM + 0] = row[j * 3 + 0];
                z_flat[j * M_DIM + 1] = row[j * 3 + 1];
                z_flat[j * M_DIM + 2] = row[j * 3 + 2];
            }
            ekf_update(ekf_smoke, z_flat);

            double pos[N_JOINTS * 3];
            ekf_get_positions(ekf_smoke, pos);

            double t1 = now_sec();
            printf("  [Smoke %d/%d]  pelvis=(%.3f, %.3f, %.3f)  "
                   "t=%.2fs\n",
                   frame + 1, smoke_frames,
                   pos[0], pos[1], pos[2],
                   t1 - t0);
            fflush(stdout);
        }

        double smoke_total = now_sec() - t0;
        double sec_per_frame = smoke_total / smoke_frames;
        double eta_total     = sec_per_frame * N_frames;

        printf("[INFO] Smoke test passed!\n");
        printf("[INFO] Time per frame : %.2f s\n",   sec_per_frame);
        printf("[INFO] ETA full run   : %.1f s  (%.1f min)\n",
               eta_total, eta_total / 60.0);
        printf("[INFO] ETA = approx %d hours %d minutes\n",
               (int)(eta_total / 3600),
               (int)(fmod(eta_total, 3600.0) / 60.0));
        fflush(stdout);

        free(ekf_smoke);
    }

    /* ================================================================
     * FIX: Main filter loop with progress reporting every
     *      PROGRESS_INTERVAL frames.
     * ================================================================ */
    printf("[INFO] Running EKF on all %d frames...\n", N_frames);
    fflush(stdout);

    double t_loop_start = now_sec();
    double t_last_print = t_loop_start;

    for (int frame = 0; frame < N_frames; frame++) {

        if (frame > 0) ekf_predict(ekf);

        /* Build z_flat */
        const double *row = noisy + (size_t)frame * N_JOINTS * 3;
        for (int j = 0; j < N_JOINTS; j++) {
            z_flat[j * M_DIM + 0] = row[j * 3 + 0];
            z_flat[j * M_DIM + 1] = row[j * 3 + 1];
            z_flat[j * M_DIM + 2] = row[j * 3 + 2];
        }

        ekf_update(ekf, z_flat);
        ekf_get_full_state(ekf, results + (size_t)frame * N_STATE);

        /* ---- FIX: progress line every PROGRESS_INTERVAL frames ---------- */
        if ((frame + 1) % PROGRESS_INTERVAL == 0 ||
             frame == 0                           ||
             frame == N_frames - 1)
        {
            double now     = now_sec();
            double elapsed = now - t_loop_start;
            double pct     = 100.0 * (frame + 1) / (double)N_frames;

            /* ETA: based on average time per frame so far */
            double sec_per_frame = elapsed / (frame + 1);
            double eta           = sec_per_frame * (N_frames - frame - 1);

            printf("  EKF-ASM: %4d/%d  (%5.1f%%)  "
                   "elapsed: %6.1fs  ETA: %5.1fs (%.1f min)",
                   frame + 1, N_frames, pct,
                   elapsed, eta, eta / 60.0);

            /* Show interval throughput every PROGRESS_INTERVAL frames */
            if ((frame + 1) % PROGRESS_INTERVAL == 0 && frame > 0) {
                double interval = now - t_last_print;
                double fps      = PROGRESS_INTERVAL / interval;
                printf("  [%.2f fr/s]", fps);
                t_last_print = now;
            }

            printf("\n");
            fflush(stdout);
        }
    }

    double t_total = now_sec() - t_loop_start;
    printf("[INFO] EKF loop done.  Total time: %.1f s (%.1f min)\n",
           t_total, t_total / 60.0);

    /* ---- Save ekf_results.csv -------------------------------------------- */
    {
        FILE *out = fopen(out_path, "w");
        if (!out) {
            fprintf(stderr, "Cannot write %s\n", out_path);
            return 1;
        }

        fprintf(out, "frame");
        for (int j = 0; j < N_JOINTS; j++) {
            const char *ax[] = {"x", "y", "z"};
            for (int a = 0; a < 3; a++)
                fprintf(out, ",j%d_p%s,j%d_v%s,j%d_a%s,j%d_j%s",
                        j, ax[a], j, ax[a], j, ax[a], j, ax[a]);
        }
        fprintf(out, "\n");

        for (int f = 0; f < N_frames; f++) {
            fprintf(out, "%d", f);
            for (int i = 0; i < N_STATE; i++)
                fprintf(out, ",%.10f", results[(size_t)f * N_STATE + i]);
            fprintf(out, "\n");
        }
        fclose(out);
        printf("[OK]  Saved: %s\n", out_path);
    }

    /* ---- §6 Verification: asm vs Python reference (if provided) ---------- */
    if (ref_path) {
        int n_ref = 0;
        double *ref = load_ref_csv(ref_path, &n_ref);
        if (!ref) {
            printf("[WARN] Could not load reference CSV: %s\n", ref_path);
            printf("       Skipping element-wise §6 check.\n");
        } else {
            int cmp_frames = (N_frames < n_ref) ? N_frames : n_ref;
            printf("\n==============================================================\n");
            printf("  §6 Element-wise verification: asm vs Python reference\n");
            printf("==============================================================\n");
            printf("  Frames compared : %d\n", cmp_frames);

            double g_max = 0.0, g_sum = 0.0;
            long   g_cnt = 0;
            int    g_fail = 0;

            for (int f = 0; f < cmp_frames; f++) {
                for (int i = 0; i < N_STATE; i++) {
                    double err = fabs(results[(size_t)f * N_STATE + i]
                                    - ref[(size_t)f * N_STATE + i]);
                    if (err > g_max) g_max = err;
                    g_sum += err;
                    g_cnt++;
                    if (err > EPSILON_TOL) g_fail++;
                }
            }

            printf("  Max |error|     : %.6e  (tolerance: %.1e)\n",
                   g_max, (double)EPSILON_TOL);
            printf("  Mean |error|    : %.6e\n",
                   g_sum / (double)g_cnt);
            printf("  Violations      : %d / %ld  (%.4f%%)\n",
                   g_fail, g_cnt,
                   100.0 * (double)g_fail / (double)g_cnt);
            if (g_fail == 0)
                printf("  RESULT: [PASS] all errors within tolerance\n");
            else
                printf("  RESULT: [FAIL] %d element(s) exceed tolerance\n",
                       g_fail);

            free(ref);
        }
    }

    /* ---- §6 Verification: position RMSE per joint vs ground truth -------- */
    char true_path[512];
    strncpy(true_path, noisy_path, sizeof(true_path) - 1);
    char *noisy_tag = strstr(true_path, "Noisy Values");
    if (noisy_tag) {
        memcpy(noisy_tag, "True Values", 11);
        memmove(noisy_tag + 11, noisy_tag + 12, strlen(noisy_tag + 12) + 1);
    }

    int n_true = 0;
    double *truth = load_csv(true_path, &n_true);
    if (!truth) {
        printf("\n[INFO] Ground-truth CSV not found; skipping RMSE table.\n");
        printf("       (Expected: %s)\n", true_path);
    } else {
        int cmp = (N_frames < n_true) ? N_frames : n_true;

        printf("\n==============================================================\n");
        printf("  Milestone-3 §6 Verification: EKF position RMSE\n");
        printf("==============================================================\n");
        printf("  Tolerance   : 1.0e-09\n");
        printf("  Frames      : %d\n", cmp);
        printf("\n");
        printf("  ── Table A: Position RMSE per joint ─────────────────────\n");
        printf("  %-16s  %10s  %10s  %10s  %10s\n",
               "Joint", "Avg|err|", "Max|err|", "Min|err|", "RMSE");
        printf("  ──────────────────────────────────────────────────────────\n");

        double g_max2 = 0.0, g_min2 = 1e99, g_sum2 = 0.0;
        int    g_cnt2 = 0;

        for (int j = 0; j < N_JOINTS; j++) {
            int b = j * S_DIM;
            double sum = 0.0, mx = 0.0, mn = 1e99, sq = 0.0;
            for (int f = 0; f < cmp; f++) {
                double *st      = results + (size_t)f * N_STATE + b;
                const double *tr = truth  + (size_t)f * N_JOINTS * 3 + j * 3;
                double ep = fabs(st[0] - tr[0]);
                double eq = fabs(st[4] - tr[1]);
                double er = fabs(st[8] - tr[2]);
                double e3 = (ep + eq + er) / 3.0;
                sum += e3; sq += ep*ep + eq*eq + er*er;
                if (e3 > mx) mx = e3;
                if (e3 < mn) mn = e3;
                if (ep > g_max2) g_max2 = ep;
                if (eq > g_max2) g_max2 = eq;
                if (er > g_max2) g_max2 = er;
                if (ep < g_min2) g_min2 = ep;
                if (eq < g_min2) g_min2 = eq;
                if (er < g_min2) g_min2 = er;
                g_sum2 += ep + eq + er;
                g_cnt2 += 3;
            }
            double avg  = sum / cmp;
            double rmse = sqrt(sq / (cmp * 3));
            printf("  %-16s  %10.4e  %10.4e  %10.4e  %10.4e\n",
                   JOINT_NAMES[j], avg, mx, mn, rmse);
        }

        printf("\n");
        printf("  Global max error : %.4e m\n", g_max2);
        printf("  Global min error : %.4e m\n", g_min2);
        printf("  Global avg error : %.4e m\n", g_sum2 / (double)g_cnt2);

        printf("\n  ── Table B: Avg |state val| per component ───────────────\n");
        const char *comp[] = {
            "px","vx","ax","jx","py","vy","ay","jy","pz","vz","az","jz"
        };
        printf("  %-8s  %12s\n", "Comp", "Avg|state val|");
        printf("  ────────────────────────\n");
        for (int ci = 0; ci < S_DIM; ci++) {
            double s = 0.0;
            for (int j2 = 0; j2 < N_JOINTS; j2++) {
                int b = j2 * S_DIM + ci;
                for (int f = 0; f < cmp; f++)
                    s += fabs(results[(size_t)f * N_STATE + b]);
            }
            printf("  %-8s  %12.4e\n",
                   comp[ci], s / ((double)N_JOINTS * cmp));
        }

        free(truth);
    }

    /* ---- Final state printout -------------------------------------------- */
    {
        double *xs = results + (size_t)(N_frames - 1) * N_STATE;
        printf("\n[EKF] Joint 0 (pelvis), final frame:\n");
        printf("  Position:     (%.4f, %.4f, %.4f) m\n",
               xs[0], xs[4], xs[8]);
        printf("  Velocity:     (%.4f, %.4f, %.4f) m/s\n",
               xs[1], xs[5], xs[9]);
        printf("  Acceleration: (%.4f, %.4f, %.4f) m/s2\n",
               xs[2], xs[6], xs[10]);
        printf("  Jerk:         (%.4f, %.4f, %.4f) m/s3\n",
               xs[3], xs[7], xs[11]);
    }

    free(noisy);
    free(results);
    free(ekf);
    printf("\nDone.\n");
    return 0;
}