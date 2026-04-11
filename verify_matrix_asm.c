/*
 * verify_matrix_asm.c  —  Numerical verification harness for matrix_asm.s
 *
 * Builds and runs under a RISC-V toolchain:
 *   riscv64-unknown-elf-gcc -O0 -march=rv64ifd -mabi=lp64d \
 *       verify_matrix_asm.c matrix_asm.s -o verify_matrix_asm -lm
 *   spike --isa=rv64ifd pk verify_matrix_asm
 *
 * Checks every function against NumPy-equivalent reference values.
 * All |error| must be <= EPSILON_TOL = 1e-9  (Milestone-3 §6).
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "matrix_asm.h"

/* ---- helpers ------------------------------------------------------------- */

static double max_abs_err(const double *a, const double *b, int len) {
    double e = 0.0;
    for (int i = 0; i < len; i++) {
        double d = fabs(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

static void print_result(const char *name, double err, int ok_thresh) {
    int pass = (err <= EPSILON_TOL);
    printf("  %-30s  max_err=%.3e  %s\n",
           name, err, pass ? "PASS" : "FAIL");
}

/* ---- reference implementations (pure C, matches Python) ------------------ */

static void ref_mat_mul(double *C, const double *A, const double *B,
                        int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0.0;
            for (int l = 0; l < k; l++)
                s += A[i*k+l] * B[l*n+j];
            C[i*n+j] = s;
        }
}

static void ref_mat_eye(double *A, int n) {
    memset(A, 0, n*n*sizeof(double));
    for (int i = 0; i < n; i++) A[i*n+i] = 1.0;
}

static void ref_mat_add(double *C, const double *A, const double *B, int m, int n) {
    for (int i = 0; i < m*n; i++) C[i] = A[i] + B[i];
}
static void ref_mat_sub(double *C, const double *A, const double *B, int m, int n) {
    for (int i = 0; i < m*n; i++) C[i] = A[i] - B[i];
}
static void ref_mat_transpose(double *B, const double *A, int m, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            B[j*m+i] = A[i*n+j];
}

static double ref_fast_atan2(double y, double x) {
    if (x == 0.0 && y == 0.0) return 0.0;
    const double PI  = 3.14159265358979323846;
    const double PI2 = 1.57079632679489661923;
    double ax = fabs(x), ay = fabs(y);
    double z, angle;
    if (ax >= ay) {
        z     = ay / ax;
        angle = (PI/4.0)*z - z*(z-1.0)*(0.2447 + 0.0663*z);
    } else {
        z     = ax / ay;
        angle = PI2 - ((PI/4.0)*z - z*(z-1.0)*(0.2447 + 0.0663*z));
    }
    if (x < 0)       angle = (y >= 0) ? (PI - angle) : (angle - PI);
    else if (y < 0)  angle = -angle;
    return angle;
}

static double ref_wrap_angle(double a) {
    const double PI  = 3.14159265358979323846;
    const double PI2 = 6.28318530717958647692;
    while (a >  PI) a -= PI2;
    while (a < -PI) a += PI2;
    return a;
}

/* Simple Gaussian-elimination inverse for reference */
static int ref_inverse(double *Ainv, const double *A, int n) {
    double *M = malloc(n*n*2*sizeof(double));
    /* Build augmented [A | I] */
    for (int i=0;i<n;i++) for (int j=0;j<n;j++) {
        M[i*(2*n)+j]   = A[i*n+j];
        M[i*(2*n)+n+j] = (i==j)?1.0:0.0;
    }
    for (int col=0;col<n;col++) {
        /* partial pivot */
        int pr=col; double pv=fabs(M[col*(2*n)+col]);
        for(int r=col+1;r<n;r++){double v=fabs(M[r*(2*n)+col]);if(v>pv){pv=v;pr=r;}}
        if(pv<1e-14){free(M);return 0;}
        if(pr!=col) for(int j=0;j<2*n;j++){double t=M[col*(2*n)+j];M[col*(2*n)+j]=M[pr*(2*n)+j];M[pr*(2*n)+j]=t;}
        double d=M[col*(2*n)+col];
        for(int j=0;j<2*n;j++) M[col*(2*n)+j]/=d;
        for(int r=0;r<n;r++) if(r!=col){
            double f=M[r*(2*n)+col];
            for(int j=0;j<2*n;j++) M[r*(2*n)+j]-=f*M[col*(2*n)+j];
        }
    }
    for(int i=0;i<n;i++) for(int j=0;j<n;j++) Ainv[i*n+j]=M[i*(2*n)+n+j];
    free(M);
    return 1;
}

/* ============================================================ */
int main(void) {
    printf("\n=== matrix_asm.s Numerical Verification (tol=%.0e) ===\n\n",
           EPSILON_TOL);
    int all_pass = 1;

    /* ------------------------------------------------------------------ */
    /* 1. mat_eye                                                          */
    {
        int n = 8;
        double A[64], R[64];
        mat_eye(A, n);
        ref_mat_eye(R, n);
        double e = max_abs_err(A, R, n*n);
        print_result("mat_eye(8)", e, 1);
        if (e > EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 2. mat_add / mat_sub                                                */
    {
        int m=7, n=5;
        double A[35], B[35], C[35], R[35];
        for(int i=0;i<35;i++){A[i]=i*0.1;B[i]=-(i*0.07)+3.14;}
        mat_add(C,A,B,m,n); ref_mat_add(R,A,B,m,n);
        double e=max_abs_err(C,R,m*n);
        print_result("mat_add(7x5)", e, 1); if(e>EPSILON_TOL) all_pass=0;

        mat_sub(C,A,B,m,n); ref_mat_sub(R,A,B,m,n);
        e=max_abs_err(C,R,m*n);
        print_result("mat_sub(7x5)", e, 1); if(e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 3. mat_transpose                                                    */
    {
        int m=6, n=9;
        double A[54], B[54], R[54];
        for(int i=0;i<m*n;i++) A[i]=i*0.13-3.0;
        mat_transpose(B,A,m,n); ref_mat_transpose(R,A,m,n);
        double e=max_abs_err(B,R,m*n);
        print_result("mat_transpose(6x9)", e, 1); if(e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 4. mat_mul — small case                                             */
    {
        int m=4, k=5, n=3;
        double A[20], B[15], C[12], R[12];
        for(int i=0;i<m*k;i++) A[i]=sin(i+1.0);
        for(int i=0;i<k*n;i++) B[i]=cos(i+1.0);
        mat_mul(C,A,B,m,k,n); ref_mat_mul(R,A,B,m,k,n);
        double e=max_abs_err(C,R,m*n);
        print_result("mat_mul(4x5 @ 5x3)", e, 1); if(e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 5. mat_mul — LKF-scale: 276x276 @ 276x69  (trimmed for speed)      */
    {
        /* Use 12x12 as a proxy for the block diagonal pattern */
        int m=12, k=12, n=3;
        double A[144], B[36], C[36], R[36];
        for(int i=0;i<m*k;i++) A[i]=(double)(i%7-3)*0.031;
        for(int i=0;i<k*n;i++) B[i]=(double)(i%5-2)*0.071;
        mat_mul(C,A,B,m,k,n); ref_mat_mul(R,A,B,m,k,n);
        double e=max_abs_err(C,R,m*n);
        print_result("mat_mul(12x12 @ 12x3)", e, 1); if(e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 6. mat_vec_mul                                                      */
    {
        int m=9, n=7;
        double A[63], x[7], y[9], r[9];
        for(int i=0;i<m*n;i++) A[i]=sin(i*0.4);
        for(int i=0;i<n;i++)   x[i]=cos(i*0.7);
        mat_vec_mul(y,A,x,m,n);
        /* reference: row-dot */
        for(int i=0;i<m;i++){double s=0;for(int j=0;j<n;j++)s+=A[i*n+j]*x[j];r[i]=s;}
        double e=max_abs_err(y,r,m);
        print_result("mat_vec_mul(9x7)", e, 1); if(e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 7. mat_inverse_nxn — 3x3                                           */
    {
        double A[9]={4,3,0, 6,3,0, 2,3,1};
        double Ainv[9], Ref[9];
        int piv[3];
        int ok = mat_inverse_nxn(Ainv, A, 3, piv);
        ref_inverse(Ref, A, 3);
        double e = max_abs_err(Ainv, Ref, 9);
        printf("  %-30s  ok=%d  max_err=%.3e  %s\n",
               "mat_inverse_nxn(3x3)", ok, e,
               (ok && e<=EPSILON_TOL) ? "PASS":"FAIL");
        if (!ok || e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 8. mat_inverse_nxn — 6x6 random                                    */
    {
        double A[36], Ainv[36], Ref[36];
        int piv[6];
        /* Well-conditioned random-ish matrix */
        for(int i=0;i<6;i++) for(int j=0;j<6;j++)
            A[i*6+j] = sin((i+1.0)*(j+2.0)) + (i==j?5.0:0.0);
        int ok = mat_inverse_nxn(Ainv, A, 6, piv);
        ref_inverse(Ref, A, 6);
        double e = max_abs_err(Ainv, Ref, 36);
        printf("  %-30s  ok=%d  max_err=%.3e  %s\n",
               "mat_inverse_nxn(6x6)", ok, e,
               (ok && e<=EPSILON_TOL) ? "PASS":"FAIL");
        if (!ok || e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 9. mat_inverse_nxn — singular detection                            */
    {
        double A[9]={1,2,3, 2,4,6, 0,0,1}; /* rows 0 and 1 linearly dependent */
        double Ainv[9]; int piv[3];
        int ok = mat_inverse_nxn(Ainv, A, 3, piv);
        printf("  %-30s  ok=%d  %s\n",
               "mat_inverse_nxn(singular)", ok,
               (ok==0) ? "PASS (correctly detected)" : "FAIL");
        if (ok != 0) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 10. fast_atan2 — sweep quadrants                                   */
    {
        double angles[][2] = {
            {1.0,  1.0}, {-1.0,  1.0}, {-1.0, -1.0}, {1.0, -1.0},
            {0.0,  1.0}, {0.0,  -1.0}, {1.0,   0.0}, {-1.0,  0.0},
            {3.0,  4.0}, {-3.0,  4.0}, {0.5,   0.1},
        };
        int N = sizeof(angles)/sizeof(angles[0]);
        double max_e = 0.0;
        for (int i=0;i<N;i++) {
            double y=angles[i][0], x=angles[i][1];
            double got = fast_atan2(y,x);
            double ref = ref_fast_atan2(y,x);
            double e = fabs(got-ref);
            if(e>max_e) max_e=e;
        }
        print_result("fast_atan2 (11 cases)", max_e, 1);
        if(max_e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 11. wrap_angle                                                      */
    {
        double vals[] = {0.0, 3.14, -3.14, 6.28, -6.28, 10.0, -10.0, 1.5};
        int N = sizeof(vals)/sizeof(vals[0]);
        double max_e = 0.0;
        for(int i=0;i<N;i++){
            double got = wrap_angle(vals[i]);
            double ref = ref_wrap_angle(vals[i]);
            double e=fabs(got-ref);
            if(e>max_e) max_e=e;
        }
        print_result("wrap_angle (8 cases)", max_e, 1);
        if(max_e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    /* 12. mat_joseph_update — 4×4 P, 2-measurement example              */
    {
        int n=4, m=2;
        /* Simple identity-like inputs for easy manual verification */
        double P[16], K[8], H[8], R[4];
        double Pout[16], Ref[16];
        double scratch[4*16];   /* 4*n*n — K^T lives in scratch[3*n*n..4*n*n) */
        ref_mat_eye(P,4);
        /* K = 0.5 * ones(4,2) */
        for(int i=0;i<8;i++) K[i]=0.5;
        /* H = [1,0,0,0; 0,0,1,0] */
        memset(H,0,sizeof(H)); H[0]=1.0; H[6]=1.0;
        /* R = 0.01*I(2) */
        memset(R,0,sizeof(R)); R[0]=0.01; R[3]=0.01;

        mat_joseph_update(Pout,P,K,H,R,n,m,scratch);

        /* reference: compute with ref functions */
        double IKH[16], tmp[16], KRKt[16], KR[8];
        ref_mat_eye(IKH,4);
        double KH[16];
        ref_mat_mul(KH,K,H,4,2,4);
        ref_mat_sub(IKH,IKH,KH,4,4);
        double IKHt[16]; ref_mat_transpose(IKHt,IKH,4,4);
        ref_mat_mul(tmp,IKH,P,4,4,4);
        ref_mat_mul(Ref,tmp,IKHt,4,4,4);
        ref_mat_mul(KR,K,R,4,2,2);
        double Kt[8]; ref_mat_transpose(Kt,K,4,2);
        ref_mat_mul(KRKt,KR,Kt,4,2,4);
        ref_mat_add(Ref,Ref,KRKt,4,4);

        /* ---- debug: print scratch slots and Pout row by row ---- */
        printf("\n  [DBG] scratch slot sizes: nn8=%d  (n=%d m=%d)\n", n*n, n, m);
        printf("  [DBG] IKH  (scratch[0..15]):\n");
        for(int r=0;r<n;r++){
            printf("    "); for(int c=0;c<n;c++) printf("%8.4f ", scratch[r*n+c]); printf("\n");
        }
        printf("  [DBG] tmp1 (scratch[16..31]):\n");
        for(int r=0;r<n;r++){
            printf("    "); for(int c=0;c<n;c++) printf("%8.4f ", scratch[n*n+r*n+c]); printf("\n");
        }
        printf("  [DBG] tmp2 (scratch[32..47]):\n");
        for(int r=0;r<n;r++){
            printf("    "); for(int c=0;c<n;c++) printf("%8.4f ", scratch[2*n*n+r*n+c]); printf("\n");
        }
        printf("  [DBG] KT   (scratch[48..55], m*n=%d elems):\n", m*n);
        for(int r=0;r<m;r++){
            printf("    "); for(int c=0;c<n;c++) printf("%8.4f ", scratch[3*n*n+r*n+c]); printf("\n");
        }
        printf("  [DBG] Pout (asm result):\n");
        for(int r=0;r<n;r++){
            printf("    "); for(int c=0;c<n;c++) printf("%8.4f ", Pout[r*n+c]); printf("\n");
        }
        printf("  [DBG] Ref  (expected):\n");
        for(int r=0;r<n;r++){
            printf("    "); for(int c=0;c<n;c++) printf("%8.4f ", Ref[r*n+c]); printf("\n");
        }
        /* ---- end debug ---- */

        double e=max_abs_err(Pout,Ref,16);
        print_result("mat_joseph_update(4,2)", e, 1);
        if(e>EPSILON_TOL) all_pass=0;
    }

    /* ------------------------------------------------------------------ */
    printf("\n%s\n\n",
        all_pass ? "=== ALL TESTS PASSED ===" : "=== SOME TESTS FAILED ===");
    return all_pass ? 0 : 1;
}