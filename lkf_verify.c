/*
 * lkf_verify.c  —  Milestone-3 §6 Numerical Verification Harness
 * ================================================================
 * Default: MAX_FRAMES = 100  (covers filter convergence, ~4-5 min under QEMU)
 * Override via third argument: ./lkf_verify noisy.csv ref.csv 200
 *
 * Outputs:
 *   lkf_asm_results.csv        estimated state (N_frames × 277 cols)
 *   lkf_asm_verification.csv   §6 error statistics table
 *
 * Build:
 *   riscv64-linux-gnu-gcc -march=rv64imfd -mabi=lp64d -O0 -g -Wall \
 *       lkf_verify.c lkf_asm.o matrix_asm.o -o lkf_verify -lm -static
 *
 * Run:
 *   qemu-riscv64 ./lkf_verify "...Noisy Values....csv" lkf_results.csv [N]
 * ================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define NUM_JOINTS   23
#define STATE_DIM    12
#define MEAS_DIM      3
#define N_STATE     276
#define N_MEAS       69
#define DT          (1.0 / 30.0)
#define EPSILON_TOL  1e-9
#define DEFAULT_MAX_FRAMES 100

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

extern size_t lkf_sizeof(void);
extern void   lkf_init              (void *lkf, double dt);
extern void   lkf_set_initial_state (void *lkf, int joint, double *pos3);
extern void   lkf_predict           (void *lkf);
extern void   lkf_update            (void *lkf, const double *z_flat);
extern void   lkf_get_full_state    (const void *lkf, double *out);

static double **alloc_matrix(int rows, int cols) {
    double *data = (double *)calloc((size_t)rows*cols, sizeof(double));
    double **m   = (double **)malloc((size_t)rows*sizeof(double *));
    if (!data||!m){fprintf(stderr,"OOM\n");exit(1);}
    for(int i=0;i<rows;i++) m[i]=data+(size_t)i*cols;
    return m;
}

static double **load_noisy_csv(const char *path, int *n) {
    FILE *f=fopen(path,"r"); if(!f){fprintf(stderr,"Cannot open: %s\n",path);exit(1);}
    char buf[65536]; int hdr=0;
    if(fgets(buf,sizeof(buf),f)){double d;hdr=(sscanf(buf,"%lf",&d)!=1);}
    rewind(f); int rows=0;
    while(fgets(buf,sizeof(buf),f)) rows++;
    if(hdr) rows--;
    rewind(f); if(hdr) fgets(buf,sizeof(buf),f);
    int cols=NUM_JOINTS*MEAS_DIM;
    double **m=alloc_matrix(rows,cols);
    for(int r=0;r<rows;r++){
        if(!fgets(buf,sizeof(buf),f)) break;
        char *p=buf;
        for(int c=0;c<cols;c++){m[r][c]=strtod(p,&p);if(*p==',')p++;}
    }
    fclose(f); *n=rows; return m;
}

static double **load_ref_csv(const char *path, int *n) {
    FILE *f=fopen(path,"r"); if(!f){fprintf(stderr,"Cannot open: %s\n",path);exit(1);}
    char buf[65536]; fgets(buf,sizeof(buf),f);
    int rows=0; long pos=ftell(f);
    while(fgets(buf,sizeof(buf),f)) rows++;
    fseek(f,pos,SEEK_SET);
    double **m=alloc_matrix(rows,N_STATE);
    for(int r=0;r<rows;r++){
        if(!fgets(buf,sizeof(buf),f)) break;
        char *p=buf; strtod(p,&p); if(*p==',')p++;
        for(int c=0;c<N_STATE;c++){m[r][c]=strtod(p,&p);if(*p==',')p++;}
    }
    fclose(f); *n=rows; return m;
}

int main(int argc, char **argv) {
    if(argc<3){
        fprintf(stderr,"Usage: %s <noisy_csv> <ref_lkf_csv> [max_frames]\n",argv[0]);
        return 1;
    }
    int max_frames=DEFAULT_MAX_FRAMES;
    if(argc>=4){max_frames=atoi(argv[3]);if(max_frames<=0)max_frames=DEFAULT_MAX_FRAMES;}

    int nn,nr;
    printf("[INFO] Loading noisy CSV:     %s\n",argv[1]);
    double **noisy=load_noisy_csv(argv[1],&nn);
    printf("[INFO] Loading reference CSV: %s\n",argv[2]);
    double **ref  =load_ref_csv  (argv[2],&nr);
    int N=nn<nr?nn:nr; if(N>max_frames) N=max_frames;
    printf("[INFO] Frames: %d  (noisy=%d, ref=%d, cap=%d)\n",N,nn,nr,max_frames);

    size_t sz=lkf_sizeof();
    printf("[INFO] LKF struct: %zu bytes (%.2f MB)\n",sz,(double)sz/(1024.*1024.));
    void *lkf=malloc(sz); if(!lkf){fprintf(stderr,"OOM\n");return 1;}
    memset(lkf,0,sz);
    lkf_init(lkf,DT);
    for(int j=0;j<NUM_JOINTS;j++){
        double p3[3]={noisy[0][j*MEAS_DIM],noisy[0][j*MEAS_DIM+1],noisy[0][j*MEAS_DIM+2]};
        lkf_set_initial_state(lkf,j,p3);
    }

    double **results=alloc_matrix(N,N_STATE);
    double z[N_MEAS];
    printf("[INFO] Running LKF assembly on %d frames...\n",N);
    for(int fr=0;fr<N;fr++){
        if(fr>0) lkf_predict(lkf);
        for(int j=0;j<NUM_JOINTS;j++){
            z[j*MEAS_DIM  ]=noisy[fr][j*MEAS_DIM  ];
            z[j*MEAS_DIM+1]=noisy[fr][j*MEAS_DIM+1];
            z[j*MEAS_DIM+2]=noisy[fr][j*MEAS_DIM+2];
        }
        lkf_update(lkf,z);
        lkf_get_full_state(lkf,results[fr]);
        if((fr+1)%10==0||fr==N-1) printf("\r  LKF-ASM: %d/%d   ",fr+1,N);
    }
    printf("\n");

    /* Save results */
    {
        FILE *out=fopen("lkf_asm_results.csv","w");
        fprintf(out,"frame");
        for(int j=0;j<NUM_JOINTS;j++) for(int ax=0;ax<MEAS_DIM;ax++){
            const char *a=(ax==0)?"x":(ax==1)?"y":"z";
            fprintf(out,",j%d_p%s,j%d_v%s,j%d_a%s,j%d_j%s",j,a,j,a,j,a,j,a);
        }
        fprintf(out,"\n");
        for(int fr=0;fr<N;fr++){
            fprintf(out,"%d",fr);
            for(int c=0;c<N_STATE;c++) fprintf(out,",%.15g",results[fr][c]);
            fprintf(out,"\n");
        }
        fclose(out); printf("[OK] Saved: lkf_asm_results.csv\n");
    }

    /* §6 verification */
    double avg_joint[NUM_JOINTS], avg_comp[STATE_DIM];
    double max_err=0.0, min_err=1e300; long violations=0;
    memset(avg_joint,0,sizeof(avg_joint)); memset(avg_comp,0,sizeof(avg_comp));
    for(int fr=0;fr<N;fr++)
        for(int j=0;j<NUM_JOINTS;j++)
            for(int c=0;c<STATE_DIM;c++){
                double err=fabs(results[fr][j*STATE_DIM+c]-ref[fr][j*STATE_DIM+c]);
                avg_joint[j]+=err; avg_comp[c]+=err;
                if(err>max_err) max_err=err;
                if(err<min_err) min_err=err;
                if(err>EPSILON_TOL) violations++;
            }
    double nj=(double)N*STATE_DIM, nc=(double)N*NUM_JOINTS;
    for(int j=0;j<NUM_JOINTS;j++) avg_joint[j]/=nj;
    for(int c=0;c<STATE_DIM; c++) avg_comp[c] /=nc;

    printf("\n==================================================================\n");
    printf("  Milestone-3 §6 Numerical Verification  (LKF Assembly)\n");
    printf("==================================================================\n");
    printf("  Tolerance          : %.1e\n",EPSILON_TOL);
    printf("  Frames compared    : %d\n",N);
    printf("  Total comparisons  : %ld\n",(long)N*N_STATE);
    printf("  Global max |error| : %.6e\n",max_err);
    printf("  Global min |error| : %.6e\n",min_err);
    printf("  Violations > eps   : %ld\n\n",violations);

    printf("  ── Table A: Avg |error| per joint ─────────────────────────\n");
    printf("  %-18s  %14s\n","Joint","Avg |error|");
    for(int j=0;j<NUM_JOINTS;j++)
        printf("  %-18s  %14.6e\n",JOINT_NAMES[j],avg_joint[j]);

    printf("\n  ── Table B: Avg |error| per state component ────────────────\n");
    printf("  %-10s  %14s\n","Component","Avg |error|");
    for(int c=0;c<STATE_DIM;c++)
        printf("  %-10s  %14.6e\n",COMP_NAMES[c],avg_comp[c]);

    printf("\n  [%s] LKF Assembly Verification\n",violations==0?"PASS ✓":"FAIL ✗");
    printf("==================================================================\n\n");

    {
        FILE *v=fopen("lkf_asm_verification.csv","w");
        fprintf(v,"section,name,avg_abs_error\n");
        for(int j=0;j<NUM_JOINTS;j++) fprintf(v,"joint,%s,%.15g\n",JOINT_NAMES[j],avg_joint[j]);
        for(int c=0;c<STATE_DIM; c++) fprintf(v,"component,%s,%.15g\n",COMP_NAMES[c],avg_comp[c]);
        fprintf(v,"\nsummary,max_err,%.15g\n",max_err);
        fprintf(v,"summary,min_err,%.15g\n",min_err);
        fprintf(v,"summary,violations,%ld\n",violations);
        fprintf(v,"summary,frames,%d\n",N);
        fprintf(v,"summary,result,%s\n",violations==0?"PASS":"FAIL");
        fclose(v); printf("[OK] Saved: lkf_asm_verification.csv\n");
    }

    free(lkf); free(results[0]); free(results);
    free(noisy[0]); free(noisy); free(ref[0]); free(ref);
    return violations==0?0:1;
}
