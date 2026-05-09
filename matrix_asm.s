# =============================================================================
#  matrix_asm.s  —  RISC-V Scalar Assembly: Matrix Utility Functions
#  Kalman Filter Milestone-3
# =============================================================================
#
#  Functions implemented (all match kalman-updated.py naming exactly):
#
#    mat_eye(A, n)
#    mat_mul(C, A, B, m, k, n)          C = A @ B
#    mat_add(C, A, B, m, n)             C = A + B
#    mat_sub(C, A, B, m, n)             C = A - B
#    mat_transpose(B, A, m, n)          B = A^T
#    mat_scale_add(C, A, B, alpha, m, n) C = A + alpha*B
#    mat_vec_mul(y, A, x, m, n)         y = A*x
#    mat_inverse_nxn(Ainv, A, n, piv)   LU-based inverse; returns 1=ok, 0=singular
#    mat_joseph_update(Pout,P,K,H,R,n,m,scratch)
#    fast_atan2(y, x)                   Scheinerman-Lyons poly atan2
#    wrap_angle(a)                      wrap to (-pi, pi]
#
#  ABI: RISC-V LP64D  (RV64IFD)
#    Integer args/return : a0-a7  (x10-x17)
#    FP args/return      : fa0-fa7 (f10-f17)
#    Callee-saved int    : s0-s11 (x8-x9, x18-x27)
#    Callee-saved FP     : fs0-fs11 (f8-f9, f18-f27)
#    Temporaries int     : t0-t6  (x5-x7, x28-x31)
#    Temporaries FP      : ft0-ft11 (f0-f7, f28-f31)
#
#  All matrices: row-major, float64 (8 bytes/element).
#  Element [i][j] of (m x n) matrix at ptr p:  p + (i*n + j)*8
#
#  Optimisations:
#    * fmadd.d / fnmsub.d for every multiply-accumulate
#    * mat_mul inner loop unrolled x4
#    * mat_add / sub / scale_add / vec_mul unrolled x4
#    * wrap_angle uses single floor-division (no loop)
#    * LU with partial pivoting (Doolittle, matches np.linalg.inv)
# =============================================================================

    .section .rodata
    .align 3

# FP constants — loaded with:  la t0, <label>  ;  fld ft, 0(t0)
fp_zero:    .double  0.0
fp_one:     .double  1.0
fp_pi:      .double  3.14159265358979323846
fp_pi2:     .double  1.57079632679489661923
fp_pi4:     .double  0.78539816339744830962
fp_2pi:     .double  6.28318530717958647692
fp_c0:      .double  0.2447
fp_c1:      .double  0.0663

    .section .text

# =============================================================================
#  mat_eye  —  A = I_n
#
#  void mat_eye(double *A, int n)
#  a0=A, a1=n   Leaf — no frame.
# =============================================================================
    .globl mat_eye
    .type  mat_eye, @function
mat_eye:
    la      t0, fp_zero;  fld ft0, 0(t0)   # ft0 = 0.0
    la      t0, fp_one;   fld ft1, 0(t0)   # ft1 = 1.0

    # Zero entire n*n block
    mul     t1, a1, a1          # element count
    slli    t1, t1, 3           # byte count
    add     t2, a0, t1          # end ptr
    mv      t3, a0              # walking ptr
.Leye_zero:
    bge     t3, t2, .Leye_diag_start
    fsd     ft0, 0(t3)
    addi    t3, t3, 8
    j       .Leye_zero

    # Set A[i][i] = 1.0
.Leye_diag_start:
    li      t0, 0
.Leye_diag:
    bge     t0, a1, .Leye_ret
    addi    t1, a1, 1           # n+1
    mul     t2, t0, t1          # i*(n+1)
    slli    t2, t2, 3
    add     t2, a0, t2
    fsd     ft1, 0(t2)
    addi    t0, t0, 1
    j       .Leye_diag
.Leye_ret:
    ret
    .size mat_eye, .-mat_eye


# =============================================================================
#  mat_mul  —  C = A @ B    (m×k) × (k×n) -> (m×n)
#
#  void mat_mul(double *C, const double *A, const double *B,
#               int m, int k, int n)
#  a0=C, a1=A, a2=B, a3=m, a4=k, a5=n
#
#  s0=C  s1=A  s2=B  s3=m  s4=k  s5=n  s6=i  s7=j  s8=l  s9=&A[i][0]
#  fs0=0.0constant   ft0=acc   ft1..ft8=temps
# =============================================================================
    .globl mat_mul
    .type  mat_mul, @function
mat_mul:
    addi    sp, sp, -96
    sd      ra,  0(sp)
    sd      s0,  8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)
    sd      s8, 72(sp)
    sd      s9, 80(sp)          # s9 = row-base ptr; must save (callee-saved)
    fsd     fs0, 88(sp)         # fs0 = 0.0 constant; must save (callee-saved FP)

    mv      s0, a0;  mv s1, a1;  mv s2, a2
    mv      s3, a3;  mv s4, a4;  mv s5, a5

    la      t0, fp_zero;  fld fs0, 0(t0)   # fs0 = 0.0

    li      s6, 0
.Lmul_i:
    bge     s6, s3, .Lmul_done
    mul     t0, s6, s4;  slli t0, t0, 3;  add s9, s1, t0   # s9=&A[i][0]

    li      s7, 0
.Lmul_j:
    bge     s7, s5, .Lmul_j_done
    fmv.d   ft0, fs0            # acc = 0.0
    li      s8, 0
    addi    t6, s4, -3          # unroll limit

.Lmul_unroll:
    bge     s8, t6, .Lmul_tail
    # l+0
    slli    t0, s8, 3;  add t1, s9, t0;  fld ft1, 0(t1)
    mul     t2, s8, s5;  add t2, t2, s7;  slli t2, t2, 3;  add t2, s2, t2;  fld ft2, 0(t2)
    fmadd.d ft0, ft1, ft2, ft0
    # l+1
    addi    t3, s8, 1
    slli    t0, t3, 3;  add t1, s9, t0;  fld ft3, 0(t1)
    mul     t2, t3, s5;  add t2, t2, s7;  slli t2, t2, 3;  add t2, s2, t2;  fld ft4, 0(t2)
    fmadd.d ft0, ft3, ft4, ft0
    # l+2
    addi    t3, s8, 2
    slli    t0, t3, 3;  add t1, s9, t0;  fld ft5, 0(t1)
    mul     t2, t3, s5;  add t2, t2, s7;  slli t2, t2, 3;  add t2, s2, t2;  fld ft6, 0(t2)
    fmadd.d ft0, ft5, ft6, ft0
    # l+3
    addi    t3, s8, 3
    slli    t0, t3, 3;  add t1, s9, t0;  fld ft7, 0(t1)
    mul     t2, t3, s5;  add t2, t2, s7;  slli t2, t2, 3;  add t2, s2, t2;  fld ft8, 0(t2)
    fmadd.d ft0, ft7, ft8, ft0
    addi    s8, s8, 4
    j       .Lmul_unroll

.Lmul_tail:
    bge     s8, s4, .Lmul_store
    slli    t0, s8, 3;  add t1, s9, t0;  fld ft1, 0(t1)
    mul     t2, s8, s5;  add t2, t2, s7;  slli t2, t2, 3;  add t2, s2, t2;  fld ft2, 0(t2)
    fmadd.d ft0, ft1, ft2, ft0
    addi    s8, s8, 1;  j .Lmul_tail

.Lmul_store:
    mul     t0, s6, s5;  add t0, t0, s7;  slli t0, t0, 3;  add t0, s0, t0
    fsd     ft0, 0(t0)
    addi    s7, s7, 1;  j .Lmul_j

.Lmul_j_done:
    addi    s6, s6, 1;  j .Lmul_i

.Lmul_done:
    ld      ra,  0(sp);  ld s0,  8(sp);  ld s1, 16(sp)
    ld      s2, 24(sp);  ld s3, 32(sp);  ld s4, 40(sp)
    ld      s5, 48(sp);  ld s6, 56(sp);  ld s7, 64(sp)
    ld      s8, 72(sp);  ld s9, 80(sp);  fld fs0, 88(sp)
    addi    sp, sp, 96
    ret
    .size mat_mul, .-mat_mul


# =============================================================================
#  mat_add  —  C = A + B  (m×n element-wise), unrolled x4
#
#  void mat_add(double *C, const double *A, const double *B, int m, int n)
#  a0=C, a1=A, a2=B, a3=m, a4=n
# =============================================================================
    .globl mat_add
    .type  mat_add, @function
mat_add:
    mul     t0, a3, a4;  slli t0, t0, 3
    add     t3, a0, t0          # end ptr
    addi    t2, t3, -32         # unroll boundary: need room for 4 doubles (32 B)
.Ladd_u:
    bgt     a0, t2, .Ladd_t
    fld ft0, 0(a1);fld ft1, 0(a2);fadd.d ft0,ft0,ft1;fsd ft0, 0(a0)
    fld ft2, 8(a1);fld ft3, 8(a2);fadd.d ft2,ft2,ft3;fsd ft2, 8(a0)
    fld ft4,16(a1);fld ft5,16(a2);fadd.d ft4,ft4,ft5;fsd ft4,16(a0)
    fld ft6,24(a1);fld ft7,24(a2);fadd.d ft6,ft6,ft7;fsd ft6,24(a0)
    addi a0,a0,32; addi a1,a1,32; addi a2,a2,32
    j .Ladd_u
.Ladd_t:
    bge a0, t3, .Ladd_done
    fld ft0,0(a1);fld ft1,0(a2);fadd.d ft0,ft0,ft1;fsd ft0,0(a0)
    addi a0,a0,8; addi a1,a1,8; addi a2,a2,8; j .Ladd_t
.Ladd_done:
    ret
    .size mat_add, .-mat_add


# =============================================================================
#  mat_sub  —  C = A - B  (m×n element-wise), unrolled x4
#
#  void mat_sub(double *C, const double *A, const double *B, int m, int n)
#  a0=C, a1=A, a2=B, a3=m, a4=n
# =============================================================================
    .globl mat_sub
    .type  mat_sub, @function
mat_sub:
    mul     t0, a3, a4;  slli t0, t0, 3
    add     t3, a0, t0
    addi    t2, t3, -32         # unroll boundary: need room for 4 doubles (32 B)
.Lsub_u:
    bgt     a0, t2, .Lsub_t
    fld ft0, 0(a1);fld ft1, 0(a2);fsub.d ft0,ft0,ft1;fsd ft0, 0(a0)
    fld ft2, 8(a1);fld ft3, 8(a2);fsub.d ft2,ft2,ft3;fsd ft2, 8(a0)
    fld ft4,16(a1);fld ft5,16(a2);fsub.d ft4,ft4,ft5;fsd ft4,16(a0)
    fld ft6,24(a1);fld ft7,24(a2);fsub.d ft6,ft6,ft7;fsd ft6,24(a0)
    addi a0,a0,32; addi a1,a1,32; addi a2,a2,32; j .Lsub_u
.Lsub_t:
    bge a0, t3, .Lsub_done
    fld ft0,0(a1);fld ft1,0(a2);fsub.d ft0,ft0,ft1;fsd ft0,0(a0)
    addi a0,a0,8; addi a1,a1,8; addi a2,a2,8; j .Lsub_t
.Lsub_done:
    ret
    .size mat_sub, .-mat_sub


# =============================================================================
#  mat_transpose  —  B = A^T   (A is m×n, B is n×m)
#
#  void mat_transpose(double *B, const double *A, int m, int n)
#  a0=B, a1=A, a2=m, a3=n
# =============================================================================
    .globl mat_transpose
    .type  mat_transpose, @function
mat_transpose:
    li      t0, 0
.Ltrp_i:
    bge     t0, a2, .Ltrp_done
    li      t1, 0
.Ltrp_j:
    bge     t1, a3, .Ltrp_jd
    mul     t2, t0, a3;  add t2, t2, t1;  slli t2, t2, 3;  add t2, a1, t2
    fld     ft0, 0(t2)
    mul     t3, t1, a2;  add t3, t3, t0;  slli t3, t3, 3;  add t3, a0, t3
    fsd     ft0, 0(t3)
    addi    t1, t1, 1;  j .Ltrp_j
.Ltrp_jd:
    addi    t0, t0, 1;  j .Ltrp_i
.Ltrp_done:
    ret
    .size mat_transpose, .-mat_transpose


# =============================================================================
#  mat_scale_add  —  C = A + alpha*B  (m×n), unrolled x4 with fmadd.d
#
#  void mat_scale_add(double *C, const double *A, const double *B,
#                     double alpha, int m, int n)
#  a0=C, a1=A, a2=B, fa0=alpha, a3=m, a4=n
# =============================================================================
    .globl mat_scale_add
    .type  mat_scale_add, @function
mat_scale_add:
    mul     t0, a3, a4;  slli t0, t0, 3
    add     t3, a0, t0
    addi    t2, t3, -32         # unroll boundary: need room for 4 doubles (32 B)
.Lsa_u:
    bgt     a0, t2, .Lsa_t
    fld ft0, 0(a1);fld ft1, 0(a2);fmadd.d ft0,fa0,ft1,ft0;fsd ft0, 0(a0)
    fld ft2, 8(a1);fld ft3, 8(a2);fmadd.d ft2,fa0,ft3,ft2;fsd ft2, 8(a0)
    fld ft4,16(a1);fld ft5,16(a2);fmadd.d ft4,fa0,ft5,ft4;fsd ft4,16(a0)
    fld ft6,24(a1);fld ft7,24(a2);fmadd.d ft6,fa0,ft7,ft6;fsd ft6,24(a0)
    addi a0,a0,32; addi a1,a1,32; addi a2,a2,32; j .Lsa_u
.Lsa_t:
    bge a0, t3, .Lsa_done
    fld ft0,0(a1);fld ft1,0(a2);fmadd.d ft0,fa0,ft1,ft0;fsd ft0,0(a0)
    addi a0,a0,8; addi a1,a1,8; addi a2,a2,8; j .Lsa_t
.Lsa_done:
    ret
    .size mat_scale_add, .-mat_scale_add


# =============================================================================
#  mat_vec_mul  —  y = A * x   (m×n matrix × n-vector -> m-vector)
#
#  void mat_vec_mul(double *y, const double *A, const double *x, int m, int n)
#  a0=y, a1=A, a2=x, a3=m, a4=n    inner loop unrolled x4 with fmadd.d
# =============================================================================
    .globl mat_vec_mul
    .type  mat_vec_mul, @function
mat_vec_mul:
    addi    sp, sp, -56
    sd      ra, 0(sp); sd s0, 8(sp); sd s1,16(sp)
    sd      s2,24(sp); sd s3,32(sp); sd s4,40(sp)
    fsd     fs0, 48(sp)          # fs0 used for 0.0 — must save

    mv s0,a0; mv s1,a1; mv s2,a2; mv s3,a3; mv s4,a4

    la      t0, fp_zero;  fld fs0, 0(t0)

    li      t0, 0
.Lmv_i:
    bge     t0, s3, .Lmv_done
    fmv.d   ft0, fs0
    mul     t1, t0, s4;  slli t1, t1, 3;  add t1, s1, t1   # &A[i][0]
    li      t2, 0
    addi    t3, s4, -3
.Lmv_u:
    bge     t2, t3, .Lmv_t
    slli t4,t2,3; add t5,t1,t4;fld ft1,0(t5); add t5,s2,t4;fld ft2,0(t5); fmadd.d ft0,ft1,ft2,ft0
    addi t6,t2,1; slli t4,t6,3; add t5,t1,t4;fld ft3,0(t5); add t5,s2,t4;fld ft4,0(t5); fmadd.d ft0,ft3,ft4,ft0
    addi t6,t2,2; slli t4,t6,3; add t5,t1,t4;fld ft5,0(t5); add t5,s2,t4;fld ft6,0(t5); fmadd.d ft0,ft5,ft6,ft0
    addi t6,t2,3; slli t4,t6,3; add t5,t1,t4;fld ft7,0(t5); add t5,s2,t4;fld ft8,0(t5); fmadd.d ft0,ft7,ft8,ft0
    addi t2,t2,4; j .Lmv_u
.Lmv_t:
    bge t2, s4, .Lmv_s
    slli t4,t2,3; add t5,t1,t4;fld ft1,0(t5); add t5,s2,t4;fld ft2,0(t5); fmadd.d ft0,ft1,ft2,ft0
    addi t2,t2,1; j .Lmv_t
.Lmv_s:
    slli t4,t0,3; add t5,s0,t4; fsd ft0,0(t5)
    addi t0,t0,1; j .Lmv_i
.Lmv_done:
    ld ra,0(sp); ld s0,8(sp); ld s1,16(sp)
    ld s2,24(sp); ld s3,32(sp); ld s4,40(sp)
    fld fs0, 48(sp)
    addi sp,sp,56; ret
    .size mat_vec_mul, .-mat_vec_mul


# =============================================================================
#  mat_inverse_nxn  —  LU decomposition inverse (partial pivoting / Doolittle)
#
#  Matches numpy.linalg.inv to |err| <= 1e-9 for well-conditioned matrices.
#
#  int mat_inverse_nxn(double *Ainv, const double *A, int n, int *piv)
#  a0=Ainv  a1=A  a2=n  a3=piv
#  Returns a0=1 (ok) or a0=0 (singular).
#
#  Stack frame: 112 bytes (fixed) + n*n*8 (LU) + n*8 (y) dynamic.
#  s0=Ainv s1=LU s2=n s3=piv s4=k s5=i s6=j s7=col s8=yptr
#  s9..s11 scratch  fs0=pivot value
# =============================================================================
    .globl mat_inverse_nxn
    .type  mat_inverse_nxn, @function
mat_inverse_nxn:
    addi    sp, sp, -112
    sd ra,0(sp); sd s0,8(sp); sd s1,16(sp); sd s2,24(sp)
    sd s3,32(sp); sd s4,40(sp); sd s5,48(sp); sd s6,56(sp)
    sd s7,64(sp); sd s8,72(sp); sd s9,80(sp); sd s10,88(sp)
    sd s11,96(sp); fsd fs0,104(sp)

    mv s0,a0; mv s2,a2; mv s3,a3; mv s11,a1   # s11=A (preserve before dynamic alloc)

    # Dynamic alloc: LU = n*n doubles
    mul t0,s2,s2; slli t0,t0,3; sub sp,sp,t0; mv s1,sp

    # Step 1: copy A -> LU
    mul t1,s2,s2; slli t1,t1,3
    mv t2,s1; mv t3,s11; add t4,t3,t1
.Linv_cp:
    bge t3,t4,.Linv_cp_d
    fld ft0,0(t3); fsd ft0,0(t2); addi t3,t3,8; addi t2,t2,8; j .Linv_cp
.Linv_cp_d:

    # Step 2: piv[i] = i
    li s4,0
.Linv_pi:
    bge s4,s2,.Linv_lu
    slli t0,s4,2; add t0,s3,t0; sw s4,0(t0)
    addi s4,s4,1; j .Linv_pi

    # Step 3: Doolittle LU with partial pivoting
.Linv_lu:
    li s4,0
.Linv_luk:
    bge s4,s2,.Linv_sl

    # Find max |LU[i][k]| for i=k..n-1
    la t0,fp_zero; fld fs0,0(t0)
    mv s5,s4; mv s6,s4
.Linv_pf:
    bge s6,s2,.Linv_pd
    mul t0,s6,s2; add t0,t0,s4; slli t0,t0,3; add t0,s1,t0
    fld ft1,0(t0); fabs.d ft1,ft1
    flt.d t1,fs0,ft1; beqz t1,.Linv_pn
    fmv.d fs0,ft1; mv s5,s6
.Linv_pn:
    addi s6,s6,1; j .Linv_pf
.Linv_pd:

    # Singular check
    la t0,fp_zero; fld ft2,0(t0)
    feq.d t0,fs0,ft2; bnez t0,.Linv_sing

    # Swap rows s4 <-> s5
    beq s4,s5,.Linv_ns
    slli t0,s4,2; add t0,s3,t0; lw t1,0(t0)
    slli t2,s5,2; add t2,s3,t2; lw t3,0(t2)
    sw t3,0(t0); sw t1,0(t2)
    mul t0,s4,s2; slli t0,t0,3; add t4,s1,t0
    mul t0,s5,s2; slli t0,t0,3; add t5,s1,t0
    li s6,0
.Linv_rs:
    bge s6,s2,.Linv_ns
    slli t0,s6,3
    add t1,t4,t0; fld ft0,0(t1)
    add t2,t5,t0; fld ft1,0(t2)
    fsd ft1,0(t1); fsd ft0,0(t2)
    addi s6,s6,1; j .Linv_rs
.Linv_ns:

    # pivot = LU[k][k]
    mul t0,s4,s2; add t0,t0,s4; slli t0,t0,3; add t0,s1,t0; fld fs0,0(t0)

    # Eliminate: for i=k+1..n-1
    addi s5,s4,1
.Linv_ei:
    bge s5,s2,.Linv_ed
    mul t0,s5,s2; add t0,t0,s4; slli t0,t0,3; add t0,s1,t0
    fld ft1,0(t0); fdiv.d ft1,ft1,fs0; fsd ft1,0(t0)   # m = LU[i][k]/pivot
    addi s6,s4,1
.Linv_ej:
    bge s6,s2,.Linv_ejd
    mul t1,s4,s2; add t1,t1,s6; slli t1,t1,3; add t1,s1,t1; fld ft2,0(t1)
    mul t2,s5,s2; add t2,t2,s6; slli t2,t2,3; add t2,s1,t2; fld ft3,0(t2)
    fnmsub.d ft3,ft1,ft2,ft3; fsd ft3,0(t2)   # LU[i][j] -= m*LU[k][j]
    addi s6,s6,1; j .Linv_ej
.Linv_ejd:
    addi s5,s5,1; j .Linv_ei
.Linv_ed:
    addi s4,s4,1; j .Linv_luk

    # Steps 4-5: solve for each column
.Linv_sl:
    slli t0,s2,3; sub sp,sp,t0; mv s8,sp   # y vector

    li s4,0   # col
.Linv_cl:
    bge s4,s2,.Linv_cd

    # Build permuted rhs
    la t0,fp_zero; fld ft0,0(t0)
    la t0,fp_one;  fld ft1,0(t0)
    li s5,0
.Linv_rb:
    bge s5,s2,.Linv_fwd
    slli t0,s5,2; add t0,s3,t0; lw t1,0(t0)
    slli t2,s5,3; add t2,s8,t2
    bne t1,s4,.Linv_r0
    fsd ft1,0(t2); j .Linv_rn
.Linv_r0:
    fsd ft0,0(t2)
.Linv_rn:
    addi s5,s5,1; j .Linv_rb

    # Forward substitution
.Linv_fwd:
    li s5,1
.Linv_fi:
    bge s5,s2,.Linv_bck
    slli t0,s5,3; add t0,s8,t0; fld ft0,0(t0)
    li s6,0
.Linv_fj:
    bge s6,s5,.Linv_fjd
    mul t1,s5,s2; add t1,t1,s6; slli t1,t1,3; add t1,s1,t1; fld ft1,0(t1)
    slli t2,s6,3; add t2,s8,t2; fld ft2,0(t2)
    fnmsub.d ft0,ft1,ft2,ft0
    addi s6,s6,1; j .Linv_fj
.Linv_fjd:
    slli t0,s5,3; add t0,s8,t0; fsd ft0,0(t0)
    addi s5,s5,1; j .Linv_fi

    # Backward substitution
.Linv_bck:
    addi s5,s2,-1
.Linv_bi:
    bltz s5,.Linv_sc
    slli t0,s5,3; add t0,s8,t0; fld ft0,0(t0)
    addi s6,s5,1
.Linv_bj:
    bge s6,s2,.Linv_bjd
    mul t1,s5,s2; add t1,t1,s6; slli t1,t1,3; add t1,s1,t1; fld ft1,0(t1)
    slli t2,s6,3; add t2,s8,t2; fld ft2,0(t2)
    fnmsub.d ft0,ft1,ft2,ft0
    addi s6,s6,1; j .Linv_bj
.Linv_bjd:
    mul t1,s5,s2; add t1,t1,s5; slli t1,t1,3; add t1,s1,t1; fld ft1,0(t1)
    fdiv.d ft0,ft0,ft1
    slli t0,s5,3; add t0,s8,t0; fsd ft0,0(t0)
    addi s5,s5,-1; j .Linv_bi

    # Store column into Ainv
.Linv_sc:
    li s5,0
.Linv_sti:
    bge s5,s2,.Linv_cn
    slli t0,s5,3; add t0,s8,t0; fld ft0,0(t0)
    mul t1,s5,s2; add t1,t1,s4; slli t1,t1,3; add t1,s0,t1; fsd ft0,0(t1)
    addi s5,s5,1; j .Linv_sti
.Linv_cn:
    addi s4,s4,1; j .Linv_cl

.Linv_cd:
    slli t0,s2,3; add sp,sp,t0               # free y
    mul t0,s2,s2; slli t0,t0,3; add sp,sp,t0 # free LU
    li a0,1; j .Linv_ret

.Linv_sing:
    mul t0,s2,s2; slli t0,t0,3; add sp,sp,t0 # free LU only
    li a0,0

.Linv_ret:
    ld ra,0(sp); ld s0,8(sp); ld s1,16(sp); ld s2,24(sp)
    ld s3,32(sp); ld s4,40(sp); ld s5,48(sp); ld s6,56(sp)
    ld s7,64(sp); ld s8,72(sp); ld s9,80(sp); ld s10,88(sp)
    ld s11,96(sp); fld fs0,104(sp)
    addi sp,sp,112; ret
    .size mat_inverse_nxn, .-mat_inverse_nxn


# =============================================================================
#  mat_joseph_update  —  P = (I-KH)*P*(I-KH)^T + K*R*K^T
#
#  void mat_joseph_update(double *Pout,
#                         const double *P,   // n×n
#                         const double *K,   // n×m
#                         const double *H,   // m×n
#                         const double *R,   // m×m
#                         int n, int m,
#                         double *scratch)
#  a0=Pout a1=P a2=K a3=H a4=R a5=n a6=m a7=scratch
#
#  scratch layout:
#    [0       .. n*n-1 ]  IKH  (n×n)
#    [n*n     .. 2*n*n ]  tmp1 (n×n)
#    [2*n*n   .. 3*n*n ]  tmp2 (n×n)
#    [3*n*n   .. 4*n*n-1] K^T  (m×n)
#  KT buffer placed in scratch[3*n*n .. 4*n*n-1]  (n*m doubles = n*n worst-case).
#  Caller MUST provide >= 4*n*n doubles of scratch.
# =============================================================================
    .globl mat_joseph_update
    .type  mat_joseph_update, @function
mat_joseph_update:
    addi    sp, sp, -104
    sd ra,0(sp); sd s0,8(sp); sd s1,16(sp); sd s2,24(sp)
    sd s3,32(sp); sd s4,40(sp); sd s5,48(sp); sd s6,56(sp)
    sd s7,64(sp); sd s8,72(sp); sd s9,80(sp); sd s10,88(sp)
    sd s11,96(sp)

    mv s0,a0; mv s1,a1; mv s2,a2; mv s3,a3
    mv s4,a4; mv s5,a5; mv s6,a6; mv s7,a7

    mul t0,s5,s5; slli t0,t0,3            # nn8 = n*n*8
    mv  s8,s7                             # IKH  = scratch[0      .. n*n-1   ]
    add s9,s7,t0                          # tmp1 = scratch[n*n    .. 2*n*n-1 ]
    add s10,s9,t0                         # tmp2 = scratch[2*n*n  .. 3*n*n-1 ]
    add s11,s10,t0                        # K^T  = scratch[3*n*n  .. 4*n*n-1 ]
    # K^T lives in scratch (caller must provide >= 4*n*n doubles).
    # No dynamic stack alloc — avoids corrupting C-stack scratch buffers.

    # 1. IKH = I_n - K@H
    mv a0,s8; mv a1,s5; call mat_eye
    mv a0,s9; mv a1,s2; mv a2,s3; mv a3,s5; mv a4,s6; mv a5,s5
    call mat_mul                           # tmp1 = K@H  (n×n)
    mv a0,s8; mv a1,s8; mv a2,s9; mv a3,s5; mv a4,s5
    call mat_sub                           # IKH  = I - K@H

    # 2. tmp1 = IKH @ P
    mv a0,s9; mv a1,s8; mv a2,s1; mv a3,s5; mv a4,s5; mv a5,s5
    call mat_mul

    # 3. Pout = (IKH@P) @ IKH^T
    mv a0,s10; mv a1,s8; mv a2,s5; mv a3,s5
    call mat_transpose                     # tmp2 = IKH^T
    mv a0,s0; mv a1,s9; mv a2,s10; mv a3,s5; mv a4,s5; mv a5,s5
    call mat_mul                           # Pout = tmp1 @ IKH^T

    # 4. tmp1 = K @ R  (n×m)
    mv a0,s9; mv a1,s2; mv a2,s4; mv a3,s5; mv a4,s6; mv a5,s6
    call mat_mul

    # 5. K^T  (m×n) stored in scratch[3*n*n] — safe, no stack alloc needed
    mv a0,s11; mv a1,s2; mv a2,s5; mv a3,s6
    call mat_transpose

    # 6. tmp2 = (K@R) @ K^T  (n×n)
    mv a0,s10; mv a1,s9; mv a2,s11; mv a3,s5; mv a4,s6; mv a5,s5
    call mat_mul

    # 7. Pout += tmp2
    mv a0,s0; mv a1,s0; mv a2,s10; mv a3,s5; mv a4,s5
    call mat_add

    ld ra,0(sp); ld s0,8(sp); ld s1,16(sp); ld s2,24(sp)
    ld s3,32(sp); ld s4,40(sp); ld s5,48(sp); ld s6,56(sp)
    ld s7,64(sp); ld s8,72(sp); ld s9,80(sp); ld s10,88(sp)
    ld s11,96(sp)
    addi sp,sp,104; ret
    .size mat_joseph_update, .-mat_joseph_update


# =============================================================================
#  fast_atan2  —  Scheinerman-Lyons polynomial atan2 (matches kalman-updated.py)
#
#  double fast_atan2(double y, double x)
#  fa0=y, fa1=x  ->  fa0=result
#
#  ft0=ax  ft1=ay  ft2=z  ft3=angle
#  ft4=pi4 ft5=pi2 ft6=pi  ft7=c0  ft8=c1  ft9=0.0  ft10=1.0  ft11=tmp
# =============================================================================
    .globl fast_atan2
    .type  fast_atan2, @function
fast_atan2:
    la t0,fp_pi4;  fld ft4,0(t0)
    la t0,fp_pi2;  fld ft5,0(t0)
    la t0,fp_pi;   fld ft6,0(t0)
    la t0,fp_c0;   fld ft7,0(t0)
    la t0,fp_c1;   fld ft8,0(t0)
    la t0,fp_zero; fld ft9,0(t0)
    la t0,fp_one;  fld ft10,0(t0)

    # if x==0 && y==0 -> return 0
    feq.d t0,fa0,ft9; feq.d t1,fa1,ft9; and t0,t0,t1
    bnez t0,.Lat2_zero

    fabs.d ft0,fa1    # ax=|x|
    fabs.d ft1,fa0    # ay=|y|

    flt.d t0,ft0,ft1  # ax < ay?
    bnez t0,.Lat2_ay

    fdiv.d ft2,ft1,ft0; j .Lat2_poly   # z = ay/ax
.Lat2_ay:
    fdiv.d ft2,ft0,ft1                 # z = ax/ay

.Lat2_poly:
    # angle = (pi/4)*z - z*(z-1)*(c0 + c1*z)
    fmadd.d  ft11,ft8,ft2,ft7          # c0 + c1*z
    fsub.d   ft3, ft2,ft10             # z-1
    fmul.d   ft3, ft2,ft3              # z*(z-1)
    fmul.d   ft3, ft3,ft11             # z*(z-1)*(c0+c1*z)
    fmul.d   ft11,ft4,ft2              # (pi/4)*z
    fsub.d   ft3, ft11,ft3             # angle

    # if ax < ay: angle = pi/2 - angle
    fabs.d ft0,fa1; fabs.d ft1,fa0     # recompute ax,ay for branch
    flt.d t0,ft0,ft1
    beqz t0,.Lat2_quad
    fsub.d ft3,ft5,ft3

.Lat2_quad:
    la t0,fp_zero; fld ft9,0(t0)
    flt.d t0,fa1,ft9    # x < 0?
    beqz t0,.Lat2_xpos
    flt.d t1,fa0,ft9    # y < 0?
    bnez t1,.Lat2_xn_yn
    fsub.d ft3,ft6,ft3; j .Lat2_done   # x<0,y>=0: pi-angle
.Lat2_xn_yn:
    fsub.d ft3,ft3,ft6; j .Lat2_done   # x<0,y<0: angle-pi
.Lat2_xpos:
    flt.d t1,fa0,ft9    # y < 0?
    beqz t1,.Lat2_done
    fneg.d ft3,ft3                     # x>=0,y<0: -angle
.Lat2_done:
    fmv.d fa0,ft3; ret
.Lat2_zero:
    fmv.d fa0,ft9; ret
    .size fast_atan2, .-fast_atan2


# =============================================================================
#  wrap_angle  —  wrap to (-pi, pi]
#
#  a = a - 2*pi * floor((a+pi)/(2*pi))   [O(1), matches Python loop]
#
#  double wrap_angle(double a)
#  fa0=a -> fa0=result
# =============================================================================
    .globl wrap_angle
    .type  wrap_angle, @function
wrap_angle:
    la t0,fp_pi;  fld ft0,0(t0)
    la t0,fp_2pi; fld ft1,0(t0)
    fadd.d ft2,fa0,ft0          # a + pi
    fdiv.d ft2,ft2,ft1          # (a+pi)/(2*pi)
    fcvt.w.d t0,ft2,rdn         # floor (round-down)
    fcvt.d.w ft3,t0
    fnmsub.d fa0,ft3,ft1,fa0    # fa0 = fa0 - ft3*ft1
    ret
    .size wrap_angle, .-wrap_angle


# =============================================================================
#  END OF matrix_asm.s
# =============================================================================

