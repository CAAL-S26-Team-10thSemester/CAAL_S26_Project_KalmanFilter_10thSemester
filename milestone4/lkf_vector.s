# =============================================================================
#  lkf_vector.s  —  RISC-V RVV Vectorised Assembly: Linear Kalman Filter
#  Kalman Filter Milestone-4
# =============================================================================
#
#  Implements the LKF class from kalman-updated.py exactly:
#
#    lkf_vec_sizeof            ()                    — return heap size needed
#    lkf_vec_init              (lkf, dt)             — __init__
#    lkf_vec_set_initial_state (lkf, joint, pos3)    — set_initial_state
#    lkf_vec_predict           (lkf)                 — predict
#    lkf_vec_update            (lkf, z_flat)         — update
#    lkf_vec_get_positions     (lkf, pos_out)        — get_positions
#    lkf_vec_get_full_state    (lkf, out)            — get_full_state
#
#  Matrix helpers (used at init only):
#    state_vec_init_F  (F, dt)
#    state_vec_init_Q  (Q)
#    meas_vec_init_H   (H)
#    meas_vec_init_R   (R)
#
#  All matrix algebra delegates to matrix_vec.s and ekf_utils_vector.s.
#
# =============================================================================
#  LKF STRUCT LAYOUT
#
#  Offset      Bytes   Field
#  ----------  ------  -------
#           0       8  dt          (double)
#           8    2208  x[276]      state vector
#        2216  609408  P[276×276]  covariance
#      611624  609408  F[276×276]  state transition
#     1221032  609408  Q[276×276]  process noise
#     1830440  152208  H[69×276]   measurement matrix
#     1982792   38088  R[69×69]    measurement noise
#     2020880  152352  PHt[276×69] (Note: kept original scalar layout offsets)
#     2173232   38088  S[69×69]
#     2211320   38088  Sinv[69×69]
#     2249408     552  y[69]
#     2249960     276  piv[69]
#     2250240          jscratch base (adjusted to align to 8, same as original: 2250240)
#
#  jscratch sub-region offsets (bytes from jscratch base):
#     JS_HT      [0        .. 152351 ]  H^T      276x69  = 152352 bytes
#     JS_K       [152352   .. 304703 ]  K        276x69  = 152352 bytes
#     JS_xnew    [304704   .. 306911 ]  x_new    276     =   2208 bytes
#     JS_Ky      [306912   .. 309119 ]  Ky       276     =   2208 bytes
#     JS_Ftmp    [309120   .. 918527 ]  Ftmp     276x276 = 609408 bytes
#     JS_FT      [918528   .. 1527935]  FT       276x276 = 609408 bytes
#     JS_joseph  [1527936  .. 3965567]  joseph           = 2437632 bytes
#
#  TOTAL: 2250240 + 3965568 = 6215808 bytes
#
# =============================================================================
#  ABI: RISC-V LP64D + V (rv64gcv, lp64d)
# =============================================================================

    # --- Vectorised matrix library (matrix_vec.s) ---
    .extern mat_mul_vec
    .extern mat_add_vec
    .extern mat_sub_vec
    .extern mat_vec_mul_vec
    .extern mat_transpose_vec
    # --- Vectorised EKF helpers (ekf_utils_vector.s) ---
    .extern mat_joseph_update_vec
    # --- Scalar helpers (matrix_asm.s / shared) ---
    .extern mat_eye
    .extern mat_inverse_nxn

# =============================================================================
#  .rodata — constants
# =============================================================================
    .section .rodata
    .align 3

lkf_NUM_JOINTS:  .quad  23
lkf_STATE_DIM:   .quad  12
lkf_MEAS_DIM:    .quad   3
lkf_N:           .quad  276
lkf_M:           .quad   69

lkf_EST_R_PX:    .double  0.29472279
lkf_EST_R_PY:    .double  0.09632091
lkf_EST_R_PZ:    .double  0.00204269

lkf_noise_0:     .double  1.0e-6
lkf_noise_1:     .double  1.0e-5
lkf_noise_2:     .double  1.0e-4
lkf_noise_3:     .double  1.0e-4

lkf_fp_zero:     .double  0.0
lkf_fp_one:      .double  1.0
lkf_fp_half:     .double  0.5
lkf_fp_sixth:    .double  0.16666666666666666667

# Struct offsets — identical values to working scalar version
lkf_OFF_dt:       .quad       0
lkf_OFF_x:        .quad       8
lkf_OFF_P:        .quad    2216
lkf_OFF_F:        .quad  611624
lkf_OFF_Q:        .quad 1221032
lkf_OFF_H:        .quad 1830440
lkf_OFF_R:        .quad 1982792
lkf_OFF_PHt:      .quad 2020880
lkf_OFF_S:        .quad 2173232
lkf_OFF_Sinv:     .quad 2211320
lkf_OFF_y:        .quad 2249408
lkf_OFF_piv:      .quad 2249960
lkf_OFF_jscratch: .quad 2250240

# jscratch sub-region offsets (bytes from jscratch base)
lkf_JS_HT:        .quad       0   # H^T      276x69  = 152352 bytes
lkf_JS_K:         .quad  152352   # K        276x69  = 152352 bytes
lkf_JS_xnew:      .quad  304704   # x_new    276     =   2208 bytes
lkf_JS_Ky:        .quad  306912   # Ky       276     =   2208 bytes
lkf_JS_Ftmp:      .quad  309120   # Ftmp     276x276 = 609408 bytes
lkf_JS_FT:        .quad  918528   # FT       276x276 = 609408 bytes
lkf_JS_joseph:    .quad 1527936   # joseph           = 2437632 bytes

lkf_TOTAL_BYTES:  .quad 6215808

    .section .text

# =============================================================================
#  lkf_vec_sizeof  —  return total allocation size
#  size_t lkf_vec_sizeof(void)
#  Leaf.
# =============================================================================
    .globl lkf_vec_sizeof
    .type  lkf_vec_sizeof, @function
lkf_vec_sizeof:
    la      t0, lkf_TOTAL_BYTES
    ld      a0, 0(t0)
    ret
    .size lkf_vec_sizeof, .-lkf_vec_sizeof

# =============================================================================
#  state_vec_init_F  —  build F (276×276 block-diagonal state transition)
#  void state_vec_init_F(double *F, double dt)
#  a0=F,  fa0=dt
# =============================================================================
    .globl state_vec_init_F
    .type  state_vec_init_F, @function
state_vec_init_F:
    addi    sp, sp, -72
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); sd s3, 32(sp); sd s4, 40(sp)
    fsd     fs0, 48(sp); fsd fs1, 56(sp); fsd fs2, 64(sp)

    mv      s0, a0
    fmv.d   fs0, fa0            # dt

    # F = eye(276)
    li      a1, 276
    call    mat_eye

    # dt2 = 0.5 * dt²
    la      t0, lkf_fp_half;   fld ft0, 0(t0)
    fmul.d  fs1, fs0, fs0       # dt*dt
    fmul.d  fs1, ft0, fs1       # 0.5 * dt²

    # dt3 = (1/6) * dt³
    la      t0, lkf_fp_sixth;  fld ft0, 0(t0)
    fmul.d  fs2, fs0, fs0       # dt²
    fmul.d  fs2, fs2, fs0       # dt³
    fmul.d  fs2, ft0, fs2       # dt³/6

    li      s4, 276             # column stride
    li      s1, 0               # j = 0

.LFvec_j:
    li      t0, 23; bge s1, t0, .LFvec_done
    li      t1, 12; mul t2, s1, t1    # base = j*12
    li      s2, 0

.LFvec_axis:
    li      t0, 3; bge s2, t0, .LFvec_axis_done
    slli    t0, s2, 2; add s3, t2, t0  # r = base + axis*4

    # F[r, r+1] = dt
    mul     t0, s3, s4; add t0, t0, s3; addi t0, t0, 1
    slli    t0, t0, 3; add t0, s0, t0; fsd fs0, 0(t0)
    # F[r, r+2] = dt2
    mul     t0, s3, s4; add t0, t0, s3; addi t0, t0, 2
    slli    t0, t0, 3; add t0, s0, t0; fsd fs1, 0(t0)
    # F[r, r+3] = dt3
    mul     t0, s3, s4; add t0, t0, s3; addi t0, t0, 3
    slli    t0, t0, 3; add t0, s0, t0; fsd fs2, 0(t0)
    # F[r+1, r+2] = dt
    addi    t1, s3, 1
    mul     t0, t1, s4; add t0, t0, s3; addi t0, t0, 2
    slli    t0, t0, 3; add t0, s0, t0; fsd fs0, 0(t0)
    # F[r+1, r+3] = dt2
    mul     t0, t1, s4; add t0, t0, s3; addi t0, t0, 3
    slli    t0, t0, 3; add t0, s0, t0; fsd fs1, 0(t0)
    # F[r+2, r+3] = dt
    addi    t1, s3, 2
    mul     t0, t1, s4; add t0, t0, s3; addi t0, t0, 3
    slli    t0, t0, 3; add t0, s0, t0; fsd fs0, 0(t0)

    addi    s2, s2, 1; j .LFvec_axis
.LFvec_axis_done:
    addi    s1, s1, 1; j .LFvec_j
.LFvec_done:
    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); ld s3, 32(sp); ld s4, 40(sp)
    fld     fs0, 48(sp); fld fs1, 56(sp); fld fs2, 64(sp)
    addi    sp, sp, 72; ret
    .size state_vec_init_F, .-state_vec_init_F

# =============================================================================
#  state_vec_init_Q  —  build Q (276×276 block-diagonal process noise)
#  void state_vec_init_Q(double *Q)
# =============================================================================
    .globl state_vec_init_Q
    .type  state_vec_init_Q, @function
state_vec_init_Q:
    addi    sp, sp, -32
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp); sd s2, 24(sp)

    mv      s0, a0

    # Zero Q: 276*276 = 76176 doubles — RVV vectorised
    li      t0, 76176
    fmv.d.x ft0, zero
    mv      t2, s0
.LQvec_zero_v:
    beqz    t0, .LQvec_zero_done
    vsetvli t1, t0, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t2)
    slli    t3, t1, 3
    add     t2, t2, t3
    sub     t0, t0, t1
    j       .LQvec_zero_v
.LQvec_zero_done:

    la      t0, lkf_noise_0; fld ft1, 0(t0)   # 1e-6
    la      t0, lkf_noise_1; fld ft2, 0(t0)   # 1e-5
    la      t0, lkf_noise_2; fld ft3, 0(t0)   # 1e-4

    li      s1, 0               # j = 0
.LQvec_j:
    li      t3, 23; bge s1, t3, .LQvec_done
    li      t4, 12; mul t4, s1, t4   # base = j*12
    li      s2, 0

.LQvec_i:
    li      t3, 12; bge s2, t3, .LQvec_i_done
    add     t5, t4, s2          # row = col = base+i
    li      t6, 276
    mul     t0, t5, t6; add t0, t0, t5
    slli    t0, t0, 3; add t0, s0, t0

    andi    t3, s2, 3
    beqz    t3, .LQvec_n0
    li      t6, 1; beq t3, t6, .LQvec_n1
    fsd     ft3, 0(t0); j .LQvec_next    # i%4 == 2 or 3
.LQvec_n0: fsd ft1, 0(t0); j .LQvec_next
.LQvec_n1: fsd ft2, 0(t0)
.LQvec_next:
    addi    s2, s2, 1; j .LQvec_i
.LQvec_i_done:
    addi    s1, s1, 1; j .LQvec_j
.LQvec_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp); ld s2, 24(sp)
    addi    sp, sp, 32; ret
    .size state_vec_init_Q, .-state_vec_init_Q


# =============================================================================
#  meas_vec_init_H  —  build H (69×276 block-diagonal measurement matrix)
#  void meas_vec_init_H(double *H)
# =============================================================================
    .globl meas_vec_init_H
    .type  meas_vec_init_H, @function
meas_vec_init_H:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)

    mv      s0, a0

    # Zero H: 69*276 = 19044 doubles — RVV vectorised
    li      t0, 19044
    fmv.d.x ft0, zero
    mv      t2, s0
.LHvec_zero_v:
    beqz    t0, .LHvec_zero_done
    vsetvli t1, t0, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t2)
    slli    t3, t1, 3
    add     t2, t2, t3
    sub     t0, t0, t1
    j       .LHvec_zero_v
.LHvec_zero_done:

    la      t0, lkf_fp_one; fld ft1, 0(t0)
    li      s1, 0
.LHvec_j:
    li      t0, 23; bge s1, t0, .LHvec_done
    li      t1, 3;  mul t2, s1, t1   # rb = j*3
    li      t1, 12; mul t3, s1, t1   # cb = j*12
    li      t4, 276

    # H[rb+0, cb+0]
    mul     t0, t2, t4; add t0, t0, t3
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)
    # H[rb+1, cb+4]
    addi    t5, t2, 1
    mul     t0, t5, t4; add t0, t0, t3; addi t0, t0, 4
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)
    # H[rb+2, cb+8]
    addi    t5, t2, 2
    mul     t0, t5, t4; add t0, t0, t3; addi t0, t0, 8
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)

    addi    s1, s1, 1; j .LHvec_j
.LHvec_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp)
    addi    sp, sp, 24; ret
    .size meas_vec_init_H, .-meas_vec_init_H


# =============================================================================
#  meas_vec_init_R  —  build R (69×69 block-diagonal measurement noise)
#  void meas_vec_init_R(double *R)
# =============================================================================
    .globl meas_vec_init_R
    .type  meas_vec_init_R, @function
meas_vec_init_R:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)

    mv      s0, a0

    # Zero R: 69*69 = 4761 doubles — RVV vectorised
    li      t0, 4761
    fmv.d.x ft0, zero
    mv      t2, s0
.LRvec_zero_v:
    beqz    t0, .LRvec_zero_done
    vsetvli t1, t0, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t2)
    slli    t3, t1, 3
    add     t2, t2, t3
    sub     t0, t0, t1
    j       .LRvec_zero_v
.LRvec_zero_done:

    la      t0, lkf_EST_R_PX; fld ft1, 0(t0)
    la      t0, lkf_EST_R_PY; fld ft2, 0(t0)
    la      t0, lkf_EST_R_PZ; fld ft3, 0(t0)

    li      s1, 0
.LRvec_j:
    li      t3, 23; bge s1, t3, .LRvec_done
    li      t4, 3; mul t4, s1, t4    # b = j*3
    li      t5, 69
    # R[b+0,b+0]
    mul     t0, t4, t5; add t0, t0, t4
    slli    t0, t0, 3; add t0, s0, t0; fsd ft1, 0(t0)
    # R[b+1,b+1]
    addi    t6, t4, 1
    mul     t0, t6, t5; add t0, t0, t6
    slli    t0, t0, 3; add t0, s0, t0; fsd ft2, 0(t0)
    # R[b+2,b+2]
    addi    t6, t4, 2
    mul     t0, t6, t5; add t0, t0, t6
    slli    t0, t0, 3; add t0, s0, t0; fsd ft3, 0(t0)

    addi    s1, s1, 1; j .LRvec_j
.LRvec_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp)
    addi    sp, sp, 24; ret
    .size meas_vec_init_R, .-meas_vec_init_R


# =============================================================================
#  lkf_vec_init  —  initialise all fields of an LKF struct
#  void lkf_vec_init(void *lkf, double dt)
# =============================================================================
    .globl lkf_vec_init
    .type  lkf_vec_init, @function
lkf_vec_init:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); fsd fs0, 16(sp)

    mv      s0, a0
    fmv.d   fs0, fa0

    # store dt
    fsd     fs0, 0(s0)

    # x = zeros(276) — RVV vectorised
    fmv.d.x ft0, zero
    addi    t1, s0, 8           # &x[0]
    li      t0, 276
.Linitvec_xz_v:
    beqz    t0, .Linitvec_xz_done
    vsetvli t2, t0, e64, m1, ta, ma
    vfmv.v.f v0, ft0
    vse64.v v0, (t1)
    slli    t3, t2, 3
    add     t1, t1, t3
    sub     t0, t0, t2
    j       .Linitvec_xz_v
.Linitvec_xz_done:

    # P = eye(276)
    la      t0, lkf_OFF_P; ld t0, 0(t0); add a0, s0, t0
    li      a1, 276; call mat_eye

    # F = state_vec_init_F(dt)
    la      t0, lkf_OFF_F; ld t0, 0(t0); add a0, s0, t0
    fmv.d   fa0, fs0; call state_vec_init_F

    # Q = state_vec_init_Q()
    la      t0, lkf_OFF_Q; ld t0, 0(t0); add a0, s0, t0
    call    state_vec_init_Q

    # H = meas_vec_init_H()
    la      t0, lkf_OFF_H; ld t0, 0(t0); add a0, s0, t0
    call    meas_vec_init_H

    # R = meas_vec_init_R()
    la      t0, lkf_OFF_R; ld t0, 0(t0); add a0, s0, t0
    call    meas_vec_init_R

    ld      ra, 0(sp); ld s0, 8(sp); fld fs0, 16(sp)
    addi    sp, sp, 24; ret
    .size lkf_vec_init, .-lkf_vec_init


# =============================================================================
#  lkf_vec_set_initial_state
#  void lkf_vec_set_initial_state(void *lkf, int joint_idx, const double *pos3)
# =============================================================================
    .globl lkf_vec_set_initial_state
    .type  lkf_vec_set_initial_state, @function
lkf_vec_set_initial_state:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 12; mul t1, a1, t1; slli t1, t1, 3; add t0, t0, t1
    la      t1, lkf_fp_zero; fld ft0, 0(t1)
    fld     ft1, 0(a2); fld ft2, 8(a2); fld ft3, 16(a2)
    fsd     ft1,  0(t0); fsd ft0,  8(t0); fsd ft0, 16(t0); fsd ft0, 24(t0)
    fsd     ft2, 32(t0); fsd ft0, 40(t0); fsd ft0, 48(t0); fsd ft0, 56(t0)
    fsd     ft3, 64(t0); fsd ft0, 72(t0); fsd ft0, 80(t0); fsd ft0, 88(t0)
    ret
    .size lkf_vec_set_initial_state, .-lkf_vec_set_initial_state


# =============================================================================
#  lkf_vec_predict  —  Kalman prediction step
#  void lkf_vec_predict(void *lkf)
# =============================================================================
    .globl lkf_vec_predict
    .type  lkf_vec_predict, @function
lkf_vec_predict:
    addi    sp, sp, -72
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); sd s3, 32(sp); sd s4, 40(sp)
    sd      s5, 48(sp); sd s6, 56(sp); sd s7, 64(sp)

    mv      s0, a0

    la      t0, lkf_OFF_x;        ld t0, 0(t0); add s1, s0, t0
    la      t0, lkf_OFF_P;        ld t0, 0(t0); add s2, s0, t0
    la      t0, lkf_OFF_F;        ld t0, 0(t0); add s3, s0, t0
    la      t0, lkf_OFF_Q;        ld t0, 0(t0); add s4, s0, t0
    
    # Compute jscratch bases
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, lkf_JS_Ftmp;      ld t1, 0(t1); add s5, t0, t1   # &Ftmp
    la      t1, lkf_JS_FT;        ld t1, 0(t1); add s6, t0, t1   # &FT
    la      t1, lkf_JS_xnew;      ld t1, 0(t1); add s7, t0, t1   # &xnew

    # x_new = F @ x
    mv      a0, s7; mv a1, s3; mv a2, s1; li a3, 276; li a4, 276
    call    mat_vec_mul_vec

    # copy x_new → x  — RVV vectorised
    mv      t2, s1; mv t3, s7
    li      t0, 276
.Lprvec_xcopy_v:
    beqz    t0, .Lprvec_xcopy_done
    vsetvli t1, t0, e64, m1, ta, ma
    vle64.v v0, (t3)
    vse64.v v0, (t2)
    slli    t4, t1, 3
    add     t2, t2, t4
    add     t3, t3, t4
    sub     t0, t0, t1
    j       .Lprvec_xcopy_v
.Lprvec_xcopy_done:

    # Ftmp = F @ P
    mv      a0, s5; mv a1, s3; mv a2, s2; li a3, 276; li a4, 276; li a5, 276
    call    mat_mul_vec

    # FT = F^T
    mv      a0, s6; mv a1, s3; li a2, 276; li a3, 276
    call    mat_transpose_vec

    # P = Ftmp @ FT
    mv      a0, s2; mv a1, s5; mv a2, s6; li a3, 276; li a4, 276; li a5, 276
    call    mat_mul_vec

    # P = P + Q
    mv      a0, s2; mv a1, s2; mv a2, s4; li a3, 276; li a4, 276
    call    mat_add_vec

    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); ld s3, 32(sp); ld s4, 40(sp)
    ld      s5, 48(sp); ld s6, 56(sp); ld s7, 64(sp)
    addi    sp, sp, 72; ret
    .size lkf_vec_predict, .-lkf_vec_predict


# =============================================================================
#  lkf_vec_update  —  Kalman measurement update
#  void lkf_vec_update(void *lkf, const double *z_flat)
# =============================================================================
    .globl lkf_vec_update
    .type  lkf_vec_update, @function
lkf_vec_update:
    addi    sp, sp, -112
    sd      ra,   0(sp); sd s0,   8(sp); sd s1,  16(sp); sd s2,  24(sp)
    sd      s3,  32(sp); sd s4,  40(sp); sd s5,  48(sp); sd s6,  56(sp)
    sd      s7,  64(sp); sd s8,  72(sp); sd s9,  80(sp); sd s10, 88(sp)
    sd      s11, 96(sp)

    mv      s0, a0; mv s9, a1

    la      t0, lkf_OFF_x;        ld t0, 0(t0); add s1,  s0, t0
    la      t0, lkf_OFF_P;        ld t0, 0(t0); add s2,  s0, t0
    la      t0, lkf_OFF_H;        ld t0, 0(t0); add s3,  s0, t0
    la      t0, lkf_OFF_R;        ld t0, 0(t0); add s4,  s0, t0
    la      t0, lkf_OFF_PHt;      ld t0, 0(t0); add s5,  s0, t0
    la      t0, lkf_OFF_S;        ld t0, 0(t0); add s6,  s0, t0
    la      t0, lkf_OFF_Sinv;     ld t0, 0(t0); add s7,  s0, t0
    la      t0, lkf_OFF_y;        ld t0, 0(t0); add s8,  s0, t0
    la      t0, lkf_OFF_piv;      ld t0, 0(t0); add s10, s0, t0
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add s11, s0, t0

    # Step 1: y = z - H @ x
    mv      a0, s8; mv a1, s3; mv a2, s1; li a3, 69; li a4, 276
    call    mat_vec_mul_vec     # y = H @ x  (z_pred)
    mv      a0, s8; mv a1, s9; mv a2, s8; li a3, 1; li a4, 69
    call    mat_sub_vec         # y = z - z_pred

    # Step 2: PHt = P @ H^T
    # H^T written into jscratch
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, lkf_JS_HT;        ld t1, 0(t1); add a0, t0, t1   # &H^T
    mv      a1, s3; li a2, 69; li a3, 276
    call    mat_transpose_vec
    
    # Recompute &H^T since t-regs are clobbered
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, lkf_JS_HT;        ld t1, 0(t1); add t2, t0, t1   # &H^T
    mv      a0, s5; mv a1, s2; mv a2, t2; li a3, 276; li a4, 276; li a5, 69
    call    mat_mul_vec

    # Step 3: S = H @ PHt + R
    mv      a0, s6; mv a1, s3; mv a2, s5; li a3, 69; li a4, 276; li a5, 69
    call    mat_mul_vec
    mv      a0, s6; mv a1, s6; mv a2, s4; li a3, 69; li a4, 69
    call    mat_add_vec

    # Step 4: Sinv = S^{-1}
    mv      a0, s7; mv a1, s6; li a2, 69; mv a3, s10
    call    mat_inverse_nxn
    bnez    a0, .Lupdvec_ok
    la      a0, .Lwarnvec_singular; call puts
    j       .Lupdvec_done

.Lupdvec_ok:
    # Step 5: K = PHt @ Sinv  (stored in jscratch)
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, lkf_JS_K;         ld t1, 0(t1); add a0, t0, t1   # &K
    mv      a1, s5; mv a2, s7; li a3, 276; li a4, 69; li a5, 69
    call    mat_mul_vec

    # Step 6: x = x + K @ y
    # Ky stored in jscratch
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, lkf_JS_Ky;        ld t1, 0(t1); add a0, t0, t1   # &Ky
    la      t1, lkf_JS_K;         ld t1, 0(t1); add a1, t0, t1   # &K
    mv      a2, s8; li a3, 276; li a4, 69
    call    mat_vec_mul_vec
    
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, lkf_JS_Ky;        ld t1, 0(t1); add t2, t0, t1   # &Ky
    mv      a0, s1; mv a1, s1; mv a2, t2; li a3, 1; li a4, 276
    call    mat_add_vec

    # Step 7: P = (I-KH)P(I-KH)^T + KRK^T  (Joseph form)
    # joseph scratch starts inside jscratch
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add t0, s0, t0
    la      t1, lkf_JS_joseph;    ld t1, 0(t1); add t2, t0, t1   # &joseph
    la      t1, lkf_JS_K;         ld t1, 0(t1); add a2, t0, t1   # &K
    mv      a0, s2; mv a1, s2
    mv      a3, s3; mv a4, s4; li a5, 276; li a6, 69; mv a7, t2
    call    mat_joseph_update_vec

.Lupdvec_done:
    ld      ra,   0(sp); ld s0,   8(sp); ld s1,  16(sp); ld s2,  24(sp)
    ld      s3,  32(sp); ld s4,  40(sp); ld s5,  48(sp); ld s6,  56(sp)
    ld      s7,  64(sp); ld s8,  72(sp); ld s9,  80(sp); ld s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 112; ret

    .section .rodata
.Lwarnvec_singular:
    .asciz  "[WARN] LKF-VEC: singular 69x69 S, skipping update\n"
    .section .text

    .size lkf_vec_update, .-lkf_vec_update


# =============================================================================
#  lkf_vec_get_positions  —  extract (23, 3) position array
#  void lkf_vec_get_positions(const void *lkf, double *pos_out)
# =============================================================================
    .globl lkf_vec_get_positions
    .type  lkf_vec_get_positions, @function
lkf_vec_get_positions:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 0
.Lgpvec_j:
    li      t2, 23; bge t1, t2, .Lgpvec_done
    li      t2, 12; mul t2, t1, t2; slli t2, t2, 3; add t3, t0, t2
    li      t4,  3; mul t4, t1, t4; slli t4, t4, 3; add t4, a1, t4
    fld     ft0,  0(t3); fsd ft0,  0(t4)   # px
    fld     ft0, 32(t3); fsd ft0,  8(t4)   # py
    fld     ft0, 64(t3); fsd ft0, 16(t4)   # pz
    addi    t1, t1, 1; j .Lgpvec_j
.Lgpvec_done:
    ret
    .size lkf_vec_get_positions, .-lkf_vec_get_positions


# =============================================================================
#  lkf_vec_get_full_state  —  copy x[276] to output buffer
#  void lkf_vec_get_full_state(const void *lkf, double *out)
# =============================================================================
    .globl lkf_vec_get_full_state
    .type  lkf_vec_get_full_state, @function
lkf_vec_get_full_state:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 276
.Lgfsvec_v:
    beqz    t1, .Lgfsvec_done
    vsetvli t2, t1, e64, m1, ta, ma
    vle64.v v0, (t0)
    vse64.v v0, (a1)
    slli    t3, t2, 3
    add     t0, t0, t3
    add     a1, a1, t3
    sub     t1, t1, t2
    j       .Lgfsvec_v
.Lgfsvec_done:
    ret
    .size lkf_vec_get_full_state, .-lkf_vec_get_full_state

# =============================================================================
#  END OF lkf_vector.s
# =============================================================================
