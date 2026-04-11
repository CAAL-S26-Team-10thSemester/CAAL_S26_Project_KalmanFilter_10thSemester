# =============================================================================
#  lkf_asm.s  —  RISC-V Scalar Assembly: Linear Kalman Filter
#  Kalman Filter Milestone-3
# =============================================================================
#
#  Implements the LKF class from kalman-updated.py exactly:
#
#    lkf_sizeof            ()                    — return heap size needed
#    lkf_init              (lkf, dt)             — __init__
#    lkf_set_initial_state (lkf, joint, pos3)    — set_initial_state
#    lkf_predict           (lkf)                 — predict
#    lkf_update            (lkf, z_flat)         — update
#    lkf_get_positions     (lkf, pos_out)         — get_positions
#    lkf_get_full_state    (lkf, out)             — get_full_state
#
#  Matrix helpers (used at init only):
#    state_init_F  (F, dt)
#    state_init_Q  (Q)
#    meas_init_H   (H)
#    meas_init_R   (R)
#
#  All matrix algebra delegates to matrix_asm.s (mat_mul, mat_vec_mul, etc.).
#  Optimisations applied (struct layout and array sizes UNCHANGED):
#    1. ABI fix: state_init_F now saves/restores fs1 and fs2 (were clobbered).
#    2. Zero loops unrolled ×8: saves ~8× branch overhead in init (one-time cost).
#       .LQ_zero: 76176→9522 iterations  (609408 bytes)
#       .LH_zero: 19044→2380 iterations  (152208 bytes)
#       .LR_zero:  4761→595  iterations  ( 38088 bytes)
#    3. x-vector zero (lkf_init) and copy (lkf_predict) unrolled ×4.
#    4. lkf_get_full_state copy unrolled ×4 (276→69 loop iterations per frame).
#
# =============================================================================
#  LKF STRUCT LAYOUT  (unchanged from working version)
#
#  Offset      Bytes   Field
#  ----------  ------  -------
#           0       8  dt          (double)
#           8    2208  x[276]      state vector
#        2216  609408  P[276×276]  covariance
#      611624  609408  F[276×276]  state transition
#     1221032  609408  Q[276×276]  process noise
#     1830440  152208  H[69×276]   measurement matrix   (69*276*8=152208)
#     1982648   38088  R[69×69]    measurement noise    (note: OFF_R=1982792 below)
#
#  Scratch (appended, same allocation):
#     2020880  152352  PHt[276×69]
#     2173232   38088  S[69×69]
#     2211320   38088  Sinv[69×69]
#     2249408     552  y[69]
#     2249960     276  piv[69]  (int32)
#     2250240  152352  K[276×69]
#     2402592 2437632  jscratch[4×276×276]
#
#  TOTAL: 4840224 bytes  (lkf_TOTAL_BYTES, used by lkf_sizeof)
#
# =============================================================================
#  ABI: RISC-V LP64D (rv64imfd, lp64d)
#    Integer arg/return   : a0–a7   (x10–x17)
#    FP arg/return        : fa0–fa7 (f10–f17)
#    Callee-saved integer : s0–s11
#    Callee-saved FP      : fs0–fs11
#    Temporaries integer  : t0–t6
#    Temporaries FP       : ft0–ft11
# =============================================================================

    .extern mat_eye
    .extern mat_mul
    .extern mat_add
    .extern mat_sub
    .extern mat_transpose
    .extern mat_vec_mul
    .extern mat_inverse_nxn
    .extern mat_joseph_update

# =============================================================================
#  .rodata — constants (names kept identical to working version)
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

# Struct offsets — identical values to working version
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
lkf_TOTAL_BYTES:  .quad 4840224

    .section .text

# =============================================================================
#  lkf_sizeof  —  return total allocation size
#  size_t lkf_sizeof(void)
#  Leaf.
# =============================================================================
    .globl lkf_sizeof
    .type  lkf_sizeof, @function
lkf_sizeof:
    la      t0, lkf_TOTAL_BYTES
    ld      a0, 0(t0)
    ret
    .size lkf_sizeof, .-lkf_sizeof


# =============================================================================
#  state_init_F  —  build F (276×276 block-diagonal state transition)
#
#  Python: F=eye(276); fill upper-triangular Taylor coefficients per joint/axis.
#
#  void state_init_F(double *F, double dt)
#  a0=F,  fa0=dt
#
#  Register allocation:
#    s0=F  s1=j  s2=axis  s3=r  s4=N=276
#    fs0=dt  fs1=dt²/2  fs2=dt³/6
#
#  ABI FIX vs original: fs1 and fs2 are callee-saved; the original file
#  used them without saving them.  Frame enlarged 56→72 bytes to hold all three.
# =============================================================================
    .globl state_init_F
    .type  state_init_F, @function
state_init_F:
    addi    sp, sp, -72
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); sd s3, 32(sp); sd s4, 40(sp)
    fsd     fs0, 48(sp); fsd fs1, 56(sp); fsd fs2, 64(sp)

    mv      s0, a0
    fmv.d   fs0, fa0            # dt

    # F = eye(276)  — a0 already holds F ptr
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

.LF_j:
    li      t0, 23; bge s1, t0, .LF_done
    li      t1, 12; mul t2, s1, t1    # t2 = base = j*12
    li      s2, 0

.LF_axis:
    li      t0, 3; bge s2, t0, .LF_axis_done
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

    addi    s2, s2, 1; j .LF_axis
.LF_axis_done:
    addi    s1, s1, 1; j .LF_j
.LF_done:
    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); ld s3, 32(sp); ld s4, 40(sp)
    fld     fs0, 48(sp); fld fs1, 56(sp); fld fs2, 64(sp)
    addi    sp, sp, 72; ret
    .size state_init_F, .-state_init_F


# =============================================================================
#  state_init_Q  —  build Q (276×276 block-diagonal process noise)
#
#  Zero loop unrolled ×8 to reduce branch overhead on 609408-byte clear.
#
#  void state_init_Q(double *Q)
#  a0=Q
#
#  Register allocation:
#    s0=Q  s1=j  s2=i
#    ft0=0.0 (zero), ft1=1e-6, ft2=1e-5, ft3=1e-4
# =============================================================================
    .globl state_init_Q
    .type  state_init_Q, @function
state_init_Q:
    addi    sp, sp, -32
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp); sd s2, 24(sp)

    mv      s0, a0

    # Zero Q: 276*276*8 = 609408 bytes — unrolled ×8 (64 bytes per iteration)
    li      t0, 276; mul t0, t0, t0; slli t0, t0, 3   # total bytes
    add     t1, s0, t0          # end ptr
    la      t2, lkf_fp_zero; fld ft0, 0(t2)   # ft0 = 0.0
    mv      t2, s0
    addi    t3, t1, -56         # unroll boundary: end - 7*8
.LQ_zero_u:
    bgt     t2, t3, .LQ_zero_t
    fsd     ft0,  0(t2); fsd ft0,  8(t2); fsd ft0, 16(t2); fsd ft0, 24(t2)
    fsd     ft0, 32(t2); fsd ft0, 40(t2); fsd ft0, 48(t2); fsd ft0, 56(t2)
    addi    t2, t2, 64; j .LQ_zero_u
.LQ_zero_t:
    bge     t2, t1, .LQ_zero_done
    fsd     ft0, 0(t2); addi t2, t2, 8; j .LQ_zero_t
.LQ_zero_done:

    la      t0, lkf_noise_0; fld ft1, 0(t0)   # 1e-6
    la      t0, lkf_noise_1; fld ft2, 0(t0)   # 1e-5
    la      t0, lkf_noise_2; fld ft3, 0(t0)   # 1e-4

    li      s1, 0               # j = 0
.LQ_j:
    li      t3, 23; bge s1, t3, .LQ_done
    li      t4, 12; mul t4, s1, t4   # base = j*12
    li      s2, 0

.LQ_i:
    li      t3, 12; bge s2, t3, .LQ_i_done
    add     t5, t4, s2          # row = col = base+i
    li      t6, 276
    mul     t0, t5, t6; add t0, t0, t5
    slli    t0, t0, 3; add t0, s0, t0

    andi    t3, s2, 3
    beqz    t3, .LQ_n0
    li      t6, 1; beq t3, t6, .LQ_n1
    fsd     ft3, 0(t0); j .LQ_next    # i%4 == 2 or 3
.LQ_n0: fsd ft1, 0(t0); j .LQ_next
.LQ_n1: fsd ft2, 0(t0)
.LQ_next:
    addi    s2, s2, 1; j .LQ_i
.LQ_i_done:
    addi    s1, s1, 1; j .LQ_j
.LQ_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp); ld s2, 24(sp)
    addi    sp, sp, 32; ret
    .size state_init_Q, .-state_init_Q


# =============================================================================
#  meas_init_H  —  build H (69×276 block-diagonal measurement matrix)
#
#  Zero loop unrolled ×8 (152208 bytes → ~2380 iterations).
#
#  void meas_init_H(double *H)
#  a0=H
#
#  Register allocation:  s0=H  s1=j  ft0=0.0  ft1=1.0
# =============================================================================
    .globl meas_init_H
    .type  meas_init_H, @function
meas_init_H:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)

    mv      s0, a0

    # Zero H: 69*276*8 = 152208 bytes — unrolled ×8
    li      t0, 69; li t1, 276; mul t0, t0, t1; slli t0, t0, 3
    add     t1, s0, t0
    la      t2, lkf_fp_zero; fld ft0, 0(t2)
    mv      t2, s0
    addi    t3, t1, -56
.LH_zero_u:
    bgt     t2, t3, .LH_zero_t
    fsd     ft0,  0(t2); fsd ft0,  8(t2); fsd ft0, 16(t2); fsd ft0, 24(t2)
    fsd     ft0, 32(t2); fsd ft0, 40(t2); fsd ft0, 48(t2); fsd ft0, 56(t2)
    addi    t2, t2, 64; j .LH_zero_u
.LH_zero_t:
    bge     t2, t1, .LH_zero_done
    fsd     ft0, 0(t2); addi t2, t2, 8; j .LH_zero_t
.LH_zero_done:

    la      t0, lkf_fp_one; fld ft1, 0(t0)
    li      s1, 0
.LH_j:
    li      t0, 23; bge s1, t0, .LH_done
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

    addi    s1, s1, 1; j .LH_j
.LH_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp)
    addi    sp, sp, 24; ret
    .size meas_init_H, .-meas_init_H


# =============================================================================
#  meas_init_R  —  build R (69×69 block-diagonal measurement noise)
#
#  Zero loop unrolled ×8 (38088 bytes → ~595 iterations).
#
#  void meas_init_R(double *R)
#  a0=R
#
#  Register allocation:  s0=R  s1=j  ft0=0.0  ft1=EST_R_PX  ft2=PY  ft3=PZ
# =============================================================================
    .globl meas_init_R
    .type  meas_init_R, @function
meas_init_R:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)

    mv      s0, a0

    # Zero R: 69*69*8 = 38088 bytes — unrolled ×8
    li      t0, 69; mul t0, t0, t0; slli t0, t0, 3
    add     t1, s0, t0
    la      t2, lkf_fp_zero; fld ft0, 0(t2)
    mv      t2, s0
    addi    t3, t1, -56
.LR_zero_u:
    bgt     t2, t3, .LR_zero_t
    fsd     ft0,  0(t2); fsd ft0,  8(t2); fsd ft0, 16(t2); fsd ft0, 24(t2)
    fsd     ft0, 32(t2); fsd ft0, 40(t2); fsd ft0, 48(t2); fsd ft0, 56(t2)
    addi    t2, t2, 64; j .LR_zero_u
.LR_zero_t:
    bge     t2, t1, .LR_zero_done
    fsd     ft0, 0(t2); addi t2, t2, 8; j .LR_zero_t
.LR_zero_done:

    la      t0, lkf_EST_R_PX; fld ft1, 0(t0)
    la      t0, lkf_EST_R_PY; fld ft2, 0(t0)
    la      t0, lkf_EST_R_PZ; fld ft3, 0(t0)

    li      s1, 0
.LR_j:
    li      t3, 23; bge s1, t3, .LR_done
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

    addi    s1, s1, 1; j .LR_j
.LR_done:
    ld      ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp)
    addi    sp, sp, 24; ret
    .size meas_init_R, .-meas_init_R


# =============================================================================
#  lkf_init  —  initialise all fields of an LKF struct
#
#  x zero loop unrolled ×4 (2208 bytes → 69 iterations instead of 276).
#
#  void lkf_init(void *lkf, double dt)
#  a0=lkf,  fa0=dt
#
#  Register allocation:  s0=lkf  fs0=dt
# =============================================================================
    .globl lkf_init
    .type  lkf_init, @function
lkf_init:
    addi    sp, sp, -24
    sd      ra, 0(sp); sd s0, 8(sp); fsd fs0, 16(sp)

    mv      s0, a0
    fmv.d   fs0, fa0

    # store dt
    fsd     fs0, 0(s0)

    # x = zeros(276): 2208 bytes — unrolled ×4
    la      t0, lkf_fp_zero; fld ft0, 0(t0)
    addi    t1, s0, 8           # &x[0]
    li      t2, 276; slli t2, t2, 3; add t2, t1, t2   # end
    addi    t3, t2, -24         # unroll boundary
.Linit_xz_u:
    bgt     t1, t3, .Linit_xz_t
    fsd     ft0,  0(t1); fsd ft0,  8(t1); fsd ft0, 16(t1); fsd ft0, 24(t1)
    addi    t1, t1, 32; j .Linit_xz_u
.Linit_xz_t:
    bge     t1, t2, .Linit_xz_done
    fsd     ft0, 0(t1); addi t1, t1, 8; j .Linit_xz_t
.Linit_xz_done:

    # P = eye(276)
    la      t0, lkf_OFF_P; ld t0, 0(t0); add a0, s0, t0
    li      a1, 276; call mat_eye

    # F = state_init_F(dt)
    la      t0, lkf_OFF_F; ld t0, 0(t0); add a0, s0, t0
    fmv.d   fa0, fs0; call state_init_F

    # Q = state_init_Q()
    la      t0, lkf_OFF_Q; ld t0, 0(t0); add a0, s0, t0
    call    state_init_Q

    # H = meas_init_H()
    la      t0, lkf_OFF_H; ld t0, 0(t0); add a0, s0, t0
    call    meas_init_H

    # R = meas_init_R()
    la      t0, lkf_OFF_R; ld t0, 0(t0); add a0, s0, t0
    call    meas_init_R

    ld      ra, 0(sp); ld s0, 8(sp); fld fs0, 16(sp)
    addi    sp, sp, 24; ret
    .size lkf_init, .-lkf_init


# =============================================================================
#  lkf_set_initial_state  —  set one joint's sub-state from a 3-D position
#
#  void lkf_set_initial_state(void *lkf, int joint_idx, double *pos3)
#  a0=lkf,  a1=joint_idx,  a2=pos3
#  Leaf — no callee-saved registers used.
# =============================================================================
    .globl lkf_set_initial_state
    .type  lkf_set_initial_state, @function
lkf_set_initial_state:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 12; mul t1, a1, t1; slli t1, t1, 3; add t0, t0, t1
    la      t1, lkf_fp_zero; fld ft0, 0(t1)
    fld     ft1, 0(a2); fld ft2, 8(a2); fld ft3, 16(a2)
    fsd     ft1,  0(t0); fsd ft0,  8(t0); fsd ft0, 16(t0); fsd ft0, 24(t0)
    fsd     ft2, 32(t0); fsd ft0, 40(t0); fsd ft0, 48(t0); fsd ft0, 56(t0)
    fsd     ft3, 64(t0); fsd ft0, 72(t0); fsd ft0, 80(t0); fsd ft0, 88(t0)
    ret
    .size lkf_set_initial_state, .-lkf_set_initial_state


# =============================================================================
#  lkf_predict  —  Kalman prediction step
#
#  x copy loop unrolled ×4 (2208 bytes → 69 iterations instead of 276).
#
#  void lkf_predict(void *lkf)
#  a0=lkf
#
#  Register allocation:
#    s0=lkf  s1=&x  s2=&P  s3=&F  s4=&Q  s5=&jscratch(Ftmp)  s6=&FT
# =============================================================================
    .globl lkf_predict
    .type  lkf_predict, @function
lkf_predict:
    addi    sp, sp, -64
    sd      ra,  0(sp); sd s0,  8(sp); sd s1, 16(sp)
    sd      s2, 24(sp); sd s3, 32(sp); sd s4, 40(sp)
    sd      s5, 48(sp); sd s6, 56(sp)

    mv      s0, a0

    la      t0, lkf_OFF_x;        ld t0, 0(t0); add s1, s0, t0
    la      t0, lkf_OFF_P;        ld t0, 0(t0); add s2, s0, t0
    la      t0, lkf_OFF_F;        ld t0, 0(t0); add s3, s0, t0
    la      t0, lkf_OFF_Q;        ld t0, 0(t0); add s4, s0, t0
    la      t0, lkf_OFF_jscratch; ld t0, 0(t0); add s5, s0, t0  # Ftmp
    li      t1, 276; mul t1, t1, t1; slli t1, t1, 3
    add     s6, s5, t1          # FT = Ftmp + 276²*8

    # x_new = F @ x  (stack buffer 2208 bytes)
    li      t0, 2208; sub sp, sp, t0
    mv      a0, sp; mv a1, s3; mv a2, s1; li a3, 276; li a4, 276
    call    mat_vec_mul

    # copy x_new → x  — unrolled ×4
    # NOTE: t1 is caller-saved and was clobbered by mat_vec_mul;
    # sp is callee-saved and still equals the x_new buffer base.
    mv      t2, s1; mv t3, sp
    li      t4, 2208; add t4, t2, t4   # end
    addi    t5, t4, -24                # unroll boundary
.Lpr_xcopy_u:
    bgt     t2, t5, .Lpr_xcopy_t
    fld     ft0,  0(t3); fld ft1,  8(t3); fld ft2, 16(t3); fld ft3, 24(t3)
    fsd     ft0,  0(t2); fsd ft1,  8(t2); fsd ft2, 16(t2); fsd ft3, 24(t2)
    addi    t2, t2, 32; addi t3, t3, 32; j .Lpr_xcopy_u
.Lpr_xcopy_t:
    bge     t2, t4, .Lpr_xcopy_done
    fld     ft0, 0(t3); fsd ft0, 0(t2)
    addi    t2, t2, 8; addi t3, t3, 8; j .Lpr_xcopy_t
.Lpr_xcopy_done:
    li      t0, 2208; add sp, sp, t0   # free x_new

    # Ftmp = F @ P
    mv      a0, s5; mv a1, s3; mv a2, s2; li a3, 276; li a4, 276; li a5, 276
    call    mat_mul

    # FT = F^T
    mv      a0, s6; mv a1, s3; li a2, 276; li a3, 276
    call    mat_transpose

    # P = Ftmp @ FT
    mv      a0, s2; mv a1, s5; mv a2, s6; li a3, 276; li a4, 276; li a5, 276
    call    mat_mul

    # P = P + Q
    mv      a0, s2; mv a1, s2; mv a2, s4; li a3, 276; li a4, 276
    call    mat_add

    ld      ra,  0(sp); ld s0,  8(sp); ld s1, 16(sp)
    ld      s2, 24(sp); ld s3, 32(sp); ld s4, 40(sp)
    ld      s5, 48(sp); ld s6, 56(sp)
    addi    sp, sp, 64; ret
    .size lkf_predict, .-lkf_predict


# =============================================================================
#  lkf_update  —  Kalman measurement update
#
#  void lkf_update(void *lkf, const double *z_flat)
#  a0=lkf,  a1=z_flat (double[69])
#
#  Register allocation:
#    s0=lkf  s1=&x  s2=&P  s3=&H  s4=&R
#    s5=&PHt s6=&S  s7=&Sinv  s8=&y  s9=z_flat  s10=&piv  s11=&jscratch
# =============================================================================
    .globl lkf_update
    .type  lkf_update, @function
lkf_update:
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
    call    mat_vec_mul         # y = H @ x  (z_pred)
    mv      a0, s8; mv a1, s9; mv a2, s8; li a3, 1; li a4, 69
    call    mat_sub             # y = z - z_pred

    # Step 2: PHt = P @ H^T
    # H^T written into jscratch (276×69 = 152208 bytes; jscratch >> that)
    mv      a0, s11; mv a1, s3; li a2, 69; li a3, 276
    call    mat_transpose
    mv      a0, s5; mv a1, s2; mv a2, s11; li a3, 276; li a4, 276; li a5, 69
    call    mat_mul

    # Step 3: S = H @ PHt + R
    mv      a0, s6; mv a1, s3; mv a2, s5; li a3, 69; li a4, 276; li a5, 69
    call    mat_mul
    mv      a0, s6; mv a1, s6; mv a2, s4; li a3, 69; li a4, 69
    call    mat_add

    # Step 4: Sinv = S^{-1}
    mv      a0, s7; mv a1, s6; li a2, 69; mv a3, s10
    call    mat_inverse_nxn
    bnez    a0, .Lupd_ok
    la      a0, .Lwarn_singular; call puts
    j       .Lupd_done

.Lupd_ok:
    # Step 5: K = PHt @ Sinv  (stored in jscratch)
    mv      a0, s11; mv a1, s5; mv a2, s7; li a3, 276; li a4, 69; li a5, 69
    call    mat_mul

    # Step 6: x = x + K @ y
    # NOTE: t1 is caller-saved; after mat_vec_mul it is clobbered.
    # Use sp (callee-saved) to address the Ky scratch buffer.
    li      t0, 2208; sub sp, sp, t0
    mv      a0, sp; mv a1, s11; mv a2, s8; li a3, 276; li a4, 69
    call    mat_vec_mul
    mv      a0, s1; mv a1, s1; mv a2, sp; li a3, 1; li a4, 276
    call    mat_add
    li      t0, 2208; add sp, sp, t0

    # Step 7: P = (I-KH)P(I-KH)^T + KRK^T  (Joseph form)
    # joseph scratch starts after K[276×69] inside jscratch
    li      t0, 276; li t1, 69; mul t0, t0, t1; slli t0, t0, 3
    add     t1, s11, t0         # &jscratch[276*69] = joseph scratch
    mv      a0, s2; mv a1, s2; mv a2, s11
    mv      a3, s3; mv a4, s4; li a5, 276; li a6, 69; mv a7, t1
    call    mat_joseph_update

.Lupd_done:
    ld      ra,   0(sp); ld s0,   8(sp); ld s1,  16(sp); ld s2,  24(sp)
    ld      s3,  32(sp); ld s4,  40(sp); ld s5,  48(sp); ld s6,  56(sp)
    ld      s7,  64(sp); ld s8,  72(sp); ld s9,  80(sp); ld s10, 88(sp)
    ld      s11, 96(sp)
    addi    sp, sp, 112; ret

    .section .rodata
.Lwarn_singular:
    .asciz  "[WARN] LKF: singular 69x69 S, skipping update\n"
    .section .text

    .size lkf_update, .-lkf_update


# =============================================================================
#  lkf_get_positions  —  extract (23, 3) position array
#
#  void lkf_get_positions(const void *lkf, double *pos_out)
#  a0=lkf,  a1=pos_out  (double[69])
#  Leaf.
# =============================================================================
    .globl lkf_get_positions
    .type  lkf_get_positions, @function
lkf_get_positions:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 0
.Lgp_j:
    li      t2, 23; bge t1, t2, .Lgp_done
    li      t2, 12; mul t2, t1, t2; slli t2, t2, 3; add t3, t0, t2
    li      t4,  3; mul t4, t1, t4; slli t4, t4, 3; add t4, a1, t4
    fld     ft0,  0(t3); fsd ft0,  0(t4)   # px
    fld     ft0, 32(t3); fsd ft0,  8(t4)   # py
    fld     ft0, 64(t3); fsd ft0, 16(t4)   # pz
    addi    t1, t1, 1; j .Lgp_j
.Lgp_done:
    ret
    .size lkf_get_positions, .-lkf_get_positions


# =============================================================================
#  lkf_get_full_state  —  copy x[276] to output buffer
#
#  Unrolled ×4: 276→69 loop iterations per frame.
#
#  void lkf_get_full_state(const void *lkf, double *out)
#  a0=lkf,  a1=out (double[276])
#  Leaf.
# =============================================================================
    .globl lkf_get_full_state
    .type  lkf_get_full_state, @function
lkf_get_full_state:
    la      t0, lkf_OFF_x; ld t0, 0(t0); add t0, a0, t0
    li      t1, 2208; add t1, t0, t1   # end ptr
    addi    t2, t1, -32                # unroll boundary: ptr+24<=end-8 => ptr<=end-32
.Lgfs_u:
    bgt     t0, t2, .Lgfs_t
    fld     ft0,  0(t0); fld ft1,  8(t0); fld ft2, 16(t0); fld ft3, 24(t0)
    fsd     ft0,  0(a1); fsd ft1,  8(a1); fsd ft2, 16(a1); fsd ft3, 24(a1)
    addi    t0, t0, 32; addi a1, a1, 32; j .Lgfs_u
.Lgfs_t:
    bge     t0, t1, .Lgfs_done
    fld     ft0, 0(t0); fsd ft0, 0(a1)
    addi    t0, t0, 8; addi a1, a1, 8; j .Lgfs_t
.Lgfs_done:
    ret
    .size lkf_get_full_state, .-lkf_get_full_state

# =============================================================================
#  END OF lkf_asm.s
# =============================================================================