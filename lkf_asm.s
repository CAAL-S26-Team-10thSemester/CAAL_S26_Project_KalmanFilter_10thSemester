# =============================================================================
# lkf_asm.s  --  Linear Kalman Filter  (RISC-V RV64GD scalar assembly)
# =============================================================================
#
# Mirrors kalman-updated.py §§ LKF class exactly:
#
#   State layout per joint j  (base offset = j*12 doubles):
#       [px, vx, ax, jx,  py, vy, ay, jy,  pz, vz, az, jz]
#
#   Global matrices:
#       F  : 276×276  block-diagonal constant-jerk state transition
#       Q  : 276×276  block-diagonal process noise
#       H  :  69×276  block-diagonal Cartesian measurement extractor
#       R  :  69× 69  block-diagonal Cartesian measurement noise
#       S  :  69× 69  innovation covariance  (scratch, per step)
#       K  : 276× 69  Kalman gain            (scratch, per step)
#
#   Per-frame algorithm:
#       Predict:
#           x      = F * x
#           P      = F * P * F^T + Q
#       Update:
#           y      = z  - H*x           (69-D Cartesian innovation)
#           PHt    = P  * H^T           (276×69)
#           S      = H  * PHt + R       (69×69)
#           S^{-1} via LU partial pivot
#           K      = PHt * S^{-1}       (276×69)
#           x      = x + K*y
#           P      = (I-K*H)*P*(I-K*H)^T + K*R*K^T   (Joseph form)
#
# Public entry points (C ABI, a0-a7 for first 8 args, stack for rest):
#
#   lkf_init_F(double *F, double dt)
#   lkf_init_Q(double *Q)
#   lkf_init_H(double *H)
#   lkf_init_R_cart(double *R)
#   lkf_init_state(double *x, int joint, double px, double py, double pz)
#   lkf_predict(double *x, double *P, const double *F, const double *Q,
#               int N, double *tmp276, double *tmpNN)
#   lkf_update(double *x, double *P,
#              const double *H, const double *R_cart,
#              const double *meas_cart,
#              double *K_buf, double *S_buf,
#              double *PHt_buf, double *y_buf,
#              double *IKH_buf,
#              int N_joints, int *pivot_buf)
#   lkf_lu_inverse(double *A, double *Ainv, int n, int *pivot)
#       Returns 0 (OK) or -1 (singular).
#
# =============================================================================
# ABI / register conventions
# =============================================================================
# Caller-saved: a0-a7, fa0-fa7, t0-t6, ft0-ft11
# Callee-saved: s0-s11, fs0-fs11, ra, sp
# All matrices ROW-MAJOR.  Element (i,j) of M×N matrix at base p:
#   byte address = p + (i*N + j)*8
# =============================================================================

    .section .rodata
    .align 3

# ---------------------------------------------------------------------------
# Double-precision constants
# ---------------------------------------------------------------------------
.lkf_const_one:
    .double  1.0
.lkf_const_half:
    .double  0.5
.lkf_const_sixth:
    .double  0.16666666666666666667
.lkf_const_eps:
    .double  1.0e-10

# Cartesian measurement noise (EST_R_PX, EST_R_PY, EST_R_PZ from Python)
.lkf_r_cart:
    .double  0.29472279   # sigma2_px
    .double  0.09632091   # sigma2_py
    .double  0.00204269   # sigma2_pz

# Process noise per state index mod 4
.lkf_q_noise:
    .double  1.0e-6   # index 0 : position
    .double  1.0e-5   # index 1 : velocity
    .double  1.0e-4   # index 2 : acceleration
    .double  1.0e-4   # index 3 : jerk

    .section .text
    .align 2

# =============================================================================
# lkf_zero_matrix -- Zero N doubles starting at ptr a0
# =============================================================================
# Args: a0 = double *ptr,  a1 = int count
# Internal helper (not exported).
# =============================================================================
lkf_zero_matrix:
    fcvt.d.w ft0, zero
.lzm_loop:
    beq     a1, zero, .lzm_done
    fsd     ft0, 0(a0)
    addi    a0, a0, 8
    addi    a1, a1, -1
    j       .lzm_loop
.lzm_done:
    ret

# =============================================================================
# lkf_init_F -- Build 276×276 block-diagonal state transition matrix F
# =============================================================================
# Args:
#   a0 = double *F   (caller-alloc 276*276*8 bytes)
#   fa0= double dt
#
# F is identity with upper-triangle offsets inside each 4×4 sub-block:
#   F[r,   r+1] = dt
#   F[r,   r+2] = dt^2/2
#   F[r,   r+3] = dt^3/6
#   F[r+1, r+2] = dt
#   F[r+1, r+3] = dt^2/2
#   F[r+2, r+3] = dt
# 23 joints × 3 axes = 69 such sub-blocks.
# =============================================================================
    .globl lkf_init_F
lkf_init_F:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    fsd     fs0, 24(sp)
    fsd     fs1, 32(sp)
    fsd     fs2, 40(sp)
    fsd     fs3, 48(sp)

    mv      s0, a0                  # F base ptr

    # Zero F
    mv      a0, s0
    li      a1, 76176               # 276*276
    call    lkf_zero_matrix

    # Diagonal = 1.0
    la      t0, .lkf_const_one
    fld     fs0, 0(t0)              # fs0 = 1.0
    li      t0, 0                   # i
    li      t2, 277                 # diagonal stride element count
    slli    t2, t2, 3               # 277*8 bytes
    mv      t1, s0
.lF_diag:
    li      t3, 276
    bge     t0, t3, .lF_diag_done
    fsd     fs0, 0(t1)
    add     t1, t1, t2
    addi    t0, t0, 1
    j       .lF_diag
.lF_diag_done:

    # Precompute dt powers into callee-saved FP regs
    la      t0, .lkf_const_half
    fld     ft0, 0(t0)
    la      t0, .lkf_const_sixth
    fld     ft1, 0(t0)
    fmul.d  fs1, fa0, fa0           # fs1 = dt^2
    fmul.d  fs2, fs1, ft0           # fs2 = dt^2/2
    fmul.d  ft2, fs1, fa0           # ft2 = dt^3
    fmul.d  fs3, ft2, ft1           # fs3 = dt^3/6
    fmv.d   fs0, fa0                # fs0 = dt (overwrite 1.0; diagonal done)

    # 23 joints × 3 axes
    li      s1, 0                   # joint
    li      s2, 23
.lF_jloop:
    bge     s1, s2, .lF_jdone
    li      t0, 0                   # axis
.lF_aloop:
    li      t3, 3
    bge     t0, t3, .lF_adone

    # r = joint*12 + axis*4
    li      t1, 12
    mul     t1, s1, t1
    slli    t2, t0, 2
    add     t1, t1, t2              # t1 = r

    # &F[r][r]  = F + r*277*8
    li      t3, 277
    mul     t4, t1, t3
    slli    t4, t4, 3
    add     t4, s0, t4              # t4 = &F[r][r]

    fsd     fs0,  8(t4)             # F[r,r+1]   = dt
    fsd     fs2, 16(t4)             # F[r,r+2]   = dt2/2
    fsd     fs3, 24(t4)             # F[r,r+3]   = dt3/6

    # &F[r+1][r+1]
    addi    t5, t1, 1
    mul     t6, t5, t3
    slli    t6, t6, 3
    add     t6, s0, t6
    fsd     fs0,  8(t6)             # F[r+1,r+2] = dt
    fsd     fs2, 16(t6)             # F[r+1,r+3] = dt2/2

    # &F[r+2][r+2]
    addi    t5, t1, 2
    mul     t6, t5, t3
    slli    t6, t6, 3
    add     t6, s0, t6
    fsd     fs0,  8(t6)             # F[r+2,r+3] = dt

    addi    t0, t0, 1
    j       .lF_aloop
.lF_adone:
    addi    s1, s1, 1
    j       .lF_jloop
.lF_jdone:

    fld     fs3, 48(sp)
    fld     fs2, 40(sp)
    fld     fs1, 32(sp)
    fld     fs0, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret

# =============================================================================
# lkf_init_Q -- Build 276×276 block-diagonal process noise matrix Q
# =============================================================================
# Args:  a0 = double *Q
# Noise per state-index mod 4: {0:1e-6, 1:1e-5, 2:1e-4, 3:1e-4}
# =============================================================================
    .globl lkf_init_Q
lkf_init_Q:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)

    mv      s0, a0

    mv      a1, a0
    li      a0, 76176 
    # (a0 = Q, a1 = count): reuse zero helper signature
    mv      a1, a0
    mv      a0, s0
    call    lkf_zero_matrix

    la      t2, .lkf_q_noise
    fld     ft1,  0(t2)             # 1e-6
    fld     ft2,  8(t2)             # 1e-5
    fld     ft3, 16(t2)             # 1e-4 (acc)
    fld     ft4, 24(t2)             # 1e-4 (jerk)

    li      s1, 0                   # i
    li      t5, 276
.lQ_loop:
    bge     s1, t5, .lQ_done
    li      t3, 277
    mul     t4, s1, t3
    slli    t4, t4, 3
    add     t4, s0, t4              # &Q[i][i]

    andi    t6, s1, 3               # i % 4
    beq     t6, zero, .lQ_c0
    li      t0, 1
    beq     t6, t0,  .lQ_c1
    li      t0, 2
    beq     t6, t0,  .lQ_c2
    fsd     ft4, 0(t4); j .lQ_next  # case 3: jerk
.lQ_c0: fsd ft1, 0(t4); j .lQ_next
.lQ_c1: fsd ft2, 0(t4); j .lQ_next
.lQ_c2: fsd ft3, 0(t4)
.lQ_next:
    addi    s1, s1, 1
    j       .lQ_loop
.lQ_done:
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 24(sp)
    addi    sp, sp, 32
    ret

# Suppress register alias:
a0_v = a0

# =============================================================================
# lkf_init_H -- Build 69×276 block-diagonal Cartesian measurement matrix H
# =============================================================================
# Args:  a0 = double *H   (caller-alloc 69*276*8 = 152064 bytes)
#
# For joint j (0..22):
#   H[j*3+0, j*12+0] = 1.0   (px)
#   H[j*3+1, j*12+4] = 1.0   (py)
#   H[j*3+2, j*12+8] = 1.0   (pz)
# =============================================================================
    .globl lkf_init_H
lkf_init_H:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)

    mv      s0, a0

    # Zero H (69*276 = 19044 doubles)
    li      a1, 19044
    call    lkf_zero_matrix

    la      t0, .lkf_const_one
    fld     ft1, 0(t0)              # ft1 = 1.0

    li      s1, 0                   # joint
    li      t6, 23
.lH_loop:
    bge     s1, t6, .lH_done
    # rb = j*3, cb = j*12
    li      t0, 3
    mul     t0, s1, t0              # rb
    li      t1, 12
    mul     t1, s1, t1              # cb

    # H[rb+0, cb+0]: element offset (rb*276 + cb)
    li      t2, 276
    mul     t3, t0, t2
    add     t3, t3, t1
    slli    t3, t3, 3
    add     t3, s0, t3
    fsd     ft1, 0(t3)              # H[rb,cb] = 1.0  (px row)

    # H[rb+1, cb+4]: ((rb+1)*276 + cb+4)*8
    addi    t4, t0, 1               # rb+1
    mul     t3, t4, t2
    addi    t5, t1, 4               # cb+4
    add     t3, t3, t5
    slli    t3, t3, 3
    add     t3, s0, t3
    fsd     ft1, 0(t3)              # H[rb+1,cb+4] = 1.0  (py row)

    # H[rb+2, cb+8]: ((rb+2)*276 + cb+8)*8
    addi    t4, t0, 2               # rb+2
    mul     t3, t4, t2
    addi    t5, t1, 8               # cb+8
    add     t3, t3, t5
    slli    t3, t3, 3
    add     t3, s0, t3
    fsd     ft1, 0(t3)              # H[rb+2,cb+8] = 1.0  (pz row)

    addi    s1, s1, 1
    j       .lH_loop
.lH_done:
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 24(sp)
    addi    sp, sp, 32
    ret

# =============================================================================
# lkf_init_R_cart -- Build 69×69 Cartesian measurement noise matrix R
# =============================================================================
# Args:  a0 = double *R   (caller-alloc 69*69*8 = 38088 bytes)
# 23 identical 3×3 diagonal blocks: diag(0.29472279, 0.09632091, 0.00204269)
# =============================================================================
    .globl lkf_init_R_cart
lkf_init_R_cart:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)

    mv      s0, a0

    # Zero R (69*69 = 4761 doubles)
    li      a1, 4761
    call    lkf_zero_matrix

    la      t2, .lkf_r_cart
    fld     ft1,  0(t2)             # EST_R_PX
    fld     ft2,  8(t2)             # EST_R_PY
    fld     ft3, 16(t2)             # EST_R_PZ

    li      s1, 0
    li      t6, 23
.lR_loop:
    bge     s1, t6, .lR_done
    li      t0, 3
    mul     t0, s1, t0              # rb = j*3

    # diagonal element stride for 69×69 matrix = 70*8 bytes
    li      t3, 70

    # R[rb+0][rb+0]
    mul     t4, t0, t3
    slli    t4, t4, 3
    add     t4, s0, t4
    fsd     ft1, 0(t4)

    # R[rb+1][rb+1]
    addi    t5, t0, 1
    mul     t4, t5, t3
    slli    t4, t4, 3
    add     t4, s0, t4
    fsd     ft2, 0(t4)

    # R[rb+2][rb+2]
    addi    t5, t0, 2
    mul     t4, t5, t3
    slli    t4, t4, 3
    add     t4, s0, t4
    fsd     ft3, 0(t4)

    addi    s1, s1, 1
    j       .lR_loop
.lR_done:
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 24(sp)
    addi    sp, sp, 32
    ret

# =============================================================================
# lkf_init_state -- Initialise joint j sub-state in x vector
# =============================================================================
# Args:
#   a0 = double *x
#   a1 = int joint
#   fa0= double px
#   fa1= double py
#   fa2= double pz
# =============================================================================
    .globl lkf_init_state
lkf_init_state:
    li      t0, 96
    mul     t0, a1, t0              # t0 = joint*12*8
    add     t0, a0, t0              # t0 = &x[joint*12]

    fcvt.d.w ft0, zero
    fsd     ft0,  0(t0)
    fsd     ft0,  8(t0)
    fsd     ft0, 16(t0)
    fsd     ft0, 24(t0)
    fsd     ft0, 32(t0)
    fsd     ft0, 40(t0)
    fsd     ft0, 48(t0)
    fsd     ft0, 56(t0)
    fsd     ft0, 64(t0)
    fsd     ft0, 72(t0)
    fsd     ft0, 80(t0)
    fsd     ft0, 88(t0)

    fsd     fa0,  0(t0)             # x[b+0]  = px
    fsd     fa1, 32(t0)             # x[b+4]  = py  (4*8=32)
    fsd     fa2, 64(t0)             # x[b+8]  = pz  (8*8=64)
    ret

# =============================================================================
# lkf_predict -- LKF prediction step
# =============================================================================
# Args:
#   a0 = double *x         (276-D, in/out)
#   a1 = double *P         (276×276, in/out)
#   a2 = const double *F   (276×276)
#   a3 = const double *Q   (276×276)
#   a4 = int N             (= 276)
#   a5 = double *tmp276    (276-double scratch)
#   a6 = double *tmpNN     (276×276-double scratch for F*P)
#
# x = F*x  (matrix-vector)
# P = F*P*F^T + Q  (two matrix-matrix multiplies + element-wise add)
# fmadd.d used throughout.
# =============================================================================
    .globl lkf_predict
lkf_predict:
    addi    sp, sp, -80
    sd      ra, 72(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    sd      s7, 56(sp)

    mv      s0, a0                  # x
    mv      s1, a1                  # P
    mv      s2, a2                  # F
    mv      s3, a3                  # Q
    mv      s4, a4                  # N = 276
    mv      s5, a5                  # tmp276
    mv      s6, a6                  # tmpNN  (F*P scratch)

    # ---- Step 1: tmp276 = F * x ----
    li      s7, 0                   # i
.lp_Fx_i:
    bge     s7, s4, .lp_Fx_done
    fcvt.d.w ft0, zero
    li      t5, 0                   # k
    # &F[i][0] = s2 + i*276*8
    mul     t0, s7, s4
    slli    t0, t0, 3
    add     t0, s2, t0
    mv      t1, s0                  # &x[0]
.lp_Fx_k:
    bge     t5, s4, .lp_Fx_kd
    fld     ft1, 0(t0)
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t0, t0, 8
    addi    t1, t1, 8
    addi    t5, t5, 1
    j       .lp_Fx_k
.lp_Fx_kd:
    slli    t0, s7, 3
    add     t0, s5, t0
    fsd     ft0, 0(t0)              # tmp276[i]
    addi    s7, s7, 1
    j       .lp_Fx_i
.lp_Fx_done:

    # Copy tmp276 -> x
    li      t0, 0
.lp_cpx:
    bge     t0, s4, .lp_cpx_done
    slli    t1, t0, 3
    add     t2, s5, t1
    fld     ft0, 0(t2)
    add     t2, s0, t1
    fsd     ft0, 0(t2)
    addi    t0, t0, 1
    j       .lp_cpx
.lp_cpx_done:

    # ---- Step 2: tmpNN = F * P  (276×276 × 276×276) ----
    li      s7, 0                   # i
.lp_FP_i:
    bge     s7, s4, .lp_FP_done
    li      t2, 0                   # j
.lp_FP_j:
    bge     t2, s4, .lp_FP_jd
    fcvt.d.w ft0, zero
    li      t3, 0                   # k
    # &F[i][0]
    mul     t0, s7, s4
    slli    t0, t0, 3
    add     t0, s2, t0
.lp_FP_k:
    bge     t3, s4, .lp_FP_kd
    fld     ft1, 0(t0)              # F[i][k]
    mul     t1, t3, s4
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s1, t1
    fld     ft2, 0(t1)              # P[k][j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t0, t0, 8
    addi    t3, t3, 1
    j       .lp_FP_k
.lp_FP_kd:
    mul     t3, s7, s4
    add     t3, t3, t2
    slli    t3, t3, 3
    add     t3, s6, t3
    fsd     ft0, 0(t3)              # tmpNN[i][j]
    addi    t2, t2, 1
    j       .lp_FP_j
.lp_FP_jd:
    addi    s7, s7, 1
    j       .lp_FP_i
.lp_FP_done:

    # ---- Step 3: P = tmpNN * F^T + Q ----
    # P[i][j] = sum_k tmpNN[i][k] * F[j][k]  + Q[i][j]
    li      s7, 0                   # i
.lp_PF_i:
    bge     s7, s4, .lp_PF_done
    li      t2, 0                   # j
.lp_PF_j:
    bge     t2, s4, .lp_PF_jd
    fcvt.d.w ft0, zero
    li      t3, 0                   # k
.lp_PF_k:
    bge     t3, s4, .lp_PF_kd
    # tmpNN[i][k]
    mul     t0, s7, s4
    add     t0, t0, t3
    slli    t0, t0, 3
    add     t0, s6, t0
    fld     ft1, 0(t0)
    # F[j][k]  (= F^T[k][j])
    mul     t1, t2, s4
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s2, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lp_PF_k
.lp_PF_kd:
    # Add Q[i][j]
    mul     t0, s7, s4
    add     t0, t0, t2
    slli    t0, t0, 3
    add     t0, s3, t0
    fld     ft1, 0(t0)
    fadd.d  ft0, ft0, ft1
    # Store P[i][j]
    mul     t0, s7, s4
    add     t0, t0, t2
    slli    t0, t0, 3
    add     t0, s1, t0
    fsd     ft0, 0(t0)
    addi    t2, t2, 1
    j       .lp_PF_j
.lp_PF_jd:
    addi    s7, s7, 1
    j       .lp_PF_i
.lp_PF_done:

    ld      s7, 56(sp)
    ld      s6, 48(sp)
    ld      s5, 40(sp)
    ld      s4, 32(sp)
    ld      s3, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 72(sp)
    addi    sp, sp, 80
    ret

# =============================================================================
# lkf_lu_inverse -- LU decomposition + back-substitution matrix inverse
# =============================================================================
# Args:
#   a0 = double *A      (n×n, OVERWRITTEN with LU factors)
#   a1 = double *Ainv   (n×n output)
#   a2 = int n          (= 69 for LKF)
#   a3 = int *pivot     (n ints, pivot index array)
# Return: a0 = 0 (OK) or -1 (singular)
#
# Algorithm: Gaussian elimination with partial pivoting.
# fnmsub.d used for elimination step: A[i][j] -= m * A[k][j]
# =============================================================================
    .globl lkf_lu_inverse
lkf_lu_inverse:
    addi    sp, sp, -80
    sd      ra, 72(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    fsd     fs0, 56(sp)
    fsd     fs1, 64(sp)

    mv      s0, a0                  # A
    mv      s1, a1                  # Ainv
    mv      s2, a2                  # n
    mv      s3, a3                  # pivot

    la      t0, .lkf_const_eps
    fld     fs1, 0(t0)              # fs1 = eps (singularity threshold)

    # ---- Phase 1: LU factorisation with partial pivoting ----
    li      s4, 0                   # col k
.lu_col:
    bge     s4, s2, .lu_factor_done

    # Find max |A[i][k]| for i >= k
    fcvt.d.w fs0, zero              # running max = 0
    mv      t5, s4                  # best pivot row
    mv      t0, s4
.lu_ps:                             # pivot search
    bge     t0, s2, .lu_pd
    mul     t1, t0, s2
    add     t1, t1, s4
    slli    t1, t1, 3
    add     t1, s0, t1
    fld     ft0, 0(t1)
    fabs.d  ft0, ft0
    flt.d   t2, fs0, ft0
    beq     t2, zero, .lu_psn
    fmv.d   fs0, ft0
    mv      t5, t0
.lu_psn:
    addi    t0, t0, 1
    j       .lu_ps
.lu_pd:

    # Store pivot index
    slli    t0, s4, 2
    add     t0, s3, t0
    sw      t5, 0(t0)

    # Singularity check
    flt.d   t2, fs0, fs1
    feq.d   t3, fs0, fs1
    or      t2, t2, t3
    beq     t2, zero, .lu_ns
    li      a0, -1
    j       .lu_ret
.lu_ns:

    # Swap rows k and t5
    beq     t5, s4, .lu_no_swap
    mv      s5, zero
.lu_sw:
    bge     s5, s2, .lu_no_swap
    mul     t0, s4, s2
    add     t0, t0, s5
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    mul     t1, t5, s2
    add     t1, t1, s5
    slli    t1, t1, 3
    add     t1, s0, t1
    fld     ft1, 0(t1)
    fsd     ft1, 0(t0)
    fsd     ft0, 0(t1)
    addi    s5, s5, 1
    j       .lu_sw
.lu_no_swap:

    # Pivot element A[k][k]
    mul     t0, s4, s2
    add     t0, t0, s4
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     fs0, 0(t0)              # fs0 = A[k][k]

    # Eliminate rows below k
    addi    s5, s4, 1               # i = k+1
.lu_er:
    bge     s5, s2, .lu_er_done
    mul     t0, s5, s2
    add     t0, t0, s4
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    fdiv.d  ft0, ft0, fs0           # m = A[i][k] / A[k][k]
    fsd     ft0, 0(t0)              # A[i][k] = m  (stores L)

    addi    s6, s4, 1               # j = k+1
.lu_ec:
    bge     s6, s2, .lu_ec_done
    mul     t1, s4, s2
    add     t1, t1, s6
    slli    t1, t1, 3
    add     t1, s0, t1
    fld     ft1, 0(t1)              # A[k][j]
    mul     t2, s5, s2
    add     t2, t2, s6
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft2, 0(t2)              # A[i][j]
    fnmsub.d ft2, ft0, ft1, ft2    # A[i][j] = A[i][j] - m*A[k][j]
    fsd     ft2, 0(t2)
    addi    s6, s6, 1
    j       .lu_ec
.lu_ec_done:
    addi    s5, s5, 1
    j       .lu_er
.lu_er_done:
    addi    s4, s4, 1
    j       .lu_col
.lu_factor_done:

    # ---- Phase 2: Solve A*X = I column by column ----
    li      s4, 0                   # column c of identity
.lu_sc:
    bge     s4, s2, .lu_all_done

    # Initialise Ainv column c: e_c
    li      t0, 0
.lu_rhs:
    bge     t0, s2, .lu_rhs_done
    mul     t1, t0, s2
    add     t1, t1, s4
    slli    t1, t1, 3
    add     t1, s1, t1
    fcvt.d.w ft0, zero
    beq     t0, s4, .lu_rhs1
    fsd     ft0, 0(t1)
    j       .lu_rhs_n
.lu_rhs1:
    la      t2, .lkf_const_one
    fld     ft0, 0(t2)
    fsd     ft0, 0(t1)
.lu_rhs_n:
    addi    t0, t0, 1
    j       .lu_rhs
.lu_rhs_done:

    # Apply row permutations
    li      t0, 0
.lu_perm:
    bge     t0, s2, .lu_perm_done
    slli    t1, t0, 2
    add     t1, s3, t1
    lw      t2, 0(t1)               # pivot row
    beq     t2, t0, .lu_pn
    mul     t3, t0, s2
    add     t3, t3, s4
    slli    t3, t3, 3
    add     t3, s1, t3
    fld     ft0, 0(t3)
    mul     t4, t2, s2
    add     t4, t4, s4
    slli    t4, t4, 3
    add     t4, s1, t4
    fld     ft1, 0(t4)
    fsd     ft1, 0(t3)
    fsd     ft0, 0(t4)
.lu_pn:
    addi    t0, t0, 1
    j       .lu_perm
.lu_perm_done:

    # Forward substitution: L*y = b (L has 1s on diagonal)
    li      t0, 1
.lu_fwd:
    bge     t0, s2, .lu_fwd_done
    fcvt.d.w ft0, zero
    li      t1, 0
.lu_fi:
    bge     t1, t0, .lu_fi_done
    mul     t2, t0, s2
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft1, 0(t2)              # L[i][j]
    mul     t3, t1, s2
    add     t3, t3, s4
    slli    t3, t3, 3
    add     t3, s1, t3
    fld     ft2, 0(t3)              # y[j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t1, t1, 1
    j       .lu_fi
.lu_fi_done:
    mul     t2, t0, s2
    add     t2, t2, s4
    slli    t2, t2, 3
    add     t2, s1, t2
    fld     ft1, 0(t2)
    fsub.d  ft1, ft1, ft0
    fsd     ft1, 0(t2)
    addi    t0, t0, 1
    j       .lu_fwd
.lu_fwd_done:

    # Back substitution: U*x = y
    addi    t0, s2, -1              # i = n-1
.lu_bck:
    bltz    t0, .lu_bck_done
    fcvt.d.w ft0, zero
    addi    t1, t0, 1               # j = i+1
.lu_bi:
    bge     t1, s2, .lu_bi_done
    mul     t2, t0, s2
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft1, 0(t2)              # U[i][j]
    mul     t3, t1, s2
    add     t3, t3, s4
    slli    t3, t3, 3
    add     t3, s1, t3
    fld     ft2, 0(t3)              # x[j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t1, t1, 1
    j       .lu_bi
.lu_bi_done:
    mul     t2, t0, s2
    add     t2, t2, t0
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft1, 0(t2)              # U[i][i]
    mul     t3, t0, s2
    add     t3, t3, s4
    slli    t3, t3, 3
    add     t3, s1, t3
    fld     ft2, 0(t3)
    fsub.d  ft2, ft2, ft0
    fdiv.d  ft2, ft2, ft1
    fsd     ft2, 0(t3)
    addi    t0, t0, -1
    j       .lu_bck
.lu_bck_done:
    addi    s4, s4, 1
    j       .lu_sc

.lu_all_done:
    li      a0, 0
.lu_ret:
    fld     fs1, 64(sp)
    fld     fs0, 56(sp)
    ld      s6, 48(sp)
    ld      s5, 40(sp)
    ld      s4, 32(sp)
    ld      s3, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 72(sp)
    addi    sp, sp, 80
    ret

# =============================================================================
# lkf_update -- LKF update step (linear Cartesian measurement model)
# =============================================================================
# Args (first 8 in registers, rest on stack relative to caller sp):
#   a0 = double *x             (276-D state, in/out)
#   a1 = double *P             (276×276 covariance, in/out)
#   a2 = const double *H       (69×276 fixed measurement matrix)
#   a3 = const double *R_cart  (69×69 Cartesian noise)
#   a4 = const double *meas_cart (NUM_JOINTS*3 = 69 doubles, flat Cartesian)
#   a5 = double *K_buf         (276×69 scratch)
#   a6 = double *S_buf         (69×69 scratch, filled then inverted)
#   a7 = double *PHt_buf       (276×69 scratch)
#   [sp+0]  = double *y_buf    (69 scratch — innovation vector)
#   [sp+8]  = double *Sinv_buf (69×69 scratch — S inverse)
#   [sp+16] = double *IKH_buf  (2×276×276 scratch — Joseph form)
#   [sp+24] = int     N_joints (= 23)
#   [sp+32] = int    *pivot_buf (69 ints)
#
# Steps:
#   1. Build z (69-D): z[j*3..j*3+2] = meas_cart[j*3..j*3+2]
#   2. y = z - H*x                                  (69-D innovation)
#   3. PHt = P * H^T                                (276×69)
#   4. S = H * PHt + R                              (69×69)
#   5. S^{-1} via LU  -> Sinv_buf
#   6. K = PHt * S^{-1}                             (276×69)
#   7. x += K * y
#   8. P = (I-K*H)*P*(I-K*H)^T + K*R*K^T  (Joseph form)
# =============================================================================
    .globl lkf_update
lkf_update:
    addi    sp, sp, -128
    sd      ra, 120(sp)
    sd      s0,   0(sp)
    sd      s1,   8(sp)
    sd      s2,  16(sp)
    sd      s3,  24(sp)
    sd      s4,  32(sp)
    sd      s5,  40(sp)
    sd      s6,  48(sp)
    sd      s7,  56(sp)
    sd      s8,  64(sp)
    sd      s9,  72(sp)
    sd      s10, 80(sp)
    sd      s11, 88(sp)
    fsd     fs0, 96(sp)
    fsd     fs1,104(sp)
    fsd     fs2,112(sp)

    mv      s0, a0              # x
    mv      s1, a1              # P
    mv      s2, a2              # H
    mv      s3, a3              # R_cart
    mv      s4, a4              # meas_cart
    mv      s5, a5              # K_buf
    mv      s6, a6              # S_buf
    mv      s7, a7              # PHt_buf

    # Stack args: caller's original sp = our sp+128
    ld      s8,  128(sp)        # y_buf
    ld      s9,  136(sp)        # Sinv_buf
    ld      s10, 144(sp)        # IKH_buf
    # s11 = N_joints, pivot loaded lazily

    # ---- Step 1: Build z = meas_cart flattened (already 69-D contiguous) ----
    # The Python copies meas[j*3..+2] -> z[j*3..+2].  Since meas_cart is
    # already flat and contiguous, z IS meas_cart.  We use s4 directly.

    # ---- Step 2: y = z - H*x  (69-D) ----
    # y[i] = z[i] - sum_j H[i][j]*x[j]
    li      t6, 69
    li      t5, 276
    li      t0, 0               # i
.lu_y_i:
    bge     t0, t6, .lu_y_done
    fcvt.d.w ft0, zero
    li      t1, 0               # j
    mul     t2, t0, t5          # H row byte start (element offset)
    slli    t2, t2, 3
    add     t2, s2, t2          # &H[i][0]
    mv      t3, s0              # &x[0]
.lu_y_k:
    bge     t1, t5, .lu_y_kd
    fld     ft1, 0(t2)          # H[i][j]
    fld     ft2, 0(t3)          # x[j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t2, t2, 8
    addi    t3, t3, 8
    addi    t1, t1, 1
    j       .lu_y_k
.lu_y_kd:
    # y[i] = z[i] - Hx[i]
    slli    t1, t0, 3
    add     t2, s4, t1          # z[i]
    fld     ft1, 0(t2)
    fsub.d  ft0, ft1, ft0
    add     t2, s8, t1
    fsd     ft0, 0(t2)          # y[i]
    addi    t0, t0, 1
    j       .lu_y_i
.lu_y_done:

    # ---- Step 3: PHt = P * H^T  (276×276 * 276×69 = 276×69) ----
    # H is 69×276 so H^T is 276×69.
    # PHt[i][j] = sum_k P[i][k] * H[j][k]  (H^T[k][j] = H[j][k])
    li      s11, 276            # row count
    li      t6, 69
    li      t0, 0               # i
.lu_PHt_i:
    bge     t0, s11, .lu_PHt_done
    li      t2, 0               # j
.lu_PHt_j:
    bge     t2, t6, .lu_PHt_jd
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.lu_PHt_k:
    bge     t3, s11, .lu_PHt_kd
    mul     t4, t0, s11
    add     t4, t4, t3
    slli    t4, t4, 3
    add     t4, s1, t4
    fld     ft1, 0(t4)          # P[i][k]
    mul     t4, t2, s11
    add     t4, t4, t3
    slli    t4, t4, 3
    add     t4, s2, t4
    fld     ft2, 0(t4)          # H[j][k] = H^T[k][j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_PHt_k
.lu_PHt_kd:
    mul     t4, t0, t6
    add     t4, t4, t2
    slli    t4, t4, 3
    add     t4, s7, t4
    fsd     ft0, 0(t4)          # PHt[i][j]
    addi    t2, t2, 1
    j       .lu_PHt_j
.lu_PHt_jd:
    addi    t0, t0, 1
    j       .lu_PHt_i
.lu_PHt_done:

    # ---- Step 4: S = H * PHt + R  (69×276 * 276×69 + 69×69 = 69×69) ----
    li      t6, 69
    li      t5, 276
    li      t0, 0               # i
.lu_S_i:
    bge     t0, t6, .lu_S_done
    li      t2, 0               # j
.lu_S_j:
    bge     t2, t6, .lu_S_jd
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.lu_S_k:
    bge     t3, t5, .lu_S_kd
    mul     t4, t0, t5
    add     t4, t4, t3
    slli    t4, t4, 3
    add     t4, s2, t4
    fld     ft1, 0(t4)          # H[i][k]
    mul     t4, t3, t6
    add     t4, t4, t2
    slli    t4, t4, 3
    add     t4, s7, t4
    fld     ft2, 0(t4)          # PHt[k][j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_S_k
.lu_S_kd:
    # Add R[i][j]
    mul     t3, t0, t6
    add     t3, t3, t2
    slli    t3, t3, 3
    add     t3, s3, t3
    fld     ft1, 0(t3)
    fadd.d  ft0, ft0, ft1
    # Store S[i][j]
    mul     t3, t0, t6
    add     t3, t3, t2
    slli    t3, t3, 3
    add     t3, s6, t3
    fsd     ft0, 0(t3)
    addi    t2, t2, 1
    j       .lu_S_j
.lu_S_jd:
    addi    t0, t0, 1
    j       .lu_S_i
.lu_S_done:

    # ---- Step 5: S^{-1} via LU  (s6 = S -> LU factors; s9 = Sinv) ----
    ld      t6, 160(sp)         # pivot_buf  (original sp+32 = our sp+32+128=160)
    mv      a0, s6              # A = S (overwritten with LU)
    mv      a1, s9              # Ainv
    li      a2, 69
    mv      a3, t6
    call    lkf_lu_inverse
    beq     a0, zero, .lu_inv_ok
    j       .lu_upd_done        # singular: skip update
.lu_inv_ok:

    # ---- Step 6: K = PHt * S^{-1}  (276×69 * 69×69 = 276×69) ----
    li      t6, 276
    li      t5, 69
    li      t0, 0               # i
.lu_K_i:
    bge     t0, t6, .lu_K_done
    li      t2, 0               # j
.lu_K_j:
    bge     t2, t5, .lu_K_jd
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.lu_K_k:
    bge     t3, t5, .lu_K_kd
    mul     t4, t0, t5
    add     t4, t4, t3
    slli    t4, t4, 3
    add     t4, s7, t4
    fld     ft1, 0(t4)          # PHt[i][k]
    mul     t4, t3, t5
    add     t4, t4, t2
    slli    t4, t4, 3
    add     t4, s9, t4
    fld     ft2, 0(t4)          # Sinv[k][j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_K_k
.lu_K_kd:
    mul     t4, t0, t5
    add     t4, t4, t2
    slli    t4, t4, 3
    add     t4, s5, t4
    fsd     ft0, 0(t4)          # K[i][j]
    addi    t2, t2, 1
    j       .lu_K_j
.lu_K_jd:
    addi    t0, t0, 1
    j       .lu_K_i
.lu_K_done:

    # ---- Step 7: x += K * y  (276×69 * 69×1 = 276×1) ----
    li      t6, 276
    li      t5, 69
    li      t0, 0               # i
.lu_xu_i:
    bge     t0, t6, .lu_xu_done
    fcvt.d.w ft0, zero
    li      t1, 0               # k
.lu_xu_k:
    bge     t1, t5, .lu_xu_kd
    mul     t2, t0, t5
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t2, s5, t2
    fld     ft1, 0(t2)          # K[i][k]
    slli    t2, t1, 3
    add     t2, s8, t2
    fld     ft2, 0(t2)          # y[k]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t1, t1, 1
    j       .lu_xu_k
.lu_xu_kd:
    slli    t1, t0, 3
    add     t1, s0, t1
    fld     ft1, 0(t1)
    fadd.d  ft1, ft1, ft0
    fsd     ft1, 0(t1)
    addi    t0, t0, 1
    j       .lu_xu_i
.lu_xu_done:

    # ---- Step 8: Joseph-form covariance update ----
    # P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
    #
    # 8a. IKH = I  (276×276)  -- stored in first half of IKH_buf
    mv      t6, s10             # IKH_buf
    li      t0, 76176
    mv      t1, t6
    fcvt.d.w ft0, zero
.lu_IKH_z:
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    bne     t0, zero, .lu_IKH_z
    la      t0, .lkf_const_one
    fld     fs0, 0(t0)
    li      t0, 0
    li      t2, 2216            # 277*8
    mv      t1, t6
.lu_IKH_d:
    li      t3, 276
    bge     t0, t3, .lu_IKH_dd
    fsd     fs0, 0(t1)
    add     t1, t1, t2
    addi    t0, t0, 1
    j       .lu_IKH_d
.lu_IKH_dd:

    # 8b. IKH -= K*H  (276×276)
    li      t5, 276
    li      t4, 69
    li      t0, 0               # i
.lu_KH_i:
    bge     t0, t5, .lu_KH_done
    li      t2, 0               # j
.lu_KH_j:
    bge     t2, t5, .lu_KH_jd
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.lu_KH_k:
    bge     t3, t4, .lu_KH_kd
    mul     t1, t0, t4
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s5, t1
    fld     ft1, 0(t1)          # K[i][k]
    mul     t1, t3, t5
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s2, t1
    fld     ft2, 0(t1)          # H[k][j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_KH_k
.lu_KH_kd:
    mul     t1, t0, t5
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, t6, t1
    fld     ft1, 0(t1)          # IKH[i][j]
    fsub.d  ft1, ft1, ft0
    fsd     ft1, 0(t1)
    addi    t2, t2, 1
    j       .lu_KH_j
.lu_KH_jd:
    addi    t0, t0, 1
    j       .lu_KH_i
.lu_KH_done:

    # 8c. tmp1 = IKH * P  -> second half of IKH_buf
    li      t5, 76176
    slli    t5, t5, 3
    add     s11, t6, t5         # s11 = IKH_buf + 276*276*8

    li      t5, 276
    li      t0, 0               # i
.lu_IKHP_i:
    bge     t0, t5, .lu_IKHP_done
    li      t2, 0               # j
.lu_IKHP_j:
    bge     t2, t5, .lu_IKHP_jd
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.lu_IKHP_k:
    bge     t3, t5, .lu_IKHP_kd
    mul     t1, t0, t5
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, t6, t1
    fld     ft1, 0(t1)          # IKH[i][k]
    mul     t1, t3, t5
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s1, t1
    fld     ft2, 0(t1)          # P[k][j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_IKHP_k
.lu_IKHP_kd:
    mul     t1, t0, t5
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s11, t1
    fsd     ft0, 0(t1)          # tmp1[i][j]
    addi    t2, t2, 1
    j       .lu_IKHP_j
.lu_IKHP_jd:
    addi    t0, t0, 1
    j       .lu_IKHP_i
.lu_IKHP_done:

    # 8d. P = tmp1 * IKH^T  -> stored in s1 (P)
    li      t5, 276
    li      t0, 0
.lu_PF_i:
    bge     t0, t5, .lu_PF_done
    li      t2, 0
.lu_PF_j:
    bge     t2, t5, .lu_PF_jd
    fcvt.d.w ft0, zero
    li      t3, 0
.lu_PF_k:
    bge     t3, t5, .lu_PF_kd
    mul     t1, t0, t5
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s11, t1
    fld     ft1, 0(t1)          # tmp1[i][k]
    mul     t1, t2, t5          # IKH^T[k][j] = IKH[j][k]
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, t6, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_PF_k
.lu_PF_kd:
    mul     t1, t0, t5
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s1, t1
    fsd     ft0, 0(t1)
    addi    t2, t2, 1
    j       .lu_PF_j
.lu_PF_jd:
    addi    t0, t0, 1
    j       .lu_PF_i
.lu_PF_done:

    # 8e. KR = K * R  (276×69 * 69×69 = 276×69)  -> PHt_buf reuse
    li      t5, 276
    li      t4, 69
    li      t0, 0
.lu_KR_i:
    bge     t0, t5, .lu_KR_done
    li      t2, 0
.lu_KR_j:
    bge     t2, t4, .lu_KR_jd
    fcvt.d.w ft0, zero
    li      t3, 0
.lu_KR_k:
    bge     t3, t4, .lu_KR_kd
    mul     t1, t0, t4
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s5, t1
    fld     ft1, 0(t1)          # K[i][k]
    mul     t1, t3, t4
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s3, t1
    fld     ft2, 0(t1)          # R[k][j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_KR_k
.lu_KR_kd:
    mul     t1, t0, t4
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s7, t1          # reuse PHt_buf for KR
    fsd     ft0, 0(t1)
    addi    t2, t2, 1
    j       .lu_KR_j
.lu_KR_jd:
    addi    t0, t0, 1
    j       .lu_KR_i
.lu_KR_done:

    # 8f. P += KR * K^T  (276×69 * 69×276 = 276×276, add into P)
    li      t5, 276
    li      t4, 69
    li      t0, 0
.lu_KKT_i:
    bge     t0, t5, .lu_KKT_done
    li      t2, 0
.lu_KKT_j:
    bge     t2, t5, .lu_KKT_jd
    fcvt.d.w ft0, zero
    li      t3, 0
.lu_KKT_k:
    bge     t3, t4, .lu_KKT_kd
    mul     t1, t0, t4
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s7, t1          # KR[i][k]
    fld     ft1, 0(t1)
    mul     t1, t2, t4          # K^T[k][j] = K[j][k]
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s5, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .lu_KKT_k
.lu_KKT_kd:
    mul     t1, t0, t5
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s1, t1
    fld     ft1, 0(t1)
    fadd.d  ft1, ft1, ft0
    fsd     ft1, 0(t1)
    addi    t2, t2, 1
    j       .lu_KKT_j
.lu_KKT_jd:
    addi    t0, t0, 1
    j       .lu_KKT_i
.lu_KKT_done:

.lu_upd_done:
    fld     fs2,112(sp)
    fld     fs1,104(sp)
    fld     fs0, 96(sp)
    ld      s11, 88(sp)
    ld      s10, 80(sp)
    ld      s9,  72(sp)
    ld      s8,  64(sp)
    ld      s7,  56(sp)
    ld      s6,  48(sp)
    ld      s5,  40(sp)
    ld      s4,  32(sp)
    ld      s3,  24(sp)
    ld      s2,  16(sp)
    ld      s1,   8(sp)
    ld      s0,   0(sp)
    ld      ra,  120(sp)
    addi    sp, sp, 128
    ret

# =============================================================================
# End of lkf_asm.s
# =============================================================================
