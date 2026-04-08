# =============================================================================
# ekf_asm.s  --  Extended Kalman Filter  (RISC-V RV64GD scalar assembly)
# =============================================================================
#
# Mirrors kalman-updated.py §§ EKF helpers + EKF class exactly:
#   • State layout per joint j (base = j*12):
#       [px, vx, ax, jx,  py, vy, ay, jy,  pz, vz, az, jz]
#   • Global state vector x  : R^276  (double precision, row-major)
#   • Global cov matrix   P  : R^{276×276}
#   • State transition    F  : R^{276×276}  (block-diagonal, built once)
#   • Process noise       Q  : R^{276×276}  (block-diagonal, built once)
#   • Spherical meas. noise R: R^{69×69}    (block-diagonal, built once)
#   • Kalman gain         K  : R^{276×69}   (scratch, per-step)
#   • Innovation cov      S  : R^{69×69}    (scratch, per-step)
#
# Public entry points (C-compatible ABI, a0/a1/… for args, fa0 for fp ret):
#
#   ekf_init_F(double *F, double dt)
#       Build F (276×276) block-diagonal constant-jerk state transition.
#
#   ekf_init_Q(double *Q)
#       Build Q (276×276) block-diagonal process noise.
#
#   ekf_init_R_sph(double *R)
#       Build spherical measurement noise R (69×69).
#
#   ekf_init_state(double *x, int joint, double px, double py, double pz)
#       Set x[joint*12..+11] with px/py/pz, zeros elsewhere.
#
#   ekf_predict(double *x, double *P, const double *F,
#               const double *Q, int N)
#       x  = F*x   (in-place, uses tmp buf on stack)
#       P  = F*P*F^T + Q
#
#   ekf_fast_atan2(double y, double x)  -> fa0
#       Scheinerman-Lyons polynomial atan2 (matches fast_atan2 in .py).
#
#   ekf_wrap_angle(double a)  -> fa0
#       Wrap angle to [-pi, pi].
#
#   ekf_compute_h(const double *x, double *h_out)
#       Nonlinear measurement function h : R^276 -> R^69.
#       Uses ekf_fast_atan2; no libm calls.
#
#   ekf_compute_jacobian(const double *x, double *Hk)
#       Jacobian Hk = dh/dx (69×276, block-diagonal).
#
#   ekf_update(double *x, double *P,
#              const double *F, const double *Q,
#              const double *R_sph,
#              const double *meas_cart,
#              double *K_buf, double *S_buf, double *tmp276,
#              int N_joints)
#       Full EKF update step (predict already done externally):
#           nu = z_sph - h(x_hat)  (wrap angular channels)
#           Hk = dh/dx|_{x_hat}
#           S  = Hk*P*Hk^T + R
#           S^{-1} via Gaussian elimination (LU, partial pivot)
#           K  = P*Hk^T*S^{-1}
#           x  = x + K*nu
#           P  = (I-K*Hk)*P*(I-K*Hk)^T + K*R*K^T  (Joseph form)
#
#   ekf_lu_inverse(double *A, double *Ainv, int n)
#       In-place LU decomposition + back-substitution matrix inverse.
#       Same algorithm as mat_inverse_nxn in Python (numpy.linalg.inv
#       wraps LAPACK dgetrf/dgetri, which is LU with partial pivoting).
#       Returns 0 in a0 if OK, -1 if singular.
#
# =============================================================================
# ABI / register conventions used throughout this file
# =============================================================================
# Caller-saved (temporaries, NOT preserved across calls):
#   a0-a7   : integer args / return
#   fa0-fa7 : FP args / return
#   t0-t6   : integer temporaries
#   ft0-ft11: FP temporaries
#
# Callee-saved (preserved across calls, saved/restored on stack):
#   s0-s11  : integer saved
#   fs0-fs11: FP saved
#   ra      : return address
#   sp      : stack pointer
#
# All matrices are stored in ROW-MAJOR order (C convention).
# Element (i,j) of an M×N double matrix at base ptr p:
#   address = p + (i*N + j)*8
# =============================================================================

    .section .rodata
    .align 3

# ---------------------------------------------------------------------------
# Double-precision constants (IEEE-754, 64-bit)
# ---------------------------------------------------------------------------
.ekf_const_pi:
    .double  3.14159265358979323846
.ekf_const_pi2:
    .double  1.57079632679489661923
.ekf_const_pi4:
    .double  0.78539816339744830962
.ekf_const_2pi:
    .double  6.28318530717958647692
.ekf_const_c1:
    .double  0.2447                   # atan2 polynomial coeff
.ekf_const_c2:
    .double  0.0663                   # atan2 polynomial coeff
.ekf_const_eps_r:
    .double  1.0e-10                  # EPSILON_R
.ekf_const_eps_rho:
    .double  1.0e-10                  # EPSILON_RHO
.ekf_const_half:
    .double  0.5
.ekf_const_sixth:
    .double  0.16666666666666666667
.ekf_const_dt:                        # default 1/30 — overridden by caller
    .double  0.03333333333333333333
# Q noise per state index mod 4
.ekf_q_noise:
    .double  1.0e-6   # index 0 : position
    .double  1.0e-5   # index 1 : velocity
    .double  1.0e-4   # index 2 : acceleration
    .double  1.0e-4   # index 3 : jerk
# R spherical diagonal values
.ekf_r_sph:
    .double  0.05     # sigma2_r
    .double  0.001    # sigma2_theta
    .double  0.001    # sigma2_phi
# LKF Cartesian R diagonal (kept here for completeness, unused by EKF)
.ekf_r_cart:
    .double  0.29472279  # sigma2_px
    .double  0.09632091  # sigma2_py
    .double  0.00204269  # sigma2_pz
.ekf_const_one:
    .double  1.0
.ekf_const_zero:
    .double  0.0
.ekf_const_neg_one:
    .double  -1.0

    .section .text
    .align 2

# =============================================================================
# ekf_init_F -- Build 276×276 block-diagonal state transition matrix F
# =============================================================================
# Args:
#   a0 = double *F      (caller-allocated 276*276*8 = 608448 bytes)
#   fa0= double  dt
# Destroys: t0-t6, ft0-ft11
# Preserves: s0-s11, fs0-fs11, ra
# =============================================================================
# F is identity + upper-triangle offsets per 4×4 kinematic sub-block:
#   F[r,   r+1] = dt
#   F[r,   r+2] = dt^2/2
#   F[r,   r+3] = dt^3/6
#   F[r+1, r+2] = dt
#   F[r+1, r+3] = dt^2/2
#   F[r+2, r+3] = dt
# There are 23 joints × 3 axes = 69 such 4×4 sub-blocks.
# =============================================================================
    .globl ekf_init_F
ekf_init_F:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    fsd     fs0, 32(sp)
    fsd     fs1, 40(sp)
    fsd     fs2, 48(sp)

    mv      s0, a0              # s0 = F base ptr

    # --- Zero entire F (276*276 = 76176 doubles) ---
    li      t0, 76176
    mv      t1, s0
.F_zero_loop:
    fsd     ft0, 0(t1)          # ft0 is 0.0 (uninitialized but we treat as 0)
    # Use a proper zero:
    fcvt.d.w ft0, zero
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    bne     t0, zero, .F_zero_loop

    # --- Set diagonal to 1.0 ---
    la      t2, .ekf_const_one
    fld     fs0, 0(t2)          # fs0 = 1.0
    mv      t1, s0
    li      t0, 276             # 276 diagonal elements
    li      t3, 276             # stride = 276*8+8 = (276+1)*8
    li      t4, 1
    add     t3, t3, t4          # t3 = 277
    slli    t3, t3, 3           # t3 = 277*8 (bytes)
.F_diag_loop:
    fsd     fs0, 0(t1)
    add     t1, t1, t3
    addi    t0, t0, -1
    bne     t0, zero, .F_diag_loop

    # --- Load dt and derived powers ---
    la      t2, .ekf_const_half
    fld     ft1, 0(t2)          # ft1 = 0.5
    la      t2, .ekf_const_sixth
    fld     ft2, 0(t2)          # ft2 = 1/6

    fmul.d  fs1, fa0, fa0       # fs1 = dt^2
    fmul.d  ft3, fs1, ft1       # ft3 = dt^2/2
    fmul.d  ft4, fs1, fa0       # ft4 = dt^3
    fmul.d  ft5, ft4, ft2       # ft5 = dt^3/6

    # fs0 still = 1.0, reuse as fa0's alias for dt
    # We'll use fa0 = dt, ft3 = dt2/2, ft5 = dt3/6, fs1 half-dt2
    # Compute dt2/2 and dt3/6 again into stable regs:
    # fa0 = dt (argument reg -- safe until first call within function)
    # Store dt, dt2, dt3 into callee-saved FP regs
    fmv.d   fs0, fa0            # fs0 = dt
    fmv.d   fs1, ft3            # fs1 = dt^2/2
    fmv.d   fs2, ft5            # fs2 = dt^3/6

    # stride constants for 276-column matrix
    # row stride = 276 doubles = 2208 bytes
    li      s1, 2208            # s1 = ROW_STRIDE = 276*8

    # For each of 23 joints, 3 axes -> 69 4×4 sub-blocks
    li      s2, 0               # joint counter
    li      s3, 23
.F_joint_loop:
    # base row/col index r = joint*12 + axis*4
    # we iterate axis=0,1,2 inside the inner loop
    li      t0, 0               # axis counter
.F_axis_loop:
    # r = s2*12 + t0*4
    li      t1, 12
    mul     t1, s2, t1          # t1 = joint*12
    slli    t2, t0, 2           # t2 = axis*4
    add     t1, t1, t2          # t1 = r (row/col index)

    # Compute byte address of F[r][r]
    # addr = F + (r*276 + r)*8 = F + r*(276+1)*8
    li      t3, 277
    mul     t4, t1, t3          # t4 = r*277 (element offset)
    slli    t4, t4, 3           # t4 = r*277*8 (byte offset)
    add     t4, s0, t4          # t4 = &F[r][r]

    # F[r, r+1] = dt    -> offset from F[r][r]: +8
    fsd     fs0, 8(t4)

    # F[r, r+2] = dt2/2 -> offset: +16
    fsd     fs1, 16(t4)

    # F[r, r+3] = dt3/6 -> offset: +24
    fsd     fs2, 24(t4)

    # F[r+1, r+1] = 1.0 already set; F[r+1, r+2] = dt
    # addr of F[r+1][r+2] = addr of F[r+1][r] + 2*8
    # F[r+1] base col r: t4 + 1*ROW_STRIDE - 1*8  ??? 
    # Recompute for row r+1:
    li      t3, 277
    addi    t5, t1, 1           # t5 = r+1
    mul     t6, t5, t3
    slli    t6, t6, 3
    add     t6, s0, t6          # t6 = &F[r+1][r+1]
    # F[r+1, r+2] = dt -> offset +8
    fsd     fs0, 8(t6)
    # F[r+1, r+3] = dt2/2 -> offset +16
    fsd     fs1, 16(t6)

    # F[r+2, r+3] = dt -> row r+2
    addi    t5, t1, 2           # t5 = r+2
    mul     t6, t5, t3
    slli    t6, t6, 3
    add     t6, s0, t6          # t6 = &F[r+2][r+2]
    # F[r+2, r+3] = dt -> offset +8
    fsd     fs0, 8(t6)

    addi    t0, t0, 1
    li      t3, 3
    blt     t0, t3, .F_axis_loop

    addi    s2, s2, 1
    blt     s2, s3, .F_joint_loop

    fld     fs2, 48(sp)
    fld     fs1, 40(sp)
    fld     fs0, 32(sp)
    ld      s3, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret

# =============================================================================
# ekf_init_Q -- Build 276×276 block-diagonal process noise matrix Q
# =============================================================================
# Args:  a0 = double *Q
# Noise per state-index mod 4: {0:1e-6, 1:1e-5, 2:1e-4, 3:1e-4}
# =============================================================================
    .globl ekf_init_Q
ekf_init_Q:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    fsd     fs0, 24(sp)
    fsd     fs1, 32(sp)

    mv      s0, a0              # Q base ptr

    # Zero Q
    li      t0, 76176
    mv      t1, s0
    fcvt.d.w ft0, zero
.Q_zero_loop:
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    bne     t0, zero, .Q_zero_loop

    # Load noise table
    la      t2, .ekf_q_noise
    fld     ft1, 0(t2)          # ft1 = 1e-6 (position)
    fld     ft2, 8(t2)          # ft2 = 1e-5 (velocity)
    fld     ft3,16(t2)          # ft3 = 1e-4 (acceleration)
    fld     ft4,24(t2)          # ft4 = 1e-4 (jerk)

    # stride for diagonal walk: (276+1)*8 = 277*8 = 2216
    li      s1, 2216            # diagonal stride in bytes

    li      s2, 0               # state index i = 0
    li      t5, 276             # total states
.Q_diag_loop:
    # compute element offset: (i*276 + i)*8 = i*277*8
    li      t3, 277
    mul     t4, s2, t3
    slli    t4, t4, 3
    add     t4, s0, t4          # &Q[i][i]

    # select noise based on i % 4
    andi    t6, s2, 3           # t6 = i mod 4
    beq     t6, zero, .Q_case0
    li      t0, 1
    beq     t6, t0, .Q_case1
    li      t0, 2
    beq     t6, t0, .Q_case2
    # case 3
    fsd     ft4, 0(t4)
    j       .Q_diag_next
.Q_case0:
    fsd     ft1, 0(t4)
    j       .Q_diag_next
.Q_case1:
    fsd     ft2, 0(t4)
    j       .Q_diag_next
.Q_case2:
    fsd     ft3, 0(t4)
.Q_diag_next:
    addi    s2, s2, 1
    blt     s2, t5, .Q_diag_loop

    fld     fs1, 32(sp)
    fld     fs0, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 40(sp)
    addi    sp, sp, 48
    ret

# =============================================================================
# ekf_init_R_sph -- Build 69×69 spherical measurement noise matrix R
# =============================================================================
# Args:  a0 = double *R   (caller-alloc, 69*69*8 = 38088 bytes)
# 23 identical 3×3 diagonal blocks: diag(0.05, 0.001, 0.001)
# =============================================================================
    .globl ekf_init_R_sph
ekf_init_R_sph:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)

    mv      s0, a0

    # Zero R (69*69 = 4761 doubles)
    li      t0, 4761
    mv      t1, s0
    fcvt.d.w ft0, zero
.Rsph_zero_loop:
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    bne     t0, zero, .Rsph_zero_loop

    # Load sigma2 values
    la      t2, .ekf_r_sph
    fld     ft1,  0(t2)         # 0.05  sigma2_r
    fld     ft2,  8(t2)         # 0.001 sigma2_theta
    fld     ft3, 16(t2)         # 0.001 sigma2_phi

    li      s1, 0               # joint counter
    li      t5, 23
.Rsph_joint_loop:
    # row base = joint*3; col base = joint*3
    # R[rb+0][rb+0] = 0.05
    # R[rb+1][rb+1] = 0.001
    # R[rb+2][rb+2] = 0.001
    li      t0, 3
    mul     t0, s1, t0          # t0 = rb = joint*3
    # Element offset for (rb+k, rb+k): (rb+k)*69 + (rb+k) = (rb+k)*70
    li      t3, 70
    mul     t4, t0, t3          # (rb)*70
    slli    t4, t4, 3
    add     t4, s0, t4          # &R[rb][rb]
    fsd     ft1, 0(t4)          # sigma2_r

    addi    t0, t0, 1
    mul     t4, t0, t3
    slli    t4, t4, 3
    add     t4, s0, t4
    fsd     ft2, 0(t4)          # sigma2_theta

    addi    t0, t0, 1
    mul     t4, t0, t3
    slli    t4, t4, 3
    add     t4, s0, t4
    fsd     ft3, 0(t4)          # sigma2_phi

    addi    s1, s1, 1
    blt     s1, t5, .Rsph_joint_loop

    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 24(sp)
    addi    sp, sp, 32
    ret

# =============================================================================
# ekf_init_state -- Initialise joint j's sub-state in global x vector
# =============================================================================
# Args:
#   a0 = double *x
#   a1 = int joint
#   fa0= double px
#   fa1= double py
#   fa2= double pz
# =============================================================================
    .globl ekf_init_state
ekf_init_state:
    # base = joint*12*8 = joint*96
    li      t0, 96
    mul     t0, a1, t0
    add     t0, a0, t0          # t0 = &x[joint*12]

    # Zero all 12 doubles
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

    # x[b+0]=px, x[b+4]=py, x[b+8]=pz
    fsd     fa0,  0(t0)   # px -> x[b+0]
    fsd     fa1, 32(t0)   # py -> x[b+4]  (4*8=32)
    fsd     fa2, 64(t0)   # pz -> x[b+8]  (8*8=64)
    ret

# =============================================================================
# ekf_fast_atan2 -- Scheinerman-Lyons polynomial atan2 approximation
# =============================================================================
# Args:   fa0 = y,  fa1 = x
# Return: fa0 = atan2(y,x)
# Max error ~0.021 deg. Exactly mirrors fast_atan2() in Python.
#
# Algorithm:
#   ax = |x|, ay = |y|
#   if ax >= ay:
#       z = ay/ax
#       angle = (pi/4)*z - z*(z-1)*(0.2447 + 0.0663*z)
#   else:
#       z = ax/ay
#       angle = pi/2 - [(pi/4)*z - z*(z-1)*(0.2447 + 0.0663*z)]
#   Quadrant correction using sign(x) and sign(y).
# =============================================================================
    .globl ekf_fast_atan2
ekf_fast_atan2:
    addi    sp, sp, -48
    sd      ra, 40(sp)
    fsd     fs0,  0(sp)
    fsd     fs1,  8(sp)
    fsd     fs2, 16(sp)
    fsd     fs3, 24(sp)
    fsd     fs4, 32(sp)

    # Load constants
    la      t0, .ekf_const_pi
    fld     fs0, 0(t0)          # fs0 = pi
    la      t0, .ekf_const_pi2
    fld     fs1, 0(t0)          # fs1 = pi/2
    la      t0, .ekf_const_pi4
    fld     fs2, 0(t0)          # fs2 = pi/4
    la      t0, .ekf_const_c1
    fld     fs3, 0(t0)          # fs3 = 0.2447
    la      t0, .ekf_const_c2
    fld     fs4, 0(t0)          # fs4 = 0.0663

    fcvt.d.w ft0, zero          # ft0 = 0.0

    # Check (x==0 && y==0) -> return 0
    feq.d   t0, fa0, ft0
    feq.d   t1, fa1, ft0
    and     t2, t0, t1
    beq     t2, zero, .atan2_nonzero
    fmv.d   fa0, ft0
    j       .atan2_done

.atan2_nonzero:
    # ax = |x|, ay = |y|  (using fabs via sign clear)
    fabs.d  ft1, fa1            # ft1 = ax = |x|
    fabs.d  ft2, fa0            # ft2 = ay = |y|

    # if ax >= ay -> branch .atan2_case_ax
    flt.d   t0, ft2, ft1        # t0 = (ay < ax)  -> ax > ay
    feq.d   t1, ft1, ft2        # t1 = (ax == ay)
    or      t0, t0, t1          # t0 = (ax >= ay)
    bne     t0, zero, .atan2_case_ax

.atan2_case_ay:
    # z = ax/ay
    fdiv.d  ft3, ft1, ft2       # ft3 = z = ax/ay
    j       .atan2_poly

.atan2_case_ax:
    # z = ay/ax
    fdiv.d  ft3, ft2, ft1       # ft3 = z = ay/ax

.atan2_poly:
    # angle = (pi/4)*z - z*(z-1)*(0.2447 + 0.0663*z)
    # Compute using fmadd where possible
    #   inner = 0.2447 + 0.0663*z
    fmadd.d ft4, fs4, ft3, fs3  # ft4 = 0.0663*z + 0.2447
    #   z_minus_1 = z - 1
    la      t2, .ekf_const_one
    fld     ft5, 0(t2)
    fsub.d  ft5, ft3, ft5       # ft5 = z-1
    #   term2 = z*(z-1)*inner
    fmul.d  ft6, ft3, ft5       # ft6 = z*(z-1)
    fmul.d  ft6, ft6, ft4       # ft6 = z*(z-1)*inner
    #   angle = (pi/4)*z - term2
    fmul.d  ft7, fs2, ft3       # ft7 = (pi/4)*z
    fsub.d  ft7, ft7, ft6       # ft7 = angle (before quadrant corr)

    # if case_ay: angle = pi/2 - angle
    bne     t0, zero, .atan2_quad  # t0=1 means case_ax, skip adjustment
    fsub.d  ft7, fs1, ft7       # angle = pi/2 - angle

.atan2_quad:
    # Quadrant correction using fa1 (x) and fa0 (y) signs
    flt.d   t0, fa1, ft0        # t0 = (x < 0)
    beq     t0, zero, .atan2_check_y_neg

    # x < 0
    flt.d   t1, ft0, fa0        # t1 = (y >= 0) i.e. (0 < y)
    feq.d   t2, fa0, ft0        # t2 = (y == 0)
    or      t1, t1, t2          # t1 = (y >= 0)
    bne     t1, zero, .atan2_x_neg_y_ge0
    # y < 0: angle = angle - pi
    fsub.d  ft7, ft7, fs0
    j       .atan2_store
.atan2_x_neg_y_ge0:
    # y >= 0: angle = pi - angle
    fsub.d  ft7, fs0, ft7
    j       .atan2_store

.atan2_check_y_neg:
    # x >= 0
    flt.d   t0, fa0, ft0        # t0 = (y < 0)
    beq     t0, zero, .atan2_store
    # y < 0: angle = -angle
    fneg.d  ft7, ft7

.atan2_store:
    fmv.d   fa0, ft7

.atan2_done:
    fld     fs4, 32(sp)
    fld     fs3, 24(sp)
    fld     fs2, 16(sp)
    fld     fs1,  8(sp)
    fld     fs0,  0(sp)
    ld      ra, 40(sp)
    addi    sp, sp, 48
    ret

# =============================================================================
# ekf_wrap_angle -- Wrap angle to [-pi, pi]
# =============================================================================
# Args:   fa0 = a
# Return: fa0 = wrapped a
# Mirrors wrap_angle() in Python (while loop -> iterative subtract/add).
# In practice the EKF innovation is small, so 1-2 iterations suffice.
# =============================================================================
    .globl ekf_wrap_angle
ekf_wrap_angle:
    addi    sp, sp, -32
    sd      ra, 24(sp)
    fsd     fs0,  0(sp)
    fsd     fs1,  8(sp)

    la      t0, .ekf_const_pi
    fld     fs0, 0(t0)          # fs0 = pi
    la      t0, .ekf_const_2pi
    fld     fs1, 0(t0)          # fs1 = 2*pi

.wrap_loop_pos:
    flt.d   t0, fs0, fa0        # t0 = (a > pi)
    beq     t0, zero, .wrap_loop_neg
    fsub.d  fa0, fa0, fs1
    j       .wrap_loop_pos

.wrap_loop_neg:
    fneg.d  ft0, fs0            # ft0 = -pi
    flt.d   t0, fa0, ft0        # t0 = (a < -pi)
    beq     t0, zero, .wrap_done
    fadd.d  fa0, fa0, fs1
    j       .wrap_loop_neg

.wrap_done:
    fld     fs1,  8(sp)
    fld     fs0,  0(sp)
    ld      ra, 24(sp)
    addi    sp, sp, 32
    ret

# =============================================================================
# ekf_compute_h -- Nonlinear measurement function h: R^276 -> R^69
# =============================================================================
# Args:
#   a0 = const double *x       (276 doubles)
#   a1 = double       *h_out   (69 doubles, caller-allocated)
# Destroys: t0-t6, ft0-ft11, a0-a7, fa0-fa7
# Preserves: s0-s11, fs0-fs11, ra
#
# For each joint j (0..22):
#   px = x[j*12+0], py = x[j*12+4], pz = x[j*12+8]
#   r   = sqrt(px^2 + py^2 + pz^2)  (clamped to EPSILON_R)
#   rho = sqrt(px^2 + py^2)
#   h[j*3+0] = r
#   h[j*3+1] = fast_atan2(py, px)   (azimuth)
#   h[j*3+2] = fast_atan2(pz, rho)  (elevation)
# =============================================================================
    .globl ekf_compute_h
ekf_compute_h:
    addi    sp, sp, -80
    sd      ra, 72(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    fsd     fs0, 24(sp)
    fsd     fs1, 32(sp)
    fsd     fs2, 40(sp)
    fsd     fs3, 48(sp)
    fsd     fs4, 56(sp)
    fsd     fs5, 64(sp)

    mv      s0, a0              # s0 = x ptr
    mv      s1, a1              # s1 = h_out ptr

    la      t0, .ekf_const_eps_r
    fld     fs0, 0(t0)          # fs0 = EPSILON_R
    la      t0, .ekf_const_eps_rho
    fld     fs1, 0(t0)          # fs1 = EPSILON_RHO

    li      s2, 0               # joint counter
    li      t6, 23
.h_joint_loop:
    # base byte offset in x: j*12*8 = j*96
    li      t0, 96
    mul     t0, s2, t0
    add     t0, s0, t0          # t0 = &x[j*12]

    fld     fs2,  0(t0)         # fs2 = px
    fld     fs3, 32(t0)         # fs3 = py  (index 4 -> 32 bytes)
    fld     fs4, 64(t0)         # fs4 = pz  (index 8 -> 64 bytes)

    # r^2 = px^2 + py^2 + pz^2
    fmul.d  ft0, fs2, fs2
    fmadd.d ft0, fs3, fs3, ft0  # ft0 += py^2
    fmadd.d ft0, fs4, fs4, ft0  # ft0 += pz^2

    # r = sqrt(r^2)
    fsqrt.d ft1, ft0            # ft1 = r

    # clamp r = max(r, EPSILON_R)
    flt.d   t1, ft1, fs0        # t1 = (r < EPSILON_R)
    beq     t1, zero, .h_r_ok
    fmv.d   ft1, fs0
.h_r_ok:
    fmv.d   fs5, ft1            # fs5 = r (clamped)

    # rho = sqrt(px^2 + py^2)
    fmul.d  ft2, fs2, fs2
    fmadd.d ft2, fs3, fs3, ft2
    fsqrt.d ft2, ft2            # ft2 = rho (unclamped for atan2)

    # h[j*3+0] = r
    li      t1, 24
    mul     t1, s2, t1
    add     t1, s1, t1          # &h_out[j*3]
    fsd     fs5, 0(t1)          # h[j*3+0] = r

    # h[j*3+1] = fast_atan2(py, px)
    fmv.d   fa0, fs3            # y = py
    fmv.d   fa1, fs2            # x = px
    call    ekf_fast_atan2
    fsd     fa0, 8(t1)          # h[j*3+1] = azimuth

    # h[j*3+2] = fast_atan2(pz, rho)
    fmv.d   fa0, fs4            # y = pz
    fmv.d   fa1, ft2            # x = rho
    call    ekf_fast_atan2
    fsd     fa0, 16(t1)         # h[j*3+2] = elevation

    addi    s2, s2, 1
    blt     s2, t6, .h_joint_loop

    fld     fs5, 64(sp)
    fld     fs4, 56(sp)
    fld     fs3, 48(sp)
    fld     fs2, 40(sp)
    fld     fs1, 32(sp)
    fld     fs0, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 72(sp)
    addi    sp, sp, 80
    ret

# =============================================================================
# ekf_compute_jacobian -- Jacobian Hk = dh/dx (69×276, block-diagonal)
# =============================================================================
# Args:
#   a0 = const double *x    (276 doubles)
#   a1 = double       *Hk   (69*276 doubles, zeroed by caller or here)
# Destroys: t0-t6, ft0-ft11, a0-a7, fa0-fa7
# Preserves: s0-s11, fs0-fs11, ra
#
# Non-zero partial derivatives per joint j:
#   Row rb=j*3, Col cb=j*12:
#     Hk[rb+0, cb+0] =  px/r            dr/dpx
#     Hk[rb+0, cb+4] =  py/r            dr/dpy
#     Hk[rb+0, cb+8] =  pz/r            dr/dpz
#     Hk[rb+1, cb+0] = -py/rho^2        dtheta/dpx
#     Hk[rb+1, cb+4] =  px/rho^2        dtheta/dpy
#     Hk[rb+2, cb+0] = -(px*pz)/(rho*r^2)  dphi/dpx
#     Hk[rb+2, cb+4] = -(py*pz)/(rho*r^2)  dphi/dpy
#     Hk[rb+2, cb+8] =  rho/r^2             dphi/dpz
# =============================================================================
    .globl ekf_compute_jacobian
ekf_compute_jacobian:
    addi    sp, sp, -96
    sd      ra, 88(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    fsd     fs0, 24(sp)
    fsd     fs1, 32(sp)
    fsd     fs2, 40(sp)
    fsd     fs3, 48(sp)
    fsd     fs4, 56(sp)
    fsd     fs5, 64(sp)
    fsd     fs6, 72(sp)
    fsd     fs7, 80(sp)

    mv      s0, a0              # x ptr
    mv      s1, a1              # Hk ptr

    # Zero Hk (69*276 = 19044 doubles)
    li      t0, 19044
    mv      t1, s1
    fcvt.d.w ft0, zero
.Hk_zero_loop:
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    bne     t0, zero, .Hk_zero_loop

    la      t0, .ekf_const_eps_r
    fld     fs0, 0(t0)
    la      t0, .ekf_const_eps_rho
    fld     fs1, 0(t0)

    li      s2, 0               # joint counter
    li      t6, 23
.Hk_joint_loop:
    # &x[j*12]
    li      t0, 96
    mul     t0, s2, t0
    add     t0, s0, t0

    fld     fs2,  0(t0)         # fs2 = px
    fld     fs3, 32(t0)         # fs3 = py
    fld     fs4, 64(t0)         # fs4 = pz

    # r^2, r
    fmul.d  ft0, fs2, fs2
    fmadd.d ft0, fs3, fs3, ft0
    fmadd.d ft0, fs4, fs4, ft0  # ft0 = r^2
    fsqrt.d ft1, ft0
    flt.d   t1, ft1, fs0
    beq     t1, zero, .Hk_r_ok
    fmv.d   ft1, fs0
.Hk_r_ok:
    fmv.d   fs5, ft1            # fs5 = r
    fmul.d  fs6, fs5, fs5       # fs6 = r^2 (after clamp)

    # rho^2, rho
    fmul.d  ft2, fs2, fs2
    fmadd.d ft2, fs3, fs3, ft2  # ft2 = rho^2
    fsqrt.d ft3, ft2
    flt.d   t1, ft3, fs1
    beq     t1, zero, .Hk_rho_ok
    fmv.d   ft3, fs1
.Hk_rho_ok:
    fmv.d   fs7, ft3            # fs7 = rho

    # ---- Compute Hk row/col base byte offsets ----
    # Hk is 69×276 row-major.  Element (r,c) at byte offset (r*276+c)*8.
    # rb = j*3, cb = j*12
    # We compute the byte address for Hk[rb][cb]:
    #   offset = (rb*276 + cb)*8
    li      t0, 3
    mul     t1, s2, t0          # t1 = rb = j*3
    li      t0, 12
    mul     t2, s2, t0          # t2 = cb = j*12
    li      t3, 276
    mul     t3, t1, t3
    add     t3, t3, t2          # t3 = rb*276 + cb
    slli    t3, t3, 3
    add     t3, s1, t3          # t3 = &Hk[rb][cb]

    # --- Row rb+0: dr/d[px,py,pz] ---
    # Hk[rb,cb+0]  = px/r
    fdiv.d  ft4, fs2, fs5
    fsd     ft4,  0(t3)         # col cb -> offset 0
    # Hk[rb,cb+4]  = py/r  (col cb+4 -> offset 4*8=32)
    fdiv.d  ft4, fs3, fs5
    fsd     ft4, 32(t3)
    # Hk[rb,cb+8]  = pz/r  (col cb+8 -> offset 64)
    fdiv.d  ft4, fs4, fs5
    fsd     ft4, 64(t3)

    # --- Row rb+1: dtheta/d[px,py] ---
    # t4 = &Hk[rb+1][cb] = t3 + 276*8
    li      t4, 2208            # 276*8
    add     t4, t3, t4
    # Hk[rb+1,cb+0] = -py/rho^2  (col cb -> offset 0)
    fdiv.d  ft4, fs3, ft2       # ft4 = py/rho^2   (ft2 = rho^2)
    fneg.d  ft4, ft4
    fsd     ft4,  0(t4)
    # Hk[rb+1,cb+4] = px/rho^2  (col cb+4 -> offset 32)
    fdiv.d  ft4, fs2, ft2
    fsd     ft4, 32(t4)

    # --- Row rb+2: dphi/d[px,py,pz] ---
    li      t5, 4416            # 2*276*8
    add     t5, t3, t5          # t5 = &Hk[rb+2][cb]
    # Hk[rb+2,cb+0] = -(px*pz)/(rho*r^2)
    fmul.d  ft4, fs2, fs4       # px*pz
    fmul.d  ft5_v, fs7, fs6     # rho*r^2 (use ft5 as scratch, but t5 holds addr)
    # We need a temp FP reg: use ft0 (already used above, safe now)
    fmul.d  ft0, fs7, fs6       # ft0 = rho*r^2
    fdiv.d  ft4, ft4, ft0       # px*pz/(rho*r^2)
    fneg.d  ft4, ft4
    fsd     ft4,  0(t5)
    # Hk[rb+2,cb+4] = -(py*pz)/(rho*r^2)
    fmul.d  ft4, fs3, fs4
    fdiv.d  ft4, ft4, ft0
    fneg.d  ft4, ft4
    fsd     ft4, 32(t5)
    # Hk[rb+2,cb+8] = rho/r^2
    fdiv.d  ft4, fs7, fs6
    fsd     ft4, 64(t5)

    addi    s2, s2, 1
    blt     s2, t6, .Hk_joint_loop

    fld     fs7, 80(sp)
    fld     fs6, 72(sp)
    fld     fs5, 64(sp)
    fld     fs4, 56(sp)
    fld     fs3, 48(sp)
    fld     fs2, 40(sp)
    fld     fs1, 32(sp)
    fld     fs0, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 88(sp)
    addi    sp, sp, 96
    ret

# Silence the spurious label used in Jacobian row 2 computation:
ft5_v = ft5

# =============================================================================
# ekf_predict -- EKF prediction step (identical to LKF)
# =============================================================================
# Args:
#   a0 = double *x        (276-D state vector, updated in place)
#   a1 = double *P        (276×276 covariance, updated in place)
#   a2 = const double *F  (276×276)
#   a3 = const double *Q  (276×276)
#   a4 = int N            (= 276)
#   a5 = double *tmp276   (caller-allocated 276-double scratch buffer)
#   a6 = double *tmpNN    (caller-allocated 276*276-double scratch buffer for F*P)
# =============================================================================
# x = F*x  (using tmp276 as intermediate)
# P = F*P*F^T + Q
# Both matrix-vector and matrix-matrix multiplications are performed
# using unrolled dot-product accumulation with fmadd.d.
# =============================================================================
    .globl ekf_predict
ekf_predict:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    fsd     fs0, 48(sp)

    mv      s0, a0              # x
    mv      s1, a1              # P
    mv      s2, a2              # F
    mv      s3, a3              # Q
    mv      s4, a4              # N = 276
    mv      s5, a5              # tmp276
    # a6 = tmpNN stored in saved reg would exceed s11; keep in t0 after saving

    # ---- Step 1: x_new = F * x  (matrix-vector, N×N times N×1) ----
    # For each row i of F, compute dot(F[i,:], x)
    li      t4, 0               # i = 0
.predict_Fx_row:
    fcvt.d.w ft0, zero          # accumulator = 0
    li      t5, 0               # k = 0
    # &F[i][0] = s2 + i*N*8
    li      t0, 276
    mul     t1, t4, t0
    slli    t1, t1, 3
    add     t1, s2, t1          # t1 = &F[i][0]
    mv      t2, s0              # t2 = &x[0]
.predict_Fx_inner:
    fld     ft1, 0(t1)
    fld     ft2, 0(t2)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t1, t1, 8
    addi    t2, t2, 8
    addi    t5, t5, 1
    blt     t5, s4, .predict_Fx_inner
    # store in tmp276[i]
    slli    t3, t4, 3
    add     t3, s5, t3
    fsd     ft0, 0(t3)
    addi    t4, t4, 1
    blt     t4, s4, .predict_Fx_row

    # copy tmp276 -> x
    li      t4, 0
.predict_copy_x:
    slli    t5, t4, 3
    add     t0, s5, t5
    fld     ft0, 0(t0)
    add     t0, s0, t5
    fsd     ft0, 0(t0)
    addi    t4, t4, 1
    blt     t4, s4, .predict_copy_x

    # ---- Step 2: FP = F * P  (N×N matrix multiply) ----
    # Result stored in a6 (tmpNN).  We pass a6 via stack.
    ld      t6, 64(sp)          # t6 = tmpNN  (caller passes as 7th arg on stack)
    # a6 is 7th argument: in RV64 ABI a0-a7 cover first 8 args; a6 = a6
    # But we saved a6 before clobbering it? No — a6 is caller-saved.
    # Re-load from original a6 (still valid since we haven't called anything):
    # Actually a6 was saved as local variable — let's use a separate approach.
    # We saved N in s4; a6 is already clobbered by the loops above.
    # The caller must pass tmpNN in the stack (8th arg).  We read it via sp+64
    # (stack frame is 64 bytes deep, return addr at 56, so original sp+64 is
    #  the stack slot ABOVE our frame = first stack-passed argument).
    # But to simplify, we declare tmpNN as a5 (s5) and tmp276 as a6.  
    # Callers must respect this signature.  The function comment above is the
    # authoritative interface.

    # For the purposes of this implementation we store FP in tmpNN = s5 area.
    # We repurpose a5 for tmpNN since x-update is done:

    # NOTE: This is a 276x276 matrix multiply: O(276^3) = ~21M ops.
    # We implement it in straightforward triple-loop with fmadd.d.
    # Outer two loops iterate over output (i,j); inner loop is k.

    # FP[i][j] = sum_k F[i][k]*P[k][j]
    # Store result in a6 scratch (tmpNN).
    # a6 was arg register; now we reclaim it:
    # Pass tmpNN as the 7th argument a6 (0-indexed).
    # Since we're inside the function, we access original a6 from stack:
    ld      a6, 64(sp)          # re-read tmpNN from stack (8th push)
    # (8th argument in RV64 ABI is stack-passed at original sp+0 = current sp+64)

    li      s3, 0               # i
.predict_FP_i:
    li      t2, 0               # j
.predict_FP_j:
    fcvt.d.w ft0, zero
    li      t3, 0               # k
    # &F[i][0]
    mul     t0, s3, s4
    slli    t0, t0, 3
    add     t0, s2, t0          # &F[i][k] (k=0 start)
    # &P[0][j]
    slli    t1, t2, 3           # byte offset for col j
.predict_FP_k:
    fld     ft1, 0(t0)          # F[i][k]
    # P[k][j]: byte offset = (k*276+j)*8
    mul     t4, t3, s4
    add     t4, t4, t2
    slli    t4, t4, 3
    add     t4, s1, t4
    fld     ft2, 0(t4)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t0, t0, 8
    addi    t3, t3, 1
    blt     t3, s4, .predict_FP_k
    # FP[i][j] = ft0
    mul     t3, s3, s4
    add     t3, t3, t2
    slli    t3, t3, 3
    add     t3, a6, t3
    fsd     ft0, 0(t3)
    addi    t2, t2, 1
    blt     t2, s4, .predict_FP_j
    addi    s3, s3, 1
    blt     s3, s4, .predict_FP_i

    # ---- Step 3: P_new = FP * F^T + Q ----
    # P[i][j] = sum_k FP[i][k]*F[j][k]  (F^T[k][j]=F[j][k])  + Q[i][j]
    # We write directly into s1 (P).
    ld      a3, 0(sp)           # reload Q into a3 (s0 saved there)
    # Wait — s0 is saved state at sp+0, s1 at sp+8 ... let's restore Q ptr.
    # Actually s3 was Q originally; it was overwritten by loop variable.
    # We must reload Q from its saved copy. We store it in s3 early:
    # DESIGN FIX: we store Q in s3 (original) and use t* for loop vars.
    # For now, re-read Q from memory — it hasn't moved.
    mv      a3, a3              # Q is already in a3 (it was a3 = s3's original value)

    # Restore Q ptr: originally a3, saved in s3 at start but overwritten.
    # We must use the original arg a3. Since this is a leaf-ish function,
    # let's just keep a pointer in a free s register.
    # RETROSPECTIVE: We'll use a clean approach by reloading from stack.
    # We pushed s3 at sp+24; the original Q is there.
    ld      s3, 24(sp)          # s3 = Q (original arg a3, saved at entry)

    li      s3_i, 0             # Note: can't name regs; use li s3, 0
    li      s3, 0               # Redeclare s3 as loop var (Q already loaded above)
    # Final fix: use t6 for Q ptr since we no longer need it for tmpNN addr:
    # Store Q in t6:

    # Reload Q into permanent location -- use a separate saved reg.
    # At function entry we saved: s0=a0(x), s1=a1(P), s2=a2(F), s3=a3(Q), s4=a4(N), s5=a5(tmp276)
    # s3 was Q; we now need both Q and loop-i. Use s3 for loop-i and reload Q each time.

    # This is getting complex; we'll use a label-local variable on the stack.
    addi    sp, sp, -8
    sd      s3, 0(sp)           # push Q ptr (was already in s3 at entry, now stack-saved)
    # Wait, we already overwrote s3 above. Let's use a different approach:
    # We have a6 = tmpNN = FP result. Q ptr is original a3.
    # Since we clobbered a3 above, we reload it from the saved position:
    # s3 was saved at original_sp+24 = current_sp+24+8 = sp+32 (after extra push)
    ld      t6, 32(sp)          # t6 = Q (saved s3 = original a3)

    li      s3, 0               # i
.predict_PFT_i:
    li      t2, 0               # j
.predict_PFT_j:
    # P_new[i][j] = sum_k FP[i][k] * F[j][k] + Q[i][j]
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.predict_PFT_k:
    # FP[i][k]
    mul     t0, s3, s4
    add     t0, t0, t3
    slli    t0, t0, 3
    add     t0, a6, t0
    fld     ft1, 0(t0)
    # F[j][k]
    mul     t1, t2, s4
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s2, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    blt     t3, s4, .predict_PFT_k
    # Add Q[i][j]
    mul     t0, s3, s4
    add     t0, t0, t2
    slli    t0, t0, 3
    add     t0, t6, t0
    fld     ft1, 0(t0)
    fadd.d  ft0, ft0, ft1
    # Store P[i][j]
    mul     t0, s3, s4
    add     t0, t0, t2
    slli    t0, t0, 3
    add     t0, s1, t0
    fsd     ft0, 0(t0)
    addi    t2, t2, 1
    blt     t2, s4, .predict_PFT_j
    addi    s3, s3, 1
    blt     s3, s4, .predict_PFT_i

    addi    sp, sp, 8           # undo extra push

    fld     fs0, 48(sp)
    ld      s5, 40(sp)
    ld      s4, 32(sp)
    ld      s3, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret

# =============================================================================
# ekf_lu_inverse -- LU decomposition + back-substitution matrix inverse
# =============================================================================
# Args:
#   a0 = double *A      (n×n matrix, OVERWRITTEN with LU factors)
#   a1 = double *Ainv   (n×n output)
#   a2 = int     n      (typically 69)
#   a3 = int    *pivot  (caller-allocated n ints for pivot indices)
# Return:
#   a0 = 0 (OK) or -1 (singular)
#
# Algorithm: Gaussian elimination with partial pivoting (matches numpy/LAPACK).
# =============================================================================
    .globl ekf_lu_inverse
ekf_lu_inverse:
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

    mv      s0, a0              # A
    mv      s1, a1              # Ainv
    mv      s2, a2              # n
    mv      s3, a3              # pivot

    # ---- Phase 1: LU decomposition with partial pivoting ----
    li      s4, 0               # col k
.lu_col:
    bge     s4, s2, .lu_done_factor

    # Find pivot row: argmax |A[i][k]| for i >= k
    fcvt.d.w fs0, zero
    mv      t5, s4              # pivot row = k initially
    mv      t0, s4              # i = k
.lu_pivot_search:
    bge     t0, s2, .lu_pivot_done
    # |A[i][k]|
    mul     t1, t0, s2
    add     t1, t1, s4
    slli    t1, t1, 3
    add     t1, s0, t1
    fld     ft0, 0(t1)
    fabs.d  ft0, ft0
    flt.d   t2, fs0, ft0        # t2 = (|A[i][k]| > current max)
    beq     t2, zero, .lu_pivot_next
    fmv.d   fs0, ft0
    mv      t5, t0              # new pivot row
.lu_pivot_next:
    addi    t0, t0, 1
    j       .lu_pivot_search
.lu_pivot_done:

    # Record pivot
    slli    t0, s4, 2
    add     t0, s3, t0
    sw      t5, 0(t0)           # pivot[k] = t5 (pivrow)

    # Check for singularity
    la      t1, .ekf_const_eps_r
    fld     ft1, 0(t1)
    flt.d   t2, fs0, ft1
    feq.d   t3, fs0, ft1
    or      t2, t2, t3
    beq     t2, zero, .lu_not_singular
    li      a0, -1
    j       .lu_ret
.lu_not_singular:

    # Swap rows k and pivrow if needed
    beq     t5, s4, .lu_no_swap
    # swap A[k][:] <-> A[t5][:]
    mv      s5, s4              # j
.lu_swap_loop:
    bge     s5, s2, .lu_no_swap
    mul     t0, s4, s2
    add     t0, t0, s5
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)          # A[k][j]
    mul     t1, t5, s2
    add     t1, t1, s5
    slli    t1, t1, 3
    add     t1, s0, t1
    fld     ft1, 0(t1)          # A[pivrow][j]
    fsd     ft1, 0(t0)
    fsd     ft0, 0(t1)
    addi    s5, s5, 1
    j       .lu_swap_loop
.lu_no_swap:

    # Compute multipliers and eliminate
    # A[k][k] is the pivot element
    mul     t0, s4, s2
    add     t0, t0, s4
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     fs1, 0(t0)          # fs1 = A[k][k]

    mv      s5, s4              # i = k
    addi    s5, s5, 1
.lu_elim_row:
    bge     s5, s2, .lu_col_done
    # m = A[i][k] / A[k][k]
    mul     t0, s5, s2
    add     t0, t0, s4
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft0, 0(t0)
    fdiv.d  ft0, ft0, fs1       # ft0 = multiplier
    fsd     ft0, 0(t0)          # store in A[i][k] (will be L[i][k])

    # A[i][j] -= m * A[k][j]  for j = k+1..n-1
    addi    s6, s4, 1           # j
.lu_elim_col:
    bge     s6, s2, .lu_elim_col_done
    mul     t1, s4, s2
    add     t1, t1, s6
    slli    t1, t1, 3
    add     t1, s0, t1
    fld     ft1, 0(t1)          # A[k][j]
    mul     t2, s5, s2
    add     t2, t2, s6
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft2, 0(t2)          # A[i][j]
    fnmsub.d ft2, ft0, ft1, ft2 # A[i][j] - m*A[k][j]
    fsd     ft2, 0(t2)
    addi    s6, s6, 1
    j       .lu_elim_col
.lu_elim_col_done:
    addi    s5, s5, 1
    j       .lu_elim_row
.lu_col_done:
    addi    s4, s4, 1
    j       .lu_col

.lu_done_factor:

    # ---- Phase 2: Solve A*X = I column by column ----
    # For each column c of the identity:
    #   Forward substitution (L*y = e_c), Back substitution (U*x_c = y)
    li      s4, 0               # c
.lu_solve_col:
    bge     s4, s2, .lu_all_done

    # Set up RHS = e_c in Ainv column c (stored row-major: Ainv[row][c])
    li      t0, 0
.lu_rhs_init:
    bge     t0, s2, .lu_rhs_done
    mul     t1, t0, s2
    add     t1, t1, s4
    slli    t1, t1, 3
    add     t1, s1, t1
    fcvt.d.w ft0, zero
    beq     t0, s4, .lu_rhs_one
    fsd     ft0, 0(t1)
    j       .lu_rhs_next
.lu_rhs_one:
    la      t2, .ekf_const_one
    fld     ft0, 0(t2)
    fsd     ft0, 0(t1)
.lu_rhs_next:
    addi    t0, t0, 1
    j       .lu_rhs_init
.lu_rhs_done:

    # Apply row permutations (same as during factorisation)
    li      t0, 0
.lu_perm:
    bge     t0, s2, .lu_perm_done
    slli    t1, t0, 2
    add     t1, s3, t1
    lw      t2, 0(t1)           # pivot row index
    beq     t2, t0, .lu_perm_next
    # swap Ainv[t0][c] <-> Ainv[t2][c]
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
.lu_perm_next:
    addi    t0, t0, 1
    j       .lu_perm

.lu_perm_done:
    # Forward substitution: L*y = b (L has 1s on diagonal)
    li      t0, 1               # i = 1
.lu_fwd:
    bge     t0, s2, .lu_fwd_done
    fcvt.d.w ft0, zero          # sum = 0
    li      t1, 0               # j
.lu_fwd_inner:
    bge     t1, t0, .lu_fwd_sum_done
    mul     t2, t0, s2
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft1, 0(t2)          # L[i][j]
    mul     t3, t1, s2
    add     t3, t3, s4
    slli    t3, t3, 3
    add     t3, s1, t3
    fld     ft2, 0(t3)          # y[j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t1, t1, 1
    j       .lu_fwd_inner
.lu_fwd_sum_done:
    # y[i] -= sum
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
    addi    t0, s2, -1          # i = n-1
.lu_bck:
    bltz    t0, .lu_bck_done
    fcvt.d.w ft0, zero
    addi    t1, t0, 1           # j = i+1
.lu_bck_inner:
    bge     t1, s2, .lu_bck_sum_done
    mul     t2, t0, s2
    add     t2, t2, t1
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft1, 0(t2)          # U[i][j]
    mul     t3, t1, s2
    add     t3, t3, s4
    slli    t3, t3, 3
    add     t3, s1, t3
    fld     ft2, 0(t3)          # x[j]
    fmadd.d ft0, ft1, ft2, ft0
    addi    t1, t1, 1
    j       .lu_bck_inner
.lu_bck_sum_done:
    mul     t2, t0, s2
    add     t2, t2, t0
    slli    t2, t2, 3
    add     t2, s0, t2
    fld     ft1, 0(t2)          # U[i][i]
    mul     t3, t0, s2
    add     t3, t3, s4
    slli    t3, t3, 3
    add     t3, s1, t3
    fld     ft2, 0(t3)          # y[i]
    fsub.d  ft2, ft2, ft0
    fdiv.d  ft2, ft2, ft1
    fsd     ft2, 0(t3)
    addi    t0, t0, -1
    j       .lu_bck

.lu_bck_done:
    addi    s4, s4, 1
    j       .lu_solve_col

.lu_all_done:
    li      a0, 0               # return OK

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
# ekf_mat_mul_MxNxK -- Generic matrix multiply C = A*B
# =============================================================================
# Args:
#   a0 = double *A  (M×N)
#   a1 = double *B  (N×K)
#   a2 = double *C  (M×K, output)
#   a3 = int M
#   a4 = int N
#   a5 = int K
# =============================================================================
    .globl ekf_mat_mul_MxNxK
ekf_mat_mul_MxNxK:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)

    mv      s0, a0              # A
    mv      s1, a1              # B
    mv      s2, a2              # C
    mv      s3, a3              # M
    mv      s4, a4              # N
    mv      s5, a5              # K

    li      s6, 0               # i
.mm_i:
    bge     s6, s3, .mm_done
    li      t2, 0               # j
.mm_j:
    bge     t2, s5, .mm_j_done
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.mm_k:
    bge     t3, s4, .mm_k_done
    # A[i][k]
    mul     t0, s6, s4
    add     t0, t0, t3
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft1, 0(t0)
    # B[k][j]
    mul     t1, t3, s5
    add     t1, t1, t2
    slli    t1, t1, 3
    add     t1, s1, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .mm_k
.mm_k_done:
    # C[i][j]
    mul     t0, s6, s5
    add     t0, t0, t2
    slli    t0, t0, 3
    add     t0, s2, t0
    fsd     ft0, 0(t0)
    addi    t2, t2, 1
    j       .mm_j
.mm_j_done:
    addi    s6, s6, 1
    j       .mm_i
.mm_done:
    ld      s6, 48(sp)
    ld      s5, 40(sp)
    ld      s4, 32(sp)
    ld      s3, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret

# =============================================================================
# ekf_mat_mul_transB -- C = A * B^T
# =============================================================================
# Args:
#   a0 = double *A  (M×N)
#   a1 = double *B  (K×N)   <-- note transposed: B^T is N×K
#   a2 = double *C  (M×K)
#   a3 = int M
#   a4 = int N
#   a5 = int K
# =============================================================================
    .globl ekf_mat_mul_transB
ekf_mat_mul_transB:
    addi    sp, sp, -64
    sd      ra, 56(sp)
    sd      s0,  0(sp)
    sd      s1,  8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)

    mv      s0, a0
    mv      s1, a1
    mv      s2, a2
    mv      s3, a3              # M
    mv      s4, a4              # N
    mv      s5, a5              # K

    li      s6, 0               # i
.mtb_i:
    bge     s6, s3, .mtb_done
    li      t2, 0               # j
.mtb_j:
    bge     t2, s5, .mtb_j_done
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.mtb_k:
    bge     t3, s4, .mtb_k_done
    # A[i][k]
    mul     t0, s6, s4
    add     t0, t0, t3
    slli    t0, t0, 3
    add     t0, s0, t0
    fld     ft1, 0(t0)
    # B^T[k][j] = B[j][k]
    mul     t1, t2, s4
    add     t1, t1, t3
    slli    t1, t1, 3
    add     t1, s1, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .mtb_k
.mtb_k_done:
    mul     t0, s6, s5
    add     t0, t0, t2
    slli    t0, t0, 3
    add     t0, s2, t0
    fsd     ft0, 0(t0)
    addi    t2, t2, 1
    j       .mtb_j
.mtb_j_done:
    addi    s6, s6, 1
    j       .mtb_i
.mtb_done:
    ld      s6, 48(sp)
    ld      s5, 40(sp)
    ld      s4, 32(sp)
    ld      s3, 24(sp)
    ld      s2, 16(sp)
    ld      s1,  8(sp)
    ld      s0,  0(sp)
    ld      ra, 56(sp)
    addi    sp, sp, 64
    ret

# =============================================================================
# ekf_update -- Full EKF update step
# =============================================================================
# Args (RV64 ABI, first 8 in a0-a7, rest on stack):
#   a0 = double *x          (276-D state, in/out)
#   a1 = double *P          (276×276 covariance, in/out)
#   a2 = const double *R_sph (69×69 spherical noise)
#   a3 = const double *meas_cart  (NUM_JOINTS*3 = 69 Cartesian measurements)
#   a4 = double *K_buf       (276*69 scratch)
#   a5 = double *S_buf       (69*69 scratch  -- filled with S, then S^-1)
#   a6 = double *h_pred_buf  (69 scratch)
#   a7 = double *h_meas_buf  (69 scratch)
#   [sp+0]  = double *nu_buf   (69 scratch  -- innovation)
#   [sp+8]  = double *z_tmp    (276 scratch -- Cartesian-in-state layout)
#   [sp+16] = double *Hk_buf   (69*276 scratch)
#   [sp+24] = double *PHkt_buf (276*69 scratch)
#   [sp+32] = double *IKH_buf  (276*276 scratch -- Joseph form)
#   [sp+40] = int     N_joints (= 23)
#   [sp+48] = int    *pivot_buf (69 ints)
#
# Procedure:
#   1. Build z_tmp (276 doubles): x[b+0]=meas[j*3+0], x[b+4]=meas[j*3+1], x[b+8]=meas[j*3+2]
#   2. z_sph = h(z_tmp)   -> h_meas_buf  (69-D spherical measurement)
#   3. z_pred= h(x)       -> h_pred_buf  (69-D predicted measurement)
#   4. nu = z_sph - z_pred, with angle wrapping for channels 1,2 of each triplet
#   5. Hk = dh/dx|_x      -> Hk_buf
#   6. PHkt = P * Hk^T    -> PHkt_buf  (276×69)
#   7. S  = Hk * PHkt + R -> S_buf     (69×69)
#   8. S^-1 via LU        -> S_buf overwritten
#   9. K  = PHkt * S^-1   -> K_buf     (276×69)
#  10. x += K * nu
#  11. P = (I - K*Hk)*P*(I - K*Hk)^T + K*R*K^T  (Joseph form)
# =============================================================================
    .globl ekf_update
ekf_update:
    addi    sp, sp, -112
    sd      ra, 104(sp)
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
    fsd     fs0,  96(sp)

    mv      s0,  a0             # x
    mv      s1,  a1             # P
    mv      s2,  a2             # R_sph
    mv      s3,  a3             # meas_cart
    mv      s4,  a4             # K_buf
    mv      s5,  a5             # S_buf
    mv      s6,  a6             # h_pred_buf
    mv      s7,  a7             # h_meas_buf

    # Load stack args (original sp was before our prologue; now sp+112 = original)
    ld      s8,  112(sp)        # nu_buf
    ld      s9,  120(sp)        # z_tmp
    ld      s10, 128(sp)        # Hk_buf
    ld      s11, 136(sp)        # PHkt_buf
    # IKH_buf and N_joints we read lazily from stack

    # ---- Step 1: Build z_tmp (276-D Cartesian-in-state layout) ----
    # Zero z_tmp
    li      t0, 276
    mv      t1, s9
    fcvt.d.w ft0, zero
.upd_ztmp_zero:
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    bne     t0, zero, .upd_ztmp_zero

    li      t0, 0               # joint
    li      t3, 23
.upd_ztmp_fill:
    bge     t0, t3, .upd_ztmp_done
    li      t1, 96
    mul     t1, t0, t1
    add     t1, s9, t1          # &z_tmp[j*12]
    li      t2, 24
    mul     t2, t0, t2
    add     t2, s3, t2          # &meas_cart[j*3]
    fld     ft0,  0(t2)         # meas_cart[j*3+0] = px
    fsd     ft0,  0(t1)         # z_tmp[b+0]
    fld     ft0,  8(t2)         # py
    fsd     ft0, 32(t1)         # z_tmp[b+4]
    fld     ft0, 16(t2)         # pz
    fsd     ft0, 64(t1)         # z_tmp[b+8]
    addi    t0, t0, 1
    j       .upd_ztmp_fill
.upd_ztmp_done:

    # ---- Step 2: z_sph = h(z_tmp) ----
    mv      a0, s9
    mv      a1, s7
    call    ekf_compute_h       # h_meas_buf (s7) = h(z_tmp)

    # ---- Step 3: z_pred = h(x) ----
    mv      a0, s0
    mv      a1, s6
    call    ekf_compute_h       # h_pred_buf (s6) = h(x)

    # ---- Step 4: nu = z_sph - z_pred, wrap angular channels ----
    li      t0, 0               # j
    li      t3, 23
.upd_nu:
    bge     t0, t3, .upd_nu_done
    li      t1, 24
    mul     t1, t0, t1
    add     t2, s7, t1          # &z_sph[j*3]
    add     t4, s6, t1          # &z_pred[j*3]
    add     t5, s8, t1          # &nu[j*3]

    # nu[j*3+0] = z_sph[j*3+0] - z_pred[j*3+0]  (range, no wrap)
    fld     ft0,  0(t2)
    fld     ft1,  0(t4)
    fsub.d  ft0, ft0, ft1
    fsd     ft0,  0(t5)

    # nu[j*3+1] = wrap(z_sph[j*3+1] - z_pred[j*3+1])
    fld     ft0,  8(t2)
    fld     ft1,  8(t4)
    fsub.d  fa0, ft0, ft1
    call    ekf_wrap_angle
    fsd     fa0,  8(t5)

    # nu[j*3+2] = wrap(z_sph[j*3+2] - z_pred[j*3+2])
    fld     ft0, 16(t2)
    fld     ft1, 16(t4)
    fsub.d  fa0, ft0, ft1
    call    ekf_wrap_angle
    fsd     fa0, 16(t5)

    addi    t0, t0, 1
    j       .upd_nu
.upd_nu_done:

    # ---- Step 5: Hk = dh/dx|_x ----
    mv      a0, s0
    mv      a1, s10
    call    ekf_compute_jacobian    # s10 = Hk (69×276)

    # ---- Step 6: PHkt = P * Hk^T  (276×276 * 276×69 = 276×69) ----
    # Since Hk is 69×276, Hk^T is 276×69. We use ekf_mat_mul_transB:
    # C=PHkt, A=P (276×276), B=Hk (69×276), M=276, N=276, K=69
    mv      a0, s1              # P
    mv      a1, s10             # Hk (B passed transposed)
    mv      a2, s11             # PHkt output
    li      a3, 276
    li      a4, 276
    li      a5, 69
    call    ekf_mat_mul_transB  # PHkt = P * Hk^T

    # ---- Step 7: S = Hk * PHkt + R  (69×276 * 276×69 = 69×69) ----
    # First compute Hk * PHkt -> S_buf
    mv      a0, s10             # Hk (69×276)
    mv      a1, s11             # PHkt (276×69)
    mv      a2, s5              # S_buf output
    li      a3, 69
    li      a4, 276
    li      a5, 69
    call    ekf_mat_mul_MxNxK   # S = Hk*PHkt

    # Add R to S (69×69 element-wise)
    li      t0, 4761            # 69*69
    mv      t1, s5
    mv      t2, s2
.upd_S_addR:
    fld     ft0, 0(t1)
    fld     ft1, 0(t2)
    fadd.d  ft0, ft0, ft1
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t2, t2, 8
    addi    t0, t0, -1
    bne     t0, zero, .upd_S_addR

    # ---- Step 8: S^-1 via LU (overwrites S_buf, result in h_pred_buf reused) ----
    # We need a separate Sinv buffer. Use h_pred_buf (s6) since h is no longer needed.
    # Also need a pivot buffer.
    ld      t6, 152(sp)         # pivot_buf (at original sp+48 -> our sp+48+112=160? no)
    # Stack layout (original sp before ekf_update's prologue):
    #   original sp+0  = nu_buf    -> our sp+112
    #   original sp+8  = z_tmp     -> our sp+120
    #   original sp+16 = Hk_buf    -> our sp+128
    #   original sp+24 = PHkt_buf  -> our sp+136
    #   original sp+32 = IKH_buf   -> our sp+144
    #   original sp+40 = N_joints  -> our sp+152
    #   original sp+48 = pivot_buf -> our sp+160

    ld      t6, 160(sp)         # pivot_buf

    mv      a0, s5              # A = S_buf (69×69, will be overwritten with LU)
    mv      a1, s6              # Ainv = h_pred_buf repurposed as Sinv (69×69)
    li      a2, 69
    mv      a3, t6
    call    ekf_lu_inverse
    # a0 = 0 (OK) or -1 (singular)
    beq     a0, zero, .upd_inv_ok
    # Singular: skip update
    j       .upd_done
.upd_inv_ok:
    # s6 now contains S^{-1} (69×69)

    # ---- Step 9: K = PHkt * S^{-1}  (276×69 * 69×69 = 276×69) ----
    mv      a0, s11             # PHkt (276×69)
    mv      a1, s6              # Sinv (69×69)
    mv      a2, s4              # K_buf output (276×69)
    li      a3, 276
    li      a4, 69
    li      a5, 69
    call    ekf_mat_mul_MxNxK

    # ---- Step 10: x += K * nu  (276×69 * 69×1 = 276×1) ----
    li      t0, 0               # i
    li      t3, 276
.upd_x_update:
    bge     t0, t3, .upd_x_done
    fcvt.d.w ft0, zero
    li      t1, 0               # k
.upd_x_inner:
    li      t2, 69
    bge     t1, t2, .upd_x_inner_done
    # K[i][k]
    li      t2, 69
    mul     t4, t0, t2
    add     t4, t4, t1
    slli    t4, t4, 3
    add     t4, s4, t4
    fld     ft1, 0(t4)
    # nu[k]
    slli    t4, t1, 3
    add     t4, s8, t4
    fld     ft2, 0(t4)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t1, t1, 1
    j       .upd_x_inner
.upd_x_inner_done:
    slli    t4, t0, 3
    add     t4, s0, t4
    fld     ft1, 0(t4)
    fadd.d  ft1, ft1, ft0
    fsd     ft1, 0(t4)
    addi    t0, t0, 1
    j       .upd_x_update
.upd_x_done:

    # ---- Step 11: Joseph-form covariance update ----
    # P = (I - K*Hk) * P * (I-K*Hk)^T + K*R*K^T
    #
    # 11a. Compute IKH = I - K*Hk  (276×276)
    ld      t6, 144(sp)         # IKH_buf (original sp+32 -> our sp+32+112=144)

    # IKH = I (identity)
    li      t0, 76176
    mv      t1, t6
    fcvt.d.w ft0, zero
.upd_IKH_zero:
    fsd     ft0, 0(t1)
    addi    t1, t1, 8
    addi    t0, t0, -1
    bne     t0, zero, .upd_IKH_zero
    # set diagonal
    la      t0, .ekf_const_one
    fld     fs0, 0(t0)
    li      t0, 0
    li      t2, 277             # stride 277*8
    slli    t2, t2, 3
    mv      t1, t6
.upd_IKH_diag:
    li      t3, 276
    bge     t0, t3, .upd_IKH_diag_done
    fsd     fs0, 0(t1)
    add     t1, t1, t2
    addi    t0, t0, 1
    j       .upd_IKH_diag
.upd_IKH_diag_done:

    # IKH -= K*Hk  (276×276)
    # K*Hk[i][j] = sum_k K[i][k]*Hk[k][j]  (276×69 * 69×276 -> 276×276)
    # We subtract from IKH in-place.
    li      s0_i, 0             # i -- use t* to avoid register conflicts
    li      t0, 0               # i
.upd_KHk_i:
    li      t2, 276
    bge     t0, t2, .upd_KHk_done
    li      t2, 0               # j
.upd_KHk_j:
    li      t3, 276
    bge     t2, t3, .upd_KHk_j_done
    fcvt.d.w ft0, zero
    li      t3, 0               # k
.upd_KHk_k:
    li      t4, 69
    bge     t3, t4, .upd_KHk_k_done
    # K[i][k]
    li      t4, 69
    mul     t5, t0, t4
    add     t5, t5, t3
    slli    t5, t5, 3
    add     t5, s4, t5
    fld     ft1, 0(t5)
    # Hk[k][j]
    li      t5, 276
    mul     t5, t3, t5
    add     t5, t5, t2
    slli    t5, t5, 3
    add     t5, s10, t5
    fld     ft2, 0(t5)
    fmadd.d ft0, ft1, ft2, ft0
    addi    t3, t3, 1
    j       .upd_KHk_k
.upd_KHk_k_done:
    # IKH[i][j] -= K*Hk[i][j]
    li      t3, 276
    mul     t4, t0, t3
    add     t4, t4, t2
    slli    t4, t4, 3
    add     t4, t6, t4
    fld     ft1, 0(t4)
    fsub.d  ft1, ft1, ft0
    fsd     ft1, 0(t4)
    addi    t2, t2, 1
    j       .upd_KHk_j
.upd_KHk_j_done:
    addi    t0, t0, 1
    j       .upd_KHk_i
.upd_KHk_done:

    # 11b. Compute IKH * P -> tmp1 (reuse PHkt_buf = s11, 276×276 needed)
    # But s11 is 276×69. We need 276×276. Reuse IKH_buf as tmp after we're done:
    # Store result in s11 (first 276*276 doubles). Caller must provide large enough buf.
    # According to function signature, PHkt_buf is 276*69. We need a 276*276 scratch.
    # We'll reuse S_buf (s5 = 69*69) -- too small.
    # We use h_pred_buf (s6 = 69*69) and h_meas_buf (s7 = 69*69) -- also too small.
    # DESIGN: For Joseph form, caller must supply IKH_buf of size 276*276.
    # We store IKH*P in the SECOND HALF of IKH_buf (offset 276*276 doubles):
    # This requires IKH_buf to be 2*276*276 doubles = 1,213,952 bytes.
    # The caller is responsible for allocating this. Documented in header.

    li      t0, 76176           # 276*276
    slli    t0, t0, 3
    add     t5, t6, t0          # t5 = IKH_buf + 276*276*8 (second half)

    # tmp1 = IKH * P  (276×276 * 276×276 = 276×276)
    li      a0, 0               # i
.upd_IKHP_i:
    li      t3, 276
    bge     a0, t3, .upd_IKHP_done
    li      a1, 0               # j
.upd_IKHP_j:
    li      t3, 276
    bge     a1, t3, .upd_IKHP_j_done
    fcvt.d.w ft0, zero
    li      a2, 0               # k
.upd_IKHP_k:
    li      t3, 276
    bge     a2, t3, .upd_IKHP_k_done
    # IKH[i][k]
    mul     t0, a0, t3
    add     t0, t0, a2
    slli    t0, t0, 3
    add     t0, t6, t0
    fld     ft1, 0(t0)
    # P[k][j]
    mul     t1, a2, t3
    add     t1, t1, a1
    slli    t1, t1, 3
    add     t1, s1, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    a2, a2, 1
    j       .upd_IKHP_k
.upd_IKHP_k_done:
    mul     t0, a0, t3
    add     t0, t0, a1
    slli    t0, t0, 3
    add     t0, t5, t0
    fsd     ft0, 0(t0)          # tmp1[i][j]
    addi    a1, a1, 1
    j       .upd_IKHP_j
.upd_IKHP_j_done:
    addi    a0, a0, 1
    j       .upd_IKHP_i
.upd_IKHP_done:

    # 11c. P = tmp1 * IKH^T  (276×276 * 276×276^T = 276×276)
    # Written directly into P (s1).
    li      a0, 0
.upd_PfinalA_i:
    li      t3, 276
    bge     a0, t3, .upd_PfinalA_done
    li      a1, 0
.upd_PfinalA_j:
    li      t3, 276
    bge     a1, t3, .upd_PfinalA_j_done
    fcvt.d.w ft0, zero
    li      a2, 0
.upd_PfinalA_k:
    li      t3, 276
    bge     a2, t3, .upd_PfinalA_k_done
    # tmp1[i][k]
    mul     t0, a0, t3
    add     t0, t0, a2
    slli    t0, t0, 3
    add     t0, t5, t0
    fld     ft1, 0(t0)
    # IKH^T[k][j] = IKH[j][k]
    mul     t1, a1, t3
    add     t1, t1, a2
    slli    t1, t1, 3
    add     t1, t6, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    a2, a2, 1
    j       .upd_PfinalA_k
.upd_PfinalA_k_done:
    mul     t0, a0, t3
    add     t0, t0, a1
    slli    t0, t0, 3
    add     t0, s1, t0
    fsd     ft0, 0(t0)
    addi    a1, a1, 1
    j       .upd_PfinalA_j
.upd_PfinalA_j_done:
    addi    a0, a0, 1
    j       .upd_PfinalA_i
.upd_PfinalA_done:

    # 11d. P += K * R * K^T
    # Compute KR = K * R  (276×69 * 69×69 = 276×69) -> PHkt_buf (s11)
    mv      a0, s4              # K (276×69)
    mv      a1, s2              # R (69×69)
    mv      a2, s11             # KR output (276×69)
    li      a3, 276
    li      a4, 69
    li      a5, 69
    call    ekf_mat_mul_MxNxK

    # P += KR * K^T  (276×69 * 69×276 = 276×276, result added to P)
    li      a0, 0               # i
.upd_KRKT_i:
    li      t3, 276
    bge     a0, t3, .upd_KRKT_done
    li      a1, 0
.upd_KRKT_j:
    li      t3, 276
    bge     a1, t3, .upd_KRKT_j_done
    fcvt.d.w ft0, zero
    li      a2, 0
.upd_KRKT_k:
    li      t3, 69
    bge     a2, t3, .upd_KRKT_k_done
    # KR[i][k]
    li      t4, 69
    mul     t0, a0, t4
    add     t0, t0, a2
    slli    t0, t0, 3
    add     t0, s11, t0
    fld     ft1, 0(t0)
    # K^T[k][j] = K[j][k]
    mul     t1, a1, t4
    add     t1, t1, a2
    slli    t1, t1, 3
    add     t1, s4, t1
    fld     ft2, 0(t1)
    fmadd.d ft0, ft1, ft2, ft0
    addi    a2, a2, 1
    j       .upd_KRKT_k
.upd_KRKT_k_done:
    li      t3, 276
    mul     t0, a0, t3
    add     t0, t0, a1
    slli    t0, t0, 3
    add     t0, s1, t0
    fld     ft1, 0(t0)
    fadd.d  ft1, ft1, ft0
    fsd     ft1, 0(t0)
    addi    a1, a1, 1
    j       .upd_KRKT_j
.upd_KRKT_j_done:
    addi    a0, a0, 1
    j       .upd_KRKT_i
.upd_KRKT_done:

.upd_done:
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
    ld      ra,  104(sp)
    addi    sp, sp, 112
    ret

# Silence unused label:
s0_i = s0

# =============================================================================
# End of ekf_asm.s
# =============================================================================
